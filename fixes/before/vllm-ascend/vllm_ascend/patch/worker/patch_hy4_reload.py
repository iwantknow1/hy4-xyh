# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HY4 reload / weight-update compatibility patches (vllm-ascend side only).

Three defects surface on Ascend for HY4 when a live weight update runs
(`POST /collective_rpc reload_weights`) or when an HCCL weight transfer lands.
All of them are worked around here so that no upstream (vLLM) source file has to
be modified:

1. `ExpertMapManager.map_global_to_local` reads the device expert map with
   `self._expert_map[global_id].item()`. On CANN 9.1.0 / torch_npu 2.10.0.post4
   that device->host scalar read on an offset 0-dim view traps
   (`LocalScalarDenseNpu`, `Invalid_Argument(EL0003)`, aivec trap), and it runs
   once per expert inside `finalize_layerwise_reload` (thousands of syncs).
   Fix: materialise the map once on the host and index a Python list.

2. `RoutedExperts._load_w2` / `_load_w13` copy straight out of a `narrow()` view
   of the safetensors mmap. Materialise the source slice into an independent
   contiguous host buffer first so the H2D copy uses a single mapping.

3. HY4's attention sinks are loaded by `HYV4ForCausalLM.load_weights` through a
   dedicated branch that writes `params_dict[name][:n].copy_(narrow_weight)` and
   `continue`s, i.e. it bypasses `param.weight_loader`. A live update runs inside
   vLLM's layerwise reload, which parks the layer on the meta device and only
   replays loads performed through `weight_loader`; the direct `copy_` therefore
   leaves `learnable_sink_param` on the meta device (`Failed to load weights` in
   the layerwise log) and later kernels dereference it. Same defect class as the
   `attn_sink` fix in vllm-ascend PR #18491 for DeepSeek-V4.1.
   Fix: intercept sink names, replicate the head sharding, and apply them through
   the parameter's loader (which is the layerwise-wrapped one during a reload).

All patches are idempotent, only wrap existing behaviour, and do not change any
numerical result. Drop them once the equivalent fixes land upstream.
"""

from __future__ import annotations

import functools

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# 1. Expert map: cache on host, never do a device scalar read.
# ---------------------------------------------------------------------------
def _patch_expert_map_manager() -> None:
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    if getattr(ExpertMapManager, "_hy4_host_map_patched", False):
        return

    def _host_expert_map(self) -> list[int]:
        key = (id(self._expert_map), self._expert_map.data_ptr())
        if getattr(self, "_hy4_expert_map_host_key", None) != key:
            self._hy4_expert_map_host = self._expert_map.tolist()
            self._hy4_expert_map_host_key = key
        return self._hy4_expert_map_host

    original_map_global_to_local = ExpertMapManager.map_global_to_local
    original_is_local_expert = ExpertMapManager.is_local_expert

    @functools.wraps(original_map_global_to_local)
    def map_global_to_local(self, global_id: int) -> int:
        if self._expert_map is None:
            return global_id
        return _host_expert_map(self)[global_id]

    @functools.wraps(original_is_local_expert)
    def is_local_expert(self, global_id: int) -> bool:
        if self._expert_map is None:
            return True
        return _host_expert_map(self)[global_id] != -1

    ExpertMapManager.map_global_to_local = map_global_to_local
    ExpertMapManager.is_local_expert = is_local_expert
    ExpertMapManager._hy4_host_map_patched = True
    logger.info("hy4 patch: ExpertMapManager uses a host-cached expert map")


# ---------------------------------------------------------------------------
# 2. Expert weight copies: materialise the checkpoint slice first.
# ---------------------------------------------------------------------------
def _materialise_source(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # _skip_meta_copy: during a layerwise reload the destination parameter may
        # still be parked on the meta device (no storage). Copying into it would
        # issue a device memcpy against an address that does not exist, which is
        # exactly the aclrtMemcpy 507899 trap. The layerwise replay performs the
        # real write once the layer is materialised, so skip it here.
        dst = args[0] if args else kwargs.get("expert_data")
        if isinstance(dst, torch.Tensor) and dst.is_meta:
            return None
        if len(args) >= 3 and isinstance(args[2], torch.Tensor):
            args = args[:2] + (args[2].clone(),) + args[3:]
        elif isinstance(kwargs.get("loaded_weight"), torch.Tensor):
            kwargs["loaded_weight"] = kwargs["loaded_weight"].clone()
        return fn(self, *args, **kwargs)

    return wrapper


def _patch_routed_experts_copy() -> None:
    from vllm.model_executor.layers.fused_moe import routed_experts

    cls = routed_experts.RoutedExperts
    if getattr(cls, "_hy4_copy_patched", False):
        return

    for name in ("_load_w2", "_load_w13"):
        fn = getattr(cls, name, None)
        if fn is None:
            logger.warning("hy4 patch: %s.%s not found, skipped", cls.__name__, name)
            continue
        setattr(cls, name, _materialise_source(fn))

    cls._hy4_copy_patched = True
    logger.info("hy4 patch: RoutedExperts weight copies materialise the source slice")


# ---------------------------------------------------------------------------
# 3. Attention sinks: load through `weight_loader` so layerwise can replay them.
# ---------------------------------------------------------------------------
def _patch_hy4_sink_loader() -> None:
    try:
        from vllm.models.hy_v4.nvidia.model import HYV4ForCausalLM
    except Exception as exc:  # pragma: no cover - model not in this build
        logger.warning("hy4 patch: HYV4ForCausalLM unavailable (%s), skipped", exc)
        return

    if getattr(HYV4ForCausalLM, "_hy4_sink_patched", False):
        return

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    original_load_weights = HYV4ForCausalLM.load_weights

    @functools.wraps(original_load_weights)
    def load_weights(self, weights):
        deferred: list[tuple[str, torch.Tensor]] = []

        def forwarded():
            for name, weight in weights:
                if "learnable_sink_param" in name:
                    deferred.append((name, weight))
                    continue
                yield name, weight

        loaded = original_load_weights(self, forwarded())
        if not deferred:
            return loaded

        params = dict(self.named_parameters())
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = getattr(self.config, "num_attention_heads", 0) // max(tp_size, 1)
        start = n_local_head * tp_rank
        end = n_local_head * (tp_rank + 1)

        applied = set()
        for name, weight in deferred:
            param = params.get(name)
            if param is None:
                continue
            slice_weight = weight[start:end] if weight.ndim else weight
            loader = getattr(param, "weight_loader", default_weight_loader)
            # During a reload this is the layerwise-wrapped loader, so the write is
            # recorded and replayed after the layer is materialised.
            loader(param, slice_weight)
            applied.add(name)

        if isinstance(loaded, set):
            loaded.update(applied)
        return loaded

    HYV4ForCausalLM.load_weights = load_weights
    HYV4ForCausalLM._hy4_sink_patched = True
    logger.info("hy4 patch: HY4 attention sinks load through weight_loader")


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_expert_map_manager()
    _patch_routed_experts_copy()
    _patch_hy4_sink_loader()
    _PATCHED = True


apply()
