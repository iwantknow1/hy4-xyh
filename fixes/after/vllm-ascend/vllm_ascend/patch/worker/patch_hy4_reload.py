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
"""Cache expert maps on CPU and materialize expert copy sources for HY4.

Meta-device copies must execute so layerwise reload can count loaded elements.
Sink parameters use the native HYV4Model weight loader.
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
        # Meta copies must still contribute to layerwise loading counts.
        dst = args[0] if args else kwargs.get("expert_data")
        if isinstance(dst, torch.Tensor) and dst.is_meta:
            return fn(self, *args, **kwargs)
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
# Sink loading is handled by the native HYV4Model weight loader.
# ---------------------------------------------------------------------------


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_expert_map_manager()
    _patch_routed_experts_copy()
    _PATCHED = True


apply()
