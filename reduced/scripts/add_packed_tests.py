from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py')
s = p.read_text()
s = s.replace('from vllm_ascend.distributed.weight_transfer.packed_tensor import (', '''from vllm_ascend.distributed.weight_transfer import packed_tensor as packed_module
from vllm_ascend.distributed.weight_transfer.packed_tensor import (''')
s += '''

@pytest.mark.parametrize("num_buffers", [1, 2])
def test_packed_iterator_preserves_earlier_batches(num_buffers):
    """Deferred loaders must retain intact tensors across packed batches."""
    tensors = [(str(i), torch.arange(4, dtype=torch.float32) + i) for i in range(4)]
    payloads = [torch.cat([t.view(torch.uint8) for _, t in tensors[i:i + 2]])
                for i in (0, 2)]
    pending = iter(payloads)
    group = _make_group_mock()
    group.broadcast.side_effect = lambda target, **kw: target.copy_(next(pending))
    received = list(packed_module.packed_broadcast_iterator(
        iterator=iter((name, (list(t.shape), t.dtype)) for name, t in tensors),
        group=group, src=0, buffer_size_bytes=16, num_buffers=num_buffers,
    ))
    assert [name for name, _ in received] == [name for name, _ in tensors]
    for (_, actual), (_, expected) in zip(received, tensors):
        torch.testing.assert_close(actual, expected)
'''
p.with_suffix('.py.codex_before').write_text(p.read_text())
p.write_text(s)
p = Path('/vllm-workspace/vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py')
s = p.read_text().replace('from types import SimpleNamespace', 'from contextlib import nullcontext\nfrom types import SimpleNamespace')
s = s.replace('import torch', 'import pytest\nimport torch')
s = s.replace('def test_unpacked_transfer_validates_the_complete_checkpoint():', '''@pytest.mark.parametrize("packed", [False, True])
def test_transfer_validates_the_complete_checkpoint(packed):''')
s = s.replace('        packed=False,', '        packed=packed,\n        packed_buffer_size_bytes=1,\n        packed_num_buffers=1,')
s = s.replace('    ), patch("torch.npu.current_stream"):', '''    ), patch("torch.npu.current_stream"), patch("torch.npu.Stream"), patch(
        "torch.npu.stream", side_effect=lambda stream: nullcontext()
    ):''')
p.write_text(s)
