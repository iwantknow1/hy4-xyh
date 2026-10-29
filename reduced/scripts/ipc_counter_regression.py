from pathlib import Path
p=Path('/codex/src/vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py')
s=p.read_text(); begin=s.index('def test_packed_npu_ipc_producer_splits_when_exceeding_buffer():'); end=s.index('\ndef test_packed_npu_ipc_producer_raises',begin)
s=s[:begin]+'''def test_packed_npu_ipc_producer_splits_when_exceeding_buffer():
    """Reuse storage but export a fresh IPC reference counter for each send."""
    exported = []

    def export(buffer):
        handle = (buffer.data_ptr(), len(exported))
        exported.append(handle)
        return "rebuild_func", handle

    with patch(f"{_MODULE}.reduce_tensor", side_effect=export):
        chunks = list(packed_npu_ipc_producer(
            iterator=iter([("a", torch.ones(4)), ("b", torch.full((4,), 2.0))]),
            npu_uuid="node-0", post_iter_func=lambda item: item[1], buffer_size_bytes=20,
        ))
    assert [chunk["names"] for chunk in chunks] == [["a"], ["b"]]
    handles = [chunk["ipc_handle"]["node-0"] for chunk in chunks]
    assert handles[0][0] == handles[1][0]
    assert handles[0][1] != handles[1][1]

''' + s[end:]
p.write_text(s)
