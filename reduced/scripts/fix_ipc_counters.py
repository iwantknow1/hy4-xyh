from pathlib import Path
p=Path('/codex/src/vllm-ascend/vllm_ascend/distributed/weight_transfer/packed_tensor.py')
s=p.read_text().replace('''    it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
    copied into this buffer before yielding, so only one IPC-shared
    allocation exists for the lifetime of the transfer.''','''    it for IPC for each chunk via ``reduce_tensor``. Each send needs a fresh
    reference counter, even though all chunks reuse the same allocation.''')
s=s.replace('    # Store only the rebuild args (drop the func); the consumer rebuilds with\n    # the well-known ``rebuild_npu_tensor``, mirroring upstream\'s CUDA IPC engine.\n    _, ipc_args = reduce_tensor(ipc_buffer)\n','')
s=s.replace('            torch.npu.current_stream().synchronize()\n            yield {','            torch.npu.current_stream().synchronize()\n            _, ipc_args = reduce_tensor(ipc_buffer)\n            yield {',1)
s=s.replace('    if total_bytes:\n        torch.npu.current_stream().synchronize()\n        yield {','    if total_bytes:\n        torch.npu.current_stream().synchronize()\n        _, ipc_args = reduce_tensor(ipc_buffer)\n        yield {',1)
p.write_text(s)
