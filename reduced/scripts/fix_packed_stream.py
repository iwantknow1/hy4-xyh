from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/packed_tensor.py')
s = p.read_text()
p.with_suffix('.py.codex_before').write_text(s)
start = s.index('def packed_broadcast_consumer(')
body_start = s.index('    def unpack_tensor(', start)
end = s.index('# \u2500', body_start)
header = s[start:body_start]
body = s[body_start:end]
old = '''            post_unpack_func(
                unpack_tensor(
                    packed_tensors[buffer_idx],
                    list(names),
                    list(shapes),
                    list(dtypes),
                    list(tensor_sizes),
                )
            )

        # Move to the next buffer'''
new = '''            batch = unpack_tensor(
                packed_tensors[buffer_idx],
                list(names),
                list(shapes),
                list(dtypes),
                list(tensor_sizes),
            )
            if post_unpack_func is not None:
                post_unpack_func(batch)

        if post_unpack_func is None:
            streams[buffer_idx].synchronize()
            if cpu_staging:
                batch = [(name, tensor.cpu()) for name, tensor in batch]
                packed_tensors[buffer_idx] = torch.empty(0, dtype=torch.uint8, device="npu")
            yield batch

        # Move to the next buffer'''
assert body.count(old) == 1
body = body.replace(old, new)
helper_header = header.replace('def packed_broadcast_consumer(', 'def _packed_broadcast_batches(')
helper_header = helper_header.replace(
    'post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],',
    'post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None] | None,')
helper_header = helper_header.replace(
    '    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,\n) -> None:',
    '    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,\n    cpu_staging: bool = False,\n) -> Iterator[list[tuple[str, torch.Tensor]]]:')
wrapper = header + '''    for _ in _packed_broadcast_batches(
        iterator, group, src, post_unpack_func, buffer_size_bytes, num_buffers
    ):
        pass


'''
iterator = '''def packed_broadcast_iterator(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    cpu_staging: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield a complete weight stream across received packed batches."""
    for batch in _packed_broadcast_batches(
        iterator, group, src, None, buffer_size_bytes, num_buffers, cpu_staging
    ):
        yield from batch


'''
s = s[:start] + wrapper + iterator + helper_header + body + s[end:]
# Make the stream used by HCCL match the stream synchronized around it.
producer_end = s.index('def packed_broadcast_consumer(')
producer = s[:producer_end]
producer = producer.replace(
    '    streams = [torch.npu.Stream() for _ in range(num_buffers)]\n',
    '    streams = [torch.npu.Stream() for _ in range(num_buffers)]\n    default_stream = torch.npu.current_stream()\n')
producer = producer.replace('group.broadcast(packed_tensors[buffer_idx], src=src)',
                            'group.broadcast(packed_tensors[buffer_idx], src=src, stream=default_stream)')
s = producer + s[producer_end:]
helper_start = s.index('def _packed_broadcast_batches(')
s = s[:helper_start] + s[helper_start:].replace(
    'group.broadcast(packed_tensors[buffer_idx], src=src)',
    'group.broadcast(packed_tensors[buffer_idx], src=src, stream=default_stream)', 1)
p.write_text(s)
p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py')
s = p.read_text()
s = s.replace('    packed_broadcast_consumer,', '    packed_broadcast_iterator,')
old = '''            packed_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self.model.load_weights,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
            )'''
new = '''            self.model.load_weights(packed_broadcast_iterator(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
                cpu_staging=update_info.cpu_staging,
            ))'''
assert s.count(old) == 1
s = s.replace(old, new)
needle = 'class HCCLWeightTransferUpdateInfo'
start = s.index(needle)
field_pos = s.index('    def ', start)
# Add an optional backend-specific receive setting before validation.
s = s[:field_pos] + '''    cpu_staging: bool = False
    """Stage received tensors on CPU during deferred checkpoint loading."""

''' + s[field_pos:]
s = s.replace('                    yield name, weight', '''                    if update_info.cpu_staging:
                        staged_weight = weight.cpu()
                        del weight
                        yield name, staged_weight
                    else:
                        yield name, weight''')
p.write_text(s)
