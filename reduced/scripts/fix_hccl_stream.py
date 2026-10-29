from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/weight_transfer/hccl_engine.py')
s = p.read_text()
old = '''            # Use simple one-by-one broadcasting
            for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device="npu")
                self.model_update_group.broadcast(weight, src=0, stream=torch.npu.current_stream())
                self.model.load_weights([(name, weight)])
                del weight'''
new = '''            def received_weights():
                for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                    dtype = getattr(torch, dtype_name)
                    weight = torch.empty(shape, dtype=dtype, device="npu")
                    self.model_update_group.broadcast(weight, src=0, stream=torch.npu.current_stream())
                    yield name, weight

            self.model.load_weights(received_weights())'''
assert s.count(old) == 1
p.with_suffix('.py.codex_before').write_text(s)
p.write_text(s.replace(old, new))
