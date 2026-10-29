from pathlib import Path
root = Path('/codex/scripts/skill-scripts/weight-update')
for original, target in (
    ('test_hccl_nopack_sleep.py', 'test_hccl_cpu_stage.py'),
    ('test_hccl_weight_update.py', 'test_hccl_packed_cpu_stage.py'),
):
    s = (root / original).read_text()
    needle = '        "shapes": [list(item.shape) for item in tensor_metadata],'
    assert s.count(needle) == 1
    s = s.replace(needle, needle + '\n        "cpu_staging": True,')
    (root / target).write_text(s)
p = Path('/vllm-workspace/vllm-ascend/tests/ut/distributed/weight_transfer/test_hccl_engine.py')
s = p.read_text().replace(
    '@pytest.mark.parametrize("packed", [False, True])\ndef test_transfer_validates_the_complete_checkpoint(packed):',
    '@pytest.mark.parametrize("packed", [False, True])\n@pytest.mark.parametrize("cpu_staging", [False, True])\ndef test_transfer_validates_the_complete_checkpoint(packed, cpu_staging):')
s = s.replace('        packed=packed,', '        packed=packed,\n        cpu_staging=cpu_staging,')
p.write_text(s)
p = Path('/vllm-workspace/vllm-ascend/tests/ut/distributed/weight_transfer/test_packed_tensor.py')
s = p.read_text().replace(
    '@pytest.mark.parametrize("num_buffers", [1, 2])\ndef test_packed_iterator_preserves_earlier_batches(num_buffers):',
    '@pytest.mark.parametrize("num_buffers", [1, 2])\n@pytest.mark.parametrize("cpu_staging", [False, True])\ndef test_packed_iterator_preserves_earlier_batches(num_buffers, cpu_staging):')
s = s.replace('            num_buffers=num_buffers,\n', '            num_buffers=num_buffers,\n            cpu_staging=cpu_staging,\n')
p.write_text(s)
