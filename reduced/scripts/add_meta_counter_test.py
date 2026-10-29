from pathlib import Path
p = Path('/vllm-workspace/vllm-ascend/tests/ut/models/test_hy_v4.py')
s = p.read_text()
s = s.replace('from contextlib import ExitStack', 'import inspect\n\nfrom contextlib import ExitStack')
s = s.replace('    record_metadata_for_reloading,', '    record_metadata_for_reloading,')
s = s.replace('from vllm_ascend.patch.worker import patch_hy_v4', '''from vllm.model_executor.model_loader.reload.meta import get_numel_loaded
from vllm_ascend.patch.worker import patch_hy4_reload
from vllm_ascend.patch.worker import patch_hy_v4''')
s += '''

def test_meta_expert_copy_is_counted_for_reload():
    """Meta copies must contribute to layerwise accounting and replay."""
    def copy_expert(self, expert_data, loaded_weight):
        expert_data.copy_(loaded_weight)
    copy = patch_hy4_reload._materialise_source(copy_expert)
    def loader(param, loaded_weight):
        copy(None, expert_data=param, loaded_weight=loaded_weight)
    bound = inspect.signature(loader).bind(
        torch.empty((2, 3), device="meta"), torch.ones((2, 3))
    )
    loaded, _ = get_numel_loaded(loader, bound)
    assert loaded == 6
'''
p.write_text(s)
