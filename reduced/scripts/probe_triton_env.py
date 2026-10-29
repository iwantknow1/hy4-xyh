import json,os,shutil
from vllm.triton_utils import HAS_TRITON
from vllm_ascend.ops.triton.compute_slot_mapping import _compute_slot_mapping_kernel as kernel
print(json.dumps({'bisheng':shutil.which('bisheng'),'HAS_TRITON':HAS_TRITON,'kernel_type':str(type(kernel)),'launch_indexing_available':hasattr(kernel,'__getitem__')}))
