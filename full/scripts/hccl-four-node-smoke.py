import argparse
import datetime
import json
import torch
import torch_npu
import torch.distributed as dist

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--master',required=True)
    p.add_argument('--world',type=int,default=4)
    p.add_argument('--port',type=int,default=29690)
    a=p.parse_args()
    torch.npu.set_device(0)
    dist.init_process_group('hccl',init_method=f'tcp://{a.master}:{a.port}',rank=a.rank,world_size=a.world,timeout=datetime.timedelta(seconds=180))
    x=torch.full((4096,),a.rank+1,dtype=torch.float32,device='npu')
    dist.all_reduce(x)
    torch.npu.synchronize()
    assert torch.equal(x.cpu(),torch.full((4096,),float(a.world*(a.world+1)//2)))
    y=torch.full((4096,),7.0 if a.rank==0 else 0.0,device='npu')
    dist.broadcast(y,0)
    torch.npu.synchronize()
    assert torch.equal(y.cpu(),torch.full((4096,),7.0))
    dist.barrier()
    print(json.dumps({'rank':a.rank,'status':'passed','all_reduce_elements':4096,'broadcast_elements':4096}),flush=True)
    dist.destroy_process_group()
