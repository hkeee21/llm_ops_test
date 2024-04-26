import torch
import torch.nn as nn

# from vllm._C import ops
from inficom import residual_rmsnorm

Z = [1, 2, 4, 8, 12, 16]
DIM  = [4096, 5120]

WARM = 10
FREQ = 100

def test(bs, dim): 
    x = torch.empty((bs, 1, dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    r = torch.empty((bs, 1, dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    out = torch.empty_like(x)
    weight = nn.Parameter(torch.empty((dim), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5))

    # ed_res, ed_out = residual_rmsnorm(r, x, rmsnorm_layer.weight)
    # ops.fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)

    for _ in range(WARM):
        residual_rmsnorm(r, x, weight)
        # ops.fused_add_rms_norm(x, r, weight, 1e-6)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(FREQ)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(FREQ)]
    for i in range(FREQ):
        start_event[i].record()
        residual_rmsnorm(r, x, weight)
        # ops.fused_add_rms_norm(x, r, weight, 1e-6)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    print(torch.mean(dur).item())

for d in DIM:
    for b in Z:
        test(b, d)

print('--------------------')

for d in DIM:
    for b in Z:
        test(b, d)