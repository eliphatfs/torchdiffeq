import torch

def old_way(t, step_size):
    start_time = t[0]
    end_time = t[-1]
    niters = torch.ceil((end_time - start_time) / step_size + 1).item()
    t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
    t_infer[-1] = t[-1]
    return t_infer

def new_way(t, step_size):
    start_time = t[0]
    end_time = t[-1]
    niters = torch.ceil((end_time - start_time) / step_size + 1).item()
    t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
    t_infer = t_infer.select_scatter(t[-1], 0, -1)
    return t_infer

t = torch.tensor([0.0, 1.0])
step_size = 0.3
res1 = old_way(t, step_size)
res2 = new_way(t, step_size)

print(f"Old: {res1}")
print(f"New: {res2}")
print(f"Equal: {torch.allclose(res1, res2)}")
