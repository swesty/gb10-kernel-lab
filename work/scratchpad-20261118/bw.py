"""Achievable memory bandwidth on this GPU with simple streaming kernels."""
import time
import torch
x = torch.empty(2 * 1024**3 // 2, dtype=torch.bfloat16, device="cuda").normal_()  # 2 GiB
y = torch.empty(1024**3 // 4, dtype=torch.float32, device="cuda")
tests = (
    ("sum, read 2 GiB", lambda: x.sum(dtype=torch.float32), x.numel() * 2),
    ("amax, read 2 GiB", lambda: torch.amax(x), x.numel() * 2),
    ("copy 1 GiB (read+write)", lambda: y.copy_(x[: y.numel() * 2].view(torch.float32)), 2 * y.numel() * 4),
)
for name, fn, nbytes in tests:
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.time(); n = 10
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); dt = (time.time() - t) / n
    print(f"{name:28s} {nbytes / dt / 1e9:7.1f} GB/s", flush=True)
