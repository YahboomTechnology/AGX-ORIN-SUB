import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda"
size = 2048
iterations = 100

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# 先预热，减少首次初始化对后续观察的影响。
for _ in range(10):
    result = torch.mm(a, b)
torch.cuda.synchronize()

# 重复计算，让时间轴上出现连续的 GPU Kernel。
start = time.perf_counter()
for _ in range(iterations):
    result = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Matrix size: {size} x {size}")
print(f"Iterations: {iterations}")
print(f"Total elapsed time: {elapsed:.3f} seconds")
print(f"Result verification value: {result[0, 0].item():.6f}")
