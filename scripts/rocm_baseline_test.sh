#!/bin/bash
set -e

echo "=== BUILDING rocm branch for gfx942 ==="
cd /data/kernels/nvdiffrast
git fetch origin rocm
git checkout -f rocm
git reset --hard origin/rocm
GPU_ARCHS=gfx942 pip install . --no-build-isolation 2>&1 | tail -5
echo "=== BUILD DONE ==="

export HIP_LAUNCH_BLOCKING=1
export HIP_VISIBLE_DEVICES=0

echo "=== TEST: interpolate (rocm branch on gfx942) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr
rast_out = torch.zeros(1, 256, 256, 4, dtype=torch.float32, device="cuda")
rast_out[0, 100:150, 100:150, 3] = 1.0
rast_out[0, 100:150, 100:150, 0] = 0.5
rast_out[0, 100:150, 100:150, 1] = 0.5
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
attrs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32, device="cuda")
print("calling interpolate...", flush=True)
out, _ = dr.interpolate(attrs, rast_out, tri)
print(f"interpolate PASS: {list(out.shape)}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST: texture (rocm branch on gfx942) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr
tex = torch.rand(1, 64, 64, 3, device="cuda")
uv = torch.rand(1, 16, 16, 2, device="cuda")
out = dr.texture(tex, uv)
print(f"texture PASS: {list(out.shape)}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== BASELINE DONE ==="
