#!/bin/bash
set -e

echo "=== ENVIRONMENT ==="
python3 --version
python3 -c 'import torch; print(f"torch {torch.__version__}")'
hipcc --version 2>&1 | tail -2
rocminfo 2>/dev/null | grep -m1 'gfx' || true

echo "=== BUILDING nvdiffrast for gfx942 ==="
cd /data/kernels/nvdiffrast
GPU_ARCHS=gfx942 pip install . --no-build-isolation 2>&1

echo "=== BUILD COMPLETE ==="

echo "=== TEST 1: import ==="
python3 -c '
import torch, sys
print(f"torch {torch.__version__}, hip={torch.version.hip}, devices={torch.cuda.device_count()}", flush=True)
import nvdiffrast.torch as dr
print("nvdiffrast imported OK", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 2: RasterizeCudaContext ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr
print("creating context...", flush=True)
glctx = dr.RasterizeCudaContext()
print("RasterizeCudaContext created OK", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 3: interpolate (no cudaraster) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr

rast_out = torch.zeros(1, 256, 256, 4, dtype=torch.float32, device="cuda")
rast_out[0, 100:150, 100:150, 3] = 1.0
rast_out[0, 100:150, 100:150, 0] = 0.5
rast_out[0, 100:150, 100:150, 1] = 0.5
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
attrs = torch.tensor([[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]], dtype=torch.float32, device="cuda")
print("calling interpolate...", flush=True)
interp_out, _ = dr.interpolate(attrs, rast_out, triangles)
print(f"interpolate: {list(interp_out.shape)}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 4: rasterize (cudaraster - wave64 critical) ==="
python3 -c '
import torch, sys, os; sys.stdout.reconfigure(line_buffering=True)
os.environ["HIP_VISIBLE_DEVICES"] = "0"
import nvdiffrast.torch as dr
print("creating context...", flush=True)
glctx = dr.RasterizeCudaContext()
print("context created, calling rasterize...", flush=True)

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

rast_out, rast_out_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
print(f"rasterize: {list(rast_out.shape)}, non-zero pixels: {(rast_out[..., 3] > 0).sum().item()}", flush=True)
print("rasterize PASS", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== ALL DONE ==="
