#!/bin/bash
set -e

echo "=== ENVIRONMENT ==="
python3 --version
python3 -c 'import torch; print(f"torch {torch.__version__}")'
hipcc --version 2>&1 | tail -2
rocminfo 2>/dev/null | grep -m1 'gfx' || true

echo "=== BUILDING nvdiffrast for gfx942 (clean build) ==="
cd /data/kernels/nvdiffrast
rm -rf build/ dist/ *.egg-info _nvdiffrast_c*.so
GPU_ARCHS=gfx942 pip install . --no-build-isolation 2>&1

echo "=== BUILD COMPLETE ==="

export HIP_LAUNCH_BLOCKING=1
export HIP_VISIBLE_DEVICES=0

echo "=== TEST 0: GPU sanity ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
print(f"torch {torch.__version__}", flush=True)
print(f"device count: {torch.cuda.device_count()}", flush=True)
print(f"device name: {torch.cuda.get_device_name(0)}", flush=True)
x = torch.randn(100, 100, device="cuda")
y = x @ x.T
print(f"matmul OK: {y.shape}", flush=True)
' && echo "PASS" || echo "FAIL"

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
import nvdiffrast.torch as dr
print("creating context...", flush=True)
glctx = dr.RasterizeCudaContext()
print("context created", flush=True)

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

print("calling rasterize (HIP_LAUNCH_BLOCKING=1)...", flush=True)
try:
    rast_out, rast_out_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
    torch.cuda.synchronize()
    print(f"rasterize: {list(rast_out.shape)}, non-zero pixels: {(rast_out[..., 3] > 0).sum().item()}", flush=True)
except Exception as e:
    print(f"rasterize EXCEPTION: {e}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 5: texture (no cudaraster) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr
tex = torch.rand(1, 64, 64, 3, dtype=torch.float32, device="cuda")
uv = torch.rand(1, 16, 16, 2, dtype=torch.float32, device="cuda")
print("calling texture...", flush=True)
tex_out = dr.texture(tex, uv)
print(f"texture: {list(tex_out.shape)}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 6: antialias forward + backward (wave64 ballot + match_any) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()
vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0).requires_grad_(True)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

print("rasterize...", flush=True)
rast_out, _ = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
color = torch.ones(1, 256, 256, 3, dtype=torch.float32, device="cuda")

print("antialias fwd...", flush=True)
aa_out = dr.antialias(color, rast_out, vertices, triangles)
torch.cuda.synchronize()
print(f"antialias fwd: {list(aa_out.shape)}", flush=True)

print("antialias bwd (gradient)...", flush=True)
loss = aa_out.sum()
loss.backward()
torch.cuda.synchronize()
grad = vertices.grad
print(f"antialias bwd OK, grad shape: {list(grad.shape)}, grad abs sum: {grad.abs().sum().item():.6f}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== TEST 7: full pipeline (rasterize + interpolate + antialias grad) ==="
python3 -c '
import torch, sys; sys.stdout.reconfigure(line_buffering=True)
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()
vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0).requires_grad_(True)
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
vertex_colors = torch.tensor([[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]], dtype=torch.float32, device="cuda")

print("rasterize...", flush=True)
rast_out, _ = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
print(f"rasterize: non-zero={(rast_out[...,3]>0).sum().item()}", flush=True)

print("interpolate...", flush=True)
color, _ = dr.interpolate(vertex_colors, rast_out, triangles)
print(f"interpolate: {list(color.shape)}", flush=True)

print("antialias...", flush=True)
aa_out = dr.antialias(color, rast_out, vertices, triangles)

print("backward...", flush=True)
loss = aa_out.sum()
loss.backward()
torch.cuda.synchronize()
print(f"full pipeline OK, grad abs sum: {vertices.grad.abs().sum().item():.6f}", flush=True)
' && echo "PASS" || echo "FAIL"

echo "=== ALL DONE ==="
