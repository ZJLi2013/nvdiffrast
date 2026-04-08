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

echo "=== TESTING ==="
python3 -c '
import torch
import nvdiffrast.torch as dr

print(f"nvdiffrast imported OK")
print(f"torch: {torch.__version__}, hip: {torch.version.hip}, devices: {torch.cuda.device_count()}")

glctx = dr.RasterizeCudaContext()
print("RasterizeCudaContext created OK")

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)

triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

rast_out, rast_out_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
print(f"rasterize:   {list(rast_out.shape)}, non-zero pixels: {(rast_out[..., 3] > 0).sum().item()}")

attrs = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)

interp_out, interp_out_db = dr.interpolate(attrs, rast_out, triangles)
print(f"interpolate: {list(interp_out.shape)}")

aa_out = dr.antialias(interp_out, rast_out, vertices, triangles)
print(f"antialias:   {list(aa_out.shape)}")

tex = torch.rand(1, 64, 64, 3, dtype=torch.float32, device="cuda")
uv = interp_out[..., :2]
tex_out = dr.texture(tex, uv)
print(f"texture:     {list(tex_out.shape)}")

print("ALL TESTS PASSED")
'

echo "=== ALL DONE ==="
