#!/bin/bash
set -euo pipefail

IMG="rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1"

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add render \
    --group-add video \
    --shm-size=16g \
    "$IMG" \
    bash -c '
set -euo pipefail

cd /tmp
git clone --branch rocm --depth 1 https://github.com/ZJLi2013/nvdiffrast.git 2>/dev/null
cd nvdiffrast
GPU_ARCHS=gfx1201 pip install . --no-build-isolation >/dev/null 2>&1
echo "Build OK"
cd /tmp

python3 << PYEOF
import torch
import nvdiffrast.torch as dr

print("nvdiffrast imported OK")
print(f"torch: {torch.__version__}, hip: {torch.version.hip}, devices: {torch.cuda.device_count()}")

glctx = dr.RasterizeCudaContext()
print("RasterizeCudaContext created OK")

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)

triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")

rast_out, rast_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
n_pixels = (rast_out[..., 3] > 0).sum().item()
print(f"rasterize: {rast_out.shape}, non-zero pixels: {n_pixels}")
assert n_pixels > 0, "No pixels rasterized!"

attrs = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=torch.float32, device="cuda").unsqueeze(0)

interp_out, interp_db = dr.interpolate(attrs, rast_out, triangles)
print(f"interpolate: {interp_out.shape}")
assert interp_out.shape == (1, 256, 256, 3)

aa_out = dr.antialias(interp_out, rast_out, vertices, triangles)
print(f"antialias: {aa_out.shape}")
assert aa_out.shape == (1, 256, 256, 3)

texc = torch.rand(1, 256, 256, 2, device="cuda")
tex = torch.rand(1, 16, 16, 3, device="cuda")
tex_out = dr.texture(tex, texc)
print(f"texture: {tex_out.shape}")
assert tex_out.shape == (1, 256, 256, 3)

print("ALL TESTS PASSED")
PYEOF
'
