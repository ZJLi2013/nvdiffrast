#!/bin/bash
# nvdiffrast ROCm build test on RDNA4 (gfx1201) with rocm/pytorch Docker
set -euo pipefail

DOCKER_IMAGE="rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1"
REPO_URL="https://github.com/ZJLi2013/nvdiffrast.git"
BRANCH="rocm"

echo "=== Step 1: Pull Docker image ==="
docker pull "$DOCKER_IMAGE" 2>&1 | tail -3

echo ""
echo "=== Step 2: Build & Test inside Docker ==="
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add render \
    --group-add video \
    --shm-size=16g \
    "$DOCKER_IMAGE" \
    bash -c "
set -euo pipefail

echo '--- Environment ---'
python3 -c 'import torch; print(\"torch:\", torch.__version__, \"hip:\", torch.version.hip, \"cuda:\", torch.cuda.is_available(), \"devices:\", torch.cuda.device_count())'
hipcc --version | head -2
rocminfo | grep -E 'gfx|Marketing' | head -6

echo ''
echo '--- Clone repo ---'
cd /tmp
git clone --branch $BRANCH --depth 1 $REPO_URL nvdiffrast
cd nvdiffrast

echo ''
echo '--- Build with hipcc ---'
export GPU_ARCHS=gfx1201
pip install . --no-build-isolation 2>&1 | tail -30

echo ''
echo '--- Test import ---'
cd /tmp
python3 -c '
import nvdiffrast.torch as dr
print(\"nvdiffrast imported OK\")
glctx = dr.RasterizeCudaContext()
print(\"RasterizeCudaContext created OK\")
'

echo ''
echo '--- Functional test ---'
python3 -c '
import torch
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device=\"cuda\").unsqueeze(0)

triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=\"cuda\")

rast_out, rast_out_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
print(f\"rasterize: {rast_out.shape}, non-zero pixels: {(rast_out[..., 3] > 0).sum()}\")

attrs = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=torch.float32, device=\"cuda\").unsqueeze(0)

interp_out, interp_out_db = dr.interpolate(attrs, rast_out, triangles)
print(f\"interpolate: {interp_out.shape}\")

aa_out = dr.antialias(interp_out, rast_out, vertices, triangles)
print(f\"antialias: {aa_out.shape}\")

print(\"ALL TESTS PASSED\")
'

echo '=== DONE ==='
"
