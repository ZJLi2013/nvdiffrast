## Nvdiffrast &ndash; Modular Primitives for High-Performance Differentiable Rendering

> **Note:** This is an **unofficial fork** with experimental ROCm/HIP support for AMD GPUs.
> It is intended **solely for non-commercial research purposes**.
> The original project and all source code are licensed under the
> [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).
> Please refer to the [upstream repository](https://github.com/NVlabs/nvdiffrast) for the
> official release.

![Teaser image](./docs/img/teaser.png)

**Modular Primitives for High-Performance Differentiable Rendering**<br>
Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, Timo Aila<br>
[http://arxiv.org/abs/2011.03277](http://arxiv.org/abs/2011.03277)

Nvdiffrast is a PyTorch library that provides high-performance primitive operations for rasterization-based differentiable rendering.

### Install (NVIDIA GPU &ndash; original)
```
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```

### Install (AMD GPU &ndash; ROCm, this fork)
```bash
# Requires ROCm 7.x and PyTorch with ROCm support
# Set GPU_ARCHS to your target: gfx1100 (RDNA3) or gfx1201 (RDNA4)
GPU_ARCHS=gfx1201 pip install . --no-build-isolation
```

**Supported AMD architectures:**
| Architecture | GPU examples | Wave size | Status |
|---|---|---|---|
| RDNA3 (gfx1100) | RX 7900 XTX, W7900 | wave32 | Supported |
| RDNA4 (gfx1201) | Radeon AI PRO R9700 | wave32 | Tested ✓ |
| CDNA3 (gfx942) | MI300X | wave64 | Not yet supported |

See &#x261E;&#x261E; [nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast) &#x261C;&#x261C; for more information.

## Licenses

Copyright &copy; 2020&ndash;2025, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).
**This is NOT an open-source license.** Commercial use is prohibited without explicit authorization from NVIDIA.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

The upstream repository does not accept outside code contributions.
This fork exists solely to enable AMD GPU compatibility for academic research.

Environment map stored as part of `samples/data/envphong.npz` is derived from a Wave Engine
[sample material](https://github.com/WaveEngine/Samples-2.5/tree/master/Materials/EnvironmentMap/Content/Assets/CubeMap.cubemap)
originally shared under 
[MIT License](https://github.com/WaveEngine/Samples-2.5/blob/master/LICENSE.md).
Mesh and texture stored as part of `samples/data/earth.npz` are derived from
[3D Earth Photorealistic 2K](https://www.turbosquid.com/3d-models/3d-realistic-earth-photorealistic-2k-1279125)
model originally made available under
[TurboSquid 3D Model License](https://blog.turbosquid.com/turbosquid-3d-model-license/#3d-model-license).

## Citation

```
@article{Laine2020diffrast,
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
  journal = {ACM Transactions on Graphics},
  year    = {2020},
  volume  = {39},
  number  = {6}
}
```
