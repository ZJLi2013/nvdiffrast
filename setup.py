# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os
import platform

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, IS_HIP_EXTENSION
except ImportError:
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast CUDA extension. Please ensure that:\n")
    print("1. You have PyTorch installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
IS_WINDOWS = platform.system() == "Windows"

if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
elif BUILD_TARGET == "cuda":
    IS_HIP = False
elif BUILD_TARGET == "rocm":
    IS_HIP = True
else:
    raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET}")

cxx_flags = ["-DNVDR_TORCH"]
nvcc_flags = ["-DNVDR_TORCH"]

if IS_WINDOWS:
    cxx_flags += ["/wd4067", "/wd4624", "/wd4996"]
    if not IS_HIP:
        nvcc_flags += ["-lineinfo"]
else:
    if not IS_HIP:
        nvcc_flags += ["-lineinfo"]

if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags += [f"--offload-arch={arch}" for arch in archs]

include_dirs = []
if IS_HIP:
    include_dirs += [
        os.path.join(os.path.dirname(__file__), "csrc", "common", "cudaraster", "impl"),
    ]

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=[
                "csrc/common/antialias.cu",
                "csrc/common/common.cpp",
                "csrc/common/cudaraster/impl/Buffer.cpp",
                "csrc/common/cudaraster/impl/CudaRaster.cpp",
                "csrc/common/cudaraster/impl/RasterImpl.cpp",
                "csrc/common/cudaraster/impl/RasterImpl_kernel.cu",
                "csrc/common/interpolate.cu",
                "csrc/common/rasterize.cu",
                "csrc/common/texture.cpp",
                "csrc/common/texture_kernel.cu",
                "csrc/torch/torch_antialias.cpp",
                "csrc/torch/torch_bindings.cpp",
                "csrc/torch/torch_interpolate.cpp",
                "csrc/torch/torch_rasterize.cpp",
                "csrc/torch/torch_texture.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
