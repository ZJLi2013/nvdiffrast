// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <torch/extension.h>
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <pybind11/numpy.h>
#endif
#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#ifndef cudaLaunchKernel
#define cudaLaunchKernel hipLaunchKernel
#endif
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { hipError_t err = CUDA_CALL; TORCH_CHECK(!err, "HIP error: ", hipGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#else
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#endif
#endif

//------------------------------------------------------------------------
