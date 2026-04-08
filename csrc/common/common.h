// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#if defined(__HIP_PLATFORM_AMD__)
#   ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#       define HIP_ENABLE_WARP_SYNC_BUILTINS
#   endif
#   include <hip/hip_runtime.h>
#else
#   include <cuda.h>
#endif

#include <stdint.h>

//------------------------------------------------------------------------
// ROCm warp-sync compatibility.
// RDNA (wave32): cast mask to 64-bit.
// CDNA (wave64): half-wavefront emulation via __lane_id().

#if defined(__HIP_PLATFORM_AMD__) && (defined(__CUDACC__) || defined(__HIPCC__))
#ifndef NVDR_HIP_WARP_COMPAT_DEFINED
#define NVDR_HIP_WARP_COMPAT_DEFINED

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define NVDR_WAVE64 1
#endif

#if defined(NVDR_WAVE64)

namespace _nvdr_hip_warp {

static __device__ __forceinline__ int _half()
{ return __lane_id() >> 5; }

static __device__ __forceinline__ unsigned int ballot_sync(unsigned int mask, int pred)
{
    unsigned long long full = __builtin_amdgcn_ballot_w64(pred != 0);
    return ((unsigned int)(full >> (_half() * 32))) & mask;
}

static __device__ __forceinline__ bool all_sync(unsigned int mask, int pred)
{
    unsigned int b = ballot_sync(~0u, pred);
    return (b & mask) == mask;
}

static __device__ __forceinline__ bool any_sync(unsigned int mask, int pred)
{
    return (ballot_sync(~0u, pred) & mask) != 0;
}

static __device__ __forceinline__ unsigned int match_any_sync(unsigned int mask, unsigned int val)
{
    unsigned int result = ~0u;
    for (int bit = 0; bit < 32; ++bit)
    {
        int b = (val >> bit) & 1;
        unsigned int m = ballot_sync(~0u, b);
        result &= b ? m : ~m;
    }
    return result & mask;
}

static __device__ __forceinline__ void syncwarp_nomask()
{ __builtin_amdgcn_wave_barrier(); }

static __device__ __forceinline__ void syncwarp_mask(unsigned int /*mask*/)
{ __builtin_amdgcn_wave_barrier(); }

}

#else // wave32 (RDNA) — also used in hipcc host phase when __gfx942__ is not yet defined

namespace _nvdr_hip_warp {
static __device__ __forceinline__ unsigned int ballot_sync(unsigned int mask, int pred)
{
#ifdef __HIP_DEVICE_COMPILE__
    return (unsigned int)__ballot_sync((unsigned long long)mask, pred);
#else
    return 0;
#endif
}
static __device__ __forceinline__ bool all_sync(unsigned int mask, int pred)
{
#ifdef __HIP_DEVICE_COMPILE__
    return __all_sync((unsigned long long)mask, pred);
#else
    return false;
#endif
}
static __device__ __forceinline__ bool any_sync(unsigned int mask, int pred)
{
#ifdef __HIP_DEVICE_COMPILE__
    return __any_sync((unsigned long long)mask, pred);
#else
    return false;
#endif
}
static __device__ __forceinline__ unsigned int match_any_sync(unsigned int mask, unsigned int val)
{
#ifdef __HIP_DEVICE_COMPILE__
    return (unsigned int)__match_any_sync((unsigned long long)mask, val);
#else
    return 0;
#endif
}
static __device__ __forceinline__ void syncwarp_nomask()
{
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_amdgcn_wave_barrier();
#endif
}
static __device__ __forceinline__ void syncwarp_mask(unsigned int /*mask*/)
{
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_amdgcn_wave_barrier();
#endif
}
}

#endif // NVDR_WAVE64

#define __ballot_sync(mask, pred)       _nvdr_hip_warp::ballot_sync((unsigned int)(mask), (int)(pred))
#define __all_sync(mask, pred)          _nvdr_hip_warp::all_sync((unsigned int)(mask), (int)(pred))
#define __any_sync(mask, pred)          _nvdr_hip_warp::any_sync((unsigned int)(mask), (int)(pred))
#define __match_any_sync(mask, val)     _nvdr_hip_warp::match_any_sync((unsigned int)(mask), (val))
#define _NVDR_SW_SEL(_0, _1, N, ...) N
#define __syncwarp(...) _NVDR_SW_SEL(dummy, ##__VA_ARGS__, _nvdr_hip_warp::syncwarp_mask, _nvdr_hip_warp::syncwarp_nomask)(__VA_ARGS__)
#endif
#endif

//------------------------------------------------------------------------
// C++ helper function prototypes.

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, int width, int height);
dim3 getLaunchGridSize(dim3 blockSize, int width, int height, int depth);

//------------------------------------------------------------------------
// The rest is CUDA device code specific stuff.

#if defined(__CUDACC__) || defined(__HIPCC__)

//------------------------------------------------------------------------
// Helpers for CUDA vector types.

static __device__ __forceinline__ float2&   operator*=  (float2& a, const float2& b)       { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ float2&   operator+=  (float2& a, const float2& b)       { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ float2&   operator-=  (float2& a, const float2& b)       { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ float2&   operator*=  (float2& a, float b)               { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ float2&   operator+=  (float2& a, float b)               { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ float2&   operator-=  (float2& a, float b)               { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ float2    operator*   (const float2& a, float b)         { return make_float2(a.x * b, a.y * b); }
static __device__ __forceinline__ float2    operator+   (const float2& a, float b)         { return make_float2(a.x + b, a.y + b); }
static __device__ __forceinline__ float2    operator-   (const float2& a, float b)         { return make_float2(a.x - b, a.y - b); }
static __device__ __forceinline__ float2    operator*   (float a, const float2& b)         { return make_float2(a * b.x, a * b.y); }
static __device__ __forceinline__ float2    operator+   (float a, const float2& b)         { return make_float2(a + b.x, a + b.y); }
static __device__ __forceinline__ float2    operator-   (float a, const float2& b)         { return make_float2(a - b.x, a - b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a)                  { return make_float2(-a.x, -a.y); }
static __device__ __forceinline__ float3&   operator*=  (float3& a, const float3& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ float3&   operator+=  (float3& a, const float3& b)       { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ float3&   operator-=  (float3& a, const float3& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ float3&   operator*=  (float3& a, float b)               { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ float3&   operator+=  (float3& a, float b)               { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ float3&   operator-=  (float3& a, float b)               { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ float3    operator*   (const float3& a, float b)         { return make_float3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ float3    operator+   (const float3& a, float b)         { return make_float3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ float3    operator-   (const float3& a, float b)         { return make_float3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ float3    operator*   (float a, const float3& b)         { return make_float3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ float3    operator+   (float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ float3    operator-   (float a, const float3& b)         { return make_float3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a)                  { return make_float3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ float4&   operator*=  (float4& a, const float4& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ float4&   operator+=  (float4& a, const float4& b)       { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ float4&   operator-=  (float4& a, const float4& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ float4&   operator*=  (float4& a, float b)               { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ float4&   operator+=  (float4& a, float b)               { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ float4&   operator-=  (float4& a, float b)               { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ float4    operator*   (const float4& a, float b)         { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ float4    operator+   (const float4& a, float b)         { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ float4    operator-   (const float4& a, float b)         { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ float4    operator*   (float a, const float4& b)         { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ float4    operator+   (float a, const float4& b)         { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ float4    operator-   (float a, const float4& b)         { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a)                  { return make_float4(-a.x, -a.y, -a.z, -a.w); }
static __device__ __forceinline__ int2&     operator*=  (int2& a, const int2& b)           { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ int2&     operator+=  (int2& a, const int2& b)           { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ int2&     operator-=  (int2& a, const int2& b)           { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ int2&     operator*=  (int2& a, int b)                   { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ int2&     operator+=  (int2& a, int b)                   { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ int2&     operator-=  (int2& a, int b)                   { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ int2      operator*   (const int2& a, const int2& b)     { return make_int2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ int2      operator+   (const int2& a, const int2& b)     { return make_int2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ int2      operator-   (const int2& a, const int2& b)     { return make_int2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ int2      operator*   (const int2& a, int b)             { return make_int2(a.x * b, a.y * b); }
static __device__ __forceinline__ int2      operator+   (const int2& a, int b)             { return make_int2(a.x + b, a.y + b); }
static __device__ __forceinline__ int2      operator-   (const int2& a, int b)             { return make_int2(a.x - b, a.y - b); }
static __device__ __forceinline__ int2      operator*   (int a, const int2& b)             { return make_int2(a * b.x, a * b.y); }
static __device__ __forceinline__ int2      operator+   (int a, const int2& b)             { return make_int2(a + b.x, a + b.y); }
static __device__ __forceinline__ int2      operator-   (int a, const int2& b)             { return make_int2(a - b.x, a - b.y); }
static __device__ __forceinline__ int2      operator-   (const int2& a)                    { return make_int2(-a.x, -a.y); }
static __device__ __forceinline__ int3&     operator*=  (int3& a, const int3& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ int3&     operator+=  (int3& a, const int3& b)           { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ int3&     operator-=  (int3& a, const int3& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ int3&     operator*=  (int3& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ int3&     operator+=  (int3& a, int b)                   { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ int3&     operator-=  (int3& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ int3      operator*   (const int3& a, const int3& b)     { return make_int3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ int3      operator+   (const int3& a, const int3& b)     { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ int3      operator-   (const int3& a, const int3& b)     { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ int3      operator*   (const int3& a, int b)             { return make_int3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ int3      operator+   (const int3& a, int b)             { return make_int3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ int3      operator-   (const int3& a, int b)             { return make_int3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ int3      operator*   (int a, const int3& b)             { return make_int3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ int3      operator+   (int a, const int3& b)             { return make_int3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ int3      operator-   (int a, const int3& b)             { return make_int3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ int3      operator-   (const int3& a)                    { return make_int3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ int4&     operator*=  (int4& a, const int4& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ int4&     operator+=  (int4& a, const int4& b)           { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ int4&     operator-=  (int4& a, const int4& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ int4&     operator*=  (int4& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ int4&     operator+=  (int4& a, int b)                   { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ int4&     operator-=  (int4& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ int4      operator*   (const int4& a, const int4& b)     { return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ int4      operator+   (const int4& a, const int4& b)     { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ int4      operator-   (const int4& a, const int4& b)     { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ int4      operator*   (const int4& a, int b)             { return make_int4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ int4      operator+   (const int4& a, int b)             { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ int4      operator-   (const int4& a, int b)             { return make_int4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ int4      operator*   (int a, const int4& b)             { return make_int4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ int4      operator+   (int a, const int4& b)             { return make_int4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ int4      operator-   (int a, const int4& b)             { return make_int4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __device__ __forceinline__ int4      operator-   (const int4& a)                    { return make_int4(-a.x, -a.y, -a.z, -a.w); }
static __device__ __forceinline__ uint2&    operator*=  (uint2& a, const uint2& b)         { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ uint2&    operator+=  (uint2& a, const uint2& b)         { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ uint2&    operator-=  (uint2& a, const uint2& b)         { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ uint2&    operator*=  (uint2& a, unsigned int b)         { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ uint2&    operator+=  (uint2& a, unsigned int b)         { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ uint2&    operator-=  (uint2& a, unsigned int b)         { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ uint2     operator*   (const uint2& a, const uint2& b)   { return make_uint2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ uint2     operator+   (const uint2& a, const uint2& b)   { return make_uint2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ uint2     operator-   (const uint2& a, const uint2& b)   { return make_uint2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ uint2     operator*   (const uint2& a, unsigned int b)   { return make_uint2(a.x * b, a.y * b); }
static __device__ __forceinline__ uint2     operator+   (const uint2& a, unsigned int b)   { return make_uint2(a.x + b, a.y + b); }
static __device__ __forceinline__ uint2     operator-   (const uint2& a, unsigned int b)   { return make_uint2(a.x - b, a.y - b); }
static __device__ __forceinline__ uint2     operator*   (unsigned int a, const uint2& b)   { return make_uint2(a * b.x, a * b.y); }
static __device__ __forceinline__ uint2     operator+   (unsigned int a, const uint2& b)   { return make_uint2(a + b.x, a + b.y); }
static __device__ __forceinline__ uint2     operator-   (unsigned int a, const uint2& b)   { return make_uint2(a - b.x, a - b.y); }
static __device__ __forceinline__ uint3&    operator*=  (uint3& a, const uint3& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ uint3&    operator+=  (uint3& a, const uint3& b)         { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ uint3&    operator-=  (uint3& a, const uint3& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ uint3&    operator*=  (uint3& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ uint3&    operator+=  (uint3& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ uint3&    operator-=  (uint3& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ uint3     operator*   (const uint3& a, const uint3& b)   { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ uint3     operator+   (const uint3& a, const uint3& b)   { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ uint3     operator-   (const uint3& a, const uint3& b)   { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ uint3     operator*   (const uint3& a, unsigned int b)   { return make_uint3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ uint3     operator+   (const uint3& a, unsigned int b)   { return make_uint3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ uint3     operator-   (const uint3& a, unsigned int b)   { return make_uint3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ uint3     operator*   (unsigned int a, const uint3& b)   { return make_uint3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ uint3     operator+   (unsigned int a, const uint3& b)   { return make_uint3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ uint3     operator-   (unsigned int a, const uint3& b)   { return make_uint3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ uint4&    operator*=  (uint4& a, const uint4& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ uint4&    operator+=  (uint4& a, const uint4& b)         { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ uint4&    operator-=  (uint4& a, const uint4& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ uint4&    operator*=  (uint4& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ uint4&    operator+=  (uint4& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ uint4&    operator-=  (uint4& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ uint4     operator*   (const uint4& a, const uint4& b)   { return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ uint4     operator+   (const uint4& a, const uint4& b)   { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ uint4     operator-   (const uint4& a, const uint4& b)   { return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ uint4     operator*   (const uint4& a, unsigned int b)   { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ uint4     operator+   (const uint4& a, unsigned int b)   { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ uint4     operator-   (const uint4& a, unsigned int b)   { return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ uint4     operator*   (unsigned int a, const uint4& b)   { return make_uint4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ uint4     operator+   (unsigned int a, const uint4& b)   { return make_uint4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ uint4     operator-   (unsigned int a, const uint4& b)   { return make_uint4(a - b.x, a - b.y, a - b.z, a - b.w); }

template<class T> static __device__ __forceinline__ T zero_value(void);
template<> __device__ __forceinline__ float  zero_value<float> (void)                      { return 0.f; }
template<> __device__ __forceinline__ float2 zero_value<float2>(void)                      { return make_float2(0.f, 0.f); }
template<> __device__ __forceinline__ float4 zero_value<float4>(void)                      { return make_float4(0.f, 0.f, 0.f, 0.f); }
static __device__ __forceinline__ float3 make_float3(const float2& a, float b)             { return make_float3(a.x, a.y, b); }
static __device__ __forceinline__ float4 make_float4(const float3& a, float b)             { return make_float4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ float4 make_float4(const float2& a, const float2& b)     { return make_float4(a.x, a.y, b.x, b.y); }
static __device__ __forceinline__ int3 make_int3(const int2& a, int b)                     { return make_int3(a.x, a.y, b); }
static __device__ __forceinline__ int4 make_int4(const int3& a, int b)                     { return make_int4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ int4 make_int4(const int2& a, const int2& b)             { return make_int4(a.x, a.y, b.x, b.y); }
static __device__ __forceinline__ uint3 make_uint3(const uint2& a, unsigned int b)         { return make_uint3(a.x, a.y, b); }
static __device__ __forceinline__ uint4 make_uint4(const uint3& a, unsigned int b)         { return make_uint4(a.x, a.y, a.z, b); }
static __device__ __forceinline__ uint4 make_uint4(const uint2& a, const uint2& b)         { return make_uint4(a.x, a.y, b.x, b.y); }

template<class T> static __device__ __forceinline__ void swap(T& a, T& b)                  { T temp = a; a = b; b = temp; }

//------------------------------------------------------------------------
// Triangle ID <-> float32 conversion functions to support very large triangle IDs.
//
// Values up to and including 16777216 (also, negative values) are converted trivially and retain
// compatibility with previous versions. Larger values are mapped to unique float32 that are not equal to
// the ID. The largest value that converts to float32 and back without generating inf or nan is 889192447.

static __device__ __forceinline__ int   float_to_triidx(float x) { if (x <= 16777216.f) return (int)x;   return __float_as_int(x) - 0x4a800000; }
static __device__ __forceinline__ float triidx_to_float(int x)   { if (x <= 0x01000000) return (float)x; return __int_as_float(0x4a800000 + x); }

//------------------------------------------------------------------------
// Coalesced atomics. These are all done via macros.

#if __CUDA_ARCH__ >= 700 || defined(__HIP_DEVICE_COMPILE__)

#define CA_TEMP       _ca_temp
#define CA_TEMP_PARAM float* CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) \
    __shared__ float CA_TEMP[(threads_per_block)]

#define CA_SET_GROUP_MASK(group, thread_mask)                   \
    bool   _ca_leader;                                          \
    float* _ca_ptr;                                             \
    do {                                                        \
        int tidx   = threadIdx.x + blockDim.x * threadIdx.y;    \
        int lane   = tidx & 31;                                 \
        int warp   = tidx >> 5;                                 \
        int tmask  = __match_any_sync((thread_mask), (group));  \
        int leader = __ffs(tmask) - 1;                          \
        _ca_leader = (leader == lane);                          \
        _ca_ptr    = &_ca_temp[((warp << 5) + leader)];         \
    } while(0)

#define CA_SET_GROUP(group) \
    CA_SET_GROUP_MASK((group), 0xffffffffu)

#define caAtomicAdd(ptr, value)         \
    do {                                \
        if (_ca_leader)                 \
            *_ca_ptr = 0.f;             \
        atomicAdd(_ca_ptr, (value));    \
        if (_ca_leader)                 \
            atomicAdd((ptr), *_ca_ptr); \
    } while(0)

#define caAtomicAdd3_xyw(ptr, x, y, w)  \
    do {                                \
        caAtomicAdd((ptr), (x));        \
        caAtomicAdd((ptr)+1, (y));      \
        caAtomicAdd((ptr)+3, (w));      \
    } while(0)

#define caAtomicAddTexture(ptr, level, idx, value)  \
    do {                                            \
        CA_SET_GROUP((idx) ^ ((level) << 27));      \
        caAtomicAdd((ptr)+(idx), (value));          \
    } while(0)

//------------------------------------------------------------------------
// Disable atomic coalescing for compute capability lower than 7.x

#else
#define CA_TEMP _ca_temp
#define CA_TEMP_PARAM float CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) CA_TEMP_PARAM
#define CA_SET_GROUP_MASK(group, thread_mask)
#define CA_SET_GROUP(group)
#define caAtomicAdd(ptr, value) atomicAdd((ptr), (value))
#define caAtomicAdd3_xyw(ptr, x, y, w)  \
    do {                                \
        atomicAdd((ptr), (x));          \
        atomicAdd((ptr)+1, (y));        \
        atomicAdd((ptr)+3, (w));        \
    } while(0)
#define caAtomicAddTexture(ptr, level, idx, value) atomicAdd((ptr)+(idx), (value))
#endif

//------------------------------------------------------------------------
#endif // __CUDACC__ || __HIPCC__
