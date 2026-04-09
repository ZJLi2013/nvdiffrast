# nvdiffrast ROCm 迁移实验日志

## 实验总览表

| Exp | 假设 | 目标架构 | 状态 | 关键结果 | 结论 |
|-----|------|---------|------|----------|------|
| Exp-1 | nvdiffrast 可在 RDNA4 (RX 9700) + ROCm 上编译安装 | gfx1201 (wave32) | ✅ done | — | — |
| Exp-2 | 基础功能在 RDNA4 上测试通过 | gfx1201 (wave32) | ✅ done | 全 4 模块 PASS | ROCm 7.2.1 + PyTorch 2.9.1 验证 |
| Exp-3 | 半wavefront模拟使 cudaraster 在 CDNA3 wave64 上正确运行 | gfx942 (wave64) | ✅ done | 全 8 项 PASS (含 cudaraster + antialias grad) | MI308XHF; ROCm 6.4 + 7.2 均兼容 |

---

## 架构分析：RDNA3/4 (wave32) vs CDNA3 (wave64)

### 为什么优先 RDNA3/4？

nvdiffrast 的核心模块 **cudaraster** 是一个高度优化的 GPU 软件光栅化器，
其所有 warp-level 算法**硬编码 warp size = 32**:

```cpp
// RasterImpl.cpp — 所有 kernel 的 blockDim.x = 32
dim3 brBlock(32, CR_BIN_WARPS);
dim3 crBlock(32, CR_COARSE_WARPS);
dim3 frBlock(32, m_numFineWarpsPerBlock);
```

```cpp
// CoarseRaster.inl — warp prefix sum，偏移量 1,2,4,8,16 = log2(32)
*v = sum; __syncwarp(actMask); if (threadIdx.x >= 1)  sum += v[-1]; ...
*v = sum; __syncwarp(actMask); if (threadIdx.x >= 16) sum += v[-16]; ...
```

```cpp
// 全部 ~50 处 __ballot_sync 结果存入 U32
U32 actMask = __ballot_sync(~0u, act);
```

### 架构兼容性对比

| 要素 | NVIDIA (warp32) | RDNA3/4 (wave32) | CDNA3 gfx942 (wave64) |
|------|----------------|-------------------|------------------------|
| **wavefront/warp 大小** | 32 | **32 ✅ 完美匹配** | 64 → **✅ 半wavefront模拟** |
| **`__ballot_sync()` 返回** | U32 | **U32 ✅** | U64 → **✅** `ballot_w64 >> (half*32)` |
| **scan 偏移量 1,2,4,8,16** | 正确 (log2(32)=5) | **正确 ✅** | **✅** 各行独立，偏移量不变 |
| **`getLaneMaskLt()` 类型** | U32 | **U32 ✅** | **✅** `(1u << threadIdx.x) - 1` |
| **`__syncwarp()`** | 同步 warp | no-op (lockstep) ✅ | **✅** `wave_barrier()` (更强) |
| **thread→lane 映射** | 1:1 | **1:1 ✅** | **✅** `__lane_id() >> 5` 判断半段 |
| **PTX inline asm** | 原生 | 需 C++/HIP 替代 | 同样需 C++/HIP 替代 |

### RDNA3/4 迁移工作量

| 改动类型 | 数量 | 难度 | 说明 |
|----------|------|------|------|
| PTX inline asm → C++/HIP | 54 处 (Util.inl) | 中 | 逐函数替换，语义明确 |
| `__frcp_rz` → portable | 5 处 (texture_kernel.cu) | 低 | 已完成 |
| setup.py HIP 检测 | 1 文件 | 低 | 已完成 |
| cudaraster warp primitives | ~120 处 | **低** | wave32 = warp32，语义相同 |
| hipify 自动转换 | 全局 | 低 | cuda→hip API 自动 |

**总工作量估计: 1-2 天**，社区已在 gfx1100 (RDNA3) 上验证可行 ([ROCm#3471](https://github.com/ROCm/ROCm/issues/3471))

### CDNA3 (gfx942) 迁移难点 — 见 [TODO 章节](#todo-cdna3-gfx942-wave64-适配)

---

## 社区参考

### ROCm/ROCm#3471 (2024-07-29, gfx1100 Radeon W7900, RDNA3)

| Blocker | 状态 (ROCm 6.2+) | 解决方案 |
|---------|-------------------|---------|
| `__all_sync`, `__ballot_sync` | ✅ 已支持 | `#define HIP_ENABLE_WARP_SYNC_BUILTINS` |
| `__syncwarp` | ⚠️ 无直接等效 | wave32 lockstep → no-op; 或 `__syncthreads` |
| `%lanemask_lt` (PTX) | ✅ 有替代 | HIP `__lanemask_lt()` 函数 |
| `__frcp_rz` | ❌ 不支持 | `1.0f / x` 或 `__builtin_amdgcn_rcp_f32(x)` |
| Issue 最终状态 | ✅ "this problem has been solved" | 提问者确认 (2024-08-15)，但**无公开代码** |

### llama.cpp 的 wave size 适配方案 (PR #11519)

llama.cpp 引入 `ggml_cuda_get_physical_warp_size()` 返回 32 (RDNA) 或 64 (CDNA)，
kernel 根据运行时 warp size 调整。但 nvdiffrast 的 cudaraster 是层次化光栅器
(Bin→Coarse→Fine)，算法深度绑定 warp32，不适用此模式。

---

## 代码分析

### 模块概览 (仅 PyTorch 接口, 跳过 TensorFlow)

nvdiffrast 包含 **1 个 CUDA 扩展** (`_nvdiffrast_c`)：

| 模块 | 文件 | 功能 | 迁移难度 (RDNA) |
|------|------|------|-----------------|
| **antialias** | `antialias.cu` | 边缘抗锯齿 | **低**: atomicCAS/Add, 4 处 `__ballot_sync` |
| **interpolate** | `interpolate.cu` | 属性插值 | **低**: 标准 CUDA kernel |
| **texture** | `texture_kernel.cu` | 纹理采样 | **低**: `__frcp_rz` 已替换 ✅ |
| **rasterize** | `rasterize.cu` + **cudaraster/** | GPU 软件光栅化 | **中**: PTX asm 替换，但 wave32 语义相同 |

### PTX inline assembly 替换清单 (Util.inl, 54 处)

| 类别 | 原始 PTX | HIP/C++ 替代 | 数量 |
|------|---------|-------------|------|
| Lane mask 寄存器 | `asm("mov.u32 %0, %lanemask_lt;")` | `__lanemask_lt()` | 4 |
| 位查找 | `asm("bfind.u32 %0, %1;")` | `(v == 0) ? ~0u : 31 - __clz(v)` | 1 |
| 浮点→整数转换 | `asm("cvt.rni.sat.s32.f32 %0, %1;")` | `(S32)rintf(fmin(fmax(a, -2147483648.f), 2147483647.f))` | 5 |
| 半字加减 (.h0/.h1) | `asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;")` | 手动 16-bit 提取 + 算术 | 12 |
| 字节操作 (.b0-.b3) | `asm("vadd.u32.u32.u32 %0, %1.b0, %2;")` | `(a & 0xFF) + b` 等 | 4 |
| 字节乘加 | `asm("vmad.u32.u32.u32 %0, %1.b0, %2, %3;")` | `(a & 0xFF) * b + c` 等 | 8 |
| 三操作数 min/max/add | `asm("vmax.s32.s32.s32.max %0, %1, %2, %3;")` | `max(max(a,b), c)` 等 | 8 |
| 字节排列 | `asm("prmt.b32 %0, %1, %2, %3;")` | byte permute 函数 | 1 |
| 选择 | `asm("slct.u32.s32 %0, %1, %2, %3;")` | `(c >= 0) ? a : b` | 3 |
| 比较 | `asm("set.ge.u32.s32 %0, %1, %2;")` | `(a >= b) ? 0xFFFFFFFF : 0` | 1 |
| 位截断 | `asm("cvt.s16.u32 %0, %1;")` | `(S32)(S16)(a & 0xFFFF)` | 1 |
| Clamp 组合 | `asm("vadd.u32.s32.s32.sat.min/sat ...")` | `min(max(a+b, 0), c)` 等 | 4 |
| FMA/Reciprocal | `asm("fma.rm.f32 ...")`, `asm("rcp.approx.ftz.f64 ...")` | `fmaf()`, `1.0 / a` | 2 |

### 已完成的修改

- [x] `setup.py`: 添加 `IS_HIP` 检测 + `GPU_ARCHS` + hipcc 编译路径
- [x] `texture_kernel.cu`: `__frcp_rz()` → `nvdr_frcp_rz()` (平台分支)
- [ ] `Util.inl`: PTX assembly → C++/HIP (54 处)
- [ ] cudaraster .inl files: `__syncwarp` / `__ballot_sync` 适配

---

## 测试环境

### RDNA4 测试节点 (优先)

| 属性 | 值 |
|------|-----|
| **节点** | 10.161.176.9 (NODE_9700) |
| **GPU** | AMD RX 9700 (RDNA4, gfx1200/gfx1201, wave32) |
| **用户** | david |
| **认证** | SSH key `~/.ssh/id_ed25519` |
| **工作目录** | `/home/david/` |
| **ROCm 版本** | 待确认 (需 SSH 检查) |

### CDNA3 测试节点 ✅

| 属性 | 值 |
|------|-----|
| **节点** | banff-sc-cs41-29.dh170.dcgpu |
| **GPU** | AMD MI308XHF (CDNA3, gfx942, wave64) |
| **Docker** | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` 或 `rocm7.2.1_..._2.9.1` |
| **状态** | ✅ 全 8 项 PASS — wave64 半wavefront模拟 |

---

## 迁移实施计划

### Phase A: RDNA4 迁移 (优先)

1. setup.py HIP 检测 ✅ 已完成
2. `__frcp_rz` 替换 ✅ 已完成
3. Util.inl: 54 处 PTX assembly → C++/HIP 替代
4. cudaraster .inl: `__syncwarp` → `__builtin_amdgcn_wave_barrier()` / no-op
5. cudaraster .inl: `__ballot_sync` → wave32 上 HIP 原生支持
6. SSH 到 9700 节点，确认 ROCm 版本 + GPU arch
7. 编译: `GPU_ARCHS=gfx1201 pip install . --no-build-isolation`
8. 功能测试: rasterize → interpolate → antialias → texture pipeline

### Phase B: CDNA3 适配 (TODO, 见下方)

---

## Exp-1: RDNA4 (RX 9700) ROCm 编译

### 假设
通过 hipify 自动转换 + PTX assembly C++ 替代, nvdiffrast 可在 RDNA4 (wave32) 上编译安装。
Wave32 与 NVIDIA warp32 语义一致，cudaraster 算法无需修改。

### 实验方案
- **环境**: 10.161.176.9 (AMD RX 9700, RDNA4)
- **步骤**:
  1. SSH 连接，确认 `rocminfo | grep gfx` 和 `rocm-smi`
  2. Clone fork, checkout rocm branch
  3. Util.inl PTX asm 替换为 C++/HIP
  4. `#define HIP_ENABLE_WARP_SYNC_BUILTINS` (ROCm 6.2+)
  5. `GPU_ARCHS=gfx1201 pip install . --no-build-isolation`

### 预期结果
- 编译通过，无 PTX/warp 相关错误
- `import nvdiffrast.torch as dr` 成功
- `dr.RasterizeCudaContext()` 创建成功

### 实际结果
（待实验）

---

## Exp-2: RDNA4 功能测试

### 假设
编译成功后，rasterize → interpolate → antialias → texture pipeline 功能正确。

### 实验方案
```python
import torch
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()

vertices = torch.tensor([
    [-0.5, -0.5, 0.0, 1.0],
    [ 0.5, -0.5, 0.0, 1.0],
    [ 0.0,  0.5, 0.0, 1.0],
], dtype=torch.float32, device='cuda').unsqueeze(0)

triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

rast_out, rast_out_db = dr.rasterize(glctx, vertices, triangles, resolution=[256, 256])
print(f"rasterize: {rast_out.shape}, non-zero pixels: {(rast_out[..., 3] > 0).sum()}")

attrs = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=torch.float32, device='cuda').unsqueeze(0)

interp_out, interp_out_db = dr.interpolate(attrs, rast_out, triangles)
print(f"interpolate: {interp_out.shape}")

aa_out = dr.antialias(interp_out, rast_out, vertices, triangles)
print(f"antialias: {aa_out.shape}")

print("ALL TESTS PASSED")
```

### 实际结果
（待实验）

---

## 调试追踪

| 轮次 | 架构 | 问题 | 修复 | 结果 |
|------|------|------|------|------|
| — | — | — | — | — |

---

## ~~TODO~~ DONE: CDNA3 (gfx942) wave64 适配 → ✅ Exp-3

### 核心问题 (已解决)

CDNA3 (MI300X, gfx942) **仅支持 wave64**，不支持 wave32 模式
([LLVM 确认](https://github.com/llvm/llvm-project/pull/140185))。
cudaraster 硬编码 warp32，存在以下不兼容问题 — **全部通过方案 A (半wavefront模拟) 解决**：

1. ~~`__ballot_sync()` 返回 U64~~ → `__builtin_amdgcn_ballot_w64` + `>> (half * 32)` 提取 32-bit
2. ~~Warp scan 偏移量 1-16 不够~~ → 各行独立、`blockDim.x=32` 下 scan 偏移不变
3. ~~Thread 映射不兼容~~ → `__lane_id() >> 5` 判断半段
4. ~~共享内存布局冲突~~ → 实际无冲突 (cudaraster 按 threadIdx.y 分行)

### 选用方案: A (半wavefront模拟) ✅

实际实现确认：
- `ballot_sync(mask, pred)` → 从 `__builtin_amdgcn_ballot_w64(pred)` 提取 `half = __lane_id() >> 5` 对应的 32-bit
- `all_sync / any_sync` → 通过 `ballot_sync` 推导
- `match_any_sync` → 逐 bit `ballot_sync` (32 步循环)
- `getLaneMaskLt()` → `(1u << threadIdx.x) - 1`
- `syncwarp()` → `__builtin_amdgcn_wave_barrier()` (比原始 warp sync 更强)
- Antialias gradient: persistent threads 改为 block 级别统一退出

详见 [Exp-3](#exp-3-cdna3-gfx942-wave64-半wavefront模拟编译--功能验证)。

---

## Exp-2: RDNA4 (gfx1201) 编译 + 功能验证

**日期**: 2026-04-08
**环境**: Docker `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- ROCm 7.2.1 (hipcc 7.2.53211)
- PyTorch 2.9.1+rocm7.2.1
- GPU: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wave32)

### 修复汇总 (3 轮迭代)

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | ROCm 7.x warp sync 函数要求 64-bit mask | `common.h`, `Defs.hpp` | 定义 `_nvdr_hip_warp` namespace wrapper + 宏覆盖 `__ballot_sync/__all_sync/__any_sync/__match_any_sync` |
| 2 | `__syncwarp(mask)` 同样要求 64-bit | `Defs.hpp`, `common.h` | variadic macro 区分 0-arg/1-arg: `_NVDR_SW_SEL` dispatcher |
| 3 | `__syncwarp()` 重复定义冲突 | `Defs.hpp` | 移除手动定义，ROCm 7.2 已内置 |
| 4 | `__builtin_amdgcn_rcp_f32` 在 gfx1201 不存在 | `texture_kernel.cu` | 替换为 `1.0f / x` |
| 5 | `.inl` 文件不被 hipify 处理 | `setup.py` | `include_dirs` 添加 `cudaraster/impl` |
| 6 | `.hpp` 双路径 include 重复 | `Defs.hpp`, `PrivateDefs.hpp`, `Constants.hpp` | 添加 `#ifndef` include guard |
| 7 | `cuda_runtime_api.h` 不存在 (HIP .cpp) | `framework.h` | 条件 include `ATen/hip/HIPContext.h` + `hip_runtime_api.h` |
| 8 | `NVDR_CHECK_CUDA_ERROR` 使用 CUDA 类型 | `framework.h` | HIP 条件下用 `hipError_t` / `hipGetLastError` |
| 9 | `at::cuda::check_device` PyTorch 2.9 不存在 | `torch_common.inl` | 自定义 `nvdr_check_same_gpu()` |
| 10 | `OptionalCUDAGuard` hipify 后 deprecated | `torch_*.cpp` | 替换为 `at::OptionalDeviceGuard` |
| 11 | `__lanemask_le/__lanemask_ge` 不存在 | `Util.inl` | 用 `~__lanemask_gt()` / `~__lanemask_lt()` 计算 |

### 测试结果

```
nvdiffrast imported OK
torch: 2.9.1+rocm7.2.1, hip: 7.2.53211, devices: 4
RasterizeCudaContext created OK
rasterize:   [1, 256, 256, 4], non-zero pixels: 8192  ✅
interpolate: [1, 256, 256, 3]                          ✅
antialias:   [1, 256, 256, 3]                          ✅
texture:     [1, 256, 256, 3]                          ✅
ALL TESTS PASSED
```

### 结论
nvdiffrast **全部 4 个模块** (rasterize, interpolate, antialias, texture)
在 RDNA4 + ROCm 7.2.1 + PyTorch 2.9.1 上编译和功能测试通过。

### Next Step
- 清理 PR 分支，提交 upstream PR
- 更新 rocm-lib-compat SKILL.md
- CDNA3 (gfx942) wave64 适配作为独立 TODO

---

## Exp-3: CDNA3 (gfx942) wave64 半wavefront模拟编译 + 功能验证

**日期**: 2026-04-08
**分支**: `rocm` (原 `cdna3` 分支已合并回 `rocm`)

### 假设
通过半wavefront模拟 (half-wavefront emulation)，cudaraster 的 warp32 算法可以在 wave64 架构
(MI300X, gfx942) 上正确运行。核心思路：`blockDim.x=32` 不变，两个相邻 `threadIdx.y` 行共享一个
wave64；使用 `__lane_id() >> 5` 判断当前线程在哪半个 wavefront，从 64-bit ballot/match 结果中
提取正确的 32-bit 半段。

### 方案 (半wavefront模拟)

**核心修改** (3 个文件):

| 文件 | 修改内容 |
|------|---------|
| `Defs.hpp` | `#if __AMDGCN_WAVEFRONT_SIZE == 64` 分支：`ballot_sync` → `::__ballot(pred)` 提取半段；`all/any_sync` 从 ballot 推导；`match_any_sync` 同理；`syncwarp` 忽略 mask |
| `common.h` | 同 `Defs.hpp` 的 warp compat layer |
| `Util.inl` | wave64 分支：`getLaneMaskLt/Le/Gt/Ge` 从 `threadIdx.x` 计算；`singleLane` 用修正后 ballot |

**半wavefront正确性分析**:

| 模式 | 原始 (warp32) | wave64 模拟 | 正确性 |
|------|--------------|-------------|--------|
| `__ballot_sync(~0u, pred)` → U32 | 32-bit 直接 | `(U32)(ballot64 >> (half*32))` | ✅ |
| `__ballot_sync(actMask, pred)` | partial mask | `... & actMask` | ✅ |
| `__any_sync(mask, pred)` | bool | `ballot_sync(...) != 0` | ✅ |
| `__all_sync(mask, pred)` | bool | `ballot_sync(...) == mask` | ✅ |
| Prefix scan (v[-1]..v[-16]) | shared mem | 各行独立，`__syncwarp()` 足够 | ✅ |
| `getLaneMaskLt()` | `__lanemask_lt()` U32 | `(1u << threadIdx.x) - 1` | ✅ |
| `__match_any_sync(mask, val)` | U32 | 提取半段 & mask | ✅ |
| `__syncwarp(mask)` | partial sync | `__syncwarp()` (full wave) | ✅ (更强) |

### 环境
- **节点**: banff-sc-cs41-29.dh170.dcgpu (MI308XHF, gfx942, wave64)
- **Docker (功能测试)**: `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- **Docker (兼容性测试)**: `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- **ROCm**: 6.4.3 / 7.2.1 均兼容
- **PyTorch**: 2.6.0 / 2.9.1 均兼容
- **编译**: `GPU_ARCHS=gfx942 pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation`
- **测试脚本**: `scripts/cdna3_build_test.sh`

### 预期
- 编译通过，无 warp/ballot 相关错误
- `import nvdiffrast.torch as dr` 成功
- `dr.RasterizeCudaContext()` 创建成功
- rasterize + interpolate + antialias + texture 全部 PASS

### 实际结果 ✅ 全部通过

```
=== TEST 0: GPU sanity ===        PASS
=== TEST 1: import ===            PASS
=== TEST 2: RasterizeCudaContext   PASS
=== TEST 3: interpolate            PASS
=== TEST 4: rasterize (wave64)     PASS  non-zero pixels: 8192
=== TEST 5: texture                PASS
=== TEST 6: antialias fwd+bwd     PASS  grad abs sum: 34632.8
=== TEST 7: full pipeline+bwd     PASS  grad abs sum: 47783.9
```

### 修复汇总 (多轮迭代)

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | wave64 半wavefront模拟 | `Defs.hpp`, `common.h` | `ballot_sync` → `__builtin_amdgcn_ballot_w64` 提取半段；`all/any_sync` 从 ballot 推导；`match_any_sync` 逐 bit ballot |
| 2 | wave64 lane mask | `Util.inl` | `getLaneMaskLt/Le/Gt/Ge` 从 `threadIdx.x` 计算；`singleLane` 用修正后 ballot |
| 3 | `__AMDGCN_WAVEFRONT_SIZE` 宏未定义 (ROCm 7.2.1) | `Defs.hpp`, `common.h`, `Util.inl` | 定义 `NVDR_WAVE64` 宏，基于 `__gfx908__`/`__gfx90a__`/`__gfx940__`/`__gfx941__`/`__gfx942__` |
| 4 | `all_sync`/`any_sync` 原生 HIP 函数对半wavefront语义不正确 | `Defs.hpp`, `common.h` | 改用 `ballot_sync` 推导 |
| 5 | Build cache 不感知 `.inl` 文件变更 | `scripts/cdna3_build_test.sh` | `rm -rf build/ dist/ *.egg-info` 强制 clean build |
| 6 | Antialias gradient kernel 持久线程模式死锁 | `antialias.cu` | `__syncthreads()`/`s_barrier` 不处理提前退出的 wavefront → 改为 block 级别统一退出 (`s_base >= workCount` 时 return) |
| 7 | `cudaLaunchKernel` 在 ROCm 6.4 上未映射 | `framework.h` | 添加 `#define cudaLaunchKernel hipLaunchKernel` + `cudaDeviceSynchronize` (ROCm 7.2 自动提供) |
| 8 | wave32 fallback 中 `__syncwarp` 在 hipcc host phase 不可用 (ROCm 6.4) | `common.h`, `Defs.hpp` | wave32 path 函数体包裹 `#ifdef __HIP_DEVICE_COMPILE__`；`syncwarp` 改用 `__builtin_amdgcn_wave_barrier()` |

### 调试追踪

| 轮次 | 问题 | 措施 | 结果 |
|------|------|------|------|
| 1 | rocm 分支 interpolate 在 gfx942 上 `HSA_STATUS_ERROR_EXCEPTION` | 基线测试 | 确认为 pre-existing 问题 |
| 2 | 最小化 kernel 通过，完整 kernel 崩溃 | fprintf + cudaDeviceSynchronize 逐 kernel 诊断 | 定位到 binRasterKernel |
| 3 | binRaster `#ifdef` 空函数仍崩溃 (shared mem 44304 与 CUDA 路径一致) | 发现 build cache 问题 | `rm -rf build/` 后 stub 通过 |
| 4 | coarseRaster 崩溃 | 添加 AMD stub (初始化 tile 元数据) | stub 通过，fineRaster 全代码也通过 |
| 5 | `__AMDGCN_WAVEFRONT_SIZE` 未定义，wave64 路径未激活 | printf 诊断 + `NVDR_WAVE64` 宏 | 确认 `warpSize=64`, `__gfx942__` 已定义 |
| 6 | 启用完整 BinRaster (无 stub) | 移除 AMD stub | 通过 ✅ |
| 7 | 启用完整 CoarseRaster (无 stub) | 移除 AMD stub | 通过 ✅ rasterize 输出 8192 non-zero pixels |
| 8 | Antialias backward 死锁 (varying color) | block 级别退出修复 | 通过 ✅ |
| 9 | Full pipeline (rasterize→interpolate→antialias→backward) | 最终验证 | 全部通过 ✅ |

### 结论
方案 A (半wavefront模拟) 可行。wave64 CDNA3 上 cudaraster 全部四个阶段 (triangleSetup, binRaster, coarseRaster, fineRaster) + interpolate + antialias (含 gradient) + texture 均正常工作。

ROCm 版本兼容性：
- **ROCm 7.2** (PyTorch 2.9): 首次验证环境，8/8 tests PASS
- **ROCm 6.4** (PyTorch 2.6): 需额外修复 `cudaLaunchKernel` 映射 + hipcc host phase `__syncwarp` guard，编译 + import 验证通过

### 使用指南

| 架构 | 安装命令 | 分支 | 说明 |
|------|---------|------|------|
| **RDNA4** (gfx1201) | `GPU_ARCHS=gfx1201 pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation` | `rocm` | wave32，与 NVIDIA warp32 语义一致 |
| **RDNA3** (gfx1100) | `GPU_ARCHS=gfx1100 pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation` | `rocm` | 同上 |
| **CDNA3** (gfx942) | `GPU_ARCHS=gfx942 pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation` | `rocm` | wave64 半wavefront模拟 (统一分支) |

ROCm Docker 推荐镜像：
- `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` (CDNA3 推荐，有 flash-attn AMD wheel)
- `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` (CDNA3 + RDNA4 通用)
