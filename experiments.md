# nvdiffrast ROCm 迁移实验日志

## 实验总览表

| Exp | 假设 | 目标架构 | 状态 | 关键结果 | 结论 |
|-----|------|---------|------|----------|------|
| Exp-1 | nvdiffrast 可在 RDNA4 (RX 9700) + ROCm 上编译安装 | gfx1201 (wave32) | 待实验 | — | — |
| Exp-2 | 基础功能 (rasterize, interpolate, antialias, texture) 在 RDNA4 上测试通过 | gfx1201 (wave32) | 待实验 | — | — |

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
| **wavefront/warp 大小** | 32 | **32 ✅ 完美匹配** | 64 ❌ 不兼容 |
| **`__ballot_sync()` 返回** | U32 | **U32 ✅** | U64 → 存入 U32 截断 ❌ |
| **scan 偏移量 1,2,4,8,16** | 正确 (log2(32)=5) | **正确 ✅** | 缺少 32 步 (需 log2(64)=6) ❌ |
| **`getLaneMaskLt()` 类型** | U32 | **U32 ✅** | U64 → 类型不兼容 ❌ |
| **`__syncwarp()`** | 同步 warp | no-op (lockstep) ✅ | 同步整个 wave64 (含 2 个逻辑 warp) ⚠️ |
| **thread→lane 映射** | 1:1 | **1:1 ✅** | 2 个 warp32 打包进 1 个 wave64 ⚠️ |
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

### CDNA3 测试节点 (待定)

| 属性 | 值 |
|------|-----|
| **节点** | banff-sc-cs41-29 |
| **GPU** | AMD MI300X (CDNA3, gfx942, wave64) |
| **状态** | 暂缓 — 需 cudaraster wave64 适配 |

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

## TODO: CDNA3 (gfx942) wave64 适配

### 核心问题

CDNA3 (MI300X, gfx942) **仅支持 wave64**，不支持 wave32 模式
([LLVM 确认](https://github.com/llvm/llvm-project/pull/140185))。
cudaraster 硬编码 warp32，导致以下根本性不兼容：

1. **`__ballot_sync()` 返回 U64**：~50 处存入 U32 变量 → 截断高 32 位 → 错误结果
2. **Warp scan/reduce 偏移量**：1,2,4,8,16 (5 步) → wave64 需 1,2,4,8,16,**32** (6 步)
3. **Thread 映射**：wave64 将 2 个逻辑 warp32 (相邻 threadIdx.y) 打包进 1 个 wavefront
4. **共享内存布局**：按 32-lane warp 分配，wave64 下可能地址冲突

### 候选方案

| 方案 | 工作量 | 风险 | 说明 |
|------|--------|------|------|
| **A: 半 wavefront 模拟** | 1-2 周 | 高 | 每个 wave64 内模拟 2 个 warp32；`(U32)(ballot >> (threadIdx.y & 1) * 32)` 提取正确半掩码 |
| **B: 完整 wave64 重写** | 3-4 周 | 中 | 所有 U32 mask→U64，scan 扩展 6 步，~200+ 处修改 |
| **C: 仅移植非 cudaraster 部分** | 2-3 天 | 低 | antialias + interpolate + texture 可用，但无 rasterize |

### 半 wavefront 模拟思路 (方案 A)

```cpp
// wave64 上，blockDim.x=32 时:
// wavefront 0 = threadIdx.y=0 (lanes 0-31) + threadIdx.y=1 (lanes 32-63)
// wavefront 1 = threadIdx.y=2 (lanes 0-31) + threadIdx.y=3 (lanes 32-63)

#if defined(__HIP_PLATFORM_AMD__) && __AMDGCN_WAVEFRONT_SIZE == 64
static __device__ __inline__ U32 nvdr_ballot(bool pred) {
    unsigned long long full = __ballot(pred);
    return (U32)(full >> ((threadIdx.y & 1) * 32));
}
static __device__ __inline__ U32 nvdr_getLaneMaskLt() {
    return (1u << threadIdx.x) - 1;
}
#else
#define nvdr_ballot(pred) __ballot_sync(~0u, pred)
#define nvdr_getLaneMaskLt() getLaneMaskLt()
#endif
```

**风险**：依赖 thread→lane 映射假设（blockDim.x 必须恰好 = 32），
且 `__syncwarp(mask)` 实际同步整个 wave64 而非半个，可能引入隐式依赖 bug。

### 决策
暂缓 CDNA3 适配，优先完成 RDNA3/4 验证。后续根据下游需求 (TRELLIS.2 等)
评估 CDNA3 投入产出比。
