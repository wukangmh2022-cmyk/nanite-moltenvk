# Nanite-MoltenVK: Apple Silicon 上的虚拟化微多边形几何体渲染引擎 (Virtualized Geometry Renderer)

本项目是一个在 **Apple Silicon (macOS Metal / MoltenVK Vulkan)** 上，使用底层系统级语言（**C++17 / Objective-C++ / HLSL / Metal Shading Language**）从零构建的 **仿 Unreal Engine 5 Nanite 的 GPU-Driven 虚拟化几何体（Virtualized Geometry）渲染管线**。

通过基于 MoltenVK 的 Vulkan 跨平台图形 API 以及针对 Apple GPU 硬件特性的手写 Metal MSL 扩展，本项目在 Mac M 系列芯片上实现了海量微多边形的高帧率实时渲染。

---

## 核心贡献与架构特性

1. **完全 GPU-Driven 的 Cluster 管线与海量实例驱动**
   - 场景几何数据由 GPU Object Table 与全局共享几何缓冲统一管理，CPU 不再根据物体或 Cluster 提交单独的 Draw Call。
   - 离线阶段将复杂高模细分为紧凑的 Cluster（128 顶点 / ~128 三角形），并在 GPU 上以单个 Indirect Command 批次分发。

2. **层次化 DAG / BVH 树与 GPU 动态遍历 (Persistent Traversal)**
   - 结合 `meshoptimizer` 与 `clusterlod` 算法离线构建网格的层次化多级简化 DAG（Directed Acyclic Graph）拓扑结构与误差边界（Error Bounds）。
   - GPU Compute Shader（`NanitePersistentCull.hlsl`）实时计算屏幕空间投影误差，平滑无缝地选择各区域最优的几何层级，从数学原理上杜绝传统网格 LOD 的跳变现象（Popping）。

3. **两阶段分层遮挡剔除（Two-Phase Occlusion Culling）**
   - **Main Pass（前帧 HZB 剔除）**：基于上一帧的 Hierarchical-Z 深度缓冲金字塔，快速对 Instance / Node / Cluster 进行保守视锥与遮挡剔除。
   - **Post Pass（当帧 HZB 重测）**：在渲染出本帧深度后，对视锥变化或先前被遮挡的不确定候选区域执行二次回溯剔除，兼顾极致性能与无视觉瑕疵。

4. **混合光栅化（Hardware + Software Rasterization）与 Visibility Buffer**
   - **大三角形硬件光栅**：大图元通过 Vulkan ExecuteIndirect 提交至硬件光栅化器。
   - **微多边形软光栅**：亚像素级微多边形分流至 Compute Shader 软件光栅器（`NaniteSoftRasterDepth.hlsl`），彻底解决传统硬件光栅管线渲染 sub-pixel 三角形时的过绘制与 quad-overdraw 开销。
   - **Apple Metal 原生 64-bit 原子深度写入（关键底层 Patch）**：详见下方关于底层补丁的专项说明。

5. **材质解析与延迟着色（Material Resolve & Deferred Shading）**
   - 屏幕空间仅输出精简的 Visibility Buffer（写入 Packed Instance/Cluster/Triangle ID 与深度）。
   - 屏幕空间 Resolve Pass 统一读取共享几何缓冲、重建三角形重心坐标、插值法线与 UV，最终输出至 G-Buffer 并执行延迟着色，将材质渲染复杂度与几何复杂度彻底解耦。

6. **经典压测场景还原**
   - **弥勒佛（Stanford Happy Buddha）场景**：默认内置斯坦福大学扫描的经典弥勒佛高模（`happy_vrip.ply`，~108 万三角形）。支持通过 UI 动态调整实例数量（如 100~512 个弥勒佛组成的 3D 矩阵，几何面数达数亿级别），在 M 系列芯片上实现实时稳定运行。
   - **雪山地形组合场景（Coastal Scene）**：支持加载多重卫星遥感与高精地貌网格，结合 PBR 材质与体积云渲染。

---

## 核心底层 Patch：支持 Apple Silicon 64-bit 硬件原子操作

在 `patches/0001-vulkan-msl-shader-module-override.patch` 中包含了一个至关重要的底层修改。

### 为什么需要这个 Patch？
1. **Nanite Visibility Buffer 的核心机制**：软光栅和 Visibility Buffer 依赖于一个 64-bit 原子操作，即使用 `atomic_max_explicit` 将 `[32-bit Depth : 32-bit Payload(Instance+Cluster+Triangle ID)]` 作为一个 64 位无符号整数一次性写入原子缓冲，从而保证深度测试与可见性标识的原子写入。
2. **SPIRV-Cross 与 MoltenVK 的局限**：
   - 标准 Vulkan 规范要求设备具备全套 64 位原子算术指令（Add/CAS/Exchange/Min/Max）才能置位 `shaderBufferInt64Atomics=1`。然而 Apple M 系列硬件原生只提供了 Min/Max 指令，导致 MoltenVK 只能向 Vulkan 报告该特性不支持（为 0）。
   - SPIRV-Cross 在将带有 64 位原子操作的 SPIR-V 编译到 Metal Shading Language (MSL) 时，会直接报错拒绝：`"MSL currently does not support 64-bit atomics"`。
3. **Patch 解决方案**：
   - 该补丁修改了 `DiligentCore` 的 Vulkan 后端（`PipelineStateVkImpl.cpp`）。
   - 当环境变量 `DILIGENT_MSL_OVERRIDE_DIR` 指向本项目手写的 MSL 目录（`Shaders/Nanite/msl`）时，DiligentCore 绕过 SPIRV-Cross 的翻译，通过 MoltenVK 私有接口魔数 `kMVKMagicNumberMSLSourceCode` 直接把手写 Metal 源码喂给底层 MoltenVK。
   - 借助手写 MSL，项目直接调用 Apple Silicon 芯片硬件原生的 `atomic_max_explicit(..., memory_order_relaxed)`，从而消除常规 32-bit 回退方案中必须进行的二次 coverage 遍历开销！

---

## 完整编译与运行指南 (macOS Apple Silicon)

以下步骤经过验证，任何人克隆该仓库后均可成功编译并运行：

### 1. 准备同级目录结构

建议创建一个工作根目录（例如 `operater-dev`），让项目与依赖库保持如下相对路径结构：

```text
operater-dev/
├── DiligentCore/             # 图形底座引擎 (打好 patch 并编译)
├── References/
│   └── meshoptimizer/        # 离线 Cluster 网格简化库
└── nanite-moltenvk/          # 本项目
```

### 2. 安装系统依赖

```sh
brew install cmake ninja glslang molten-vk vulkan-headers vulkan-loader git-lfs
git lfs install
```

### 3. 克隆并编译 DiligentCore（应用 64 位原子 Patch）

```sh
cd /path/to/operater-dev

# 克隆官方 DiligentCore 并切到推荐基础 commit
git clone https://github.com/DiligentGraphics/DiligentCore.git
cd DiligentCore
git checkout b402aefa3

# 应用 MSL 源码级覆写补丁（解锁 Apple 64-bit 原子特性）
git apply ../nanite-moltenvk/patches/0001-vulkan-msl-shader-module-override.patch

# 编译 DiligentCore 的 Vulkan 静态库
cmake -S . -B build-vulkan-github -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DDILIGENT_BUILD_VULKAN=ON \
      -DDILIGENT_BUILD_METAL=OFF \
      -DDILIGENT_BUILD_OPENGL=OFF \
      -DDILIGENT_NO_HLSL=ON \
      -DDILIGENT_BUILD_SAMPLES=OFF \
      -DDILIGENT_BUILD_DEMOS=OFF \
      -DDILIGENT_BUILD_TESTS=OFF

cmake --build build-vulkan-github --parallel
```

### 4. 克隆 meshoptimizer

```sh
cd /path/to/operater-dev
mkdir -p References
git clone https://github.com/zeux/meshoptimizer.git References/meshoptimizer
```

### 5. 编译与运行本项目

```sh
cd /path/to/operater-dev/nanite-moltenvk

# 1. 确保 Git LFS 已完整拉取弥勒佛模型
git lfs pull

# 2. 生成构建工程
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 3. 编译应用
cmake --build build --parallel

# 4. 运行弥勒佛场景
# 【情况 A：Apple M1 机器】
# M1 芯片硬件（Apple Family 7）原生不支持 64 位整数原子指令，使用高兼容性的便携 32 位双 Pass 模式即可全速运行：
NANITE_NATIVE_64BIT_VISIBILITY=0 NANITE_GPU_TIMINGS=1 open build/DiligentCoreVulkanDemo.app

# 【情况 B：Apple M2 / M3 / M4 机器】
# M2 及以上芯片（Apple Family 8+，如 M4 MacBook）原生支持 64 位硬件原子操作，可开启 MSL 覆写优化：
DILIGENT_MSL_OVERRIDE_DIR="$PWD/Shaders/Nanite/msl" \
NANITE_NATIVE_64BIT_VISIBILITY=1 \
NANITE_GPU_TIMINGS=1 \
open build/DiligentCoreVulkanDemo.app
```

---

## 常用环境变量调优参数

| 环境变量 | 默认值 | 作用说明 |
| :--- | :--- | :--- |
| `DILIGENT_MSL_OVERRIDE_DIR` | 无 | 指向 `Shaders/Nanite/msl`，激活 Patch 的手写 Metal 内核覆写 |
| `NANITE_NATIVE_64BIT_VISIBILITY` | 0 | 设为 `1` 启用单 Pass 原生 64-bit 原子 Visibility Buffer |
| `NANITE_MODEL` | `happy_vrip.ply` | 场景加载模型，默认加载斯坦福弥勒佛（41MB） |
| `NANITE_SCENE` | 默认矩阵 | 设为 `coastal` 切换至多材质雪山群峰场景 |
| `NANITE_GPU_TIMINGS` | 0 | 设为 `1` 在窗口标题栏实时打印各 GPU Pass 纳秒/毫秒耗时 |
| `NANITE_DISABLE_HZB` | 0 | 设为 `1` 强制关闭 HZB 遮挡剔除，用于性能 A/B 对比 |
| `NANITE_VISUALIZE_HZB` | 0 | 设为 `1` 在渲染窗口中可视化当前帧 HZB 金字塔 Mipmap |
| `NANITE_STRESS_TRIANGLES` | 0 | 开启海量复制压测（例如设为 `1000000000` 渲染 10 亿逻辑面） |

---

## 目录索引

```text
.
├── CMakeLists.txt              # 主构建脚本
├── Assets/                     # 3D 资产（内置弥勒佛 happy_vrip.ply 与 stanford_bunny.obj）
├── DemoRenderer.hpp / .cpp     # 渲染主循环、GPU 时序查询与视锥摄像机管理
├── NaniteGpuScene.hpp / .cpp   # GPU 场景资源管理、Cluster 缓冲、两阶段剔除调度
├── NaniteModel.hpp / .cpp      # PLY/OBJ 解析、ClusterLOD 离线生成与 .nanite 序列化缓存
├── NanitePipelines.hpp / .cpp  # Vulkan Pipeline State 状态机与资源绑定布局
├── NaniteTypes.hpp             # 几何、DAG 节点、Cluster、剔除常量结构定义
├── main.mm                     # macOS Cocoa 窗口、CAMetalLayer 桥接与 ImGui 控制面板
├── Shaders/Nanite/             # HLSL 着色器源码 (Cull, Raster, HZB, Shading)
│   └── msl/                    # 手写 Metal 原生内核 (供 Patch 覆写注入)
├── patches/                    # DiligentCore 底层 64-bit 原子支持补丁
└── test/                       # 体积云（Volumetric Cloud）等模块化测试代码
```

---

## 许可证

本项目遵循 MIT 开源许可证。内置模型来源于 Stanford Computer Graphics Laboratory 3D Scanning Repository。
