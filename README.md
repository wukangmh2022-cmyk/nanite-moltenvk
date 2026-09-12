# Nanite-MoltenVK: Apple Silicon 上的虚拟化微多边形几何体渲染引擎 (Virtualized Geometry Renderer)

本项目是一个在 **Apple Silicon (macOS Metal / MoltenVK Vulkan)** 上，使用底层系统级语言（**C++17 / Objective-C++ / HLSL / Metal Shading Language**）从零构建的 **仿 Unreal Engine 5 Nanite 的 GPU-Driven 虚拟化几何体（Virtualized Geometry）渲染管线**。

通过基于 MoltenVK 的 Vulkan 跨平台图形 API 以及针对 Apple GPU 架构深度调优的手写 Metal 着色器，本项目在 Mac M系列芯片上实现了海量微多边形的高帧率实时渲染。

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
   - **Apple Metal 原生 64-bit 原子深度写入**：针对 Apple Silicon 定制手写 MSL 内核，通过单次 `atomic_max_explicit` 写入 `[depth:payload]` 64位压缩缓冲，规避标准 Vulkan 在缺少 64-bit 原子操作时的双 Pass 开销。

5. **材质解析与延迟着色（Material Resolve & Deferred Shading）**
   - 屏幕空间仅输出精简的 Visibility Buffer（写入 Packed Instance/Cluster/Triangle ID 与深度）。
   - 屏幕空间 Resolve Pass 统一读取共享几何缓冲、重建三角形重心坐标、插值法线与 UV，最终输出至 G-Buffer 并执行延迟着色，将材质渲染复杂度与几何复杂度彻底解耦。

6. **经典压测场景还原**
   - **弥勒佛（Stanford Happy Buddha）场景**：默认内置斯坦福大学扫描的经典弥勒佛高模（`happy_vrip.ply`，~108 万三角形）。支持通过 UI 动态调整实例数量（如 100~512 个弥勒佛组成的 3D 矩阵，几何面数达数亿级别），在 M 系列芯片上实现实时稳定运行。
   - **雪山地形组合场景（Coastal Scene）**：支持加载多重卫星遥感与高精地貌网格，结合 PBR 材质与体积云渲染。

---

## 目录结构

```text
.
├── CMakeLists.txt              # 构建系统（支持 macOS MoltenVK 与 Windows DX12/Vulkan）
├── Assets/                     # 3D 模型资产目录（内置斯坦福弥勒佛 happy_vrip.ply 等）
├── DemoRenderer.hpp / .cpp     # 核心渲染主循环、管线调度与帧时序统计
├── NaniteGpuScene.hpp / .cpp   # GPU 场景资源管理、Cluster 缓冲、HZB 与剔除 Dispatch 调度
├── NaniteModel.hpp / .cpp      # 模型加载、ClusterLOD 离线生成与 .nanite 二进制缓存
├── NanitePipelines.hpp / .cpp  # Vulkan Pipeline State 状态机与资源绑定
├── NaniteTypes.hpp             # 几何结构体、GPU 节点、Cluster、剔除常量定义
├── main.mm                     # macOS Cocoa 原生窗口、CAMetalLayer 与 UI 控制面板
├── main_win32.cpp              # Windows 平台入口
├── Shaders/
│   └── Nanite/                 # HLSL Compute / Vertex / Pixel 着色器源码及 MSL 原生内核
├── patches/                    # DiligentCore MoltenVK MSL 覆盖补丁
├── test/                       # 体积云（Volumetric Cloud）等衍生特效测试套件
└── tools/                      # MSL 编译与着色器验证工具
```

---

## 依赖与环境准备

在 macOS (Apple Silicon) 上编译并运行本项目需要以下组件：

1. **Homebrew 基础开发工具**：
   ```sh
   brew install cmake ninja glslang
   ```
2. **Vulkan / MoltenVK 运行时**：
   ```sh
   brew install molten-vk vulkan-headers vulkan-loader
   ```
   确保 `/opt/homebrew/lib/libvulkan.dylib` 存在。

3. **引擎依赖 (DiligentCore)**：
   - 项目基于开源轻量图形库 `DiligentCore` 作为底座。
   - 在上级目录克隆并编译 Vulkan 后端：
     ```sh
     git clone https://github.com/DiligentGraphics/DiligentCore.git ../DiligentCore
     cmake -S ../DiligentCore -B ../DiligentCore/build-vulkan-github -G Ninja \
           -DCMAKE_BUILD_TYPE=Release \
           -DDILIGENT_BUILD_VULKAN=ON \
           -DDILIGENT_BUILD_METAL=OFF \
           -DDILIGENT_BUILD_OPENGL=OFF
     cmake --build ../DiligentCore/build-vulkan-github --parallel
     ```
   - 如需启用手写 Metal 64-bit 原子深度优化，可按需应用 `patches/0001-vulkan-msl-shader-module-override.patch`。

4. **网格优化算法库 (meshoptimizer)**：
   - 依赖带 `clusterlod.h` 的 meshoptimizer 仓库：
     ```sh
     git clone https://github.com/zeux/meshoptimizer.git ../References/meshoptimizer
     ```

5. **3D 扫描资产**：
   - 弥勒佛模型已经内置于 `Assets/happy_vrip.ply`（Stanford 3D Scanning Repository）。
   - 首次加载大网格时会自动计算 ClusterLOD 分层并导出 `.nanite` 缓存，后续启动实现秒开。

---

## 编译与运行 (macOS)

```sh
# 1. 配置并生成构建文件
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. 编译可执行程序
cmake --build build --parallel

# 3. 运行应用程序
open build/DiligentCoreVulkanDemo.app
```

运行后会打开主渲染视窗以及独立的 `Nanite Controls` 控制面板。可通过 UI 面板调整模型实例规模、切换 HZB 遮挡剔除开关、监控逻辑三角形总数（Logical Triangles）、可见 Cluster 数量以及微秒级 GPU 各 Pass 耗时。

---

## 常用调试与调优环境变量

| 环境变量 | 说明 |
| :--- | :--- |
| `NANITE_MODEL` | 指定加载的模型路径（默认加载 `happy_vrip.ply` 弥勒佛） |
| `NANITE_SCENE=coastal` | 切换至海滨地貌群山多材质测试场景 |
| `NANITE_GPU_TIMINGS=1` | 在窗口标题栏实时打印各 GPU Pass（剔除、光栅化、HZB构建）耗时 |
| `NANITE_NATIVE_64BIT_VISIBILITY=1` | 启用针对 Apple GPU 优化的手写 Metal 64位原子深度软件光栅化 |
| `NANITE_DISABLE_HZB=1` | 强制关闭 HZB 剔除，用于性能基准对照 |
| `NANITE_VISUALIZE_HZB=1` | 在视窗中可视化当前 HZB 深度金字塔各 Mipmap 级别 |
| `NANITE_VISUALIZE_DEPTH=1` | 在视窗中可视化原始深度图 |
| `NANITE_CLUSTER_RASTER_EXPERIMENT=1` | 启用 Cluster 级别的软硬光栅化阈值分流实验 |
| `NANITE_CLUSTER_RASTER_AREA=256` | 软硬件光栅化分流的屏幕面积阈值（像素） |
| `NANITE_STRESS_TRIANGLES=1000000000` | 开启十亿级逻辑面数的可扩展性压力测试模式 |

---

## 许可证

本项目基于 MIT 许可证开源。内置模型资产来源于 Stanford 3D Scanning Repository。
