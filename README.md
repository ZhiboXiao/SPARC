# SPARC 最小演示包

**Sparse Propagation-Aware ReCovery**。仅包含模型、已训练权重、训练/推理代码，以及一个模拟样本和一个真实样本。

## 内容

| 文件 | 用途 |
| --- | --- |
| `model.py` | SPARC 网络与 X→Y 双轴级联 |
| `common.py` | 样本读取、训练 batch、损失与波形指标 |
| `windows.py` | 原时间窗口与重叠融合逻辑 |
| `train.py` | 单模拟样本 clean 训练演示 |
| `infer.py` | 模拟 / 真实样本推理 |
| `config.json` | 模型与训练参数 |
| `weights/sparc.pt` | 当前已验证的 anatomy SPARC 权重，4,338,322 参数 |
| `data/simulation.h5` | 原训练 split 中的模拟样本 162 |
| `data/real.h5` | Figure 5C 使用的 Mouse1 满阵传感器信号 |
| `data/manifest.json` | 样本来源、尺寸、数值哈希与拷贝校验 |

Figure 5C 是 Mouse1/2 配对展示；这里按“一个真实样本”的要求选用 Mouse1 组成部分。真实信号已经按原流程适配至 150 MHz，不要再次带通或降采样。两个 HDF5 均只包含满阵参考信号与必要元数据；代码内部生成每轴四倍稀疏输入，未测量位置的参考值不进入网络。

## 下载完整演示包

请从 [2026-09-25 Release](https://github.com/ZhiboXiao/SPARC/releases/tag/demo-20260925) 下载 `SPARC_demo_with_samples_20260925.zip`。解压后进入 `SPARC_demo` 目录，即可按下方说明运行。

Git 仓库包含代码、已训练权重和样本元数据。两个 HDF5 样本通过完整 ZIP 分发；如果使用 `git clone`，请从 ZIP 中将 `data/real.h5` 和 `data/simulation.h5` 复制到仓库的 `data/` 目录。

完整 ZIP 的 SHA-256：

```text
fa433cc024a590985af8754c203285c0508f5895d492f91dd3de638791e05d8d
```

## 安装

在本目录下执行：

```bash
python -m pip install -r requirements.txt
```

依赖仅为 PyTorch、NumPy、h5py。支持 CPU 或 CUDA；需要 GPU 时安装与你机器匹配的 PyTorch。已在 Python 3.11 / PyTorch 2.5.1 上运行检查。

## 快速演示

先用两个样本各自的一段 320 点信号确认运行环境：

```bash
python infer.py --case simulation --start 6000 --length 320
python infer.py --case real --start 6000 --length 320
```

未指定设备时，有 CUDA 则使用 `cuda:0`，否则使用 CPU。可以显式添加 `--device cuda:1` 或 `--device cpu`。

推理默认读取随包权重，输出 `outputs/样本名/completion.npz` 和 `metrics.json`。前者包含补全传感器场 `grid[X,Y,T]`，后者包含缺失位置 NRMSE、Pearson r 和实测位置回填误差。代码拒绝覆盖非空输出目录。

## 完整样本推理

省略 `--start`、`--length` 即覆盖整条时间记录，使用保存的原窗口设置与重叠融合：

```bash
python infer.py --case simulation --device cuda:0 --output outputs/simulation_full
python infer.py --case real --device cuda:0 --output outputs/real_full
```

短窗口运行用于快速检查，会改变边界上下文，指标不等同于完整记录结果。所有窗口共用样本级固定尺度，不进行逐窗口归一化。

## 训练演示

只使用随包模拟样本；真实样本不参与训练。默认随机初始化，所有参数参与训练。

```bash
python train.py --steps 10 --batch-size 1 --device cuda:0
```

CPU 快速检查可以缩短时间输入：

```bash
python train.py --steps 1 --time-size 320 --device cpu --output outputs/train_cpu.pt
```

需要演示从随包权重继续更新时，显式指定：

```bash
python train.py --initialize weights/sparc.pt --steps 10 --output outputs/train_continued.pt
```

训练输出独立于随包权重，不会覆盖参考模型。损失保留 clean 阶段的波形 L1、时空差分和传播约束；不加噪、不冻结参数、不加 BP loss。这里是**单样本流程演示**，不能代替论文中的完整数据集训练或用于报告泛化精度。

## 数据格式

`signal` 的磁盘形状为 `[10404,T]`，通道序号为 `x + 102*y`。读取后转换为 `[X,Y,T]`。模拟记录 T=9061，真实记录 T=8192，采样率均为 150 MHz。HDF5 属性保存单次采集尺度、时间窗口、空间步长与预处理说明。

样本无损压缩，未裁剪、未新增滤波、未改变振幅。打包时对每个样本完整读回并核对信号哈希。

本包不包含 EDSR/PFT、BP 重建、论文绘图或视频工具。输出是**补全后的传感器信号**，不是 Figure 5C 的最终增强重建图像。

## 许可证

项目开源许可证尚待作者选择。
