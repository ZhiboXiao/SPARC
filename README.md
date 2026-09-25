# SPARC 

**Sparse Propagation-Aware ReCovery(SPARC)**。包含模型、已训练权重、训练/推理代码，以及模拟、真实真实样本。

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
