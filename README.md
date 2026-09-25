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


## 样本下载与放置

**样本网盘下载链接：待补充。**

两个 HDF5 样本不包含在 Git 仓库中。下载后，请将文件放置为：

```text
data/
  manifest.json
  simulation.h5
  real.h5
```

如果下载的是完整演示 ZIP，请将其中 `SPARC_demo/data/` 下的两个 `.h5` 文件复制到本仓库的 `data/` 目录。模型权重 `weights/sparc.pt` 已包含在仓库中。样本就位后，再运行下方训练或推理命令。

`SHA256SUMS.txt` 包含仓库文件及两个样本的校验值，可用于核对下载内容。

## 安装

在本目录下执行：

```bash
python -m pip install -r requirements.txt
```

依赖仅为 PyTorch、NumPy、h5py。支持 CPU 或 CUDA；需要 GPU 时安装与你机器匹配的 PyTorch。已在 Python 3.11 / PyTorch 2.5.1 上运行检查。
