# SD-MSCF-DET

本文仓库提供论文所需的最小训练代码，用于 RGB-红外双模态目标检测。主模型 SD-MSCF-DET 采用 P2-P5 四尺度检测结构，在双流骨干中使用 MSCF-C3 与 SD-PSA，并结合 SOEP 小目标增强金字塔和 RFPN 特征金字塔增强模块。代码标识符分别写作 `MSCF_C3` 和 `SD_PSA`。

仓库仅包含模型定义、定制的 Ultralytics 训练框架和训练入口，不包含数据集、训练权重、实验日志或可视化结果。

## Environment

建议使用 Python 3.10、CUDA 11.8 或更高版本，以及与本机 CUDA 匹配的 PyTorch。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch，再安装本仓库：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e .
```

## Dataset

数据配置模板位于 `configs/data.yaml`。将其中的 `path` 修改为数据集根目录。默认目录约定如下，RGB 与红外图像需要具有相同的相对路径和文件名：

```text
M3FD_split/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
|-- images_ir/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

如使用其他数据集，请同步修改 `names`。模型配置中的类别数会在训练时根据数据配置自动调整。

## Training

复现原始训练设置：

```bash
python train.py --data configs/data.yaml --epochs 300 --batch 8 --workers 6 --no-amp
```

常用参数可通过命令行覆盖：

```bash
python train.py --help
python train.py --device 0 --imgsz 640 --amp
```

训练结果默认保存在 `runs/train/SD-MSCF-DET`，该目录不会被 Git 跟踪。

## Files

- `configs/SD-MSCF-DET.yaml`: 论文主模型结构配置。
- `configs/data.yaml`: 双模态数据集模板。
- `train.py`: 最小训练入口。
- `ultralytics/nn/modules/sd_mscf.py`: MSCF-C3 与 SD-PSA 的完整实现。
- `ultralytics/nn/Neck/`: SOEP 与 RFPN 相关模块。
- `ultralytics/`: 仅保留 YOLOMM 双模态检测训练所需的框架代码。

## License

本项目基于 Ultralytics 代码修改，按 GNU Affero General Public License v3.0 发布，详见 `LICENSE`。使用者还需遵守所用数据集各自的许可条款。

## Citation

论文正式发表后，请在此处补充论文题目、作者、期刊和 DOI/BibTeX 信息。
