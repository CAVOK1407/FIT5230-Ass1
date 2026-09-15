# FIT5230 Assignment 1

Dark.ModelBreakers · Theme 1 · Milestone 2

本仓库包含代码、Notebook、交接文档、实验数据和已有结果。队友请先阅读 [M2 交接说明](docs/M2_HANDOVER_ZH.md)，完整项目说明见 [README_M2.md](README_M2.md)。

## 从这里开始

| 内容 | 入口 |
| --- | --- |
| 交接说明与结果解读 | [docs/M2_HANDOVER_ZH.md](docs/M2_HANDOVER_ZH.md) |
| M2 提交 Notebook | [notebooks/M2_Submission.ipynb](notebooks/M2_Submission.ipynb) |
| Mac 环境依赖与运行说明 | [docs/README_LOCAL_MAC_ZH.md](docs/README_LOCAL_MAC_ZH.md) |
| 可运行的项目代码 | [StealthDiffusion/](StealthDiffusion/) |
| 已有评测结果 | [StealthDiffusion/results/](StealthDiffusion/results/) |
| M3/M4 待办 | [docs/M3_M4_RESERVE_ZH.md](docs/M3_M4_RESERVE_ZH.md) |

## 下载并运行

```bash
git clone https://github.com/CAVOK1407/FIT5230-Ass1.git
cd FIT5230-Ass1/StealthDiffusion
```

先按 [本地环境说明](docs/README_LOCAL_MAC_ZH.md) 安装依赖。本仓库已包含官方代码和 M2 脚本，可以跳过该文档中另行克隆官方仓库、复制脚本的步骤。

### 模型权重

6 个模型权重约 2.6 GB，单独下载，不包含在 Git 仓库中。原项目 [权重说明](StealthDiffusion/README.md) 提供 [百度网盘下载链接](https://pan.baidu.com/s/1gWCPS--IUbu3QWKlWD5bRA)，提取码为 `z8se`。也可以向队内获取已有权重文件。

下载后把下面 6 个文件直接放入 `StealthDiffusion/checkpoints/`，保留文件名：

```text
checkpoints/
├── efficientnet-b0.pth
├── resnet50.pth
├── deit.pth
├── swin-t.pth
├── Controlvae.pt
└── dncnn_color_blind.pth
```

激活环境后，在 `StealthDiffusion/` 内检查并运行：

```bash
conda activate fit5230m2
bash run_local_mac.sh check
N_ATTACK=20 bash run_local_mac.sh attack
bash run_local_mac.sh eval
```

只查看文档与已有实验结果不需要下载权重。

## 协作说明

- 私有仓库需要先接受仓库拥有者发出的 GitHub 邀请才能访问。
- 日常修改请从最新 `main` 新建分支，提交后通过 Pull Request 合并。
- M2 脚本以 `StealthDiffusion/` 中的副本为准；修改后同步到 `m2_scripts/`。
- `notebooks/M1_original_DO_NOT_EDIT.ipynb` 保留为 M1 原件。

## 来源

基于 [wyczzy/StealthDiffusion](https://github.com/wyczzy/StealthDiffusion)，原有项目说明和作者致谢保留在 [StealthDiffusion/README.md](StealthDiffusion/README.md)。
