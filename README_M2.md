# FIT5230 Assignment · Milestone 2 —— 总目录

> Note (Oct 2026): this is the Milestone 2 README, kept for the M2 script and variant details. The `docs/M2_*.md` / `docs/M3_M4_RESERVE_ZH.md` working notes it mentions are no longer in the public repository; only `docs/README_LOCAL_MAC_ZH.md` remains. Start from [README.md](README.md).

**队伍**:Dark.ModelBreakers · Theme 1 · Dark(攻击方)
**基于论文**:StealthDiffusion (ACM MM 2024), [arXiv:2408.05669](https://arxiv.org/abs/2408.05669)
**官方代码**:https://github.com/wyczzy/StealthDiffusion,锁定 commit `a156fb0`
**M2 截止**:2026-09-18 23:55 MYT

---

## 先看哪个

| 你想干什么 | 打开 |
|---|---|
| **搞懂我们做了什么**(队友从这里开始) | `docs/M2_HANDOVER_ZH.md` |
| 看要发到 Ed 的帖子 | `docs/M2_ED_POST.md` |
| 知道 M3/M4 还留了什么没做 | `docs/M3_M4_RESERVE_ZH.md` |
| 联系 Light 队 | `docs/M2_LIGHT_TEAM_OUTREACH_ZH.md` |
| 在 Mac 上跑起来 | `docs/README_LOCAL_MAC_ZH.md`,或本文件下面的「怎么跑」 |

---

## 目录结构

```
FIT5230 Ass1/
├── README_M2.md            ← 本文件
│
├── docs/                   人读的文档
│   ├── M2_HANDOVER_ZH.md          交接说明:做了什么、为什么、结果如何 ★从这里开始
│   ├── M2_ED_POST.md              Ed 帖草稿,数字已填实
│   ├── M3_M4_RESERVE_ZH.md        M3/M4 的保留项清单
│   ├── M2_LIGHT_TEAM_OUTREACH_ZH.md  联系别队的话术和记录表
│   ├── M2_DIRECTIONS_ANALYSIS_ZH.md  最早的论文改进方向分析
│   ├── M2_BASIC_README_ZH.md      M2 方案说明
│   └── README_LOCAL_MAC_ZH.md     Mac 本地环境搭建
│
├── notebooks/
│   ├── M2_Submission.ipynb            我们的 M2 notebook(Colab 提交用)
│   ├── M1_original_DO_NOT_EDIT.ipynb  M1 原件,不要动
│   └── archive/                       历史版本,留档
│
├── m2_scripts/             我们写的脚本,干净副本(方便查看和交接)
│
└── StealthDiffusion/       ★ 可直接运行的完整工作目录
    ├── (官方代码,一行没改)
    ├── m2_*.py, run_local_mac.sh, tests/   我们的脚本
    ├── checkpoints/        6 个权重,2.4G
    ├── data/               数据集(52 张图 + 官方 data 包)
    ├── results/            实验产出 ★
    └── results_BUGGY/      修 bug 之前的那次运行,留作对照
```

**注意**:`m2_scripts/` 和 `StealthDiffusion/` 里的 `m2_*.py` 是同一套代码的两份副本。
- 改代码请改 **`StealthDiffusion/` 里的那份**(它必须待在那里才能 `import` 官方模块)
- 改完记得同步回 `m2_scripts/`

---

## 怎么跑

```bash
conda activate fit5230m2                       # Python 3.10, torch 2.2.2
cd ~/Documents/"FIT5230 Ass1"/StealthDiffusion

bash run_local_mac.sh check                    # 检查 6 个权重在不在
N_ATTACK=20 bash run_local_mac.sh attack       # 跑四个变体,约 16 分钟(M5 Max / MPS)
bash run_local_mac.sh eval                     # 评测,含 JPEG 和缩放
bash run_local_mac.sh challenge                # 生成给别队的盲测包
```

`results/` 里已经有跑好的结果,想直接看数字不用重跑:

```bash
python m2_diag.py --runs B00=results/B00 B10=results/B10 ENS=results/ENS FULL=results/FULL
cat results/eval/summary.csv
```

**数据集**:`data/real` 和 `data/gen` 里是实际用到的 52 张图。原始 Imagenette 压缩包(约 700M)没有收进来,`m2_data.py` 会自动重新下载,需要时跑 `bash run_local_mac.sh data` 即可。

**不需要 GPU**,Mac 的 MPS 就能跑,约 11 秒一张图。

---

## 结果速览

条件 ASR(越高表示攻击越成功)。`*` = 该检测器参与了攻击损失。

| 版本 | E | R | D | **S(黑盒)** | 平均 | JPEG 75 后 | 0.5× 缩放后 |
|---|---|---|---|---|---|---|---|
| B00(官方) | 1.00* | 0.67 | 0.58 | 0.83 | 0.77 | 0.42 | 0.17 |
| B10 | 1.00* | 0.67 | 0.58 | 0.83 | 0.77 | 0.38 | 0.17 |
| **ENS** | 1.00* | 1.00* | 1.00* | **0.92** | **0.98** | 0.56 | 0.46 |
| **FULL** | 1.00* | 1.00* | 1.00* | **0.92** | **0.98** | 0.54 | 0.46 |

三条结论:

1. **集成有效**,而且在从没参与训练的 S 上也涨了(0.83 → 0.92),这是真的迁移性提升。
2. **攻击扛不住后处理**。存一次 JPEG 掉一半,缩放一次官方版只剩两成 —— 而论文所有数字都只测了第一列。
3. **用 AUC 一看,四个检测器其实一个都没真被骗**。我们的图被推得比真照片还像真的,防守方翻一下阈值就全抓。

细节和为什么,全在 `docs/M2_HANDOVER_ZH.md`。

---

## 还没做完的

- [ ] Ed 帖里填队名、两个成员、M1 帖链接、Colab 链接
- [ ] `bash run_local_mac.sh challenge` 生成盲测包
      ⚠️ **只发 `challenge/challenge_public.zip`,`challenge/private/challenge_key.csv` 绝对不能发出去**
- [ ] **去 Ed 找 Theme 1 的 Light 队** ← 唯一的瓶颈,M3 有 14 分押在跟别队的真实互动上,没法最后突击
- [ ] M2_Submission.ipynb 还没在 Colab 里跑出带输出的版本(M3 的 Colab 评分明确要求所有 cell 有已渲染输出)
