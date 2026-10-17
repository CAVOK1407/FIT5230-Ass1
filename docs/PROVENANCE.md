# 哪些是官方的,哪些是我们的

一句话:**官方代码我们一个字节都没改。** 我们的全部工作都在新增文件里。

这可以自己验证,不用信我们的话。在还带着上游 `.git` 的那份仓库里跑:

```bash
git rev-parse --short HEAD          # a156fb0  —— 论文官方仓库锁定的 commit
git diff HEAD --name-only           # 空输出   —— 官方文件 0 处改动
git status --porcelain | grep '^??' # 我们新增的全部文件
```

实测结果(2026-09-15):

```
=== 上游 commit ===
a156fb0

=== 官方文件改动 ===
0 处 —— 官方代码一个字节未改
```

---

## 一、官方文件(作者写的,我们没动)

来自 https://github.com/wyczzy/StealthDiffusion,commit `a156fb0`。

**我们真正用到的三个:**

| 文件 | 作用 | 我们怎么用 |
|---|---|---|
| `diff_latent_attack.py` | 官方 LAO 实现 | `import` 它的 `ddim_reverse_sample`、`diffusion_step`、`preprocess`、`calculate_psnr`、`calculate_ssim`。**论文弱点 (b) 就是读这个文件发现的** |
| `other_attacks.py` | 检测器加载器 `model_selection()` | E / R / D / S 四个检测器全从这里来 |
| `ControlVAE.py` | Control-VAE 的 `NewEncoder` / `NewDecoder` | 第三步解码 |

**现在没用、但 M4 要用的:**

| 文件 | 作用 |
|---|---|
| `train_vae.py` | 训练 Control-VAE。M4 的「重训 Control-VAE」(4 分)要用它 |

**其余官方文件,我们一个都没调用**(作者训练检测器和跑他们自己实验的部分):

```
main.py  train.py  eval.py  validate.py  network.py  demo_dir.py
distances.py  earlystop.py  eval_config.py  loggers.py  util.py  utils.py
data/__init__.py  data/datasets.py
dataset_caption/__init__.py  dataset_caption/ours_label.py
models/basicblock.py  models/network_dncnn.py
networks/__init__.py  networks/base_model.py  networks/lpf.py
networks/resnet.py  networks/resnet_lpf.py  networks/trainer.py
options/__init__.py  options/base_options.py  options/test_options.py  options/train_options.py
README.md  requirements.txt
```

---

## 二、我们新增的文件(8 个,全是新文件)

| 文件 | 作用 |
|---|---|
| `m2_stealth.py` | **攻击主程序**。五个开关都在这里。默认参数 = 复现官方流程 |
| `m2_data.py` | 建数据集(真照片 + SD 生成图)+ 检测器基线检查 |
| `m2_evaluate.py` | 评测:条件 ASR、真实图 FPR、AUC、JPEG/缩放、画质 |
| `m2_diag.py` | 诊断:把攻击拆成 clean → PGD → 普通 VAE → Control-VAE 四阶段 |
| `m2_challenge.py` | 盲测包生成 + 给别队评分 |
| `m2_device.py` | 设备适配,让只支持 CUDA 的官方代码跑在 Mac 的 MPS 上 |
| `run_local_mac.sh` | 一键脚本 |
| `tests/bn_mode_repro.py` | BatchNorm bug 的最小复现(40 行,不依赖官方代码) |

运行时产生的目录:`data/real`、`data/gen`、`results/`、`results_BUGGY/`、`checkpoints/`。

---

## 三、我们"修改"的:严格说是零

这是最容易被误解的一点。我们**没有改官方代码**,而是在 `m2_stealth.py` 里**重新实现了 LAO 循环**,其余部分 import 官方的:

```python
import other_attacks                          # 检测器
from diff_latent_attack import (              # DDIM 反演、扩散步、预处理、指标
    ddim_reverse_sample, diffusion_step, preprocess,
    calculate_psnr, calculate_ssim)
from ControlVAE import NewEncoder, NewDecoder # Control-VAE
```

官方那个 `diffattack()` 函数(整个 LAO 循环)我们没有调用,而是在 `attack_one()` 里重写了。

**为什么这样做,而不是直接改官方文件:**

1. 官方的循环把所有选择写死了,没有开关 —— 做不了四个变体的消融对比。
2. 如果改官方文件,别人无法验证我们到底改了什么,也无法把我们的改动和作者的原始实现分开。
3. 现在任何人 `git diff` 一下就能确认:**官方 0 改动,我们的东西全在新文件里,边界清清楚楚。**

改写的那部分具体是什么,见 `docs/M2_HANDOVER_ZH.md` 第 3 节。

---

## 四、⚠️ 移进 GitHub 仓库之后,这个边界丢了

把 `StealthDiffusion/` 整个放进我们自己的 git 仓库之后,官方文件和我们的文件在 git 眼里变成了一样的「已跟踪文件」,`git diff` 不再能区分。

**要保住可复现性这一分,记住两件事:**

1. 上游 commit 是 **`a156fb0`**,写在 Ed 帖和 README 里。
2. 本文件第一、二节的清单就是边界。改了官方文件的话,回来更新这里。

想恢复自动验证,在仓库外另克隆一份对照:

```bash
git clone https://github.com/wyczzy/StealthDiffusion /tmp/sd-upstream
cd /tmp/sd-upstream && git checkout a156fb0
diff -rq /tmp/sd-upstream <你的 StealthDiffusion 目录> \
  --exclude=.git --exclude=checkpoints --exclude=results\* \
  --exclude=data --exclude=__pycache__ --exclude=.cache --exclude=tests \
  --exclude='m2_*' --exclude=run_local_mac.sh
# 无输出 = 官方文件未被改动
```

---

## 五、notebook 起什么作用

**`notebooks/M2_Submission.ipynb` 不是代码的源头,是投递方式。**

它做的事,按顺序:

1. 克隆官方仓库,`git checkout a156fb0`
2. 建 Python 3.10 环境(torch 2.2.2 / diffusers 0.12.1 等)
3. **把 6 个 `m2_*.py` 用 `%%writefile` 写到硬盘上**
4. 语法检查
5. 建数据集 → 跑四个变体 → 评测 → 生成盲测包

**为什么要有它**:M2 和 M3 都要交 Colab 链接。评分的人点开 Run all 就能跑,不用装环境、不用找权重、不用读 README。

### ⚠️ notebook 是**生成的**,不要手改

它由 `build_m2_notebook.py` 从 `scripts/` 目录构建,脚本被**逐字嵌入**成 `%%writefile` cell。所以:

```
scripts/m2_*.py   ←  真正的源头,改这里
      ↓  python build_m2_notebook.py --scripts scripts --out M2_Submission.ipynb
M2_Submission.ipynb  ←  生成物
```

**改了 `.py` 却没重新生成 notebook,notebook 里就是旧代码。** 这已经发生过一次:修完 BatchNorm bug 之后 notebook 没重建,里面嵌的一直是有 bug 的版本,直到 2026-09-15 才发现并重建(同时补进了 `m2_diag.py`)。

改完脚本,一条命令:

```bash
python build_m2_notebook.py --scripts scripts --out notebooks/M2_Submission.ipynb
```

### 还没做完的

notebook **从来没有在 Colab 里完整跑出过带输出的版本**。

M3 的 Colab 评分明写:「All relevant code cells must display pre-rendered execution outputs」。这是一件要专门留时间做的事,不是提交前顺手能补的。
