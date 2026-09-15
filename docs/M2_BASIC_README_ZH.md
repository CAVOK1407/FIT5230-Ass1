# M2 基础版说明（`m2/StealthDiffusion_M2_Submission.ipynb`）

日期：2026-09-14 ｜ 队伍：Dark.ModelBreakers ｜ 论文：StealthDiffusion (ACM MM 2024)

## 1. 这份 notebook 是什么

在 M1 notebook（`StealthDiffusion_Baseline_Submission_1.ipynb`）的基础上做的 M2 版本。
M1 的第 3 节（拉官方代码、uv 建 Python 3.10 环境、挂 6 个 checkpoint、单 GPU patch）原样保留；
M1 的第 4–6 节（合成数据小实验）不再放进 M2，M2 用真实的攻击流程替代。

M2 新增的内容全部在 4 个脚本里，notebook 用 `%%writefile` 把它们写进官方仓库目录，然后用官方的 Python 3.10 环境运行：

| 脚本 | 作用 |
|---|---|
| `m2_data.py` | 准备数据：Imagenette 真实照片 + Stable Diffusion 生成图（同样 10 个类），dev/test 划分，检测器极性检查 |
| `m2_stealth.py` | **改进后的攻击**。默认参数 = 官方流程（B00）；每个改进都是一个开关 |
| `m2_evaluate.py` | 对保存下来的 PNG 做评测：条件 ASR、真实图误报率、AUC、JPEG/缩放鲁棒性、PSNR/SSIM/LPIPS/Linf、耗时 |
| `m2_challenge.py` | 给其他队的挑战包（真图 / 原始生成图 / 攻击图混合，隐藏标签）+ 评分器 |

## 2. 四个变体

| 变体 | 攻击损失里的代理检测器 | LAO 损失算在哪条解码路径上 | 目标函数 | 额外项 |
|---|---|---|---|---|
| B00（官方） | E | 普通 VAE（官方） | 交叉熵（官方） | 无 |
| B10 | E | **最终 Control-VAE 路径** | 交叉熵 | 无 |
| ENS | **E + R + D 集成** | 普通 VAE | 交叉熵 | 无 |
| FULL | E + R + D 集成 | 最终 Control-VAE 路径 | **margin 损失** | **LPIPS 约束 + 输入多样性** |

S（Swin）从不进入任何损失，所以对四个变体都是真正的迁移目标。B00/B10 只有 E 是白盒，R/D/S 都是迁移。

对应 `m2_stealth.py` 的开关：
```
--surrogates E,R,D      代理集成（PGD 阶段和 LAO 损失都用平均 logits）
--decode_path final     LAO 损失算在最终 Control-VAE 解码路径上（同权重、同 0.1825 缩放、同控制特征）
--loss margin           relu(kappa - (logit_real - logit_AI))，kappa 默认 3
--lpips_weight 1.0      LPIPS(输出, 输入) 加进目标
--di_prob 0.5           输入多样性：随机缩放到 90–100% 再补零到 224
```
没改的：DDIM 反演、2 个扩散步（start_step 18 / 20）、AdamW lr 1e-2、10 次 LAO、PGD eps 4/255 × 10 步、Control-VAE 权重、224 分辨率、VAE 采样种子 8888。

## 3. 怎么跑（Colab A100）

1. 打开 notebook，运行时选 **A100**。
2. 第 3.3 格路径（默认按"用个人账号跑、StealthDiffusion 在 Shared with me"配置）：
   - 先在 Drive 网页里给共享的 `StealthDiffusion` 文件夹加快捷方式到 My Drive（Shared with me → 右键 → Organise → Add shortcut to Drive → My Drive）。Colab 只挂 My Drive，不挂 Shared with me，不加快捷方式就读不到。
   - `CKPT_SRC` = `/content/drive/MyDrive/StealthDiffusion/checkpoints`（走快捷方式，只读即可）。
   - `DRIVE_ROOT` = `/content/drive/MyDrive/FIT5230_M2_out`（放当前账号自己的 My Drive，输出不依赖对共享文件夹的写权限）。
   - 如果改用 Monash 账号（文件本来就在它的 My Drive）跑，就不用加快捷方式，把两格改成 `/content/drive/MyDrive/FIT5230_Ass1/StealthDiffusion/checkpoints` 和 `.../M2`。
3. 第 4.1 格：`N_DEV=10`、`N_TEST=100`；`GEN_MODEL` 默认 SD 1.5 社区镜像，失败会自动改用 `SD_MODEL`（M1 用过的 SD 2.1-base 镜像），manifest 里记录实际用了哪个。
4. 第 5.1 格：`N_ATTACK` 默认 20（四个变体约 40–80 分钟）。时间够就改成 50 或 100。
5. Run all。结果在 Drive：`DRIVE_ROOT/runs/<变体>/`、`DRIVE_ROOT/eval/summary.csv`、`DRIVE_ROOT/challenge/challenge_public.zip`。
6. 中断了重新 Run all 即可：攻击脚本带 `--resume`，已完成的图会跳过。

## 4. 评测口径（写 Ed 帖时照抄）

- **条件 ASR**：只在"攻击前被该检测器正确判为 AI"的图里，算攻击后被判为真实的比例；分母在 Table 1 下面单独列出。
- **真实图误报率 FPR**：真实测试照片被判为 AI 的比例。
- **AUC（真实 vs 攻击图）**：检测器调阈值还能不能分开；0.5 = 分不开。
- **JPEG75 / resize05**：原图和攻击图做相同处理后再算，分母重新确定。
- **PSNR / SSIM / LPIPS / Linf**：攻击图相对 224×224 输入。初始 PGD 的 eps 不约束最终输出，所以只报测出来的失真，不声称预算。
- 白盒格子在表里用 `*` 标出，不把白盒结果说成迁移。

## 5. 本地做过的检查

- 4 个脚本在与 Colab 相同的环境（Python 3.10 + torch 2.2.2 + diffusers 0.12.1 + timm 0.6.12）里通过编译。
- `m2_scripts/tests/cpu_stub_test.py`：用随机小模型在 CPU 上把四个变体、resume、评测、挑战包整条流程跑通（`python cpu_stub_test.py <官方仓库目录> <m2_scripts 目录>`）。
- `m2_scripts/tests/notebook_cells_test.py`：notebook 的表格/画图单元格在上面的输出上执行通过。
- **没做过的**：A100 上的真实运行。第一次跑建议先 `N_TEST=20, N_ATTACK=5` 走通全流程，再放大。

## 6. 还要你们自己填的

- notebook 顶部的队员名字；第 8 节"Engagement"（目标 Light 队、发了什么、对方返回什么、我们的回应）。
- Ed 帖：把 Table 1–5、Figure 2–5 和挑战包链接贴上去；Moodle 交 Ed 链接的 PDF/文本。
