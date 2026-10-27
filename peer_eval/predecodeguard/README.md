# PreDecodeGuard 技术迁移试点

本模块把有界扰动、早停和逐文件复验流程适配到对方的原始潜变量接口。准备阶段和正式试点分开保存，原有 M3 证据保持不变。

## 用户手动复跑

用户已要求提供可亲自运行的复跑命令。以下命令会真正重新计算同一套七个早停 PGD 样本，并复制同样的三个原样对照；所有参数和分配保持一致，仅作本地可重复性检查：

```bash
cd "/Users/wangyunjie/Documents/ChatGPT/FIT5230/github/FIT5230-Ass1"
/opt/anaconda3/envs/fit5230m2/bin/python -B -u -m peer_eval.predecodeguard.reproduce
```

终端会显示开始时间、参数、逐样本结果表、官方验证状态，以及与原七加三结果的逐文件哈希比较。每次输出到 `artifacts/local_reproduction_2026-10-18/runs/` 下的新目录；原结果和对外盲审 ZIP 不变。复跑输出含原文件名、对照身份和评分，仅供本地查看，不是新的盲审交付包。

命令末尾加 `--check-only` 时只检查路径、协议和文件完整性，不加载模型、不优化，也不创建实验结果。脚本准备阶段只执行了该检查。

## 当前交付：7 个攻击输出 + 3 个对照，匿名盲审

对方最终明确确认：只接收一批，7 个早停 PGD 输入、3 个原样对照；文件名打乱；评分和对照映射必须等其完成复核后才提供。用户确认旧的 8+2 包尚未发送。

当前交付记录位于 `artifacts/pilot_blind_7plus3_2026-10-18/`。复用了下述本地运行中的 7 个已验证输出，保留原有两个对照，并按固定文件名哈希规则选出第三个样本、恢复其原始文件；**没有新增优化或追加批次**。新增对照是在原本地运行后选定的，不能将这次调整称为看结果前完成的七样本预注册实验；原八次尝试的完整记录保留。

新版 ZIP 仅包含 10 个匿名 latent、匿名文件哈希、简短说明和空白复核表。评分、原名映射及对照身份全部留在本地 `withheld_until_review/`，等对方完成复核后再披露。内部按原名复验，再逐字节核对匿名交付文件；所有文件满足预算且官方评分一致。匿名哈希仍可能通过比对原始包揭示对照，因此说明中请对方复核前不要评分或比对原始哈希。

## 原始本地运行记录（8+2，已被新版交付替代）

2026-10-18，用户提供了对方对原始单批提案的确认回复。按材料包预算执行了**唯一一批 10 个文件：8 个早停 PGD 样本、2 个事先选定的未修改对照**，没有执行准备阶段拟议的三批实验。

- 参数在查看攻击结果前固定：原始单位 `L∞ ≤ 0.1`、零容差，最多 40 步、步长 0.01、`early_stop_margin=0.001`。早停条件为量化后 CPU 分数严格小于 `0.6518052091598511`，官方分类阈值仍为 `0.6528052091598511`。
- 51 项合成测试通过。8/8 攻击尝试改判，2 个对照保持原样；按全部输入计为 8/10。十个保存文件全部有效，官方评分器与适配器分数最大差为 0。
- 攻击实际使用 1–8 步，最大 `L∞=0.080078125`。固定步长可能越过阈值较多，早停不等于已经找到最小扰动。
- **尚未解码或人工复核，最终内容保持规避率待定。**不能把 8/8 检测器改判称为 8/8 已确认成功。

确认记录、预先固定的参数与对照分配、代码副本、测试日志及完整结果位于 `artifacts/pilot_single_batch_2026-10-18/`。该目录保存历史本地运行，旧交付包不再使用；当前实际回传按上方最终确认的 7+3 方案。

## 当前验证结果

2026-10-18 的结果位于 `artifacts/preparation_2026-10-18/`：

| 目录 | 已完成的检查 | 结果 |
| --- | --- | --- |
| `baseline_verified/` | 冻结材料包校验、CPU 基线、官方评分器复验 | 10/10 原本检出；适配器与官方分数最大差为 0 |
| `copy_verified/` | 原文件复制后重新加载、评分 | 10/10 有效；元素、判决不变，扰动为 0 |
| `roundtrip_verified/` | FP16 → FP32 → FP16 保存、重载、评分 | 10/10 有效；元素、判决不变，扰动为 0 |
| `plan/` | 生成实验计划、逐样本分配与种子 | 仅计划，尚未执行 |

准备阶段的 44 项合成测试已通过；加入固定 margin 后为 51 项。测试使用虚构清单、微型张量和玩具分类器，不优化对方样本。材料包源码、模型、源潜变量及原有 M3 结果没有修改；模型权重、潜变量文件和逐步 trace 被忽略，配置、CSV 与汇总日志可随代码保存。当前没有提交或推送这些新增文件。

`baseline/` 保留首次调用子进程时的失败记录：嵌套 Torch/OpenMP 进程在本机沙箱触发共享内存错误 OMP 179。最终复验通过 `runpy` 执行**未经修改的**对方 `score.py`，并保存标准输出、错误输出和复现命令；独立运行官方 CLI 也已成功。有效结果请读取带 `_verified` 的目录，不能把首次失败目录当成验证通过。

## 模块与约定

| 文件 | 职责 |
| --- | --- |
| `adapter.py` | 冻结 CNN，设置 `eval()`，按 checkpoint 标准化；提供 CPU logit 和评分接口 |
| `methods.py` | 原始单位的随机对照、L-PGD、L-STOP；FP16 舍入后的严格预算约束 |
| `contract.py` | 校验输入、协议、确认凭据和运行条件 |
| `package_lock.json` | 固定材料包文件哈希 |
| `protocol.pending.json` | 待确认的样本、参数、预算、批次和复核方案 |
| `run.py` | 基线、无扰动对照、计划、正式实验和既有输出审计入口 |
| `evaluate.py` | 重载实际文件、检查预算与完整性、复验官方分数 |
| `report.py` | 保留失败的统计、内容复核合并、CSV/JSON/Markdown 报告 |

输入是有限的 `[1,16,128,128]` FP16 数组。预算为原始 latent 单位的 `L∞ ≤ 0.1`，容差为零；不采用 RGB 的 `[0,1]` 裁剪或 ImageNet 标准化。阈值固定为 `0.6528052091598511`，`score >= threshold` 表示检出，只有严格低于阈值才算改判。

候选输出先投影，再向预算区间内进行 FP16 舍入修正。最终验收对保存并重载的 FP16 文件转 FP32 后计算扰动，按照官方接口检查；不能用内存中的 FP32 候选代替交付文件验收。

## 复现准备工作

在仓库根目录运行。环境已验证为 Python 环境 `fit5230m2`、Torch 2.2.2、NumPy 1.26.4；依赖版本见 `requirements.txt`。以下 `PDG_OUT` 应设为新的结果位置，每条命令的 `--out` 目录必须尚不存在，程序不会覆盖旧结果。

```bash
cd /Users/wangyunjie/Documents/ChatGPT/FIT5230/github/FIT5230-Ass1
PDG_PY=/opt/anaconda3/envs/fit5230m2/bin/python
PDG_PACKAGE=/Users/wangyunjie/Documents/ChatGPT/FIT5230/predecodeguard/PreDecodeGuard_ModelBreakers_Pilot_v1
PDG_OUT=/tmp/pdg_preparation_recheck
export PYTHONDONTWRITEBYTECODE=1

"$PDG_PY" -m peer_eval.predecodeguard.run baseline \
  --package "$PDG_PACKAGE" --out "$PDG_OUT/baseline"
"$PDG_PY" -m peer_eval.predecodeguard.run control --kind copy \
  --package "$PDG_PACKAGE" --out "$PDG_OUT/copy"
"$PDG_PY" -m peer_eval.predecodeguard.run control --kind roundtrip \
  --package "$PDG_PACKAGE" --out "$PDG_OUT/roundtrip"
"$PDG_PY" -m peer_eval.predecodeguard.run plan \
  --package "$PDG_PACKAGE" --out "$PDG_OUT/plan"
"$PDG_PY" -m unittest discover -s peer_eval/predecodeguard/tests -v
```

这些准备命令不会生成真实扰动。`plan` 只读取协议与清单、检查哈希并写计划；不加载模型或读取潜变量数组。其他准备命令只评分原始数据或不改变元素值的复制/类型回环。

## 准备阶段拟议的对照实验（并非本次执行范围）

| 组别 | 实现与固定参数 | 要比较的问题 |
| --- | --- | --- |
| 原始基线、复制、读写回环 | 不加扰动 | 评分、文件和类型转换是否一致 |
| `random` | 原始单位均匀随机噪声，一次采样，再投影与量化 | 普通扰动的分数变化 |
| `lpgd` | 从原样本开始，以良性标签为目标；最多 40 步，步长 0.01，预算 0.1 | 优化是否超出随机对照效果 |
| `lstop` | 与 L-PGD 同一路径，仅增加量化后参考分数早停 | 是否用更少步骤、较小扰动达到判决变化 |

协议明确采用 `base_seed_plus_manifest_index`：每个样本的种子是方法基础种子 42 加清单中的零起始索引。同一套样本顺序在计划、运行和回传中保持一致。`random` 的统一配置虽然保留 `steps: 40`，算法只进行**一次噪声采样**，实际记录步数为 1；其 `alpha` 不参与计算。PGD 当前没有随机初始化。

L-STOP 的每次决策使用与 FP16 保存表示等价的数组，通过独立 CPU 回调评分；最终输出还必须经过真实保存、重载和官方评分器复验。已低于阈值的初始样本或零预算输入会返回原值，相关状态保留在 trace 中。

## 协议与运行记录

准备阶段提案曾请求三个方法各十个文件，共 30 个返回文件；**扩大批次未获确认，也没有执行**。历史本地运行配置为 `artifacts/pilot_single_batch_2026-10-18/protocol.confirmed.json`。当前最终交付协议为 `artifacts/pilot_blind_7plus3_2026-10-18/protocol.confirmed.json`，只包含七个复用的 `lstop` 输出和三份原样对照。不应重新运行或根据本批结果调参后追加样本。

默认 `protocol.pending.json` 的状态是 `awaiting_peer_confirmation`。任何 `experiment` 命令（包括 `random`）都会在加载模型、创建输出或读取源数组之前阻止执行。这里的协议状态不能用来替代真实同意。

收到真实确认后，由执行者保存确认原文及来源，核对它覆盖的提案版本，再生成独立的确认协议文件。`agreement` 必须记录确认人、带时区时间、来源引用、证据文件路径及 SHA-256，以及冻结提案的 `proposal_sha256`。后者通过 `contract.proposal_digest()` 计算，覆盖预算、样本、参数、种子规则、批次和复核要求；后续改变提案必须重新确认。代码能检查文件和哈希，不能判断确认原文是否真实、是否充分，执行者仍须核实。

仅在**真实协议已确认**后，才按用户已授权的范围使用下列正式实验入口；当前不要执行：

```bash
"$PDG_PY" -m peer_eval.predecodeguard.run experiment \
  --package "$PDG_PACKAGE" \
  --protocol /absolute/path/to/genuinely_confirmed_protocol.json \
  --method lpgd --execute-approved --out /absolute/path/to/new_experiment_output
```

其他方法使用各自独立的新输出目录，且必须在已确认方法列表内。确认状态、保存的证据、匹配的提案摘要以及显式 `--execute-approved` 缺一不可。程序不会因收到确认就自动运行。

## 文件验收与结果解读

内部验证目录 `returned/` 必须恰好包含清单中的十个原始 `.npy` 文件名；CSV、配置、trace 和复核表放在目录外。对外盲审 ZIP 使用打乱后的匿名文件名，内部保留映射，并核对匿名文件与内部已验证文件的哈希一致。验证流程是：哈希校验 → 实际文件重载 → shape/dtype/有限值检查 → 零容差预算检查 → CPU 评分 → 官方评分复验。缺失、无效、超预算或生成失败都保留逐样本记录；内部验证目录出现额外文件会使整个批次无效。

运行输出包括 `run_config.json`、`per_sample.csv`、`summary.json`、`summary.md`、`review_template.csv` 和官方评分日志；正式方法另外保存 `traces/` 与 `execution_records.json`。计划输出包括 `experiment_plan.json`、`planned_samples.csv`；它们不是实验结果。

主要分母为“原本检出的实际方法尝试数”，失败及缺失输出不从分母删除，未修改对照不算尝试。另列全部尝试分母。文件有效且检测器改判只称为候选；确认有效规避还需要固定流程解码成功、图像可用且原目标内容保留。改判候选没有复核或标签不确定时保留 pending，最终确认率为 `null`，不能解释为零成功或已完成复核。若所有尝试都未产生有效改判，确认成功数可以直接确定为零，但这仍不代表所有输出已经完成人工复核。

`review_template.csv` 是**内部协调表，不是可直接发给评审者的盲审表**。协调者要将解码图片随机编号，隐藏方法、分数、源样本与对照对应关系，并保管映射。复核后按文件名与返回文件 SHA-256 合并，填写 `decode_status`（`ok/failed/uncertain`）、`usability`（`usable/unusable/uncertain`）、`content_label`（`retained/changed/uncertain`）、`reviewer` 与 `decoder_reference`。清晰的否定结果计为失败，不确定结果继续待定。

候选复核完成不代表所有输出都已解码复核；报告分别列候选状态及 `all_output_*` 全文件复核数量。审计已有输出可使用 `audit --method ... --returned ... --review ... --out ...`，并传入对应的 `--package` 和 `--protocol`；它只读取既有文件，不生成扰动。

本轮只有十个历史训练阳性源样本。多个方法重复处理仍然是十个独立源样本，不能据此报告 FPR、ROC-AUC、总体准确率或留出数据泛化结论。现阶段也没有内容保持后的攻击成功率。
