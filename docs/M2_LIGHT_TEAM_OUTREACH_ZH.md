# 联系 Light 队 —— 话术、流程、记录表

M2 的"Engagement with other teams"占 **2 分(全课 8 分里的 2 分)**。评分写的是"Did the team have meaningful attacks/defenses? Did they respond to others' ideas thoughtfully?"——**光发邀请不算**,必须真的交换了东西、拿到对方结果、并且有来有回。所以越早发出去越好。

---

## 一、怎么挑队伍

按优先级:

1. **Theme 1 的 Light 队**(最优先)。同一个主题,他们的任务就是"防御针对生成式 AI 的对抗攻击"——和我们正好对打。
2. 他们的方法**接受图片输入**(而不是只处理文本/音频)。我们的挑战包就是一堆 PNG,对方能直接跑。
3. 他们有**可复现的预测接口**(Colab / GitHub / 能返回预测结果)。
4. 如果 Theme 1 的 Light 队都挑不到,退而求其次找 **Theme 2 Light**(TTI 安全)——他们里也有做 AI 图检测的。

**挑之前先看清楚对方论文到底做什么**,不要默认每支 Theme 1 的队都在做图像取证检测器。有的可能做的是输入净化(purification)、有的是对抗训练、有的是水印。这决定了我们"针对他们哪一环"怎么写。

对应关系:

| 对方做的是 | 我们针对的那一环 | 怎么说 |
|---|---|---|
| AI 图检测器 | 他们依赖的证据类型(频域 / 噪声残差 / 语义特征) | "想测你们的检测器在潜空间攻击下还剩多少" |
| 输入净化 / 去扰动 | 净化能不能洗掉潜空间扰动 | "我们的扰动来自潜空间不是像素噪声,想看你们的净化还管不管用" |
| 对抗训练过的检测器 | 训练时没见过的攻击类型 | "你们对抗训练用的是像素域样本,我们这个是潜空间的,想看泛化" |
| 水印 / 内容凭证 | VAE 重建会不会洗掉水印 | "我们的流程必过一次 SD 的 VAE,想测你们的水印能不能扛住" |

---

## 二、第一封(公开回帖,推荐)

发在**对方 Ed 帖底下的回复**,公开的互动更容易被看到、也算分。

> Hi `<team name>`,
>
> We are Dark.ModelBreakers (Theme 1, Dark). Our M2 work improves StealthDiffusion, an attack that edits AI-generated images so that forensic detectors call them real photos. We would like to test our attack against `<their specific component>`, and give you something to test back.
>
> **What we can give you now:** a blinded challenge set of 224×224 PNGs — real photographs, unmodified Stable Diffusion images, and Stable Diffusion images processed by our attack, all shuffled with the labels hidden. You label each one `1 = generated` / `0 = real`, send back a CSV, and we run our scorer and post your four numbers publicly: overall accuracy, detection rate on **raw** generated images, detection rate on **attacked** images, and false-positive rate on real photos. The gap between the middle two is exactly what our attack costs your method.
>
> **What would help us:** if you can share a detector wrapper — a `load_detector()` returning a `torch.nn.Module` whose forward takes float RGB `[B,3,H,W]` in `[0,1]` and returns fake logits `[B]` (positive = generated) — we will plug it into our evaluation as an extra detector and report it separately from our own, including under JPEG and resizing.
>
> Either direction works, or both. Happy to go first — say the word and we will post the challenge link here.
>
> `<your names>`

**为什么这么写**:给对方一个**零成本就能答应**的选项(只要标一堆图),同时提出一个更有价值的选项(交换检测器)。而且明说"我们先来",降低对方的启动成本。

---

## 三、第二封(对方答应之后)

> Thanks `<name>`! Challenge package: `<link>`
>
> It contains `images/` (N PNGs), a `README.md` with the task, and `submission_template.csv`. Just fill the `pred_label` column and send the CSV back — no need to install anything of ours.
>
> For reference, here is what our own detectors do on the same set, so you can see where the bar is: `<fill in: our 4 detectors' detection rate on attacked images>`.
>
> One thing we would find genuinely useful in your reply: **which images did you get wrong, and does your method have a guess why?** We are trying to work out whether our attack is removing a specific kind of evidence (frequency artefacts) or just pushing samples over a decision boundary. Your errors are informative for us either way.

**为什么加最后一段**:评分要"responded to others' ideas thoughtfully"。问一个**有实质内容的问题**,比单纯要分数更容易拿到一段像样的往返对话。

---

## 四、第三封(拿到对方结果之后 —— 这封最值钱)

拿到对方 CSV 后跑我们的评分器:
```bash
python m2_challenge.py score --key challenge/private/challenge_key.csv \
  --submission <他们发来的.csv> --team <team name> --save challenge/score_<team>.json
```

然后公开回复(这段直接进 Ed 帖第 9 节):

> Results for `<team name>` on our challenge set:
>
> | metric | value |
> |---|---|
> | overall accuracy | `__` |
> | detection rate on **raw** generated images | `__` |
> | detection rate on **attacked** images | `__` |
> | false-positive rate on real photos | `__` |
>
> Reading this: your method detects `__`% of plain Stable Diffusion images but only `__`% after our attack, so our attack costs you `__` points. `<or: your method held up — detection only dropped from __ to __, which is the best result anyone has posted against us.>`
>
> What we take from this: `<one honest sentence — e.g. "our attack is much weaker against your method than against the four detectors from the original paper, which suggests it is over-fitted to low-level frequency evidence.">`
>
> We have changed `<what you will change / test next because of their result>`.

**这一封是 2 分的核心** —— 它证明了:真的互动了 + 真的分析了对方的结果 + 真的因此调整了工作。

---

## 五、给对方的东西怎么准备

代码已经写好了,等攻击跑完就能一条命令打包:
```bash
python m2_challenge.py build --real_dir data/real/test --gen_dir data/gen/test \
  --attack_dir results/FULL --n_per_group 20 --out challenge
```
- **发给对方的**:`challenge/challenge_public.zip`(不含答案)
- **自己留着的**:`challenge/private/challenge_key.csv`(**绝对不要发出去**)

**没跑完攻击也能先联系** —— 先把邀请发出去、把互动谈好,包等跑完再给。对方回复也要时间,这一步越早启动越好。

---

## 六、证据记录表(M3/M4 都要用,现在就开始填)

| 项目 | 内容 |
|---|---|
| 对方队名 | |
| 对方 Ed 帖链接 | |
| 对方方法是什么 | |
| 我们针对他们的哪一环 | |
| 发出邀请的日期 / 链接 | |
| 我们给了什么(包的版本/哈希) | |
| 对方返回了什么 / 日期 | |
| 评分结果 | |
| 我们的分析(为什么是这个结果) | |
| 我们因此改了什么 | |
| 对方的后续回应 | |

M4 的个人报告里有一整节要写"你用了什么策略对付别的队、结果如何、学到什么",这张表填满了就是现成的素材。

---

## 七、现在就能做的三件事

1. 上 Ed 找 Theme 1 的 Light 队,把他们的帖子**发给我**,我来把"针对他们哪一环"那句改成具体的。
2. 用上面第二节的模板发第一封(公开回帖)。
3. 把 Ed 帖草稿(`M2_ED_POST.md`)里的队名、成员、M1 链接先填上。

**不用等模型下载完。**
