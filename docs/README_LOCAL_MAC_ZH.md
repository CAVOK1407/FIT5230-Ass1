# 在 Mac 本地跑 M2（M5 Max / MPS）

不用 Colab、不用 Google Drive 挂载。你的 M5 Max 128GB 内存足够。
代码已加设备开关 `--device mps`（自动让 Apple 的 MPS 跑；遇到 MPS 没实现的算子会自动回退 CPU）。

> 说明：我这边没有 Mac GPU，只能验证"设备切换 + 整条流程"在 CPU 上跑通。真正的 MPS 首跑可能会有个别算子报错——真遇到把报错发我，一般加一两个小补丁就行。

---

## 0. 前提
- 已装 Anaconda/conda（你有）。
- 终端打开，下面命令一条一条复制运行。

## 1. 建环境（一次性，约 5–10 分钟）
```bash
conda create -y -n fit5230m2 python=3.10
conda activate fit5230m2
pip install torch==2.2.2 torchvision==0.17.2
pip install numpy==1.26.4 setuptools==80.9.0 diffusers==0.12.1 huggingface_hub==0.23.4 \
  transformers==4.30.2 accelerate==0.20.3 timm==0.6.12 efficientnet-pytorch==0.7.1 \
  torchattacks==3.5.1 lossers==0.0.9 eagerpy==0.30.0 natsort==8.4.0 scipy==1.11.4 \
  scikit-learn==1.3.2 opencv-python-headless==4.10.0.84 Pillow==10.4.0 tqdm==4.66.1 \
  safetensors==0.4.3 pandas==2.2.2 matplotlib==3.9.2
```
验证 MPS 可用：
```bash
python -c "import torch; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
```
应输出 `mps True`。

## 2. 拿官方代码（固定版本）
```bash
cd ~/Documents/ChatGPT/FIT5230/m2        # 工作目录，可自选
git clone https://github.com/wyczzy/StealthDiffusion.git
cd StealthDiffusion
git checkout a156fb0792732e1aabc5b94a0cbecf6829e79d73
```

## 3. 放入我们的脚本
把 `ClaudeOutPut/m2_scripts/` 里这 5 个文件复制进上面的 `StealthDiffusion/` 目录：
```bash
cp ../ClaudeOutPut/m2_scripts/m2_device.py \
   ../ClaudeOutPut/m2_scripts/m2_stealth.py \
   ../ClaudeOutPut/m2_scripts/m2_data.py \
   ../ClaudeOutPut/m2_scripts/m2_evaluate.py \
   ../ClaudeOutPut/m2_scripts/m2_challenge.py .
```
（`run_local_mac.sh` 也在 m2_scripts 里，想用便捷脚本就一起复制进来。）

## 4. 放入 6 个权重
这些是**你自己 Drive 里的文件**，直接从 Drive 网页下载到本机就行（自己的文件没有共享限制）：
Drive → 选中 `checkpoints` 里的 6 个文件 → 下载 → 解压后放到 `StealthDiffusion/checkpoints/`：
```
checkpoints/efficientnet-b0.pth
checkpoints/resnet50.pth
checkpoints/deit.pth
checkpoints/swin-t.pth
checkpoints/Controlvae.pt        # 注意是 .pt
checkpoints/dncnn_color_blind.pth
```
检查：
```bash
ls -la checkpoints
```

## 5. 准备数据（先小规模）
真实照片：用你已有的任意照片文件夹（递归找图），或不加 `--real_src` 让它自动下 Imagenette。
```bash
python m2_data.py real --out data --n_dev 6 --n_test 20 --real_src /path/to/你的真实照片文件夹
```
生成 AI 图（SD1.5，MPS，稍慢；首跑会下载 SD 权重约 4GB）：
```bash
python m2_data.py generate --out data --n_dev 6 --n_test 20 \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 --device mps
```
检测器体检（确认 0=AI、1=真实，clean 准确率正常）：
```bash
python m2_data.py check --out data --device mps
```

## 6. 跑攻击（4 个变体，先各 6 张）
首跑攻击会下载 SD2.1 权重（约 5GB）。`--resume` 断了可续跑。
```bash
COMMON="--device mps --images_root data/gen/test --max_images 6 --iterations 10 \
  --pretrained_diffusion_path Manojb/stable-diffusion-2-1-base \
  --encoder_weights ./checkpoints/Controlvae.pt --seed 42 --resume"

python m2_stealth.py $COMMON --save_dir results/B00
python m2_stealth.py $COMMON --decode_path final --save_dir results/B10
python m2_stealth.py $COMMON --surrogates E,R,D --save_dir results/ENS
python m2_stealth.py $COMMON --surrogates E,R,D --decode_path final \
  --loss margin --lpips_weight 1.0 --di_prob 0.5 --save_dir results/FULL
```

## 7. 评测
```bash
python m2_evaluate.py --device mps \
  --runs B00=results/B00 B10=results/B10 ENS=results/ENS FULL=results/FULL \
  --real_dir data/real/test --out results/eval --conditions none,jpeg75,resize05
```
看 `results/eval/summary.csv`：条件 ASR、真实图 FPR、AUC、JPEG/缩放鲁棒性、PSNR/SSIM/LPIPS、耗时。

## 8. 给其他队的挑战包
```bash
python m2_challenge.py build --real_dir data/real/test --gen_dir data/gen/test \
  --attack_dir results/FULL --n_per_group 20 --out challenge
```
`challenge/challenge_public.zip` 发给对方；`challenge/private/challenge_key.csv` 自己留着评分。

## 便捷脚本
第 5–8 步也可以一次跑完（先激活环境、cd 进 StealthDiffusion、放好权重）：
```bash
bash run_local_mac.sh all            # 小规模：各 6 张
N_TEST=100 N_ATTACK=50 bash run_local_mac.sh all   # 放大
```
或分步：`bash run_local_mac.sh data` / `attack` / `eval` / `challenge`。

## 跑通后放大
先用小规模确认整条链路没问题，再把 `--n_test` 提到 100、`--max_images` 提到 50，重跑攻击和评测。

## 常见问题
- 报某个算子 `not implemented for MPS`：`m2_device` 已自动设 `PYTORCH_ENABLE_MPS_FALLBACK=1`；若仍报错，把报错发我。
- 慢：MPS 比 A100 慢不少，先小规模。
- 首跑要下 SD1.5(~4GB)+SD2.1(~5GB) 权重到 `~/.cache/huggingface`，确保网络和磁盘够。
