# M3 课程 Colab

课程源码取自 `before_lao_architecture_optimization_20261018_145942`，没有使用后续 LAO 架构研究代码。原归档和当前论文工作区均保留。

## 运行内容

- A5、A2、X2、X4、ENS、CAPB_ENS、EOT_ENS：每种方法从头运行同样的前三张生成图，保存中间图、最终 PNG、配置和日志。
- 独立进程重新读 PNG，评估 E/R/D/S 在原图、JPEG75、resize0.5 下的结果；FPR 使用全部 100 张真实测试图。
- 单独复算归档的 100 张课程结果，核对 ASR、FPR 和 AUC；不将小批演示冒充新的留出验证。
- PreDecodeGuard 仅保留聚合结果与 pending 状态。未包含 latent、个体分数、文件映射、对照身份，也不会再次执行同伴攻击。
- CAPB 与 EOT 分开验证；不声称已经验证 CAP+EOT 联合方法。

## Colab 使用

在 Google Drive 的 `FIT5230_M3` 文件夹放置：

```text
FIT5230_M3/
  M3_Submission.ipynb
  m3_course_bundle.zip
  weights/
    checkpoints/        # 五个作者权重
    .cache/checkpoints/vgg16-397923af.pth
```

也支持用 `m3_weights.zip` 代替 `weights/` 文件夹。权重和课程包不能因为只分享 notebook 而自动对其他人可见；提交时需要给评阅者读取这些资源的权限。

选择 GPU，从头运行。本交付版已在 NVIDIA L4（22 GiB 显存、CUDA、FP32）完成默认 3 图全流程。代码会建立单独的 Python 3.10 环境，下载固定版本 SD2.1 模型，校验全部资源哈希；不训练模型。上游建议至少 32 GB GPU，A100 有更多显存余量；16 GB T4 尚未验证。首次安装/下载需要联网。

默认 `N_IMAGES=3` 是完整流程演示。改成 `100` 可重跑课程规模，耗时会明显增加。每次使用新目录，不从历史结果恢复。

最后一个单元会把输出 ZIP 保存到 Drive 的 `FIT5230_M3/outputs/`。还要保存 notebook 本身，并保留所有单元输出。

## 交付状态的边界

最终 `M3_Submission.ipynb` 的 19/19 代码单元均在 Colab L4/CUDA 完整运行，零错误，输出已保存。另保留两次本地 MPS 验证。默认每方法 3 图，历史 100 图统计单独审计；不将小批演示当成新留出集实验。

人工复核、个人贡献与新增同伴策略不能自动补全：PreDecodeGuard 复核保持 pending；刘俊鹏另行设计的策略单列待补充；王云杰的个人策略与实际工作记录仍需如实确认。

## 开发与验证文件

- `build_bundle.py`：从快照按 SHA-256 复制课程资源，记录三处可移植修改。
- `m3_runtime.py`：校验、全新运行目录、独立方法进程、保存文件复评。
- `m3_reporting.py`：从逐图记录计算表格、配对变化、FPR、AUC 和图。
- `create_notebook.py`：生成清空输出的 notebook 源稿。
- `execute_notebook.py`：使用真实 Jupyter kernel 执行并逐单元保存输出；失败不会伪装成成功。
- `validate_delivery.py`：只读核验 notebook、输出 ZIP、150 个文件哈希、102 张 PNG 和评分统计。

本地执行可通过 `M3_RESOURCE_DIR`、`M3_CHECKPOINT_DIR`、`M3_LPIPS_CACHE`、`M3_MODEL_DIR`、`M3_PYTHON` 指定现有资源。Mac 路径没有硬编码到 Colab。
