from pathlib import Path
import json,hashlib,datetime
sd=Path('/Users/wangyunjie/Documents/ChatGPT/FIT5230/github/FIT5230-Ass1/StealthDiffusion');root=sd/'results/m3_lao_v3/validate_20261018_195753_zxupemab';dev=sd/'results/m3_lao_v3/dev_20261018_184038_1mmcu3yj';snap=sd.parents[2]/'snapshots/before_lao_pgd_eot_v3_20261018_183003'
r=json.loads((root/'report/report.json').read_text()); names=['P','L','H','P_EOT','L_EOT','H_EOT']; labels=dict(P='强低失真 PGD',L='LAO',H='LAO＋PGD',P_EOT='PGD＋EOT',L_EOT='LAO＋EOT',H_EOT='LAO＋PGD＋EOT')
metric=lambda n,d,c:next(x for x in r['metrics'] if (x['run'],x['detector'],x['condition'])==(n,d,c))
joint=lambda n,d:next(x for x in r['robustness'] if (x['run'],x['detector'])==(n,d))
link=lambda title,path:f'[{title}]({path})'
lines=['# LAO v3：强低失真 PGD 与混合 EOT 的第三批验证','', '2026-10-18。已完成优化前存档、原开发集有限调参、配置冻结、另一批50张生成图＋50张真实图验证及独立审计。**低失真目标保住了，但这轮没有证明 LAO 或混合方法优于加强后的纯 PGD。** 混合EOT的四条件共同成功率数值较高，统计证据不足，失真和计算成本也更高。','',
'本轮没有训练检测器或扩散模型权重，只优化每张图的latent／像素扰动。本轮与PreDecodeGuard同伴实验分开，旧实验及盲审材料均未改动。','',
'## 存档与实验设计','',
f'- {link("修改前完整快照",snap/"RESTORE.txt")}：28,947文件，6,535,938,064字节，源／副本SHA-256全部核验。旧结果、代码和同伴资产另复核14,738个文件，零变化。',
f'- {link("预定协议",sd.parent/"docs/M3_LAO_V3_PROTOCOL_ZH.md")}；{link("开发集结果与选型",dev/"DEVELOPMENT_REVIEW_ZH.md")}；{link("实际验证冻结文件",dev/"freeze_amended.json")}。仅在原10张生成开发图上选择参数，前两批验证图不参与本轮调参。',
'- 第三批为50张SD1.5图，固定种子5233000–5233049，10类各5张；另50张未用真实照片。与旧446种规范图像内容、210个真实来源ID及236个生成身份均无重叠，内部无重复。仍是相同生成器和类别范围，不能称跨生成器／跨语义域泛化。',
'- 全部最终PNG相对同一原图的整数L∞≤8，即8/255，零容差；单图LPIPS≤0.03、PSNR≥30 dB；配置全50图平均LPIPS≤0.015。全部样本保留，失败不移除。',
'- 纯PGD新增连续亚像素状态、3个固定起点、开发集步长／LPIPS权重有限搜索、早停与5次实际评分回退。混合方法先独立运行LAO，再以LAO结果开始像素PGD，预算始终相对原图，不叠加。',
'- EOT优化视图为原PNG及真实PIL JPEG80/90，JPEG前向真实执行、反向采用identity BPDA。最终评估使用原PNG及未用于优化的JPEG75/85/95。E是唯一优化检测器；R/D/S只作正式次要评价。',
'- 共同的是扰动预算，不是算力：P上限80步×3起点，H上限40个latent步骤＋40个像素步骤×3起点。所有方法均独立运行并记录实际成本。',
'', '## 新图的正式结果','',
'下表成功均要求原PNG质量合格。原PNG分母46，是50张中被E原本检出的图；四条件共同分母33，要求原图在全部四个条件均被检出，且同一输出在四条件均改判。平均LPIPS和耗时覆盖全部50张，不能把这两个分母混用。','',
'| 方法 | 原PNG成功／46 | 四条件共同成功／33 | 平均LPIPS ↓ | 平均秒／图 ↓ |', '|---|---:|---:|---:|---:|']
for n in names:
 q=r['methods'][n]['quality'];c=r['methods'][n]['compute'];a=metric(n,'E','none');j=joint(n,'E');lines.append(f"| {labels[n]} | {a['quality_success']}/46 | {j['quality_joint_success']}/33 | {q['lpips_mean']:.6f} | {c['seconds']['mean']:.2f} |")
lines += ['', '独立零步重建Z为原PNG 4/46、四条件0/33，平均LPIPS 0.004571。全部7组均50/50质量合格且平均失真守卫通过；PGD两组实际最大像素差仅3，LAO及混合最大为8。相同预算上限不等于相同实际用量。耗时为本机MPS实测攻击阶段时间，含对应renderer准备、回退、reset、导出与核验，不含公共模型加载及独立评价；不外推其他硬件效率。', '',
'主比较为LAO＋PGD＋EOT减去PGD＋EOT：**5/33 对3/33，净增6.06个百分点**；新增成功4张、丢失成功2张。配对bootstrap 95%区间为 **[−9.09，+21.21]个百分点**，精确McNemar **p=0.6875**，未达到预定改善标准。这既不能宣称显著优越，也不能以不显著宣称两者等效。', '',
'混合EOT相对PGD＋EOT平均LPIPS为1.86倍、平均耗时为4.35倍；原PNG成功率也较低。相比独立LAO＋EOT，混合把原PNG成功从35/46提高到43/46，并把平均LPIPS从0.012426降至0.008289；但这说明PGD阶段对LAO有帮助，尚不能证明LAO给强PGD带来可靠的额外收益。', '',
'![成功率、失真与成本对照]('+str(root/'report/comparison.png')+')','',
'## EOT改善了哪些条件','',
'| 方法 | 原PNG | JPEG75 | JPEG85 | JPEG95 |','|---|---:|---:|---:|---:|']
for n in names:
 cells=[metric(n,'E',c) for c in ['none','jpeg75','jpeg85','jpeg95']];lines.append('| '+labels[n]+' | '+' | '.join(f"{x['quality_success']}/{x['n_clean_detected']}" for x in cells)+' |')
ids=set(joint('P_EOT','E')['eligible_ids']); diag={}
for n in ['P_EOT','L_EOT','H_EOT']:
 rows=[json.loads(s) for s in (root/'methods'/n/'records.jsonl').read_text().splitlines()];ok=[x['image_id'] for x in rows if x['image_id'] in ids and x['selected']['success']]
 diag[n]={'cohort_n':len(ids),'optimization_view_success':len(ok),'optimization_success_ids':ok,'formal_joint_success':joint(n,'E')['quality_joint_success']}
(root/'report/optimization_view_diagnostic.json').write_text(json.dumps({'status':'verified','role':'post-hoc descriptive diagnostic; not configuration selection or a new confirmatory endpoint','optimization_conditions':['none','jpeg80','jpeg90'],'optimization_margin_buffer':.1,'formal_conditions':['none','jpeg75','jpeg85','jpeg95'],'formal_threshold':0,'methods':diag},indent=2)+'\n')
lines += ['', 'EOT管线在JPEG85/95下有明显的描述性改善，但JPEG75仍是瓶颈，不能称已获得普遍压缩鲁棒性。P与P_EOT、H与H_EOT的LPIPS权重分别为1和10，这是各自调优管线的比较，不能把全部差异都归因于EOT单一开关。次要比较未作多重比较校正。', '',
'事后诊断：在同一33张主比较样本中，PGD＋EOT与混合EOT各有30张满足优化视图none/80/90的严格成功条件，换为正式none/75/85/95共同成功仅3张和5张。这说明对固定JPEG质量视图的成功没有充分泛化；它不是模型权重训练结果，也未用于重新选型。', '',
'## 迁移、误报与LAO归因','',
'R／D／S上的四条件共同成功全部为0，分别为0/42、0/50、0/48。原PNG上的少量迁移成功不能替代鲁棒迁移证据。真实照片误报率如下，分母均为50：','',
'| 检测器 | 原PNG | JPEG75 | JPEG85 | JPEG95 |','|---|---:|---:|---:|---:|']
for det in ['E','R','D','S']:
 fs=[next(x for x in r['real_fpr'] if x['detector']==det and x['condition']==c) for c in ['none','jpeg75','jpeg85','jpeg95']]; lines.append('| '+det+' | '+' | '.join(f"{x['false_positives']}/50 ({x['real_fpr']:.0%})" for x in fs)+' |')
lines += ['', 'E的误报为2%–8%；部分迁移检测器在压缩下的误报很高，尤其S在JPEG75为54%，因此这些条件的攻击率必须与其工作点失效一起解释。', '',
'结构证据证明LAO已真实执行并落盘：350份最终PNG预算检查、200次实际reset与独立Z一致、100份混合中间LAO图片与独立L逐字节一致；L与L_EOT分别执行352次与821次latent更新，全部有限且非零；H与H_EOT另各自独立重复相同数量的latent更新。H最终直接保留LAO图15/50，H_EOT为8/50。部分零像素梯度更新仍经过像素插值回退，不能称为纯LAO输出。', '',
'这些证据支持“LAO支路有效、混合链路可追踪”，不支持“LAO主导”。完整必要性／因果主导判断仍需要同renderer、同调参规则的消融和相应重新优化，而不能仅看中间图变化或更新次数。LPIPS／PSNR也不替代人工内容保持复核。', '',
'## 对已有结论的修正','',
'v2独立LAO的40/41旧结果仍完整保留，但其相对固定大步长PGD或有限PGD10_MATCHED的优势不能推广成“LAO普遍比PGD低失真”。本轮补强PGD后，纯PGD以更低失真获得44/46原PNG成功；PGD＋EOT为46/46。因此，论文和M3应将此前表述限定到当时的特定基线。本轮的价值是补齐强对照、识别JPEG泛化瓶颈，并诚实界定LAO的贡献。', '',
'若继续研究，下一轮可在旧开发集尝试更广的JPEG质量分布和更充分的BPDA／EOT策略，冻结后再用第四批新图验证，并加入同renderer／相近计算成本消融。第三批已经是已观察数据，不能再次充当未见测试集。本轮未执行这些后续实验。', '',
'## 可复核材料','',
f'- {link("完整统计报告",root/"report/report.md")}及{link("机器可读结果",root/"report/report.json")}：5,600条逐图评分、112个条件汇总、800条真实图评分；原始PNG与失败均保留。',
f'- {link("数据独立审计",root/"independent_data_audit.json")}、{link("结构审计",root/"independent_architecture_audit.json")}、{link("统计独立复算",root/"independent_statistics_audit.json")}、{link("旧资产保全",root/"independent_final_preservation.json")}。',
f'- {link("命令／权重／源码记录",root/"plan.json")}、{link("运行完成状态",root/"session.json")}、{link("审计工具",root/"audit_tools")}；原始源码保存在frozen_code。重放已生成输入时应使用记录中的attacks／evaluation命令并改为新输出目录；重新运行数据生成器会执行当前排除规则，不应当作本批数据的直接重放。',
f'- {link("111项测试记录",dev/"reporting_amendment/111_tests.log")}。开发完成后、第三批生成前，仅对报告器E分数一致性检查增加与核心一致的相对浮点容差；绝对容差、分类阈值、扰动／质量约束及攻击逻辑未变。原冻结文件保留，修订原因、前后源码与哈希链均可审计。',
'', '新增实现为m3_lao_v3.py、m3_lao_newdata_v3.py、m3_lao_v3_run.py、m3_lao_v3_report.py及对应测试，并保留m2_scripts镜像；旧v1/v2代码与结果不覆盖。','']
text='\n'.join(lines);dst=sd.parent/'docs/M3_LAO_V3_RESULTS_ZH.md';assert not dst.exists();dst.write_text(text);(root/'RESULTS_ZH.md').write_text(text);print(dst)
