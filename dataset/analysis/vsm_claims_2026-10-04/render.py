from pathlib import Path
import json
P=Path(__file__).parent
r=json.loads((P/'results.json').read_text());sens=json.loads((P/'repetition_sensitivity.json').read_text());fresh=json.loads((P/'source_freshness.json').read_text())
def pct(x):return f'{x*100:.2f}'
def eff(x):return f"{x['estimate']*100:+.2f} [{x['ci_95'][0]*100:.2f}, {x['ci_95'][1]*100:.2f}]"
def pv(x):return f"{x:.5f}"
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
L=['# DALIC VSM 替换、运行波动与 runtime 消融核查','',
'日期：2026-10-04。只读分析现有实验，未修改论文或原始结果，未调用模型重新运行实验。','',
'## Material Passport','',
'- Origin Skill: ARS experiment-agent','- Origin Mode: validate','- Verification Status: ANALYZED','- Version Label: vsm_claims_validation_v1','',
'## 判断','',
'VSM 数据仍支持“运行时证据提供有效但不完整的定位线索，LLM 可以补充它，完整工作流在较弱模型上主要通过完成率和排序获得优势”的主线，但不支持保留所有原有细节。不能将新的七组 VSM 结果直接替换旧 BM25 数字后继续声称全部实验条件保持不变。','',
'9B 的三次运行适合作为单一模型下的补充稳定性检查：全部实例的完成率、F1、MAP 点估计方向一致，共同成功实例的 F1 接近零而 MAP 为正。三个运行不足以证明其他模型或消融结果也稳定，且 VSM run3 包含五次 API 404。','',
'run1 runtime 消融在 4B、27B 上支持完成率和端到端质量改善，9B 方向为正但证据不明确。共同成功实例的 F1 点估计在三个模型上均为负、区间均跨零；不能归纳为 runtime 在所有模型上都改善成功输出的 F1。','',
'## 数据与统计口径','',
'全部实例使用已验证的 226 个实例，Avg3 MAP 使用三个粒度 ground truth 均非空的 170 个实例。全部 evaluation 的逐实例指标通过现有 evaluate.load_run_report 从保存的有序预测与 ground truth 重算并校验。旧 BM25 从归档 results.txt 重算，不使用 main.tex 中的结果数字作为输入。三个当前方法目录中全部 7,232 条 evaluation 记录与原始 results.txt 的 SHA-256 相符。','',
'单位：绝对指标为 %，差异为百分点。置信区间为 20,000 次配对实例 bootstrap 的 pointwise 95% percentile CI。正式比较用 exact McNemar（完成率）及 100,000 次双侧 paired sign flip（Avg3 F1/MAP）。每个指标分别对七个模型做 Holm 校正；消融为三个模型；重复运行的探索性检验为三个运行。这些分析方案是不同候选设计，不应将其挑选后包装为预先指定的单一验证方案。pointwise CI 与多重校正后的 p 值不同，前者不跨零并不保证后者小于 .05。','',
'匹配长度与当前论文和脚本一致：有效 DALIC 输出按每个实例、每个粒度的实际长度截断检索排序；无效 DALIC 输出得零分，检索方法使用 k=3。所有 run1 的选择遵循用户本次统一规则，没有按分数挑选最好运行。','',
'源清单、SHA-256、各比较的实际重采样种子、逐项目删除检查与全部统计量见 [results.json](results.json)。重算入口：[analyze.py](analyze.py)，重复运行补充检查：[check_repetitions.py](check_repetitions.py)。','',
'## 选用的运行','',
'VSM 为当前 dalic_vsm 目录的七组结果。非 Qwen 模型使用该目录现有唯一对应 config，DeepSeek Pro 为名字含 run2 的那一份。主替换分析保持论文原来的 Baseline；另计算当前 Baseline（其中 9B 用 run1）的候选组合。','']
rows=[]
for m in r['replacement']:
 rows.append([m,r['metadata']['sources']['vsm:'+m]['path'].split('/')[-2],r['metadata']['sources']['paper_baseline:'+m]['path'].split('/')[-2],r['metadata']['sources']['current_baseline:'+m]['path'].split('/')[-2]])
L += [table(['模型','VSM config','原论文 Baseline','当前 Baseline'],rows),'','## 1. BM25 换成 VSM','',
'### 1.1 绝对结果及相对旧 BM25 的差异','',table(['模型','BM25 F1','VSM F1','差异 [95% CI]','BM25 MAP','VSM MAP','差异 [95% CI]'],[[m,pct(x['b']['metrics']['avg3']['f1']),pct(x['a']['metrics']['avg3']['f1']),eff(x['f1']),pct(x['b']['metrics']['avg3']['map']),pct(x['a']['metrics']['avg3']['map']),eff(x['map'])] for m,x in r['bm25_to_vsm'].items()]),'',
'Qwen run1 与原 BM25 的绝对 F1 基本接近。上述新旧差异同时包含不同运行和可能不同服务配置的影响，不能被当成已隔离的 VSM 相对 BM25 因果效应；区间跨零也不证明两者等效。','',
'### 1.2 RQ1 的阶段性比较','',
'以下 Trace VSM 使用实际提供给 VSM agent 的 workbench 候选排序，移除测试实体。缓存候选集合完全一致，但有 78 份函数排序顺序不同，其中 2 份前 3 名顺序不同。使用缓存重新计算作为敏感性检查不改变本报告的结论，正式稿应统一实际证据与检索比较的生成口径。','',
 table(['k','Static VSM F1 / MAP','Trace VSM F1 / MAP','ΔF1 [95% CI]','ΔMAP [95% CI]'],[[k,pct(x['b']['metrics']['avg3']['f1'])+' / '+pct(x['b']['metrics']['avg3']['map']),pct(x['a']['metrics']['avg3']['f1'])+' / '+pct(x['a']['metrics']['avg3']['map']),eff(x['f1']),eff(x['map'])] for k,x in r['retrieval']['static_trace_vsm'].items()]),'',
 table(['模型','DALIC−Trace k=3：F1','DALIC−Trace k=3：MAP','DALIC−Trace 等长：F1','DALIC−Trace 等长：MAP'],[[m,eff(x['trace_vsm']['fixed_k3']['f1']),eff(x['trace_vsm']['fixed_k3']['map']),eff(x['trace_vsm']['matched']['f1']),eff(x['trace_vsm']['matched']['map'])] for m,x in r['retrieval']['dalic_vs_ir'].items()]),'',
'运行时筛选仍有明确正向结果。LLM 进一步改善的总体方向保留，但 4B 等长 F1 和 MAP 区间均跨零；只能表述为点估计正向、证据不确定。其余六组的等长 F1/MAP 区间均在零以上。若整体方法改用 VSM，应将 RQ1 链条改为 Static VSM → Trace VSM → DALIC-VSM；原 BM25 对比可以保留为补充，不能继续作为 VSM 的直接前一阶段。','',
'### 1.3 RQ2 对静态 IR','',
'七组 VSM agent 相比三个静态 IR 在 k=3 与匹配长度下均保留正向 F1/MAP 差异，匹配长度下的全部 Avg3 F1/MAP 区间均在零以上。以下列出等长比较，完整固定长度结果见 JSON。','',
 table(['模型','静态比较方法','ΔF1 [95% CI]','ΔMAP [95% CI]'],[[m,method,eff(x[method]['matched']['f1']),eff(x[method]['matched']['map'])] for m,x in r['retrieval']['dalic_vs_ir'].items() for method in ['static_vsm','static_lsi','static_bm25']]),'']
for group,title in [('replacement','1.4 固定原论文 Baseline，仅替换 DALIC'),('current_baseline','1.5 同时采用当前 Baseline 的替代组合')]:
 L += ['### '+title,'',table(['模型','B / D 有效数','Δ完成率 [95% CI]；p_H','ΔF1 [95% CI]；p_H','ΔMAP [95% CI]；p_H'],[[m,str(x['all']['b']['valid_n'])+' / '+str(x['all']['a']['valid_n']),*[eff(x['all'][k])+'; '+pv(x['all'][k]['holm_p']) for k in ['completion','f1','map']]] for m,x in r[group].items()]),'',table(['模型','共同成功 N / MAP N','ΔF1 [95% CI]','ΔMAP [95% CI]'],[[m,str(x['common']['n'])+' / '+str(x['common']['ranking_n']),eff(x['common']['f1']),eff(x['common']['map'])] for m,x in r[group].items()]),'']
L += ['对原稿具体表述的影响：','',
'- 原稿称 Qwen 完成率收益随模型大小递增；run1 新结果是 +17.70、+13.27、+14.60，已不单调。','- 原稿称 4B 在共同成功上保持明确 F1 优势；新差异 +4.46，CI [-1.91, 10.85]，证据减弱。','- 固定旧 Baseline 时，Pro 与 Pro-32K 的共同成功 F1 分别变为 -2.65 和 -9.17，区间均为负；且存在服务设置混杂，不能沿用“均保持相当 F1”的笼统表述。','- 使用当前 Baseline 时，Pro-32K 端到端完成率/F1/MAP 差异为 +1.77/+0.10/+2.36，区间均跨零，不能声称 32K 明确增强 DALIC 收益。','- 新 VSM Pro-32K 的 File/Class/Function F1 为 74.53/69.99/66.22，不再支持原稿“三种粒度相近”的特例。','- 旧轨迹百分比、上下文失败数和输出范围表需要随数据更新。七组仍都恢复了 trace 以外的正确实体，soft prior 的核心证据仍在。','',
'### 1.6 Runtime 范围','',
'完整 runtime evidence 对 ground truth 的覆盖仍为 1,071/1,399 = 76.55%。以下计数只包含有效输出。','',
 table(['模型','范围外正确数','全部正确数','范围内 precision','范围外 precision'],[[m,x['correct_outside'],x['correct_inside']+x['correct_outside'],pct(x['correct_inside']/x['pred_inside']),pct(x['correct_outside']/x['pred_outside'])] for m,x in r['runtime_scope'].items() if m!='coverage']),'',
'七组的范围内 precision 均高于范围外，原稿 Pro-32K 的例外消失。此处只核查了成员关系和命令中证据文件引用；未重算旧稿“70.64% 检查 top10 之外”这一具体轨迹统计，因此不能复用该比例。','',
'## 2. Qwen3.5-9B 三次运行','',
 table(['Run','B 有效数','D 有效数','B F1','D F1','ΔF1','B MAP','D MAP','ΔMAP'],[[run,x['all']['b']['valid_n'],x['all']['a']['valid_n'],pct(x['all']['b']['metrics']['avg3']['f1']),pct(x['all']['a']['metrics']['avg3']['f1']),f"{x['all']['f1']['estimate']*100:+.2f}",pct(x['all']['b']['metrics']['avg3']['map']),pct(x['all']['a']['metrics']['avg3']['map']),f"{x['all']['map']['estimate']*100:+.2f}"] for run,x in r['repetitions'].items()]),'',
 table(['方法','完成率：均值 ± 样本 SD','Avg3 F1：均值 ± 样本 SD','Avg3 MAP：均值 ± 样本 SD'],[[m,*[pct(x[k]['mean'])+' ± '+pct(x[k]['sd']) for k in ['completion','f1','map']]] for m,x in r['repetition_summary'].items() if m in ['baseline','vsm']]),'',
 table(['Run','Δ完成率 [95% CI]；p_H','ΔF1 [95% CI]；p_H','ΔMAP [95% CI]；p_H','共同成功 ΔF1 [95% CI]','共同成功 ΔMAP [95% CI]'],[[run,*[eff(x['all'][k])+'; '+pv(x['all'][k]['holm_p']) for k in ['completion','f1','map']],eff(x['common']['f1']),eff(x['common']['map'])] for run,x in r['repetitions'].items()]),'',
'三次同编号比较的完成率和 MAP 在各自三个探索性检验的 Holm 家族内保留支持；F1 的 pointwise CI 均高于零，但校正 p 约 .067，不支持把“每次均显著改善 F1”作为结论。共同成功 F1 的点估计会换方向、区间都跨零；MAP 三次均为正。','',
'把三次结果在每个实例内平均，再以实例为单位做配对 bootstrap，F1 差异为 '+eff(r['repetition_summary']['mean_per_instance_effect']['f1'])+'，MAP 为 '+eff(r['repetition_summary']['mean_per_instance_effect']['map'])+'。这些区间只描述给定三次运行后的实例抽样不确定性，并未充分估计未来运行随机性。不能把 3×226 当成 678 个相互独立的任务。','',
'九种 VSM run 与 Baseline run 交叉组合的端到端差异全部为正，F1 范围 +0.17 至 +12.43、MAP +5.85 至 +22.33；这是描述性敏感性检查，九组并非九个独立实验。固定到六次运行均成功的 113 个实例后，三次 F1 差异均为负且区间跨零，MAP 点估计均为正但只有 run2 的区间完全为正，提示共同成功集合选择影响证据强度。','',
'实际轨迹中的模型名、提示指纹、agent 限额、observation 模板和 mini 版本在各自方法的三次运行内一致，model_kwargs 都只显式记录 drop_params=True。没有可核验的显式 seed、temperature 或历史服务端配置，所以应称“三次重复运行的波动”，不能称“三个受控随机种子实验”。','',
'VSM run3 有 5 个模型调用 API 404（NotFoundError），属于服务错误。主分析仍保留全部 226 实例计零；补充分析从六次运行同步移除同样五个实例，剩 221 个（MAP 166 个），三组 F1/MAP 差异仍为正：','',
 table(['Run','ΔF1 [95% CI]','ΔMAP [95% CI]'],[[run,eff(x['effects']['f1']),eff(x['effects']['ap'])] for run,x in sens['pairs'].items()]),'',
'因此三次 9B 运行适合说明该模型下主要结果方向没有随本次运行改变，同时报告均值、SD、每次结果和基础设施失败。不能将这一检查推广成所有模型、上下文设定或 runtime 消融均不受随机性影响。','',
'## 3. Runtime 消融：VSM run1 对 no-runtime run1','',
 table(['模型','无 runtime / VSM 有效数','无 runtime F1','VSM F1','无 runtime MAP','VSM MAP'],[[m,str(x['all']['b']['valid_n'])+' / '+str(x['all']['a']['valid_n']),pct(x['all']['b']['metrics']['avg3']['f1']),pct(x['all']['a']['metrics']['avg3']['f1']),pct(x['all']['b']['metrics']['avg3']['map']),pct(x['all']['a']['metrics']['avg3']['map'])] for m,x in r['ablation'].items()]),'',
 table(['模型','Δ完成率 [95% CI]；p_H','ΔF1 [95% CI]；p_H','ΔMAP [95% CI]；p_H'],[[m,*[eff(x['all'][k])+'; '+pv(x['all'][k]['holm_p']) for k in ['completion','f1','map']]] for m,x in r['ablation'].items()]),'',
 table(['模型','共同成功 N / MAP N','ΔF1 [95% CI]','ΔMAP [95% CI]'],[[m,str(x['common']['n'])+' / '+str(x['common']['ranking_n']),eff(x['common']['f1']),eff(x['common']['map'])] for m,x in r['ablation'].items()]),'',
'4B 和 27B 的三个端到端指标在各自三个模型的 Holm 家族内获得支持；9B 三项区间均跨零。共同成功的 MAP 只有 4B 的区间完全为正，9B 和 27B 尚不明确。9B 消融 F1 的逐项目删除差异范围为 -0.25 至 +5.31，也不宜表述为普遍稳定提升。','',
'这组消融支持“提供 runtime 证据及其使用工作流有助于部分 Qwen 配置的端到端效果，收益明显涉及完成率”，不支持“所有规模都获得独立于完成率的 F1/MAP 改善”。单次 run1 是一致的选取规则，但仍不能估计该消融自身的运行方差。','',
'## 4. 实验条件核查','',
'1. **DeepSeek 旧/新服务配置不同。** 旧论文 Baseline 使用 openai/deepseek-v4-*；VSM 使用 openai/deepseek/deepseek-v4-*-20260423，显式 high reasoning 并锁定 Alibaba。旧 Pro-32K 的保存配置显式 reasoning_effort=none，新 VSM 为 high。不能把这组差异都归为检索器或上下文长度。当前 Pro Baseline 与 VSM 的 model/provider/reasoning 标签已对齐，但 Flash 仍沿用旧 Baseline；当前 32K Baseline 锁定 baseten，VSM 锁定 Alibaba。','',
'2. **GPT 模型标识不同。** Baseline 保存 openai/gpt-5.4，VSM 保存 openai/gpt-5.4-2026-03-05。尚未核实旧 alias 当时是否对应相同快照。','',
'3. **VSM 运行的提示仍写 BM25。** 七组 VSM 的保存模板均使用 BM25/bm25_score 的说明，而实际证据记录是 vsm_score。这是执行记录中的真实不一致，不能通过现在修改 YAML 来追溯修正既有运行。正式实验需如实说明并统一提示、字段和方法描述；若据此更改实际提示，新的结果应另行运行。','',
'4. **现有 no-runtime 不只是删掉证据文件。** 它使用不含 dynamic_evidence 的 baseline 工作台；模型名、50-call 限额、observation 模板与 VSM 相同。但提示同时删掉 Dynamic Evidence 章节中的通用阅读控制规则，如 wc、禁止直接 cat 大文件、sed 最多 80 行、rg 最多 20 项、截断后缩小搜索。因此它更准确地消融“runtime 证据及其配套使用规则”整个模块，不能严格隔离只有 runtime 信息本身的作用。若要更纯的消融，应两边保留相同通用阅读规则，只删除 runtime 文件及专属指导，然后重新运行该消融。','',
'5. **Trace VSM 缓存与实际证据存在少量排序差异。** 本报告已用实际 workbench 顺序为主要 Trace comparator，并保留缓存结果。Static VSM 与缓存 Trace VSM 的相同文档表示更适于声称严格控制检索器的比较；agent 的直接前一阶段应使用它实际接收的顺序。两种口径数值非常接近且不改变判断，但最终复现材料应统一生成方式。','',
'## 5. 统计解释检查','',
'11/11 类已检查；以下是本次检查的适用性和边界，不表示实验已消除所有偏差。','',
 table(['检查项','本次处理/发现'],[
 ['Simpson paradox','报告逐项目删除范围，9B 消融 F1 可换符号；不把总体正值推广到每个项目。'],
 ['Ecological fallacy','推断限定到此 benchmark/模型；不推断每个实例都会改善。'],
 ['Berkson selection','共同成功子集是选择后的样本，仅作补充。'],
 ['Collider bias','共同成功不能被解释成已因果隔离完成率贡献。'],
 ['Base rate neglect','同时报告 226 分母的有效输出率和 MAP 的 170 分母。'],
 ['Regression to mean','使用统一 run1 规则，三次全部展示，不选择最好 run。'],
 ['Survivorship bias','全部实例为主分析，失败计零；共同成功为次分析。'],
 ['Look elsewhere','完整保留比较及校正，不仅展示通过阈值的结果。'],
 ['Forking paths','分析为用户指定替代方案的事后敏感性检查，记录 comparator 版本、失败 k=3 和选择规则。'],
 ['Causal attribution','指出 provider/reasoning、提示阅读规则、API 404 等混杂，不能归因于单个因素。'],
 ['Reverse causality','非相关性时序回归场景，不提出反向因果主张。']]),'',
'## 复现状态','',
'ANALYZED：已重算预测指标与统计量、检查来源哈希和保存配置；未重新执行 LLM 实验，不能宣称已经复现随机生成过程。','',
'运行命令（仓库根目录）：','',
'```bash','python_venv/dalic-nexp/bin/python dataset/analysis/vsm_claims_2026-10-04/analyze.py','python_venv/dalic-nexp/bin/python dataset/analysis/vsm_claims_2026-10-04/check_repetitions.py','python_venv/dalic-nexp/bin/python dataset/analysis/vsm_claims_2026-10-04/render.py','```','']
s='\n'.join(L);assert '—' not in s
(P/'report.md').write_text(s)
print('SAVED',P/'report.md','lines',len(s.splitlines()))
