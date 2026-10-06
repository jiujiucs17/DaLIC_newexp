# 已完成 DaLIC-VSM run 与对应模型比较

仅重新运行 evaluate.py run：DeepSeek-v4-pro-0423-openrouter-alibaba，以及Qwen35-9B-runpod-run1、run2、run3。输出写回这四个目录的evaluation.json和evaluation.txt。旧DS VSM、baseline和DaLIC-BM25直接读取已有evaluation，不重新评分。

主表使用All Instances：226个验证后实例，失败或无效结果记零，ground truth为added与edited的并集。P/R/F1为实例宏平均。MAP仅对该粒度ground truth非空的实例求平均：File=226、Class=172、Function=217、Avg3=170。有效数为evaluation的valid_n，与runner summary的success_count可能不同。

11份evaluation的cohort、ground truth和scoring version完全相同；逐实例results.txt SHA256校验全部通过。

## DeepSeek v4 Pro

单位：百分数。

| Run | 有效/226 | File F1 | Class F1 | Function F1 | Avg3 F1 | Avg3 MAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 225/226 | 84.30 | 78.71 | 75.19 | 79.40 | 87.55 |
| DaLIC-BM25 | 224/226 | 87.35 | 82.74 | 77.90 | 82.67 | 87.88 |
| DaLIC-VSM 旧 | 225/226 | 85.71 | 80.58 | 75.23 | 80.51 | 88.34 |
| DaLIC-VSM 新 | 226/226 | 86.74 | 79.51 | 75.49 | 80.58 | 90.04 |

## Qwen 3.5 9B

单位：百分数。

| Run | 有效/226 | File F1 | Class F1 | Function F1 | Avg3 F1 | Avg3 MAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline run1 | 188/226 | 69.66 | 61.61 | 54.86 | 62.04 | 61.20 |
| Baseline run2 | 185/226 | 69.02 | 63.00 | 56.90 | 62.97 | 62.31 |
| Baseline run3 | 169/226 | 61.26 | 57.02 | 50.01 | 56.10 | 53.96 |
| DaLIC-BM25 | 203/226 | 75.93 | 68.26 | 59.27 | 67.82 | 72.94 |
| DaLIC-VSM run1 | 210/226 | 76.60 | 71.44 | 61.22 | 69.75 | 74.35 |
| DaLIC-VSM run2 | 210/226 | 77.04 | 68.42 | 58.36 | 67.94 | 72.75 |
| DaLIC-VSM run3 | 210/226 | 77.14 | 69.18 | 59.07 | 68.47 | 75.19 |

## 配置核对

以下从全部已保存trajectory的info.config汇总，未推断服务端未记录的默认值。

| Run | model_name | reasoning_effort | Provider配置 | Context length记录 | Step limit |
|---|---|---|---|---|---:|
| DeepSeek v4 Pro Baseline (226 trajectories) | openai/deepseek/deepseek-v4-pro-20260423 | high | {"only": ["Alibaba"], "order": ["Alibaba"], "allow_fallbacks": false} | 未记录 | 50 |
| DeepSeek v4 Pro DaLIC-BM25 (226 trajectories) | openai/deepseek-v4-pro | 未记录 | 未记录 | 未记录 | 50 |
| DeepSeek v4 Pro DaLIC-VSM 旧 (226 trajectories) | openai/deepseek/deepseek-v4-pro-20260423 | high | {"only": ["Alibaba"], "order": ["Alibaba"], "allow_fallbacks": false} | 未记录 | 50 |
| DeepSeek v4 Pro DaLIC-VSM 新 (226 trajectories) | openai/deepseek/deepseek-v4-pro-20260423 | high | {"only": ["Alibaba"], "order": ["Alibaba"], "allow_fallbacks": false} | 未记录 | 50 |
| Qwen 3.5 9B Baseline run1 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B Baseline run2 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B Baseline run3 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B DaLIC-BM25 (226 trajectories) | openai/local | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B DaLIC-VSM run1 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B DaLIC-VSM run2 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |
| Qwen 3.5 9B DaLIC-VSM run3 (226 trajectories) | openai/Qwen3.5-9B | 未记录 | 未记录 | 未记录 | 50 |

DS新旧VSM及对应baseline记录的model_name、reasoning_effort、provider设置一致。归档DS BM25为openai/deepseek-v4-pro，reasoning_effort及provider未记录。因此是对应模型系列的并列结果，不能称为完全相同模型服务配置的控制实验。Qwen baseline与VSM为openai/Qwen3.5-9B，归档BM25记录为openai/local，具体服务端配置不能仅由该alias确认。

旧DS VSM的evaluation元数据仍使用改名前的目录和model标签。本报告按实际读取路径DeepSeek-v4-pro-0423-openrouter-alibaba-1标识，未改动旧evaluation。

## 全部分粒度指标

### all_instances

valid_only为各run自身的有效实例集合，集合可能不同，不等同于Common Success。

| 模型 | Run | 粒度 | N | Ranking N | Precision | Recall | F1 | MAP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v4 Pro | Baseline | files | 226 | 226 | 84.09 | 92.82 | 84.30 | 91.28 |
| DeepSeek v4 Pro | Baseline | classes | 226 | 172 | 78.60 | 86.38 | 78.71 | 88.25 |
| DeepSeek v4 Pro | Baseline | functions | 226 | 217 | 75.39 | 87.00 | 75.19 | 84.48 |
| DeepSeek v4 Pro | Baseline | avg3 | 226 | 170 | 79.36 | 88.73 | 79.40 | 87.55 |
| DeepSeek v4 Pro | DaLIC-BM25 | files | 226 | 226 | 88.40 | 93.37 | 87.35 | 92.31 |
| DeepSeek v4 Pro | DaLIC-BM25 | classes | 226 | 172 | 83.54 | 88.15 | 82.74 | 87.65 |
| DeepSeek v4 Pro | DaLIC-BM25 | functions | 226 | 217 | 77.17 | 88.81 | 77.90 | 87.00 |
| DeepSeek v4 Pro | DaLIC-BM25 | avg3 | 226 | 170 | 83.04 | 90.11 | 82.67 | 87.88 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | files | 226 | 226 | 86.01 | 93.56 | 85.71 | 91.92 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | classes | 226 | 172 | 81.23 | 86.79 | 80.58 | 89.09 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | functions | 226 | 217 | 74.59 | 87.68 | 75.23 | 86.87 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | avg3 | 226 | 170 | 80.61 | 89.35 | 80.51 | 88.34 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | files | 226 | 226 | 86.66 | 94.58 | 86.74 | 92.86 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | classes | 226 | 172 | 79.60 | 87.28 | 79.51 | 90.95 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | functions | 226 | 217 | 73.78 | 89.66 | 75.49 | 88.73 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | avg3 | 226 | 170 | 80.01 | 90.50 | 80.58 | 90.04 |
| Qwen 3.5 9B | Baseline run1 | files | 226 | 226 | 70.92 | 73.35 | 69.66 | 71.57 |
| Qwen 3.5 9B | Baseline run1 | classes | 226 | 172 | 62.95 | 65.32 | 61.61 | 62.17 |
| Qwen 3.5 9B | Baseline run1 | functions | 226 | 217 | 57.23 | 60.11 | 54.86 | 55.37 |
| Qwen 3.5 9B | Baseline run1 | avg3 | 226 | 170 | 63.70 | 66.26 | 62.04 | 61.20 |
| Qwen 3.5 9B | Baseline run2 | files | 226 | 226 | 70.70 | 72.31 | 69.02 | 70.18 |
| Qwen 3.5 9B | Baseline run2 | classes | 226 | 172 | 64.98 | 65.07 | 63.00 | 63.00 |
| Qwen 3.5 9B | Baseline run2 | functions | 226 | 217 | 59.58 | 62.78 | 56.90 | 57.20 |
| Qwen 3.5 9B | Baseline run2 | avg3 | 226 | 170 | 65.09 | 66.72 | 62.97 | 62.31 |
| Qwen 3.5 9B | Baseline run3 | files | 226 | 226 | 62.51 | 64.25 | 61.26 | 62.49 |
| Qwen 3.5 9B | Baseline run3 | classes | 226 | 172 | 57.86 | 59.32 | 57.02 | 56.68 |
| Qwen 3.5 9B | Baseline run3 | functions | 226 | 217 | 52.50 | 54.22 | 50.01 | 49.71 |
| Qwen 3.5 9B | Baseline run3 | avg3 | 226 | 170 | 57.63 | 59.26 | 56.10 | 53.96 |
| Qwen 3.5 9B | DaLIC-BM25 | files | 226 | 226 | 77.21 | 81.29 | 75.93 | 78.53 |
| Qwen 3.5 9B | DaLIC-BM25 | classes | 226 | 172 | 69.65 | 73.31 | 68.26 | 74.12 |
| Qwen 3.5 9B | DaLIC-BM25 | functions | 226 | 217 | 59.40 | 71.47 | 59.27 | 68.17 |
| Qwen 3.5 9B | DaLIC-BM25 | avg3 | 226 | 170 | 68.76 | 75.35 | 67.82 | 72.94 |
| Qwen 3.5 9B | DaLIC-VSM run1 | files | 226 | 226 | 77.07 | 84.28 | 76.60 | 80.51 |
| Qwen 3.5 9B | DaLIC-VSM run1 | classes | 226 | 172 | 73.11 | 77.47 | 71.44 | 76.07 |
| Qwen 3.5 9B | DaLIC-VSM run1 | functions | 226 | 217 | 60.81 | 74.18 | 61.22 | 68.46 |
| Qwen 3.5 9B | DaLIC-VSM run1 | avg3 | 226 | 170 | 70.33 | 78.64 | 69.75 | 74.35 |
| Qwen 3.5 9B | DaLIC-VSM run2 | files | 226 | 226 | 78.29 | 83.93 | 77.04 | 80.38 |
| Qwen 3.5 9B | DaLIC-VSM run2 | classes | 226 | 172 | 70.01 | 74.33 | 68.42 | 73.62 |
| Qwen 3.5 9B | DaLIC-VSM run2 | functions | 226 | 217 | 55.94 | 75.88 | 58.36 | 70.73 |
| Qwen 3.5 9B | DaLIC-VSM run2 | avg3 | 226 | 170 | 68.08 | 78.05 | 67.94 | 72.75 |
| Qwen 3.5 9B | DaLIC-VSM run3 | files | 226 | 226 | 77.27 | 85.86 | 77.14 | 82.95 |
| Qwen 3.5 9B | DaLIC-VSM run3 | classes | 226 | 172 | 69.85 | 75.59 | 69.18 | 77.44 |
| Qwen 3.5 9B | DaLIC-VSM run3 | functions | 226 | 217 | 57.68 | 74.30 | 59.07 | 67.84 |
| Qwen 3.5 9B | DaLIC-VSM run3 | avg3 | 226 | 170 | 68.27 | 78.59 | 68.47 | 75.19 |

### valid_only

valid_only为各run自身的有效实例集合，集合可能不同，不等同于Common Success。

| 模型 | Run | 粒度 | N | Ranking N | Precision | Recall | F1 | MAP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v4 Pro | Baseline | files | 225 | 225 | 84.46 | 93.24 | 84.68 | 91.68 |
| DeepSeek v4 Pro | Baseline | classes | 225 | 171 | 78.95 | 86.76 | 79.06 | 88.76 |
| DeepSeek v4 Pro | Baseline | functions | 225 | 216 | 75.73 | 87.39 | 75.53 | 84.87 |
| DeepSeek v4 Pro | Baseline | avg3 | 225 | 169 | 79.71 | 89.13 | 79.75 | 88.07 |
| DeepSeek v4 Pro | DaLIC-BM25 | files | 224 | 224 | 89.19 | 94.21 | 88.13 | 93.14 |
| DeepSeek v4 Pro | DaLIC-BM25 | classes | 224 | 170 | 84.28 | 88.94 | 83.48 | 88.68 |
| DeepSeek v4 Pro | DaLIC-BM25 | functions | 224 | 215 | 77.86 | 89.60 | 78.60 | 87.81 |
| DeepSeek v4 Pro | DaLIC-BM25 | avg3 | 224 | 168 | 83.78 | 90.92 | 83.40 | 88.92 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | files | 225 | 225 | 86.39 | 93.98 | 86.09 | 92.33 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | classes | 225 | 171 | 81.59 | 87.18 | 80.94 | 89.61 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | functions | 225 | 216 | 74.92 | 88.07 | 75.56 | 87.27 |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | avg3 | 225 | 169 | 80.97 | 89.74 | 80.86 | 88.86 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | files | 226 | 226 | 86.66 | 94.58 | 86.74 | 92.86 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | classes | 226 | 172 | 79.60 | 87.28 | 79.51 | 90.95 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | functions | 226 | 217 | 73.78 | 89.66 | 75.49 | 88.73 |
| DeepSeek v4 Pro | DaLIC-VSM 新 | avg3 | 226 | 170 | 80.01 | 90.50 | 80.58 | 90.04 |
| Qwen 3.5 9B | Baseline run1 | files | 188 | 188 | 85.25 | 88.17 | 83.74 | 86.03 |
| Qwen 3.5 9B | Baseline run1 | classes | 188 | 142 | 75.67 | 78.52 | 74.06 | 75.31 |
| Qwen 3.5 9B | Baseline run1 | functions | 188 | 181 | 68.80 | 72.26 | 65.95 | 66.38 |
| Qwen 3.5 9B | Baseline run1 | avg3 | 188 | 140 | 76.57 | 79.65 | 74.58 | 74.31 |
| Qwen 3.5 9B | Baseline run2 | files | 185 | 185 | 86.37 | 88.33 | 84.31 | 85.74 |
| Qwen 3.5 9B | Baseline run2 | classes | 185 | 141 | 79.39 | 79.49 | 76.96 | 76.85 |
| Qwen 3.5 9B | Baseline run2 | functions | 185 | 178 | 72.78 | 76.69 | 69.51 | 69.73 |
| Qwen 3.5 9B | Baseline run2 | avg3 | 185 | 139 | 79.51 | 81.50 | 76.93 | 76.21 |
| Qwen 3.5 9B | Baseline run3 | files | 169 | 169 | 83.60 | 85.92 | 81.92 | 83.57 |
| Qwen 3.5 9B | Baseline run3 | classes | 169 | 125 | 77.38 | 79.33 | 76.26 | 77.99 |
| Qwen 3.5 9B | Baseline run3 | functions | 169 | 161 | 70.21 | 72.50 | 66.88 | 67.00 |
| Qwen 3.5 9B | Baseline run3 | avg3 | 169 | 123 | 77.06 | 79.25 | 75.02 | 74.59 |
| Qwen 3.5 9B | DaLIC-BM25 | files | 203 | 203 | 85.96 | 90.50 | 84.53 | 87.42 |
| Qwen 3.5 9B | DaLIC-BM25 | classes | 203 | 153 | 77.55 | 81.61 | 75.99 | 83.32 |
| Qwen 3.5 9B | DaLIC-BM25 | functions | 203 | 196 | 66.13 | 79.57 | 65.99 | 75.47 |
| Qwen 3.5 9B | DaLIC-BM25 | avg3 | 203 | 152 | 76.55 | 83.89 | 75.50 | 81.57 |
| Qwen 3.5 9B | DaLIC-VSM run1 | files | 210 | 210 | 82.94 | 90.70 | 82.43 | 86.65 |
| Qwen 3.5 9B | DaLIC-VSM run1 | classes | 210 | 160 | 78.68 | 83.37 | 76.89 | 81.78 |
| Qwen 3.5 9B | DaLIC-VSM run1 | functions | 210 | 202 | 65.45 | 79.83 | 65.89 | 73.55 |
| Qwen 3.5 9B | DaLIC-VSM run1 | avg3 | 210 | 158 | 75.69 | 84.64 | 75.07 | 80.00 |
| Qwen 3.5 9B | DaLIC-VSM run2 | files | 210 | 210 | 84.25 | 90.32 | 82.91 | 86.51 |
| Qwen 3.5 9B | DaLIC-VSM run2 | classes | 210 | 157 | 75.35 | 80.00 | 73.63 | 80.65 |
| Qwen 3.5 9B | DaLIC-VSM run2 | functions | 210 | 201 | 60.21 | 81.66 | 62.80 | 76.36 |
| Qwen 3.5 9B | DaLIC-VSM run2 | avg3 | 210 | 155 | 73.27 | 83.99 | 73.11 | 79.79 |
| Qwen 3.5 9B | DaLIC-VSM run3 | files | 210 | 210 | 83.16 | 92.40 | 83.02 | 89.27 |
| Qwen 3.5 9B | DaLIC-VSM run3 | classes | 210 | 159 | 75.17 | 81.35 | 74.45 | 83.77 |
| Qwen 3.5 9B | DaLIC-VSM run3 | functions | 210 | 201 | 62.07 | 79.96 | 63.57 | 73.24 |
| Qwen 3.5 9B | DaLIC-VSM run3 | avg3 | 210 | 157 | 73.47 | 84.57 | 73.68 | 81.41 |

## 来源

| 模型 | Run | Evaluation |
|---|---|---|
| DeepSeek v4 Pro | Baseline | [dataset/agent_results/baseline/DeepSeek-v4-pro-0423-openrouter-alibaba/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/DeepSeek-v4-pro-0423-openrouter-alibaba/evaluation.json) |
| DeepSeek v4 Pro | DaLIC-BM25 | [dataset/results_archive/agent_results/dalic/DeepSeek-v4-pro/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/results_archive/agent_results/dalic/DeepSeek-v4-pro/evaluation.json) |
| DeepSeek v4 Pro | DaLIC-VSM 旧 | [dataset/agent_results/dalic_vsm/DeepSeek-v4-pro-0423-openrouter-alibaba-1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/DeepSeek-v4-pro-0423-openrouter-alibaba-1/evaluation.json) |
| DeepSeek v4 Pro | DaLIC-VSM 新 | [dataset/agent_results/dalic_vsm/DeepSeek-v4-pro-0423-openrouter-alibaba/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/DeepSeek-v4-pro-0423-openrouter-alibaba/evaluation.json) |
| Qwen 3.5 9B | Baseline run1 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run1/evaluation.json) |
| Qwen 3.5 9B | Baseline run2 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run2/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run2/evaluation.json) |
| Qwen 3.5 9B | Baseline run3 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run3/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run3/evaluation.json) |
| Qwen 3.5 9B | DaLIC-BM25 | [dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B/evaluation.json) |
| Qwen 3.5 9B | DaLIC-VSM run1 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run1/evaluation.json) |
| Qwen 3.5 9B | DaLIC-VSM run2 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run2/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run2/evaluation.json) |
| Qwen 3.5 9B | DaLIC-VSM run3 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3/evaluation.json) |

## 本次运行命令

```bash
python_venv/dalic-nexp/bin/python scripts/evaluate.py run --selected-instances dataset/trace_completed_instance_validated.toml --workbench-path dataset/agent_workbench/dalic_vsm --result-folder dataset/agent_results/dalic_vsm/DeepSeek-v4-pro-0423-openrouter-alibaba
python_venv/dalic-nexp/bin/python scripts/evaluate.py run --selected-instances dataset/trace_completed_instance_validated.toml --workbench-path dataset/agent_workbench/dalic_vsm --result-folder dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run1
python_venv/dalic-nexp/bin/python scripts/evaluate.py run --selected-instances dataset/trace_completed_instance_validated.toml --workbench-path dataset/agent_workbench/dalic_vsm --result-folder dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run2
python_venv/dalic-nexp/bin/python scripts/evaluate.py run --selected-instances dataset/trace_completed_instance_validated.toml --workbench-path dataset/agent_workbench/dalic_vsm --result-folder dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3
```
