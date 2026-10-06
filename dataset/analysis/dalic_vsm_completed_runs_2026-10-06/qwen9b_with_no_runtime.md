# Qwen 3.5 9B：加入 no-runtime 三轮结果

所有指标直接读取现有 evaluation.json，本次没有重新运行 evaluate.py。三轮no-runtime的evaluation已经存在。全部使用226个验证后实例，cohort、ground truth和scoring version一致，结果文件SHA256校验通过。

主表为All Instances：无效结果记零，F1是实例宏平均，单位为百分数。MAP排除该粒度ground truth为空的实例，File/Class/Function/Avg3的ranking N分别为226/172/217/170。有效数使用evaluation的valid_n。

| Run | 有效/226 | File F1 | Class F1 | Function F1 | Avg3 F1 | Avg3 MAP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline run1 | 188/226 | 69.66 | 61.61 | 54.86 | 62.04 | 61.20 |
| Baseline run2 | 185/226 | 69.02 | 63.00 | 56.90 | 62.97 | 62.31 |
| Baseline run3 | 169/226 | 61.26 | 57.02 | 50.01 | 56.10 | 53.96 |
| DaLIC-no-runtime run1 | 195/226 | 73.49 | 66.66 | 59.57 | 66.57 | 66.59 |
| DaLIC-no-runtime run2 | 189/226 | 72.48 | 65.39 | 58.60 | 65.49 | 63.45 |
| DaLIC-no-runtime run3 | 196/226 | 74.93 | 66.69 | 60.03 | 67.22 | 66.55 |
| DaLIC-BM25 | 203/226 | 75.93 | 68.26 | 59.27 | 67.82 | 72.94 |
| DaLIC-VSM run1 | 210/226 | 76.60 | 71.44 | 61.22 | 69.75 | 74.35 |
| DaLIC-VSM run2 | 210/226 | 77.04 | 68.42 | 58.36 | 67.94 | 72.75 |
| DaLIC-VSM run3 | 210/226 | 77.14 | 69.18 | 59.07 | 68.47 | 75.19 |

## 三轮描述性平均

| 方法 | Avg3 Precision | Avg3 Recall | Avg3 F1 | Avg3 MAP |
|---|---:|---:|---:|---:|
| Baseline | 62.14 | 64.08 | 60.37 | 59.16 |
| DaLIC-no-runtime | 69.21 | 68.99 | 66.43 | 65.53 |
| DaLIC-VSM | 68.89 | 78.43 | 68.72 | 74.09 |

上述为三轮指标的算术平均，不合并输出。归档BM25只有一轮，不作为三轮均值。

## 配置

| Run | model_name | model_kwargs | Step limit | trajectory N |
|---|---|---|---:|---:|
| Baseline run1 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| Baseline run2 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| Baseline run3 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-no-runtime run1 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-no-runtime run2 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-no-runtime run3 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-BM25 | openai/local | {"drop_params": true} | 50 | 226 |
| DaLIC-VSM run1 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-VSM run2 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |
| DaLIC-VSM run3 | openai/Qwen3.5-9B | {"drop_params": true} | 50 | 226 |

Baseline、no-runtime、VSM均记录为openai/Qwen3.5-9B，drop_params=True、step_limit=50。归档BM25记录为openai/local；不能仅由alias确认服务端权重和采样配置完全一致。

## 全部分粒度指标

### all_instances

valid_only使用各run自己的有效集合，不是双方共同成功实例集合。

| Run | 粒度 | N | Ranking N | Precision | Recall | F1 | MAP |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline run1 | files | 226 | 226 | 70.92 | 73.35 | 69.66 | 71.57 |
| Baseline run1 | classes | 226 | 172 | 62.95 | 65.32 | 61.61 | 62.17 |
| Baseline run1 | functions | 226 | 217 | 57.23 | 60.11 | 54.86 | 55.37 |
| Baseline run1 | avg3 | 226 | 170 | 63.70 | 66.26 | 62.04 | 61.20 |
| Baseline run2 | files | 226 | 226 | 70.70 | 72.31 | 69.02 | 70.18 |
| Baseline run2 | classes | 226 | 172 | 64.98 | 65.07 | 63.00 | 63.00 |
| Baseline run2 | functions | 226 | 217 | 59.58 | 62.78 | 56.90 | 57.20 |
| Baseline run2 | avg3 | 226 | 170 | 65.09 | 66.72 | 62.97 | 62.31 |
| Baseline run3 | files | 226 | 226 | 62.51 | 64.25 | 61.26 | 62.49 |
| Baseline run3 | classes | 226 | 172 | 57.86 | 59.32 | 57.02 | 56.68 |
| Baseline run3 | functions | 226 | 217 | 52.50 | 54.22 | 50.01 | 49.71 |
| Baseline run3 | avg3 | 226 | 170 | 57.63 | 59.26 | 56.10 | 53.96 |
| DaLIC-no-runtime run1 | files | 226 | 226 | 75.12 | 76.90 | 73.49 | 75.06 |
| DaLIC-no-runtime run1 | classes | 226 | 172 | 68.75 | 68.57 | 66.66 | 67.13 |
| DaLIC-no-runtime run1 | functions | 226 | 217 | 62.73 | 63.47 | 59.57 | 59.43 |
| DaLIC-no-runtime run1 | avg3 | 226 | 170 | 68.87 | 69.65 | 66.57 | 66.59 |
| DaLIC-no-runtime run2 | files | 226 | 226 | 74.89 | 74.37 | 72.48 | 73.19 |
| DaLIC-no-runtime run2 | classes | 226 | 172 | 68.12 | 66.99 | 65.39 | 64.03 |
| DaLIC-no-runtime run2 | functions | 226 | 217 | 62.19 | 61.10 | 58.60 | 58.56 |
| DaLIC-no-runtime run2 | avg3 | 226 | 170 | 68.40 | 67.48 | 65.49 | 63.45 |
| DaLIC-no-runtime run3 | files | 226 | 226 | 77.25 | 77.83 | 74.93 | 76.34 |
| DaLIC-no-runtime run3 | classes | 226 | 172 | 69.80 | 67.77 | 66.69 | 66.74 |
| DaLIC-no-runtime run3 | functions | 226 | 217 | 64.02 | 63.95 | 60.03 | 61.47 |
| DaLIC-no-runtime run3 | avg3 | 226 | 170 | 70.35 | 69.85 | 67.22 | 66.55 |
| DaLIC-BM25 | files | 226 | 226 | 77.21 | 81.29 | 75.93 | 78.53 |
| DaLIC-BM25 | classes | 226 | 172 | 69.65 | 73.31 | 68.26 | 74.12 |
| DaLIC-BM25 | functions | 226 | 217 | 59.40 | 71.47 | 59.27 | 68.17 |
| DaLIC-BM25 | avg3 | 226 | 170 | 68.76 | 75.35 | 67.82 | 72.94 |
| DaLIC-VSM run1 | files | 226 | 226 | 77.07 | 84.28 | 76.60 | 80.51 |
| DaLIC-VSM run1 | classes | 226 | 172 | 73.11 | 77.47 | 71.44 | 76.07 |
| DaLIC-VSM run1 | functions | 226 | 217 | 60.81 | 74.18 | 61.22 | 68.46 |
| DaLIC-VSM run1 | avg3 | 226 | 170 | 70.33 | 78.64 | 69.75 | 74.35 |
| DaLIC-VSM run2 | files | 226 | 226 | 78.29 | 83.93 | 77.04 | 80.38 |
| DaLIC-VSM run2 | classes | 226 | 172 | 70.01 | 74.33 | 68.42 | 73.62 |
| DaLIC-VSM run2 | functions | 226 | 217 | 55.94 | 75.88 | 58.36 | 70.73 |
| DaLIC-VSM run2 | avg3 | 226 | 170 | 68.08 | 78.05 | 67.94 | 72.75 |
| DaLIC-VSM run3 | files | 226 | 226 | 77.27 | 85.86 | 77.14 | 82.95 |
| DaLIC-VSM run3 | classes | 226 | 172 | 69.85 | 75.59 | 69.18 | 77.44 |
| DaLIC-VSM run3 | functions | 226 | 217 | 57.68 | 74.30 | 59.07 | 67.84 |
| DaLIC-VSM run3 | avg3 | 226 | 170 | 68.27 | 78.59 | 68.47 | 75.19 |

### valid_only

valid_only使用各run自己的有效集合，不是双方共同成功实例集合。

| Run | 粒度 | N | Ranking N | Precision | Recall | F1 | MAP |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline run1 | files | 188 | 188 | 85.25 | 88.17 | 83.74 | 86.03 |
| Baseline run1 | classes | 188 | 142 | 75.67 | 78.52 | 74.06 | 75.31 |
| Baseline run1 | functions | 188 | 181 | 68.80 | 72.26 | 65.95 | 66.38 |
| Baseline run1 | avg3 | 188 | 140 | 76.57 | 79.65 | 74.58 | 74.31 |
| Baseline run2 | files | 185 | 185 | 86.37 | 88.33 | 84.31 | 85.74 |
| Baseline run2 | classes | 185 | 141 | 79.39 | 79.49 | 76.96 | 76.85 |
| Baseline run2 | functions | 185 | 178 | 72.78 | 76.69 | 69.51 | 69.73 |
| Baseline run2 | avg3 | 185 | 139 | 79.51 | 81.50 | 76.93 | 76.21 |
| Baseline run3 | files | 169 | 169 | 83.60 | 85.92 | 81.92 | 83.57 |
| Baseline run3 | classes | 169 | 125 | 77.38 | 79.33 | 76.26 | 77.99 |
| Baseline run3 | functions | 169 | 161 | 70.21 | 72.50 | 66.88 | 67.00 |
| Baseline run3 | avg3 | 169 | 123 | 77.06 | 79.25 | 75.02 | 74.59 |
| DaLIC-no-runtime run1 | files | 195 | 195 | 87.06 | 89.12 | 85.17 | 86.99 |
| DaLIC-no-runtime run1 | classes | 195 | 148 | 79.68 | 79.47 | 77.25 | 78.01 |
| DaLIC-no-runtime run1 | functions | 195 | 188 | 72.71 | 73.56 | 69.04 | 68.59 |
| DaLIC-no-runtime run1 | avg3 | 195 | 147 | 79.82 | 80.72 | 77.15 | 77.01 |
| DaLIC-no-runtime run2 | files | 189 | 189 | 89.55 | 88.93 | 86.67 | 87.52 |
| DaLIC-no-runtime run2 | classes | 189 | 144 | 81.46 | 80.10 | 78.19 | 76.48 |
| DaLIC-no-runtime run2 | functions | 189 | 181 | 74.36 | 73.06 | 70.07 | 70.21 |
| DaLIC-no-runtime run2 | avg3 | 189 | 142 | 81.79 | 80.70 | 78.31 | 75.96 |
| DaLIC-no-runtime run3 | files | 196 | 196 | 89.07 | 89.75 | 86.40 | 88.03 |
| DaLIC-no-runtime run3 | classes | 196 | 149 | 80.48 | 78.15 | 76.89 | 77.04 |
| DaLIC-no-runtime run3 | functions | 196 | 187 | 73.81 | 73.74 | 69.22 | 71.33 |
| DaLIC-no-runtime run3 | avg3 | 196 | 147 | 81.12 | 80.54 | 77.51 | 76.97 |
| DaLIC-BM25 | files | 203 | 203 | 85.96 | 90.50 | 84.53 | 87.42 |
| DaLIC-BM25 | classes | 203 | 153 | 77.55 | 81.61 | 75.99 | 83.32 |
| DaLIC-BM25 | functions | 203 | 196 | 66.13 | 79.57 | 65.99 | 75.47 |
| DaLIC-BM25 | avg3 | 203 | 152 | 76.55 | 83.89 | 75.50 | 81.57 |
| DaLIC-VSM run1 | files | 210 | 210 | 82.94 | 90.70 | 82.43 | 86.65 |
| DaLIC-VSM run1 | classes | 210 | 160 | 78.68 | 83.37 | 76.89 | 81.78 |
| DaLIC-VSM run1 | functions | 210 | 202 | 65.45 | 79.83 | 65.89 | 73.55 |
| DaLIC-VSM run1 | avg3 | 210 | 158 | 75.69 | 84.64 | 75.07 | 80.00 |
| DaLIC-VSM run2 | files | 210 | 210 | 84.25 | 90.32 | 82.91 | 86.51 |
| DaLIC-VSM run2 | classes | 210 | 157 | 75.35 | 80.00 | 73.63 | 80.65 |
| DaLIC-VSM run2 | functions | 210 | 201 | 60.21 | 81.66 | 62.80 | 76.36 |
| DaLIC-VSM run2 | avg3 | 210 | 155 | 73.27 | 83.99 | 73.11 | 79.79 |
| DaLIC-VSM run3 | files | 210 | 210 | 83.16 | 92.40 | 83.02 | 89.27 |
| DaLIC-VSM run3 | classes | 210 | 159 | 75.17 | 81.35 | 74.45 | 83.77 |
| DaLIC-VSM run3 | functions | 210 | 201 | 62.07 | 79.96 | 63.57 | 73.24 |
| DaLIC-VSM run3 | avg3 | 210 | 157 | 73.47 | 84.57 | 73.68 | 81.41 |

## 来源

| Run | Evaluation |
|---|---|
| Baseline run1 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run1/evaluation.json) |
| Baseline run2 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run2/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run2/evaluation.json) |
| Baseline run3 | [dataset/agent_results/baseline/Qwen35-9B-runpod-run3/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/baseline/Qwen35-9B-runpod-run3/evaluation.json) |
| DaLIC-no-runtime run1 | [dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run1/evaluation.json) |
| DaLIC-no-runtime run2 | [dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run2/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run2/evaluation.json) |
| DaLIC-no-runtime run3 | [dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run3/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_no_runtime/Qwen35-9B-no-runtime-run3/evaluation.json) |
| DaLIC-BM25 | [dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/results_archive/agent_results/dalic/Qwen_Qwen3.5-9B/evaluation.json) |
| DaLIC-VSM run1 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run1/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run1/evaluation.json) |
| DaLIC-VSM run2 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run2/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run2/evaluation.json) |
| DaLIC-VSM run3 | [dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3/evaluation.json](/home/mengqi/work/DaLIC_newexp/dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3/evaluation.json) |
