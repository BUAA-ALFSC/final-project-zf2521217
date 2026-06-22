# Lag-Llama MindSpore 论文复现

本项目使用 MindSpore 复现论文 **Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting** 的核心方法，并在 Huawei ModelArts 上完成资源受限实验。原始 Lag-Llama 官方代码基于 PyTorch、Lightning 和 GluonTS；本目录提供面向课程作业的 MindSpore 实现、运行脚本、实验结果和可视化输出。

课程任务要求：使用 MindSpore 深度学习框架复现 Lag-Llama，提交代码、Jupyter Notebook、README、研究报告和 PPT。

## 复现目标

Lag-Llama 是一个面向单变量概率时间序列预测的基础模型。论文的核心结论包括：

- 使用多领域时间序列数据预训练 decoder-only Transformer；
- 利用 lag 特征、时间特征和 Student-T 概率输出建模不确定性；
- 预训练模型在未见数据集上具备 zero-shot 预测能力；
- 在下游数据集上 fine-tuning 后，模型预测性能显著提升；
- fine-tuned Lag-Llama 在多个下游数据集上达到或接近强基线效果。

本项目复现重点是：

- MindSpore 版 Lag-Llama 模型结构；
- 多数据集预训练；
- 未见数据集 zero-shot 评估；
- 下游 fine-tuning；
- 概率预测指标 CRPS / mean weighted quantile loss；
- 与代表性统计和深度学习基线对比。

## 项目结构

```text
mindspore_lag_llama/
  README.md
  MODELARTS.md
  DATASETS.md
  requirements.txt
  notebooks/
    lag_llama_mindspore_reproduction.ipynb
  src/
    config.py             # 实验配置
    data.py               # CSV / JSON / 多数据集加载与滑窗构造
    device.py             # MindSpore 设备选择
    model.py              # MindSpore Lag-Llama 模型
    train.py              # 训练入口
    evaluate.py           # 评估入口
    metrics.py            # MAE/MSE/CRPS/基线指标
    baseline_models.py    # DeepAR/PatchTST 轻量基线
  scripts/
    run_smoke.sh
    run_ett_quick.sh
    run_ett_budget.sh
    run_paper10_budget.sh
    run_paper10_downstreams.sh
    prepare_paper10_datasets.py
    run_paper_baselines.py
    run_deep_baselines.py
    plot_paper10_summary.py
  data/
  results/
```

其中 `results/` 保存 ModelArts 运行得到的 JSON 结果、训练历史和图像。若本地代码目录未包含最终 `results/`，请将从 ModelArts 下载的 `lag_llama_final_report_assets_light.zip` 解压，并把其中的 `mindspore_lag_llama/results/` 放回本目录。

## 环境说明

推荐在 ModelArts Notebook 中运行。已验证的主要环境为：

```text
MindSpore 2.4.10 / CANN 8.0.0 / Ascend NPU
Python 3.10
```

也测试过 MindSpore 2.7.0rc1 / CANN 8.2.rc1。简单 `lgamma` 梯度可以通过，但完整训练图在部分 Ascend 环境下可能出现 `CustLgamma` 类型推导问题。因此代码保留了 `--student_t_fixed_df` 作为后备开关。最终主实验 JSON 记录为：

```text
loss = student_t
student_t_fixed_df = false
```

说明最终结果使用的是三参数 Student-T 输出。

安装额外依赖：

```bash
pip install -r requirements.txt
pip install gluonts==0.14.4 statsforecast
```

注意：`statsforecast` 可能升级 `pandas`，从而与 ModelArts 内置包产生版本警告。建议先完成主训练和评估，再安装 `statsforecast` 补跑统计基线。

验证 MindSpore：

```bash
python -c "import mindspore as ms; print(ms.__version__); print(ms.get_context('device_target'))"
```

## 已实现的核心技术

本项目实现了 Lag-Llama 的主要结构和训练流程：

- decoder-only Transformer；
- causal self-attention；
- RMSNorm；
- RoPE rotary position embedding；
- LLaMA-style gated MLP；
- lagged target covariates；
- loc/scale 静态缩放特征；
- robust scaling；
- 六维 calendar time features；
- Student-T negative log likelihood；
- autoregressive multi-step forecasting；
- training-only frequency masking / frequency mixing 数据增强；
- early stopping；
- best/latest/epoch checkpoint 保存；
- MAE、MSE、sample CRPS、GluonTS-style mean weighted quantile loss；
- last-value、moving average、drift、seasonal naive 等轻量基线；
- AutoETS、DynOptTheta 统计基线；
- DeepAR、PatchTST 轻量 MindSpore 深度基线。

## 数据集设置

论文完整实验使用 27 个真实数据集，其中一部分用于预训练，一部分作为未见下游数据集。受课程预算和时间限制，本项目采用真实数据子集进行资源受限复现。

### 预训练数据集

`paper10` 主实验使用 10 个论文列表中的真实数据集作为预训练数据：

```text
ETTh1
ETTh2
ETTm1
electricity_hourly
solar_10_minutes
traffic
kdd_cup_2018_without_missing
sunspot_without_missing
australian_electricity_demand
london_smart_meters_without_missing
```

数据准备脚本会导出为 CSV，并存放在：

```text
data/paper10_pretrain/
```

### 未见下游数据集

保留 3 个真实数据集作为未见下游数据集：

```text
ETTm2
exchange_rate
weather
```

对应目录：

```text
data/paper10_downstream/
```

这些数据集不参与预训练，用于 zero-shot 和 fine-tuning 评估。

## 运行流程

### 1. 冒烟测试

用于确认代码和 MindSpore 环境是否可运行，不作为论文复现结果。

```bash
cd mindspore_lag_llama
bash scripts/run_smoke.sh
```

期望输出：

```text
results/metrics_smoke.json
results/train_history_smoke.json
results/figures/forecast_smoke.png
```

### 2. 主实验：10 数据集预训练 + 3 下游数据集

```bash
cd mindspore_lag_llama
bash scripts/run_paper10_budget.sh
```

该脚本执行：

1. 下载/转换 10 个预训练数据集；
2. 在 Ascend 上预训练 Lag-Llama；
3. 在 ETTm2、exchange_rate、weather 上 zero-shot 评估；
4. 分别 fine-tune 三个下游数据集；
5. 评估 fine-tuned 模型；
6. 运行统计基线和深度基线；
7. 生成汇总图。

可调预算参数：

```bash
PRETRAIN_EPOCHS=150 \
FINETUNE_EPOCHS=20 \
MAX_TRAIN_BATCHES=256 \
MAX_VAL_BATCHES=256 \
EVAL_SAMPLES=10 \
bash scripts/run_paper10_budget.sh
```

### 3. 仅重跑下游评估和基线

如果预训练 checkpoint 已存在，可只运行下游流程：

```bash
bash scripts/run_paper10_downstreams.sh
```

默认预训练 checkpoint：

```text
results/checkpoints/lag_llama_paper10_pretrain_best.ckpt
```

### 4. 补充 samples=30 CRPS 稳定性评估

论文使用更多 empirical samples 计算 CRPS。本实验主结果使用 `num_samples=10`，并额外对 fine-tuned 模型补充 `num_samples=30` 的稳定性复核。为控制 CPU 评估时间，补充实验限制 `max_eval_windows=64`。

示例命令：

```bash
python -m src.evaluate \
  --dataset csv \
  --data_path data/paper10_downstream/ETTm2.csv \
  --target_column OT \
  --time_column date \
  --checkpoint_path results/checkpoints/lag_llama_paper10_ettm2_finetune_best.ckpt \
  --output_checkpoint lag_llama_paper10_ettm2_finetune_best.ckpt \
  --metrics_file metrics_paper10_finetuned_ettm2_samples30.json \
  --figure_file forecast_paper10_finetuned_ettm2_samples30.png \
  --device_target CPU \
  --num_samples 30 \
  --batch_size 8 \
  --window_stride 96 \
  --max_eval_windows 64 \
  --log_every 1
```

同样方式用于 `weather` 和 `exchange_rate`。

## 最终实验配置

主实验关键配置如下：

| 配置项 | 值 |
|---|---:|
| context length | 32 |
| prediction length | 24 |
| lag sequence | 0-8, 23-25, 47-49, 95-97, 167-169, 671-673 |
| layers | 8 |
| heads | 9 |
| embedding per head | 16 |
| hidden size | 144 |
| scaling | robust |
| loss | Student-T NLL |
| pretrain batch size | 64 |
| pretrain max epochs | 150 |
| pretrain max train batches | 256 |
| pretrain max val batches | 256 |
| early stopping patience | 15 |
| fine-tune epochs | 20 |
| evaluation samples | 10 |
| samples30 supplement | 30 samples, 64 eval windows |

预训练结果：

| 项目 | 结果 |
|---|---:|
| stopped epoch | 121 |
| best epoch | 106 |
| early stopped | true |

## 最终结果

### Zero-shot 与 fine-tuned 主结果

主实验使用 `num_samples=10`。下表列出 Lag-Llama 与轻量 classical baseline 的核心结果。指标越低越好。

| Dataset | Setting | MAE | MSE | CRPS / mean_wQuantileLoss | sample CRPS | Best lightweight baseline |
|---|---|---:|---:|---:|---:|---|
| ETTm2 | zero-shot | 3.3172 | 23.1991 | 0.0935 | 2.5787 | moving_average, MAE 2.4137 |
| ETTm2 | fine-tuned | 1.1281 | 3.1182 | 0.0381 | 1.0659 | moving_average, MAE 2.4137 |
| exchange_rate | zero-shot | 0.0464 | 0.004707 | 0.0626 | 0.0596 | last_value, MAE 0.0158 |
| exchange_rate | fine-tuned | 0.0187 | 0.000692 | 0.0204 | 0.0190 | last_value, MAE 0.0158 |
| weather | zero-shot | 1.3155 | 16.4003 | 1.3838 | 1.7408 | moving_average, MAE 1.9219 |
| weather | fine-tuned | 1.2851 | 16.4687 | 1.3667 | 1.7855 | moving_average, MAE 1.9219 |

结论：

- ETTm2：fine-tuning 后显著优于 zero-shot、naive 和 moving average。
- exchange_rate：fine-tuning 明显优于 zero-shot，但略弱于 last-value / DynOptTheta；该数据集接近随机游走，last-value 是很强的基线。
- weather：Lag-Llama 在 MAE 上明显优于 naive 和统计基线；MSE 略弱于 AutoETS/DynOptTheta，说明可能存在少数较大误差点。

### 论文统计基线补充

使用 StatsForecast 跑论文中代表性统计基线 AutoETS 和 DynOptTheta。

| Dataset | AutoETS MAE | AutoETS MSE | DynOptTheta MAE | DynOptTheta MSE | Best by MAE |
|---|---:|---:|---:|---:|---|
| ETTm2 | 4.2697 | 33.3526 | 2.0446 | 7.8431 | DynOptTheta |
| exchange_rate | 0.01598 | 0.000519 | 0.01585 | 0.000517 | DynOptTheta |
| weather | 2.1972 | 15.1928 | 2.0544 | 15.6282 | DynOptTheta |

对比结论：

- ETTm2：fine-tuned Lag-Llama MAE 1.1281，优于 DynOptTheta MAE 2.0446。
- weather：fine-tuned Lag-Llama MAE 1.2851，优于 DynOptTheta MAE 2.0544；但 MSE 不占优。
- exchange_rate：fine-tuned Lag-Llama MAE 0.0187，略弱于 DynOptTheta MAE 0.01585。

### 深度基线补充

为避免只和简单统计方法比较，项目实现了轻量 MindSpore DeepAR 和 PatchTST 作为代表性深度基线。由于训练预算和调参远小于论文，这部分只作为补充参考，不作为严格论文级基线。

| Dataset | Best deep baseline | MAE | MSE |
|---|---|---:|---:|
| ETTm2 | DeepAR | 6.3409 | 56.0598 |
| exchange_rate | PatchTST | 0.0738 | 0.008536 |
| weather | DeepAR | 2.0877 | 15.6358 |

### samples=30 稳定性复核

该补充实验使用 fine-tuned checkpoint、`num_samples=30`、`max_eval_windows=64`。它用于验证概率预测指标稳定性，不直接替代主实验。

| Dataset | MAE | MSE | CRPS / mean_wQuantileLoss | sample CRPS |
|---|---:|---:|---:|---:|
| ETTm2 | 0.9792 | 1.8841 | 0.0322 | 0.8666 |
| exchange_rate | 0.01871 | 0.000692 | 0.0203 | 0.0180 |
| weather | 1.2851 | 16.4687 | 1.3668 | 1.7006 |

结果显示，ETTm2 和 weather 上 fine-tuned 模型保持较好性能；exchange_rate 的结论与 samples=10 一致，即 fine-tuning 显著改善 zero-shot，但仍略弱于 last-value / DynOptTheta。

## 结果文件

主要结果文件：

```text
results/train_history_paper10_pretrain.json
results/metrics_paper10_zero_shot_ettm2.json
results/metrics_paper10_finetuned_ettm2.json
results/metrics_paper10_zero_shot_exchange_rate.json
results/metrics_paper10_finetuned_exchange_rate.json
results/metrics_paper10_zero_shot_weather.json
results/metrics_paper10_finetuned_weather.json
results/metrics_paper10_finetuned_ettm2_samples30.json
results/metrics_paper10_finetuned_exchange_rate_samples30.json
results/metrics_paper10_finetuned_weather_samples30.json
results/paper_baselines_ettm2.json
results/paper_baselines_exchange_rate.json
results/paper_baselines_weather.json
results/paper_deep_baselines_ettm2.json
results/paper_deep_baselines_exchange_rate.json
results/paper_deep_baselines_weather.json
```

主要图像：

```text
results/figures/paper10_downstream_mae_comparison.png
results/figures/paper10_downstream_mse_comparison.png
results/figures/forecast_paper10_zero_shot_ettm2.png
results/figures/forecast_paper10_finetuned_ettm2.png
results/figures/forecast_paper10_zero_shot_exchange_rate.png
results/figures/forecast_paper10_finetuned_exchange_rate.png
results/figures/forecast_paper10_zero_shot_weather.png
results/figures/forecast_paper10_finetuned_weather.png
results/figures/paper_baselines.png
results/figures/paper_baselines_exchange_rate.png
results/figures/paper_baselines_weather.png
results/figures/paper_deep_baselines_ettm2.png
results/figures/paper_deep_baselines_exchange_rate.png
results/figures/paper_deep_baselines_weather.png
```

## 与论文的差距

本项目完成的是资源受限复现，不是论文完整 Table 1 的严格复现。主要差距如下：

1. 论文使用更完整的 27 数据集语料；本实验使用其中 10 个真实数据集进行预训练，3 个数据集作为下游评估。
2. 论文在更多未见数据集上报告平均排名；本实验只在 ETTm2、exchange_rate、weather 三个下游数据集上评估。
3. 论文使用更多 empirical samples 计算 CRPS；本实验主结果使用 10 samples，并补充 30 samples 稳定性复核。
4. 论文基线更多，包括 AutoARIMA、CrostonSBA、NPTS、TFT、N-BEATS、Informer、AutoFormer、ETSFormer、OneFitsAll 等；本实验选取 AutoETS、DynOptTheta、DeepAR、PatchTST 作为代表性基线。
5. DeepAR / PatchTST 为轻量 MindSpore 实现，训练预算和调参不等同于论文官方基线。
6. 受 Ascend / MindSpore 算子兼容性影响，Student-T 相关算子在不同镜像上表现不完全一致。本次最终结果 JSON 记录使用三参数 Student-T。
7. 本实验部分评估为了控制时间使用 `window_stride=96` 和 samples30 的 `max_eval_windows=64`，与论文完整评估设置不同。

因此报告中应表述为：

> 在资源受限条件下，本实验复现了 Lag-Llama 的主要技术路线和实验趋势。结果支持论文中“多数据集预训练带来跨数据集迁移能力，fine-tuning 可显著提升下游性能”的核心结论，但未达到论文完整规模和全部基线的严格复现水平。

## 最终结论

本实验结果表明：

- 多数据集预训练后的 Lag-Llama 在未见数据集上具有一定 zero-shot 能力；
- fine-tuning 后模型在 ETTm2 和 weather 上显著提升，并在 MAE 指标上优于代表性统计基线；
- exchange_rate 上 fine-tuned 模型明显优于 zero-shot，但略弱于 last-value / DynOptTheta，说明随机游走型序列对简单基线更友好；
- CRPS / mean weighted quantile loss 在 fine-tuned 后整体下降，说明概率预测质量提升；
- 结果总体符合论文核心趋势，但受数据规模、训练预算、评估窗口和基线完整性限制，不应夸大为完整论文排行榜复现。

## ModelArts 使用建议

详见 [MODELARTS.md](MODELARTS.md)。简要流程：

1. 创建 ModelArts Notebook，选择 MindSpore + Ascend 镜像；
2. 上传项目压缩包并解压；
3. 安装 `gluonts==0.14.4` 和 `statsforecast`；
4. 先运行 `bash scripts/run_smoke.sh`；
5. 再运行 `bash scripts/run_paper10_budget.sh`；
6. 下载 `results/`、Notebook、README、报告和 PPT；
7. 停止或删除 Notebook，避免继续计费。

## 提交建议

建议提交材料包括：

```text
mindspore_lag_llama/src/
mindspore_lag_llama/scripts/
mindspore_lag_llama/notebooks/
mindspore_lag_llama/README.md
mindspore_lag_llama/MODELARTS.md
mindspore_lag_llama/DATASETS.md
mindspore_lag_llama/results/*.json
mindspore_lag_llama/results/figures/
研究报告.docx 或 PDF
PPT
```

如果压缩包体积过大，checkpoint 可不全部提交；至少保留最佳 checkpoint 名称、训练历史 JSON、指标 JSON 和图像。
