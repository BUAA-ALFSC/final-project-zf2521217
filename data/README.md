# 数据目录说明

本目录在提交包中有意保持轻量，不直接包含完整原始数据集。

原因如下：

- 原始时间序列数据可以通过脚本重新下载和转换；
- 完整数据体积较大，直接提交会显著增加压缩包大小；
- 本次实验的训练、评估结果已经保存在 `../results/` 目录中。

## 生成 Paper10 主实验数据

主实验使用 Lag-Llama 论文数据集列表中的 10 个真实数据集作为预训练数据，并保留 3 个下游数据集用于 zero-shot 和 fine-tuning 评估。

在项目根目录下执行：

```bash
python scripts/prepare_paper10_datasets.py
```

执行后会生成：

```text
data/paper10_pretrain/
data/paper10_downstream/
```

预训练数据集包括：

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

下游未见数据集包括：

```text
ETTm2
exchange_rate
weather
```

## 仅生成 ETT 数据

如果只需要运行较小规模的 ETT 复现实验，可以执行：

```bash
bash scripts/download_ett.sh
```

该脚本会下载：

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
```

## 已提交的实验结果

最终训练和评估输出保存在：

```text
../results/*.json
../results/figures/
```

如果需要重新运行实验，可以根据上述脚本重新生成数据。
