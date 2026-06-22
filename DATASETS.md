# 数据集获取与生成说明

本文档说明本项目如何获取和生成 Lag-Llama MindSpore 复现实验所需数据。

本次最终实验采用 **paper10 资源受限复现方案**：从 Lag-Llama 论文数据集列表中选择 10 个真实数据集用于预训练，并保留 3 个真实下游数据集用于 zero-shot 和 fine-tuning 评估。

为控制提交包体积，`data/` 目录默认不包含完整原始数据。数据可通过脚本重新下载和转换，最终训练与评估结果保存在 `results/` 目录。

## 1. 最终实验数据设置

### 1.1 预训练数据集

主实验使用以下 10 个真实数据集作为预训练数据：

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

其中：

- `ETTh1`、`ETTh2`、`ETTm1` 来自 ETT 原始数据仓库；
- 其余数据集通过 GluonTS public dataset repository 导出；
- 所有数据最终统一转换为包含 `date` 和 `OT` 两列的 CSV，便于 MindSpore 版本统一读取。

生成目录：

```text
data/paper10_pretrain/
```

### 1.2 下游未见数据集

以下 3 个数据集不参与预训练，用于验证 zero-shot 和 fine-tuning 能力：

```text
ETTm2
exchange_rate
weather
```

生成目录：

```text
data/paper10_downstream/
```

## 2. 生成 paper10 数据

在 ModelArts 的项目根目录下执行：

```bash
cd mindspore_lag_llama
python scripts/prepare_paper10_datasets.py
```

该脚本会自动完成：

1. 下载 ETT 系列 CSV；
2. 将 `ETTh1`、`ETTh2`、`ETTm1` 转换到 `data/paper10_pretrain/`；
3. 将 `ETTm2` 转换到 `data/paper10_downstream/`；
4. 通过 GluonTS 导出 7 个预训练数据集；
5. 通过 GluonTS 导出 `exchange_rate` 和 `weather` 两个下游数据集。

默认参数：

```text
max_series_per_dataset = 8
max_points_per_series = 20000
```

也可以手动调整：

```bash
python scripts/prepare_paper10_datasets.py \
  --max_series_per_dataset 8 \
  --max_points_per_series 20000
```

如果想减少数据规模用于调试：

```bash
python scripts/prepare_paper10_datasets.py \
  --max_series_per_dataset 2 \
  --max_points_per_series 5000
```

## 3. 依赖项

生成 paper10 数据需要：

```bash
pip install gluonts==0.14.4
```

统计基线需要：

```bash
pip install statsforecast
```

注意：`statsforecast` 可能升级 `pandas`，从而与 ModelArts 内置包产生版本警告。建议先完成主训练和评估，再安装 `statsforecast` 补跑统计基线。

## 4. ETT 数据来源

ETT 数据来自公开仓库：

```text
https://github.com/zhouhaoyi/ETDataset
```

脚本中使用的原始地址包括：

```text
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv
```

如果只想下载 ETT 系列数据，可执行：

```bash
bash scripts/download_ett.sh
```

下载后通常包含以下列：

```text
date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
```

本项目最终实验使用 `OT` 作为目标列。

## 5. GluonTS 数据来源

以下数据通过 GluonTS public dataset repository 获取：

```text
electricity_hourly
solar_10_minutes
traffic
kdd_cup_2018_without_missing
sunspot_without_missing
australian_electricity_demand
london_smart_meters_without_missing
exchange_rate
weather
```

脚本会读取每个数据集的训练 split，将前若干条 series 导出为 CSV：

```text
date, OT
```

导出逻辑位于：

```text
scripts/prepare_paper10_datasets.py
```

如果 ModelArts 无法访问外网或 GluonTS 下载失败，可先使用 ETT-only 流程完成调试，但最终报告中的主结果应以 paper10 真实数据子集为准。

## 6. 检查数据

生成数据后可检查某个 CSV：

```bash
python scripts/inspect_dataset.py \
  --dataset csv \
  --data_path data/paper10_downstream/ETTm2.csv \
  --target_column OT
```

也可以检查目录：

```bash
find data/paper10_pretrain -maxdepth 1 -name "*.csv" | wc -l
find data/paper10_downstream -maxdepth 1 -name "*.csv" | wc -l
```

## 7. 与论文完整数据设置的差异

论文完整实验使用更大规模的多数据集语料和更多未见下游数据集。本项目受课程预算和运行时间限制，采用 10 个真实数据集预训练、3 个真实数据集下游评估。

因此报告中应说明：

```text
本实验不是完整 27 数据集语料复现，而是资源受限条件下选取论文真实数据子集进行复现。
```

## 8. 官方 Google Drive 数据包说明

Lag-Llama 官方脚本曾提供一个 Google Drive 数据包：

```text
https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing
```

如果网络可访问，可尝试：

```bash
pip install gdown
gdown --id 1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE -O nonmonash_datasets.tar.gz
mkdir -p data/nonmonash
tar -xvzf nonmonash_datasets.tar.gz -C data/nonmonash
```

但在浏览器或 ModelArts 无法访问 Google Drive 时，不建议卡在该数据包上。本项目最终采用 ETT 公开仓库 + GluonTS 数据仓库的方式生成真实数据子集。

## 9. 数据不随提交包完整提供的原因

提交包中 `data/` 目录可以为空或只包含说明文件，原因是：

- 原始数据可以由脚本重新生成；
- 完整数据体积较大；
- 作业评阅主要依赖代码、Notebook、README、报告、PPT 和 `results/` 中的指标/图像；
- `results/` 已保存本次实验的训练历史、评估 JSON 和可视化结果。

最终结果文件位于：

```text
results/*.json
results/figures/
```

如需完整复跑，请先按本文档生成数据，再运行：

```bash
bash scripts/run_paper10_budget.sh
```
