# tools · 辅助分析与检查脚本

这些脚本是项目开发过程中的辅助工具，**不参与 Web 运行**。正式的数据清洗、特征构建、训练与导出流程仍以 `wheat-ml/scripts/run_pipeline.py` 和 `wheat-ml/src/` 为准。

脚本内统一通过 `Path(__file__).resolve().parents[2] / "wheat-ml"` 定位运行项目，因此可以在任意目录执行。

## checks · 数据与对齐检查

| 脚本 | 用途 |
|---|---|
| `check_alignment_v2.py` | 按市级目录扫描原始气象文件，与 `source_yield/河南省小麦产量标签_总表.csv` 做区县对齐检查 |
| `check_leakage.py` | 对比“全量县均值”和“仅训练集县均值”两种口径，检查数据泄露影响 |
| `check_outliers.py` | 检查单产异常值（>700 或 <150 kg/亩）与波动最大的县 |
| `list_counties.py` | 列出产量标签中的区县、记录数、均值与标准差 |
| `show_mapping.py` | 打印原始气象文件名 → 标准区县名的映射与匹配状态 |
| `county_mapping.py` | 上述脚本共用的区县映射与跳过规则 |

## model · 模型实验

| 脚本 | 用途 |
|---|---|
| `validate_model.py` | 原始单产 vs 归一化单产对比，并按年份做滚动交叉验证 |
| `compare_models.py` | 线性回归 / 随机森林 / LightGBM 对比；安装 `xgboost` 时自动加入 XGBoost |
| `tune_rf_params.py` | 随机森林参数组合试验 |

## 运行示例

```bash
cd 人工智能大赛数据集
python tools/checks/check_alignment_v2.py
python tools/checks/check_outliers.py
python tools/model/validate_model.py
```

## 注意

- 这些脚本读取 `wheat-ml/data/processed/feature_table.csv`、`wheat-ml/data/raw/county_yield.csv` 等运行数据，不做写入。
- `compare_models.py` 的 XGBoost 为可选项，未安装时会自动跳过；主项目依赖不要求 XGBoost。
- 历史一次性脚本（`add_*.py`、DB 初始化、旧清洗脚本等）已归档到 `archive/root_scripts_oneoff/` 与 `archive/root_scripts_superseded/`。
