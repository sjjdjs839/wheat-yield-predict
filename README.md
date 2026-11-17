# 河南省小麦气象灾害影响分析与预警平台

工作区总入口。运行项目、原始数据、辅助工具、文档和历史归档已经分层，根目录只保留必要入口。

## 快速开始

### Windows

双击 [启动小麦系统.bat](启动小麦系统.bat)。脚本按顺序自检，不需要手动做任何事：

1. 定位 Python（`python` → `py -3` → 常见安装路径），找不到会提示下载地址
2. 检查依赖，缺什么就自动 `pip install -r requirements.txt`（首次约 2—5 分钟）
3. 检查 `data/processed/weather_clean.csv` 与 `output/models/model_rf.joblib`，
   缺失则自动从 `data/raw` 重建一次（实测约 50 秒，仅首次）
4. 清理占用 5000 端口的旧服务，启动 Flask，并自动打开 `http://127.0.0.1:5000`

关闭控制台窗口即停止服务。任一步失败，窗口都会停在原地并给出可操作提示。

### macOS / Linux

```bash
chmod +x 启动小麦系统.sh && ./启动小麦系统.sh
```

### 手动启动（调试用）

```bash
# 在仓库根目录
pip install -r requirements.txt      # 含 lightgbm / reportlab / xlrd / shap

cd wheat-ml
python -m scripts.run_pipeline       # 首次需要：清洗 821850 行 -> 特征 -> 训练三模型
python run.py                        # 加 --no-browser 则不弹浏览器
```

### 换电脑为什么能直接跑

仓库里已提交原始气象 CSV（`data/raw`，41 MB）与特征表、县单产基准、指标报告；
`weather_clean.csv`（72 MB）与模型 `.joblib`（18 MB）是可重建产物，**不随仓库提交**，
由启动脚本在首次运行时自动生成。因此：

- 有 MySQL：走库表（更快，`python -m scripts.init_database` 可导入）
- 没有 MySQL：自动回退 CSV，功能完整，只是少了库查询的便利
- 没有模型/清洗数据：启动脚本自动重建，指标可复现（R² 0.9499 / 0.9552 / 0.9174，对应 RF / LGBM / LR，真源为 `wheat-ml/output/reports/metrics_*.json`）

常见问题：

| 现象 | 处理 |
|---|---|
| `pip install` 卡住或超时 | 加国内源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 提示没有 Python | 安装 3.12+，勾选 “Add Python to PATH” |
| 数据重建报错 | 把窗口报错截图发团队；多为依赖未装全 |
| 端口 5000 被占用 | 脚本会自动清理；仍失败则关闭旧控制台窗口 |

## 目录结构

```text
人工智能大赛数据集/
├─ README.md                     # 本文件
├─ 启动小麦系统.bat / .sh         # 一键启动入口（Windows / macOS·Linux）
├─ wheat-ml/                     # 唯一运行项目（Flask + 前端 + 数据 + 模型）
│  ├─ app.py
│  ├─ run.py                    # 一键启动器（健康检查 + 自动开浏览器）
│  ├─ config/
│  ├─ src/
│  ├─ scripts/                   # 正式清洗 / 特征 / 训练 / 导入流水线
│  ├─ web/                       # 模板、样式、本地图表库、麦田摄影
│  ├─ data/
│  │  ├─ raw/
│  │  │  ├─ county_yield.csv
│  │  │  ├─ county_yield_train.csv
│  │  │  ├─ weather/             # 18 个市、150 个县的原始气象 CSV
│  │  │  └─ source_yield/        # 16 个市级产量标签 CSV + 河南省总表
│  │  ├─ processed/              # feature_table.csv（入库）；weather_clean.csv 首次运行自动生成
│  │  └─ demo/
│  ├─ output/                    # models / reports / dashboard
│  └─ README.md
├─ tools/                        # 可复用的数据检查与模型实验脚本
│  ├─ checks/
│  ├─ model/
│  └─ README.md
├─ docs/                         # 项目文档、报告与验收截图
│  ├─ 项目介绍_指导老师版.md
│  ├─ 项目技术讲解_答辩用.md
│  ├─ 项目交付检查清单.md
│  ├─ 前端优化说明.md
│  ├─ reports/                   # 数据质检报告 / chk_all.pdf / chk_all.xlsx
│  └─ qa/                        # 七页四档视口验收截图
└─ archive/                      # 历史版本与一次性脚本，不参与运行
   ├─ pre_reorg_20261111/        # 整理前代码与文档备份
   ├─ old_full_project_20261109/
   ├─ old_frontend_copy_20261110/
   ├─ backup_旧前端_20261110/
   ├─ root_scripts_oneoff/
   ├─ root_scripts_superseded/
   ├─ legacy_tests/
   └─ uploads_old/
```

## 数据说明

- 运行项目自带原始气象数据：`wheat-ml/data/raw/weather/{城市}/{县}.csv`。
- 原始包共 150 个县文件，配置中的 `洛阳市/龙管会.csv` 原始文件缺失；清洗后的 `wheat-ml/data/processed/weather_clean.csv` 已包含该县，页面与模型不受影响。
- 重新运行 `wheat-ml/scripts/run_pipeline.py` 前请先备份 `wheat-ml/data/processed/`，否则新结果会因缺少上述原始文件而少一个县。
- 市级产量 CSV 与《河南省小麦产量标签_总表.csv》已迁入 `wheat-ml/data/raw/source_yield/`，用于追溯；运行项目实际读取的是 `wheat-ml/data/raw/county_yield.csv`。
- `wheat-ml/data/processed/` 与 `wheat-ml/output/models/` 是运行必需的大文件，不要随代码清理误删。

## Git 仓库说明

为控制仓库体积，以下内容已在 `.gitignore` 中排除，不随代码入库：

- `wheat-ml/data/processed/weather_clean.csv`（清洗后的逐日气象宽表，约 73MB）
- `wheat-ml/output/models/*.joblib`、`wheat-ml/output/dashboard/model.json`（训练产出的模型与缓存）
- 各类 `.zip` 压缩包、`archive/` 历史归档、`__pycache__/` 缓存

复现方式：

- 清洗宽表与特征表：运行 `wheat-ml/scripts/run_pipeline.py`，由 `data/raw/weather/` 原始 CSV 重建。
- 模型文件：运行 `wheat-ml/src/models/train_model.py` 重新训练导出。
- 若需随仓库分发大模型或大数据表，可使用 Git LFS，或在 GitHub Release 中作为附件上传。

## 工具与文档

- 数据检查：`tools/checks/`（对齐、泄露、异常值、区县清单、映射）。
- 模型实验：`tools/model/`（原始 vs 归一化、模型对比、RF 调参）。
- 前端移植与验收：`wheat-ml/前端移植说明.md`。
- 验收截图：`docs/qa/`；整理前代码备份：`archive/pre_reorg_20261111/`。

## 归档说明

`archive/` 中的旧完整项目、旧前端副本、一次性 `add_*.py`、旧 DB/清洗脚本仅用于回溯，不参与运行，也不要作为当前交付物。确认不再需要旧模型或旧数据后，可再整体删除 `archive/` 以节省空间。
