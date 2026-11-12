# 河南省小麦气象数据分析与产量预测系统

> 基于 2010—2024 年河南省 151 个县（市、区）逐日气象数据 + 分县小麦产量，提供数据上传校验、
> 气象数据分析、产量地图、干热风预警、多模型融合产量预测与农事建议的可交互 Web 应用。

## 功能模块

| 模块 | 说明 |
|---|---|
| 数据上传 | 支持 CSV / Excel（.xlsx、.xls），自动识别中英文字段名、数据预览，检查缺失值、重复记录与异常值 |
| 气象数据分析 | 最高温/最低温/平均温趋势、日降水、2m 风速、地表短波辐射、露点与湿度、年月对比、综合评估 |
| 产量地图 | 按年份查看县级小麦单产与生育期降水的空间分布，支持缩放、指标切换与 Top 5 排名 |
| 干热风预警 | 按城市展开干热风天数、等级与产量影响预估，支持按年查看 |
| 产量预测与预警 | 随机森林 + LightGBM + 线性回归融合预测单产，输出 MAE 近似区间、历史实测对比、自定义预警线、情景模拟与 Excel / PDF 报告 |
| 农事建议 | 结合气象条件、小麦生育期（返青/拔节/抽穗扬花/灌浆/成熟）与产量预测生成建议 |

## 前端界面

界面采用「麦浪智域 · 现代生态农业」设计语言：真实麦田摄影、森林绿与麦穗金、米白纸感，
固定六项主导航：概览 / 气象分析 / 产量地图 / 干热风预警 / 产量预测 / 农事建议，上传数据为页头按钮。

- 首页实景麦田首屏 + 真实数据带 + 三模块工作流 + 空间与灾害工具 + 航拍数据底座 + 深绿模型状态区
- 气象分析页 7 组图表 / 11 个 canvas，使用本地 Chart.js 渲染；产量地图使用本地 ECharts
- 预测页包含融合模型结果、预警标尺、MAE 近似区间、趋势图、特征重要性、情景模拟与报告导出
- 图片全部本地化（WebP + JPG 回退），首屏预加载，其余懒加载；断网可用
- 适配 1440 / 1024 / 768 / 390 四档视口，验收零横向溢出、零控制台错误
- 详细改动与验收记录见 [前端移植说明.md](前端移植说明.md)

## 快速开始

```bash
# 1. 安装依赖（Python 3.12+）
pip install -r requirements.txt

# 2. 一键跑数据处理与训练（首次运行或数据更新后）
python -m scripts.run_pipeline

# 3.（可选）启动本机 MySQL 并导入数据；不启动时自动回退 CSV
# powershell -ExecutionPolicy Bypass -File scripts\mysql_server.ps1 start
# python -m scripts.init_database   # 建库建表 + 全量导入

# 4. 启动 Web 应用（服务就绪后自动打开默认浏览器）
python run.py
# 也可只启动服务、不打开浏览器：
# python run.py --no-browser
# 关闭控制台窗口即停止服务
```

### 分步运行

```bash
python -m src.data.clean            # 清洗气象包 → data/processed/weather_clean.csv
python -m src.data.build_features   # 18 个生育期特征 → data/processed/feature_table.csv
python -m src.models.train_model    # 按年份划分训练/测试集，输出模型与指标
python -m src.models.export_model   # 导出浏览器端模型 output/dashboard/model.json
python -m scripts.init_database     # CSV 成果导入 MySQL
```

## 数据库（MySQL，可选）

项目默认使用 CSV 数据链路，不依赖 MySQL；数据库未启动时页面功能不受影响。

如本机已有 MySQL 8，可在 `data/mysql/mysql-8.0.24-winx64` 放置安装目录，并通过 `scripts/mysql_server.ps1` 启动；当前仓库不内置 MySQL 二进制。

- 连接配置：[config.yaml](config/config.yaml) 的 `database` 段，默认 `127.0.0.1:3306`、库名 `wheat_ml`

```bash
powershell -ExecutionPolicy Bypass -File scripts\mysql_server.ps1 start   # 启动
powershell -ExecutionPolicy Bypass -File scripts\mysql_server.ps1 status  # 状态
powershell -ExecutionPolicy Bypass -File scripts\mysql_server.ps1 stop    # 停止
```

表结构（`src/db/schema.py` 定义，主键 / 唯一键 / 外键 / 索引齐全）：

| 表 | 内容 | 行数 |
|---|---|---|
| counties | 151 个县（市、区）及所属市、区域类型 | 151 |
| weather_daily | 逐日气象（8 参数，按县 + 日期唯一） | 821,850 |
| feature_table | 年度 18 个生育期特征（按县 + 年份唯一） | 2,250 |
| yield_labels | 建模用分县分年产量标签（`source=real/demo`） | 1,635 |

导入方式：`python -m scripts.init_database` 建库建表并全量写入；采用分批 `executemany` 批量插入，
不依赖 `LOAD DATA LOCAL INFILE`，无需开启 `local_infile`，也不受 `secure_file_priv` 目录限制。

读取顺序：**上传数据 → MySQL → 内置 CSV**。应用优先查询 MySQL（单县气象、产量标签、年度特征、库内统计），
数据库未启动、查询失败或表为空时自动回退内置 CSV，页面功能不受影响。

## 数据要求

### 气象数据（data/raw/weather/{城市}/{县}.csv）

数据集为中文表头、年份 + 年内日序格式，每行一天：

```
年份,日,最高温,最低温,平均温,日降水量,2m风速,地表短波辐射,露点温度,相对湿度
```

同时兼容 NASA POWER 英文表头（前 16 行注释头，第 17 行列名）：
`T2M_MAX / T2M_MIN / T2M / PRECTOTCORR / WS2M / ALLSKY_SFC_SW_DWN / T2MDEW / RH2M`。

> 注意：当前原始包缺 `洛阳市/龙管会.csv`，而 `data/processed/weather_clean.csv` 已包含该县。重新运行 `scripts/run_pipeline.py` 前请先备份 `data/processed/`，否则新结果会少一个县。

校验规则：
- 「日」字段使用 `YYYY-MM-DD` 或年内日序（1—366），两种口径不可混用
- 缺失值明确标注（空 / NA / -999），不会被当作 0
- 最高温 ≥ 最低温、相对湿度 0—100、降水非负
- 短波辐射以中位数判断单位，W/m² 自动换算为 MJ/m²/day
- 保留闰年 2 月 29 日记录

### 产量数据（data/raw/county_yield.csv）

| 字段 | 说明 | 是否必需 |
|---|---|---|
| `county` | 县/市/区名称（可带「县」「市」后缀） | 强烈建议 |
| `year` | 小麦收获年份 | 必需 |
| `wheat_yield` | 小麦单位面积产量（统一为 kg/亩） | 必需 |

真实产量缺失时，系统可读取 `data/demo/county_yield_demo.csv` 的**演示标签**跑通预测链路，
此时界面与指标均明确标注「演示数据」，R² / MAE / RMSE 不构成任何准确率结论。

## 建模口径

| 项 | 规则 |
|---|---|
| 目标变量 | `yield_norm = 单产 ÷ 县域多年均值`，预测后还原真实单产再算指标 |
| 划分方式 | **按年份划分**（`config.yaml → model.split.test_years`，默认 2022—2024），禁止随机拆分 |
| 特征 | 按生育阶段汇总：阶段均温、累计降水、累计辐射、高温日数、霜冻日数、干热风日数等，共 18 个 |
| 模型 | 多模型融合：随机森林 50% + LightGBM 30% + 线性回归 20% |
| 指标 | R² / RMSE / MAE，仅在具备真实产量标签的测试年份上计算 |

## 目录结构

```
wheat-ml/
├─ app.py                     # Flask Web 应用（七页路由）
├─ run.py                     # 一键启动器（健康检查 + 自动打开默认浏览器）
├─ config/config.yaml         # 路径、县白名单、生育期、校验规则、模型与预警配置
├─ data/
│  ├─ raw/weather/            # 150 个县的原始气象文件（18 个市；洛阳市/龙管会原始包缺失）
│  ├─ raw/source_yield/       # 16 个市级产量标签 CSV + 河南省小麦产量标签_总表.csv
│  ├─ demo/                   # 演示产量标签（界面强制标注）
│  └─ processed/              # weather_clean.csv, feature_table.csv
├─ src/
│  ├─ data/clean.py           # NASA POWER 清洗
│  ├─ data/build_features.py  # 18 个生育期特征
│  ├─ data/validate.py        # 上传文件校验（CSV/Excel、字段映射、异常检查）
│  ├─ analysis/weather_charts.py  # 离线 PNG 图表（导出用）
│  ├─ analysis/web_charts.py      # 分析页 Chart.js 图表数据层
│  ├─ advice/agronomy.py      # 按生育期生成农事建议
│  └─ models/                 # 训练与模型导出
├─ web/
│  ├─ templates/              # 概览 / 分析 / 地图 / 干热风 / 预测 / 建议 / 上传
│  ├─ static/style.css        # 麦浪智域设计系统
│  ├─ static/app.js           # 图表渲染、动效、上传校验
│  ├─ static/assets/field/    # 麦田摄影（WebP + JPG）
│  ├─ static/vendor/          # Chart.js 4.4.7 + ECharts（本地）
│  └─ uploads/                # 上传文件暂存
├─ output/
│  ├─ models/                 # model_rf.joblib, model_lr.joblib, county_stats.json, model_meta.json
│  ├─ reports/                # metrics_rf.json, importance_rf.csv, predictions_rf.csv
│  └─ dashboard/              # 单文件离线大屏 predict_dashboard.html
└─ scripts/run_pipeline.py    # 一键流水线
```

## 离线大屏（附加）

`output/dashboard/predict_dashboard.html` 为内嵌模型树的单文件页面，无需服务器与网络，
双击即可进行情景模拟演示。

## 参考案例

| 县 | 年份 | 类型 | 说明 |
|---|---|---|---|
| 兰考 | 2011 | 春旱 | 3-4 月降水约 16 mm（常年约 40 mm） |
| 尉氏 | 2012 | 干热风 | 5-6 月干热风约 10 天 |
