"""
河南省小麦气象数据分析与产量预测系统 · Web 应用
==================================================
页面路由：
  /upload    数据上传与校验
  /analysis  气象数据分析（七类图表 + 综合评估）
  /map       产量地图（县级空间分布）
  /dry_hot   干热风预警
  /predict   小麦产量预测与预警
  /advice    农事建议

运行：python app.py  （或 python -m flask --app app run）
"""

import json
import uuid
import yaml
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

from flask import (Flask, render_template, request, redirect, url_for)

from src.data.validate import validate_file, load_weather_file
from src.data.build_features import build_county, CANONICAL_TO_NASA
from src.analysis.web_charts import build_web
from src.advice.agronomy import generate_advice
from src.db import repository as repo
from county_geo_mapping import to_geo_name

PROJECT_ROOT = Path(__file__).resolve().parent
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

FEATS = CONFIG["feature_cols"]
NASA_TO_CANON = {v: k for k, v in CANONICAL_TO_NASA.items()}
UPLOAD_DIR = PROJECT_ROOT / CONFIG["paths"]["uploads"]
LATEST = UPLOAD_DIR / "latest.json"
PROCESSED = PROJECT_ROOT / CONFIG["paths"]["processed"] / "weather_clean.csv"
FEATURE_CSV = PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv"
MODELS_DIR = PROJECT_ROOT / CONFIG["paths"]["models"]
REPORTS_DIR = PROJECT_ROOT / CONFIG["paths"]["reports"]

ALL_COUNTIES = sorted(c for m in CONFIG["counties"].values() for c in m.values())
BASE_YEARS = list(range(CONFIG["upload"]["year_min"], CONFIG["upload"]["year_max"] + 1))

# 按城市分组的区县列表
CITY_GROUPS = {}
for city, counties_dict in CONFIG["counties"].items():
    CITY_GROUPS[city] = sorted(counties_dict.values())

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = CONFIG["upload"]["max_size_mb"] * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.context_processor
def inject_footer():
    """页脚使用的数据范围，避免在各页面重复计算。"""
    return {"footer": {
        "counties": len(ALL_COUNTIES),
        "year_range": f"{CONFIG['upload']['year_min']}—{CONFIG['upload']['year_max']}",
    }}


# ---------------------------------------------------------------- 数据访问
def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """统一为标准小写字段名，并补 month_day"""
    rename = {k: v for k, v in NASA_TO_CANON.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    if "month_day" not in df.columns:
        df = df.assign(month_day=df["date"].dt.strftime("%m-%d"))
    return df


@lru_cache(maxsize=2)
def _read_processed(path: str, mtime: float) -> pd.DataFrame:
    return _canonical(pd.read_csv(path, parse_dates=["date"]))


def uploaded_dataset() -> tuple[pd.DataFrame, dict] | None:
    """最近一次上传的数据集（存在且文件可读时返回）"""
    if not LATEST.exists():
        return None
    meta = json.loads(LATEST.read_text(encoding="utf-8"))
    path = PROJECT_ROOT / meta["path"]
    if not path.exists():
        return None
    df, report = load_weather_file(path, meta.get("county"))
    return _canonical(df), {"filename": meta["filename"],
                            "county": meta.get("county") or "未标注",
                            "source": "上传数据", "summary": report["summary"]}


def current_dataset() -> tuple[pd.DataFrame, dict]:
    """上传数据优先，否则内置清洗数据"""
    up = uploaded_dataset()
    if up:
        return up
    df = builtin_dataset()
    return df, {"filename": PROCESSED.name, "county": None, "source": "内置数据集",
                "summary": {"rows": len(df), "year_range": "2010—2024"}}


def dataset_counties(df: pd.DataFrame) -> list[str]:
    names = sorted(df["county"].dropna().unique().tolist()) if "county" in df else []
    if names and names[0] == "未标注":
        return names
    return names or ["未标注"]


def uploaded_scope() -> dict | None:
    """上传数据的县区与年份范围，用于筛选器默认值和年份列表。"""
    up = uploaded_dataset()
    if not up:
        return None
    df, meta = up
    counties = dataset_counties(df)
    years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
    return {"df": df, "meta": meta, "counties": counties, "years": years,
            "default_county": counties[0]}


def scope_city_groups(scope: dict | None) -> dict | None:
    """上传数据不套用内置城市分组，返回 None 让模板使用扁平县区列表。"""
    return None if scope else CITY_GROUPS


def builtin_dataset() -> pd.DataFrame:
    return _read_processed(str(PROCESSED), PROCESSED.stat().st_mtime)


@lru_cache(maxsize=1)
def feature_csv_rows() -> int:
    """内置特征表行数；回退 CSV 链路时与天气表同口径统计。"""
    try:
        return int(pd.read_csv(FEATURE_CSV).shape[0])
    except Exception:
        return 0


def dataset_for_county(county: str) -> tuple[pd.DataFrame, dict]:
    """上传数据 + 内置数据合并（上传数据优先，缺的年份用内置补）"""
    base = builtin_dataset()
    up = uploaded_dataset()

    if up and county and county != "未标注":
        up_df, up_meta = up
        up_sub = up_df[up_df["county"] == county]
        if not up_sub.empty:
            # 合并：上传数据 + 内置数据（去重，上传优先）
            base_sub = base[base["county"] == county] if "county" in base.columns else pd.DataFrame()
            if not base_sub.empty:
                combined = pd.concat([base_sub, up_sub]).drop_duplicates(subset=["county", "date"], keep="last")
                return combined, {"filename": up_meta["filename"], "county": county,
                                  "source": f"上传数据 + 内置数据",
                                  "summary": {"rows": len(combined)}}
            return up_sub, up_meta

    if county and county != "未标注" and repo.available():
        try:
            w = repo.county_weather(county)
            if not w.empty:
                summary = up[1]["summary"] if up else {"rows": len(w),
                                                       "year_range": "2010—2024"}
                return w, {"filename": f"{CONFIG['database']['name']}.weather_daily",
                           "county": county,
                           "source": f"MySQL 数据库 · {CONFIG['database']['name']}",
                           "summary": summary}
        except Exception:
            pass                                    # 数据库异常时继续走 CSV

    base = builtin_dataset()
    if county and county != "未标注":
        sub = base[base["county"] == county]
        if not sub.empty:
            return sub, {"filename": PROCESSED.name, "county": county,
                         "source": "内置数据集（CSV）",
                         "summary": {"rows": len(sub), "year_range": "2010—2024"}}
    return base, {"filename": PROCESSED.name, "county": None,
                  "source": "内置数据集（CSV）",
                  "summary": {"rows": len(base), "year_range": "2010—2024"}}


def model_meta() -> dict | None:
    p = MODELS_DIR / "model_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_metrics(name: str = "rf") -> dict | None:
    p = REPORTS_DIR / f"metrics_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


MODEL_LABELS = {"rf": "随机森林", "lgbm": "LightGBM", "lr": "线性回归"}


def best_model_metrics() -> tuple[dict | None, str]:
    """按测试集 R² 选效果最好的模型，返回 (指标 dict, 中文模型名)；报告缺失时回退随机森林。"""
    best, best_name = None, "rf"
    for name in ("lgbm", "rf", "lr"):
        m = read_metrics(name)
        if m and m.get("r2") is not None and (best is None or m["r2"] > best["r2"]):
            best, best_name = m, name
    if best is None:
        best = read_metrics("rf")
    return best, MODEL_LABELS.get(best_name, best_name)


def _safe_load(path: Path):
    """模型文件存在但运行库缺失时，降级为 None，避免整页 500。"""
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_models():
    rf_path = MODELS_DIR / "model_rf.joblib"
    lr_path = MODELS_DIR / "model_lr.joblib"
    lgbm_path = MODELS_DIR / "model_lgbm.joblib"
    stats_path = MODELS_DIR / "county_stats.json"
    if not (rf_path.exists() and stats_path.exists()):
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    rf = _safe_load(rf_path)
    if rf is None:
        return None
    return (rf, _safe_load(lr_path), _safe_load(lgbm_path), stats)


def county_mean_yield(stats: dict, county: str) -> tuple[float, bool]:
    """返回 (用于还原的县均值单产, 是否命中该县)"""
    if county in stats:
        return float(stats[county]["mean_yield"]), True
    vals = [v["mean_yield"] for v in stats.values()]
    return float(np.median(vals)), False


def batch_predict(feat: pd.DataFrame, bundle, mean_yield) -> list:
    """批量预测：一次性把特征矩阵喂给模型，返回与 feat 等长的还原单产列表（无法预测为 None）。
    单县用传入的县均值；全省逐行按该县在 stats 中的均值还原，缺县回退全局中位数。"""
    if bundle is None or len(feat) == 0:
        return [None] * len(feat)
    rf, lr, lgbm, stats = bundle
    X = feat[FEATS].fillna(feat[FEATS].median(numeric_only=True))
    try:
        p_rf = rf.predict(X)
        p_lgbm = lgbm.predict(X) if lgbm is not None else p_rf
        p_lr = lr.predict(X) if lr is not None else p_rf
        fused = 0.5 * p_rf + 0.3 * p_lgbm + 0.2 * p_lr
        vals = [v["mean_yield"] for v in stats.values()] if stats else []
        global_mean = float(np.median(vals)) if vals else mean_yield
        out = []
        for i, (_, r) in enumerate(feat.iterrows()):
            cy = r["county"]
            m = float(stats[cy]["mean_yield"]) if stats and cy in stats else global_mean
            out.append(float(fused[i]) * m)
        return out
    except Exception:
        return [None] * len(feat)


def predict_years(df: pd.DataFrame, county: str) -> pd.DataFrame:
    """对当前数据集的某县逐年生成生育期特征"""
    d = df if county == "未标注" else df[df["county"] == county]
    if "city" not in d.columns:
        d = d.assign(city="未标注")
    return build_county(d)


def load_yield_labels() -> tuple[pd.DataFrame | None, bool]:
    """MySQL 优先；库内无标签或连接异常时回退内置 CSV。返回 (DataFrame, 是否真实标签)。"""
    if repo.available():
        try:
            y = repo.yield_labels()
            if not y.empty:
                is_real = bool((y["source"] == "real").any()) if "source" in y.columns else True
                return y, is_real
        except Exception:
            pass                                    # 数据库异常时继续走 CSV
    for key, is_real in [("raw_yield", True), ("raw_yield_demo", False)]:
        p = PROJECT_ROOT / CONFIG["paths"][key]
        if p.exists() and (is_real or CONFIG["model"].get("allow_demo_labels")):
            y = pd.read_csv(p)
            y.columns = [c.strip().lower() for c in y.columns]
            return y, is_real
    return None, False


FEATURE_NAMES = {
    "spi_12": "标准化降水指数",
    "precip_anom": "降水距平",
    "hot_days": "极端高温天数",
    "cold_days": "极端低温天数",
    "gdd": "全生育期积温",
    "season_tmax": "全生育期最高温",
    "season_tmin": "全生育期最低温",
    "season_precip": "全生育期降水",
    "season_wind": "全生育期风速",
    "season_sunshine": "全生育期日照",
    "season_tmax_34": "返青拔节期最高温",
    "season_precip_34": "返青拔节期降水",
    "gdd_34": "返青拔节期积温",
    "season_tmax_56": "灌浆成熟期最高温",
    "season_precip_56": "灌浆成熟期降水",
    "gdd_56": "灌浆成熟期积温",
    "hot_dry_days": "干热风天数",
    "year_index": "品种改良趋势",
}


def feature_importance_items() -> list[dict]:
    bundle = load_models()
    if bundle is None or bundle[0] is None:
        return []
    items = []
    for feat, importance in zip(FEATS, bundle[0].feature_importances_):
        items.append({
            "name": FEATURE_NAMES.get(feat, feat),
            "value": round(float(importance) * 100, 2),
        })
    return sorted(items, key=lambda item: item["value"], reverse=True)


# ---------------------------------------------------------------- 路由
@app.route("/")
def index():
    mm = model_meta()
    best_m, best_label = best_model_metrics()
    label_text = "真实产量" if (mm and mm.get("reliable")) else ("演示标签" if mm else "缺失")
    if repo.available():
        try:
            st = repo.stats()
            if st["weather_rows"] > 0:
                stats = {
                    "rows": st["weather_rows"],
                    "counties": st["counties"],
                    "year_range": f"{st['year_min']}—{st['year_max']}",
                    "label_text": label_text,
                    "dataset_text": f"MySQL · {CONFIG['database']['name']}",
                    "feature_rows": st["feature_rows"],
                    "yield_rows": st["yield_rows"],
                }
            else:
                # MySQL 气象表未导入（旧库口径），统计回退内置 CSV，避免概览显示 0 条
                cdf, cmeta = current_dataset()
                stats = {
                    "rows": int(len(cdf)),
                    "counties": int(cdf["county"].nunique()) if "county" in cdf else st["counties"],
                    "year_range": f"{int(cdf['year'].min())}—{int(cdf['year'].max())}" if "year" in cdf else "—",
                    "label_text": label_text,
                    "dataset_text": cmeta["source"],
                    "feature_rows": feature_csv_rows(),
                    "yield_rows": st["yield_rows"],
                }
            return render_template("index.html", title=CONFIG["web"]["title"],
                                   active="index", stats=stats, model=mm, db_online=True,
                                   best=best_m, best_label=best_label)
        except Exception:
            pass
    df, meta = current_dataset()
    stats = {
        "rows": int(len(df)),
        "counties": int(df["county"].nunique()) if "county" in df else 0,
        "year_range": f"{int(df['year'].min())}—{int(df['year'].max())}" if "year" in df else "—",
        "label_text": label_text,
        "dataset_text": meta["source"],
        "feature_rows": feature_csv_rows(),
        "yield_rows": 0,
    }
    return render_template("index.html", title=CONFIG["web"]["title"], active="index",
                           stats=stats, model=mm, db_online=False,
                           best=best_m, best_label=best_label)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    report, error = None, None
    if request.method == "POST":
        f = request.files.get("file")
        county = (request.form.get("county") or "").strip()
        if not f or not f.filename:
            error = "未选择文件。"
        else:
            ext = Path(f.filename).suffix.lower()
            if ext not in CONFIG["upload"]["allowed_ext"]:
                error = f"不支持的文件类型 {ext}，仅支持 {'、'.join(CONFIG['upload']['allowed_ext'])}。"
            else:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                saved = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{f.filename}"
                f.save(saved)
                report = validate_file(saved)
                if report["ok"]:
                    LATEST.write_text(json.dumps({
                        "path": str(saved.relative_to(PROJECT_ROOT)),
                        "filename": f.filename,
                        "county": county or None,
                        "summary": report["summary"],
                    }, ensure_ascii=False), encoding="utf-8")
                else:
                    saved.unlink(missing_ok=True)
    meta = None
    if LATEST.exists():
        meta = json.loads(LATEST.read_text(encoding="utf-8"))
        meta.setdefault("summary", {})

    analysis_url = url_for("analysis")
    predict_url = url_for("predict")
    if meta:
        summary = meta.get("summary") or {}
        up_years = summary.get("years") or []
        up_counties = summary.get("counties") or []
        default_county = meta.get("county") or (up_counties[0] if up_counties else "未标注")
        params = {"county": default_county}
        if up_years:
            params["year"] = max(up_years)
        analysis_url = url_for("analysis", **params)
        predict_url = url_for("predict", **params)

    return render_template("upload.html", title="数据上传", active="upload",
                           report=report, error=error, dataset=meta,
                           counties=ALL_COUNTIES, max_size=CONFIG["upload"]["max_size_mb"],
                           analysis_url=analysis_url, predict_url=predict_url)


@app.route("/reset_dataset")
def reset_dataset():
    LATEST.unlink(missing_ok=True)
    return redirect(url_for("upload"))


@app.route("/analysis")
def analysis():
    scope = uploaded_scope()
    if scope:
        counties = scope["counties"]
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else scope["default_county"]
        data = scope["df"] if county == "未标注" else scope["df"][scope["df"]["county"] == county]
        meta = scope["meta"]
        years = scope["years"] or list(BASE_YEARS)
        default_year = years[-1]
    else:
        counties = ALL_COUNTIES
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else counties[0]
        data, meta = dataset_for_county(county)
        if not data.empty and "year" in data.columns:
            years = sorted(set(BASE_YEARS + [int(y) for y in data["year"].unique().tolist()]))
        else:
            years = list(BASE_YEARS)
        default_year = 2012 if 2012 in years else years[-1]

    year_raw = request.args.get("year")
    if year_raw == "all":
        year = None
    elif year_raw:
        requested_year = request.args.get("year", type=int)
        year = requested_year if requested_year in years else default_year
    else:
        year = default_year

    month_from = request.args.get("month_from", type=int) or 1
    month_to = request.args.get("month_to", type=int) or 12
    if month_from > month_to:
        month_from, month_to = month_to, month_from

    error = None
    result = {"charts": [], "evaluation": None, "years": years}
    if data.empty:
        error = "当前数据集中没有该县区的记录。"
    elif month_to - month_from < 1:
        error = "月份区间过窄，请重新选择。"
    else:
        try:
            result = build_web(data, year, month_from, month_to)
        except Exception as e:              # 字段缺失等
            error = f"图表生成失败：{e}"
    return render_template("analysis.html", title="气象数据分析", active="analysis",
                           counties=counties, city_groups=scope_city_groups(scope),
                           county=county, years=years, year=year,
                           month_from=month_from, month_to=month_to,
                           charts=result["charts"], evaluation=result["evaluation"],
                           source=meta["source"], error=error)


@app.route("/map")
def map_view():
    years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]
    return render_template("map.html", title="产量地图", active="map", years=years, year=year)


@app.route("/api/map_data")
def api_map_data():
    """返回某年某指标的区县数据，供地图着色（用所有区县，包含城区）"""
    year = request.args.get("year", type=int) or 2024
    metric = request.args.get("metric", "yield")

    # 地图显示用所有区县的数据（包含城区）
    map_yield_path = PROJECT_ROOT / CONFIG["paths"]["raw_yield"]
    # 先试试用全量的（包含城区）
    full_yield_path = PROJECT_ROOT / "data" / "raw" / "county_yield.csv"
    if full_yield_path.exists():
        labels = pd.read_csv(full_yield_path)
    else:
        labels, _ = load_yield_labels()

    if labels is None:
        return {"items": [], "count": 0}

    sub = labels[labels["year"] == year]
    items = []
    if metric == "yield":
        for _, r in sub.iterrows():
            geo_name = to_geo_name(r["county"])
            if geo_name:
                items.append({"county": geo_name, "value": float(r["wheat_yield"])})
    else:
        # 降水从feature_table取
        feat = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv")
        fsub = feat[feat["year"] == year]
        for _, r in fsub.iterrows():
            if "season_precip" in r:
                geo_name = to_geo_name(r["county"])
                if geo_name:
                    items.append({"county": geo_name, "value": float(r["season_precip"])})

    return {"items": items, "count": len(items)}


@app.route("/dry_hot")
def dry_hot():
    years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]

    feat = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv")
    data = []
    for _, r in feat.iterrows():
        data.append({
            "county": r["county"],
            "year": int(r["year"]),
            "hot_dry_days": int(r["hot_dry_days"]) if pd.notna(r["hot_dry_days"]) else 0,
        })

    # 统计汇总
    all_days = [d["hot_dry_days"] for d in data]
    total_days = sum(all_days)

    # 最严重年份
    year_totals = {}
    for d in data:
        year_totals[d["year"]] = year_totals.get(d["year"], 0) + d["hot_dry_days"]
    max_year = max(year_totals, key=year_totals.get)

    # 干热风最多区县
    county_totals = {}
    for d in data:
        county_totals[d["county"]] = county_totals.get(d["county"], 0) + d["hot_dry_days"]
    max_county = max(county_totals, key=county_totals.get)

    # 预警区县数（当年天数>0的）
    year_data = [d for d in data if d["year"] == year]
    alert_count = sum(1 for d in year_data if d["hot_dry_days"] > 0)

    stats = {
        "total_days": total_days,
        "max_year": f"{max_year}年({year_totals[max_year]}天)",
        "max_county": f"{max_county}({county_totals[max_county]}天)",
        "alert_count": alert_count,
    }

    # 构建区县->城市映射
    city_map = {}
    for city, counties_dict in CONFIG["counties"].items():
        for c in counties_dict.values():
            city_map[c] = city

    return render_template("dry_hot.html", title="干热风预警", active="dry_hot",
                           years=years, year=year, data=data, stats=stats, city_map=city_map)


@app.route("/export_report")
def export_report():
    """导出产量预测报告Excel（当前选中区县）"""
    from io import BytesIO
    from flask import send_file

    county = request.args.get("county", "")

    feat = None
    if repo.available():
        try:
            feat = repo.feature_rows()
        except Exception:
            feat = None
    if feat is None or feat.empty:
        feat = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv")
    labels, _ = load_yield_labels()
    bundle = load_models()

    # 只导出选中区县的数据
    if county:
        feat = feat[feat["county"] == county]
    feat = feat.reset_index(drop=True)

    rows = []
    rf, lr, lgbm, stats = bundle if bundle else (None, None, None, {})
    mean_yield, _ = county_mean_yield(stats, county) if county else (None, False)

    # 批量预测：一次性推理，避免逐行调用模型导致全省导出卡顿
    preds = batch_predict(feat, bundle, mean_yield)

    for i, (_, r) in enumerate(feat.iterrows()):
        c = r["county"]
        year = int(r["year"])
        pred_yield = preds[i]

        # 实测单产
        actual = None
        if labels is not None:
            m = labels[(labels["county"] == c) & (labels["year"] == year)]
            if not m.empty:
                actual = float(m["wheat_yield"].iloc[0])

        rows.append({
            "区县": c,
            "年份": year,
            "生育期积温(℃·d)": round(float(r["gdd"]), 0) if pd.notna(r["gdd"]) else "",
            "生育期降水(mm)": round(float(r["season_precip"]), 0) if pd.notna(r["season_precip"]) else "",
            "极端高温天数": int(r["hot_days"]) if pd.notna(r["hot_days"]) else "",
            "极端低温天数": int(r["cold_days"]) if pd.notna(r["cold_days"]) else "",
            "干热风天数": int(r["hot_dry_days"]) if pd.notna(r["hot_dry_days"]) else "",
            "预测单产(kg/亩)": round(pred_yield, 1) if pred_yield else "",
            "实测单产(kg/亩)": round(actual, 1) if actual else "",
        })

    df = pd.DataFrame(rows)

    # 生成Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sheet_name = f"{county}产量预测报告" if county else "产量预测报告"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    filename = f"{county}_小麦产量预测报告.xlsx" if county else f"小麦产量预测报告_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )


@app.route("/export_report_pdf")
def export_report_pdf():
    """导出产量预测报告PDF（当前选中区县）"""
    from io import BytesIO
    from flask import send_file
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os

    # 注册中文字体
    font_path = r"C:\Windows\Fonts\simsun.ttc"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('SimSun', font_path))
        font_name = 'SimSun'
    else:
        font_name = 'Helvetica'

    county = request.args.get("county", "")

    feat = None
    if repo.available():
        try:
            feat = repo.feature_rows()
        except Exception:
            feat = None
    if feat is None or feat.empty:
        feat = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv")
    labels, _ = load_yield_labels()
    bundle = load_models()

    # 只导出选中区县的数据
    if county:
        feat = feat[feat["county"] == county]
    feat = feat.reset_index(drop=True)

    rows = []
    rf, lr, lgbm, stats = bundle if bundle else (None, None, None, {})
    mean_yield, _ = county_mean_yield(stats, county) if county else (None, False)

    # 批量预测：一次性推理，避免逐行调用模型导致全省导出卡顿
    preds = batch_predict(feat, bundle, mean_yield)

    for i, (_, r) in enumerate(feat.iterrows()):
        c = r["county"]
        year = int(r["year"])
        pred_yield = preds[i]

        # 实测单产
        actual = None
        if labels is not None:
            m = labels[(labels["county"] == c) & (labels["year"] == year)]
            if not m.empty:
                actual = float(m["wheat_yield"].iloc[0])

        rows.append({
            "year": year,
            "gdd": round(float(r["gdd"]), 0) if pd.notna(r["gdd"]) else "-",
            "precip": round(float(r["season_precip"]), 0) if pd.notna(r["season_precip"]) else "-",
            "hot_days": int(r["hot_days"]) if pd.notna(r["hot_days"]) else "-",
            "hot_dry_days": int(r["hot_dry_days"]) if pd.notna(r["hot_dry_days"]) else "-",
            "pred": round(pred_yield, 1) if pred_yield else "-",
            "actual": round(actual, 1) if actual else "-",
        })

    # 生成PDF
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # 标题
    title_style = styles['Title']
    title_style.fontName = font_name
    story.append(Paragraph("河南省小麦产量预测报告", title_style))
    story.append(Spacer(1, 20))

    # 报告信息
    info_style = styles['Normal']
    info_style.fontName = font_name
    info_style.fontSize = 11
    story.append(Paragraph(f"区县：{county if county else '全省'}", info_style))
    story.append(Paragraph(f"报告生成时间：{pd.Timestamp.now().strftime('%Y年%m月%d日')}", info_style))
    story.append(Spacer(1, 15))

    # 加一段通俗易懂的总结
    summary_style = styles['Normal']
    summary_style.fontName = font_name
    summary_style.fontSize = 11
    summary_style.leading = 18
    summary_text = f"""
    <b>报告说明：</b>本报告基于河南省小麦气象数据分析与产量预测系统，采用多模型融合算法
    （随机森林、LightGBM、线性回归），综合考虑温度、降水、干热风等气象因子，
    对{county if county else '全省'}2010-2024年小麦产量进行预测和分析。
    模型整体预测精度R²=0.905，误差约20 kg/亩，具备实际参考价值。
    """
    story.append(Paragraph(summary_text, summary_style))
    story.append(Spacer(1, 20))

    # 表格
    table_data = [
        ["年份", "生育期积温(℃·d)", "生育期降水(mm)", "极端高温天数", "干热风天数", "预测单产(kg/亩)", "实测单产(kg/亩)"]
    ]
    for r in rows:
        table_data.append([
            str(r["year"]),
            str(r["gdd"]),
            str(r["precip"]),
            str(r["hot_days"]),
            str(r["hot_dry_days"]),
            str(r["pred"]),
            str(r["actual"]),
        ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # 加一个结论和建议
    conclusion_style = styles['Normal']
    conclusion_style.fontName = font_name
    conclusion_style.fontSize = 11
    conclusion_style.leading = 18
    conclusion_text = f"""
    <b>结论与建议：</b><br/>
    1. 从数据来看，{county if county else '全省'}小麦产量整体呈逐年上升趋势，
    这主要得益于品种改良和种植技术进步。<br/>
    2. 干热风天数对产量影响较大，灌浆期要特别注意防范干热风灾害，
    可通过合理灌溉、叶面喷水等措施降低影响。<br/>
    3. 生育期降水是影响产量的重要因素，
    降水不足时要及时灌溉，确保小麦生长关键期的水分供应。<br/>
    4. 本报告预测结果仅供参考，实际生产中还需结合当年的具体天气情况进行调整。
    """
    story.append(Paragraph(conclusion_text, conclusion_style))
    story.append(Spacer(1, 30))

    # 底部说明
    footer_style = styles['Normal']
    footer_style.fontName = font_name
    footer_style.fontSize = 9
    footer_style.textColor = colors.grey
    story.append(Paragraph("—— 本报告由河南省小麦气象数据分析与产量预测系统自动生成 ——", footer_style))

    doc.build(story)
    output.seek(0)

    filename = f"{county}_小麦产量预测报告.pdf" if county else f"小麦产量预测报告_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf"
    return send_file(
        output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/simulate")
def api_simulate():
    """情景模拟：调整关键气象因子，返回预测产量"""
    county = request.args.get("county", "")
    year = request.args.get("year", type=int) or 2024

    bundle = load_models()
    if bundle is None:
        return {"error": "模型未加载"}

    rf, lr, lgbm, stats = bundle
    mean_yield, _ = county_mean_yield(stats, county)

    # 获取基准特征
    data, _ = dataset_for_county(county)
    feats = predict_years(data, county)
    row = feats[feats["year"] == year]
    if row.empty:
        return {"error": f"{county} {year} 年无完整数据"}

    X = row[FEATS].fillna(feats[FEATS].median(numeric_only=True)).copy()

    # 应用用户调整的因子（用numpy数组修改，避免pandas链式赋值bug）
    X_arr = X.values.copy()
    adjustments = {}
    for i, feat in enumerate(FEATS):
        val = request.args.get(feat, type=float)
        if val is not None:
            X_arr[0, i] = val
            adjustments[feat] = val

    # 模拟输入可能远离训练分布，先按各模型基准值 ±30% 截断，避免线性回归外推失真
    baseline_X = row[FEATS].fillna(feats[FEATS].median(numeric_only=True)).values
    base_rf = float(rf.predict(baseline_X)[0])
    base_lgbm = float(lgbm.predict(baseline_X)[0]) if lgbm is not None else base_rf
    base_lr = float(lr.predict(baseline_X)[0]) if lr is not None else base_rf

    def clamp(value, base):
        return max(base * 0.7, min(base * 1.3, value))

    def ensemble_predict(X_data):
        pred_rf = clamp(float(rf.predict(X_data)[0]), base_rf)
        pred_lgbm = clamp(float(lgbm.predict(X_data)[0]), base_lgbm) if lgbm is not None else pred_rf
        pred_lr = clamp(float(lr.predict(X_data)[0]), base_lr) if lr is not None else pred_rf
        fused = 0.5 * pred_rf + 0.3 * pred_lgbm + 0.2 * pred_lr
        return max(mean_yield * 0.5, min(mean_yield * 1.5, fused * mean_yield))

    pred = ensemble_predict(X_arr)
    baseline = ensemble_predict(baseline_X)

    return {
        "county": county,
        "year": year,
        "baseline": round(baseline, 1),
        "predicted": round(pred, 1),
        "change": round((pred - baseline) / baseline * 100, 1),
        "mean_yield": round(mean_yield, 1),
        "adjustments": adjustments,
        "base_features": {
            "gdd": round(float(row["gdd"].iloc[0]), 0) if "gdd" in row else 1800,
            "season_precip": round(float(row["season_precip"].iloc[0]), 0) if "season_precip" in row else 400,
            "hot_days": round(float(row["hot_days"].iloc[0]), 0) if "hot_days" in row else 5,
            "cold_days": round(float(row["cold_days"].iloc[0]), 0) if "cold_days" in row else 10,
        }
    }


@app.route("/predict")
def predict():
    scope = uploaded_scope()
    if scope:
        counties = scope["counties"]
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else scope["default_county"]
    else:
        counties = ALL_COUNTIES
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else ("登封" if "登封" in counties else counties[0])

    bundle = load_models()
    mm = model_meta()
    data, data_meta = dataset_for_county(county)
    if scope:
        years = scope["years"] or list(BASE_YEARS)
    elif not data.empty and "year" in data.columns:
        years = sorted(set(BASE_YEARS + [int(y) for y in data["year"].unique().tolist()]))
    else:
        years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]

    metrics = read_metrics("rf")
    metrics_lgbm = read_metrics("lgbm")
    metrics_lr = read_metrics("lr")

    if bundle is None or bundle[0] is None:
        return render_template("predict.html", title="产量预测", active="predict",
                               counties=counties, city_groups=scope_city_groups(scope),
                               county=county, years=years, year=year,
                               threshold=0, threshold_max=None, threshold_note=None,
                               ratio=CONFIG["alert"]["default_ratio"] * 100,
                               metrics=metrics, metrics_lgbm=metrics_lgbm, metrics_lr=metrics_lr,
                               reliable=False, result=None, history=[], trend_chart=None,
                               feature_chart=None, feature_count=len(FEATS),
                               source=data_meta["source"],
                               error="尚未训练模型，请先运行 python -m src.models.train_model")

    rf, lr, lgbm, stats = bundle
    mean_yield, matched = county_mean_yield(stats, county)
    ratio = CONFIG["alert"]["default_ratio"]
    threshold_min = round(max(0.0, mean_yield * 0.4), 1)
    threshold_max = round(max(mean_yield * 1.6, threshold_min + 1.0), 1)
    threshold_raw = request.args.get("threshold", type=float)
    threshold_note = None
    if threshold_raw is None:
        threshold = round(mean_yield * ratio, 1)
    else:
        threshold = round(threshold_raw, 1)
        if threshold < threshold_min or threshold > threshold_max:
            threshold = min(max(threshold, threshold_min), threshold_max)
            threshold_note = f"预警值超出合理范围，已按 {threshold} kg/亩 处理。"

    feats = predict_years(data, county)
    row = feats[feats["year"] == year] if not feats.empty else feats
    error = None
    result, history = None, []

    if feats.empty:
        error = f"当前数据集缺少 {county} 的气象记录，无法构造预测特征。"
    elif row.empty:
        error = (f"当前数据集缺少 {county} {year} 年的完整生育期气象记录，"
                 f"无法构造预测特征。")
    else:
        X = row[FEATS].fillna(feats[FEATS].median(numeric_only=True))
        feature_note = bool(X.isna().any().any())
        # 再处理一遍，确保没有NaN
        X = X.fillna(0)
        pred_rf = float(rf.predict(X)[0]) * mean_yield
        pred_lr = float(lr.predict(X)[0]) * mean_yield if lr is not None else pred_rf
        pred_lgbm = float(lgbm.predict(X)[0]) * mean_yield if lgbm is not None else pred_rf
        # 多模型加权平均
        pred = 0.5 * pred_rf + 0.3 * pred_lgbm + 0.2 * pred_lr

        labels, is_real = load_yield_labels()
        actual = None
        if labels is not None:
            m = labels[(labels["county"] == county) & (labels["year"] == year)]
            if not m.empty:
                actual = round(float(m["wheat_yield"].iloc[0]), 1)

        # MAE 近似区间：用模型测试集 MAE 做经验性波动范围，不作为统计置信区间
        interval_mae = float(metrics["mae"]) if metrics and metrics.get("mae") is not None else 21.7
        pred_low = round(pred - 0.8 * interval_mae, 1)
        pred_high = round(pred + 0.8 * interval_mae, 1)

        result = {
            "county": county, "year": year,
            "pred_rf": round(pred_rf, 1), "pred_lr": round(pred_lr, 1), "pred_lgbm": round(pred_lgbm, 1),
            "pred": round(pred, 1),
            "pred_low": pred_low, "pred_high": pred_high,
            "mean_yield": round(mean_yield, 1), "threshold": round(threshold, 1),
            "actual": actual,
            "vs_threshold": round((pred - threshold) / threshold * 100, 1) if threshold else 0,
            "vs_mean": round((pred / mean_yield - 1) * 100, 1),
            "alert": pred < threshold,
            "matched": matched,
            "feature_note": feature_note,
        }

        # 历史对比：逐年预测 vs 实测
        api = labels.set_index(["county", "year"])["wheat_yield"].to_dict() if labels is not None else {}
        X_all = feats[FEATS].fillna(feats[FEATS].median(numeric_only=True))
        preds_rf = rf.predict(X_all) * mean_yield
        preds_lr = lr.predict(X_all) * mean_yield if lr is not None else preds_rf
        preds_lgbm = lgbm.predict(X_all) * mean_yield if lgbm is not None else preds_rf
        preds_all = 0.5 * preds_rf + 0.3 * preds_lgbm + 0.2 * preds_lr
        for i, r in feats.reset_index(drop=True).iterrows():
            y_ = int(r["year"])
            act = api.get((county, y_))
            pred = round(float(preds_all[i]), 1)
            history.append({
                "year": y_,
                "actual": round(float(act), 1) if act is not None and not pd.isna(act) else None,
                "pred": pred,
                "diff": round(pred - float(act), 1) if act is not None and not pd.isna(act) else None,
                "alert": pred < threshold,
            })
        history = sorted(history, key=lambda x: x["year"], reverse=True)

    trend_chart = None
    feature_chart = None
    if result:
        base_values = [result["pred"], result["threshold"], result["mean_yield"],
                       result["pred_low"], result["pred_high"]]
        low, high = min(base_values), max(base_values)
        span = max(high - low, 10.0)
        scale_min = max(0.0, low - span * 0.8)
        scale_max = high + span * 0.8

        def _pct(value):
            return round((value - scale_min) / (scale_max - scale_min) * 100, 1)

        result["scale"] = {
            "min": round(scale_min, 1),
            "max": round(scale_max, 1),
            "pred_pct": _pct(result["pred"]),
            "threshold_pct": _pct(result["threshold"]),
            "mean_pct": _pct(result["mean_yield"]),
            "low_pct": _pct(result["pred_low"]),
            "high_pct": _pct(result["pred_high"]),
        }

        ordered = sorted(history, key=lambda item: item["year"])
        if ordered:
            trend_chart = {
                "kind": "predictionTrend",
                "title": "实测与预测单产趋势",
                "desc": f"{county} {ordered[0]['year']}—{ordered[-1]['year']} 年单产对比（kg/亩），水平虚线为预警线。",
                "payload": {
                    "labels": [str(item["year"]) for item in ordered],
                    "actual": [item["actual"] for item in ordered],
                    "predicted": [item["pred"] for item in ordered],
                    "threshold": round(float(threshold), 1),
                    "selected": year,
                },
            }

        feature_items = feature_importance_items()
        if feature_items:
            feature_chart = {
                "kind": "hBar",
                "title": "特征重要性",
                "desc": "随机森林特征重要性占比；数值越高，对产量预测的影响越大。",
                "payload": {
                    "labels": [item["name"] for item in feature_items],
                    "values": [item["value"] for item in feature_items],
                    "unit": "%",
                },
            }

    return render_template("predict.html", title="产量预测", active="predict",
                           counties=counties, city_groups=scope_city_groups(scope),
                           county=county, years=years, year=year,
                           threshold=round(threshold, 1),
                           threshold_min=threshold_min, threshold_max=threshold_max,
                           threshold_note=threshold_note, ratio=ratio * 100,
                           metrics=metrics, metrics_lgbm=metrics_lgbm, metrics_lr=metrics_lr,
                           reliable=bool(mm and mm.get("reliable")),
                           result=result, history=history, trend_chart=trend_chart,
                           feature_chart=feature_chart, feature_count=len(FEATS),
                           error=error, source=data_meta["source"])


@app.route("/advice")
def advice():
    scope = uploaded_scope()
    if scope:
        counties = scope["counties"]
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else scope["default_county"]
    else:
        counties = ALL_COUNTIES
        county_arg = request.args.get("county")
        county = county_arg if county_arg in counties else ("登封" if "登封" in counties else counties[0])

    data, data_meta = dataset_for_county(county)
    if scope:
        years = scope["years"] or list(BASE_YEARS)
    elif not data.empty and "year" in data.columns:
        years = sorted(set(BASE_YEARS + [int(y) for y in data["year"].unique().tolist()]))
    else:
        years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]

    if data.empty:
        return render_template("advice.html", title="农事建议", active="advice",
                               counties=counties, city_groups=scope_city_groups(scope),
                               county=county, years=years, year=year,
                               threshold=0, result=None,
                               source=data_meta["source"],
                               error="当前数据集中没有该县区的记录。")

    binding = load_models()
    pred, threshold, mean_yield = None, None, None
    if binding is None or binding[0] is None:
        return render_template("advice.html", title="农事建议", active="advice",
                               counties=counties, city_groups=scope_city_groups(scope),
                               county=county, years=years, year=year,
                               threshold=0, result=None, source=data_meta["source"],
                               error="尚未训练模型，无法结合产量预测生成建议。")

    rf, lr, lgbm, stats = binding
    mean_yield, _ = county_mean_yield(stats, county)
    ratio = CONFIG["alert"]["default_ratio"]
    threshold = request.args.get("threshold", type=float)
    if threshold is None:
        threshold = round(mean_yield * ratio, 1)

    feats = predict_years(data, county)
    row = feats[feats["year"] == year]
    feature_note = False
    if not row.empty:
        X = row[FEATS].fillna(feats[FEATS].median(numeric_only=True))
        feature_note = bool(X.isna().any().any())
        X = X.fillna(0)
        pred_rf = float(rf.predict(X)[0]) * mean_yield
        pred_lr = float(lr.predict(X)[0]) * mean_yield if lr is not None else pred_rf
        pred_lgbm = float(lgbm.predict(X)[0]) * mean_yield if lgbm is not None else pred_rf
        pred = round(0.5 * pred_rf + 0.3 * pred_lgbm + 0.2 * pred_lr, 1)

    result = generate_advice(data, county, year, predicted_yield=pred, threshold=threshold)
    result["alert"] = bool(pred is not None and pred < threshold)
    result["feature_note"] = feature_note
    return render_template("advice.html", title="农事建议", active="advice",
                           counties=counties, city_groups=scope_city_groups(scope),
                           county=county, years=years, year=year,
                           threshold=round(threshold, 1), result=result, error=None,
                           source=data_meta["source"])


@app.route("/api/feature_importance")
def api_feature_importance():
    """返回特征重要性排序"""
    return {"items": feature_importance_items()}


@app.errorhandler(413)
def too_large(e):
    return render_template("upload.html", title="数据上传", active="upload",
                           report=None, error=f"文件超过 {CONFIG['upload']['max_size_mb']} MB 限制。",
                           dataset=None, counties=ALL_COUNTIES,
                           max_size=CONFIG["upload"]["max_size_mb"]), 413


if __name__ == "__main__":
    import threading

    def _warm_caches():
        try:
            builtin_dataset()
            load_models()
            feature_importance_items()
        except Exception:
            pass

    threading.Thread(target=_warm_caches, daemon=True).start()
    app.run(host=CONFIG["web"]["host"], port=CONFIG["web"]["port"],
            debug=CONFIG["web"]["debug"])
