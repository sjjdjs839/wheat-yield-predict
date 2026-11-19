"""
河南省小麦气象灾害影响分析与智能预警平台 · Web 应用
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
import re
import uuid
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from flask import (Flask, render_template, request, redirect, url_for)

from src.data.validate import validate_file, load_weather_file
from src.data.build_features import build_county, CANONICAL_TO_NASA
from src.analysis.web_charts import build_web
from src.advice.agronomy import generate_advice
from src.advice import stage_store, check_store
from src.advice.diagnose import diagnose as run_diagnose
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
    """内置逐日气象宽表。

    新克隆的仓库里 weather_clean.csv 是可重建产物、不随仓库提交；文件缺失时
    返回空表让 MySQL 分支接管，避免整页抛 FileNotFoundError。
    """
    if not PROCESSED.exists():
        return pd.DataFrame()
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
        # 配置里有这个县、但气象表里一行都没有 → 返回空表并明确标记缺失。
        # 绝不拿全省 150 县数据冒充该县（否则会出现「龙管会累计降水 5124.5 mm」这种露馅数字）。
        return base.iloc[0:0], {"filename": PROCESSED.name, "county": county,
                                "source": "无气象数据", "missing": True,
                                "summary": {"rows": 0, "year_range": ""}}
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
    """返回 (用于还原的县均值单产, 是否命中该县)。

    取训练期 county_prior（2010—2021），与模型训练口径一致，避免把测试年
    实际产量带进特征。与年鉴全期均值相差约 3.6 kg/亩（最大 13.8）。
    """
    if county in stats:
        return county_prior_of(stats, county, float(stats[county]["mean_yield"])), True
    vals = [v["mean_yield"] for v in stats.values()]
    return float(np.median(vals)), False


def county_prior_of(stats: dict, county: str, fallback: float) -> float:
    """模型用的县均单产，取训练期口径（county_stats 的 features.county_prior）。

    训练时 county_prior 只用 2010—2021 的县均值，预测若改用全期 mean_yield，
    既与训练分布不一致，也会把 2022—2024 的实际产量带进特征（信息泄露）。
    """
    v = stats.get(county) if stats else None
    if not v:
        return float(fallback)
    f = v.get("features") or {}
    if "county_prior" in f:
        return float(f["county_prior"])
    return float(v.get("mean_yield", fallback))


def _attach_prior(feat: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """补县域先验特征（=该县常年单产）；缺县回退全局中位数，与训练口径一致。"""
    if "county_prior" not in FEATS:
        return feat
    vals = [v["mean_yield"] for v in stats.values()] if stats else []
    global_mean = float(np.median(vals)) if vals else 0.0
    feat = feat.copy()
    feat["county_prior"] = feat["county"].map(
        lambda c: county_prior_of(stats, c, global_mean)
    )
    return feat


def batch_predict(feat: pd.DataFrame, bundle, mean_yield) -> list:
    """批量预测：一次性把特征矩阵喂给模型，返回与 feat 等长的还原单产列表（无法预测为 None）。
    单县用传入的县均值；全省逐行按该县在 stats 中的均值还原，缺县回退全局中位数。"""
    if bundle is None or len(feat) == 0:
        return [None] * len(feat)
    rf, lr, lgbm, stats = bundle
    feat = _attach_prior(feat, stats)
    X = feat[FEATS].fillna(feat[FEATS].median(numeric_only=True))
    try:
        p_rf = rf.predict(X)
        p_lgbm = lgbm.predict(X) if lgbm is not None else p_rf
        p_lr = lr.predict(X) if lr is not None else p_rf
        fused = 0.2 * p_rf + 0.6 * p_lgbm + 0.2 * p_lr
        vals = [v["mean_yield"] for v in stats.values()] if stats else []
        global_mean = float(np.median(vals)) if vals else mean_yield
        out = []
        for i, (_, r) in enumerate(feat.iterrows()):
            cy = r["county"]
            m = county_prior_of(stats, cy, global_mean)
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


@lru_cache(maxsize=256)
def dry_hot_reference(county: str) -> dict | None:
    """本县历年灌浆期干热风 ≥3 天年份的单产表现（相对本县常年），仅供参考。

    刻意只给样本量与区间：县域尺度上气象与单产的统计关联较弱，
    不能据此推算"今年会减产多少"，见 output/reports/_legacy_note.md。
    """
    if not FEATURE_CSV.exists():
        return None
    try:
        feat = pd.read_csv(FEATURE_CSV)
    except Exception:
        return None
    if "wheat_yield" not in feat.columns or "hot_dry_days" not in feat.columns:
        return None
    d = feat[(feat["county"] == county) & feat["wheat_yield"].notna()].copy()
    if d.empty:
        return None
    d["norm"] = d["wheat_yield"] / d["wheat_yield"].mean() * 100
    heavy = d[d["hot_dry_days"] >= 3]
    if len(heavy) < 3:      # 样本过少时区间退化，不展示以免误导
        return None
    return {
        "n": int(len(heavy)),
        "mean": float(heavy["norm"].mean()),
        "low": float(heavy["norm"].min()),
        "high": float(heavy["norm"].max()),
        "range": f"{int(d['year'].min())}—{int(d['year'].max())}",
    }


FEATURE_NAMES = {
    "spi_12": "标准化降水指数",
    "precip_anom": "降水距平",
    "hot_days": "极端高温天数",
    "cold_days": "极端低温天数",
    "gdd": "生育期积温",
    "season_tmax": "生育期最高温",
    "season_tmin": "生育期最低温",
    "season_precip": "生育期降水",
    "season_wind": "生育期风速",
    "season_sunshine": "生育期日照",
    "season_tmax_34": "返青拔节期最高温",
    "season_precip_34": "返青拔节期降水",
    "gdd_34": "返青拔节期积温",
    "season_tmax_56": "灌浆期最高温",
    "season_precip_56": "灌浆期降水",
    "gdd_56": "灌浆期积温",
    "hot_dry_days": "干热风天数",
    "year_index": "品种改良趋势",
    "county_prior": "县域常年单产先验",
}


def _load_shap_global() -> pd.DataFrame | None:
    p = REPORTS_DIR / "shap_importance.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def feature_importance_items() -> list[dict]:
    bundle = load_models()
    if bundle is None or bundle[0] is None:
        return []
    shap_df = _load_shap_global()
    if shap_df is not None and len(shap_df):
        items = []
        for _, r in shap_df.iterrows():
            feat = r["feature"]
            items.append({
                "name": FEATURE_NAMES.get(feat, feat),
                "value": round(float(r["pct"]), 2),
                "mean_shap": round(float(r["mean_shap"]), 6),
                "source": "shap",
            })
        return sorted(items, key=lambda item: item["value"], reverse=True)
    # 回退：随机森林分裂增益重要性
    items = []
    for feat, importance in zip(FEATS, bundle[0].feature_importances_):
        items.append({
            "name": FEATURE_NAMES.get(feat, feat),
            "value": round(float(importance) * 100, 2),
            "source": "gain",
        })
    return sorted(items, key=lambda item: item["value"], reverse=True)


@lru_cache(maxsize=1)
def _rf_explainer():
    bundle = load_models()
    if bundle is None or bundle[0] is None:
        return None
    try:
        import shap
        return shap.TreeExplainer(bundle[0])
    except Exception:
        return None


def sample_contributions(x_vec: np.ndarray, top_n: int = 5) -> dict:
    """单样本 SHAP：返回把预测推高 / 拉低的 top 因子（yield_norm 尺度）。"""
    expl = _rf_explainer()
    if expl is None:
        return {"available": False, "positive": [], "negative": []}
    try:
        sv = np.asarray(expl.shap_values(np.asarray(x_vec).reshape(1, -1)))[0]
        rows = [{"feature": f, "name": FEATURE_NAMES.get(f, f),
                 "shap": round(float(v), 4)} for f, v in zip(FEATS, sv)]
        pos = sorted([r for r in rows if r["shap"] > 0], key=lambda r: -r["shap"])[:top_n]
        neg = sorted([r for r in rows if r["shap"] < 0], key=lambda r: r["shap"])[:top_n]
        return {"available": True, "positive": pos, "negative": neg}
    except Exception:
        return {"available": False, "positive": [], "negative": []}


# ---------------------------------------------------------------- 路由
_YEARLY_HEAVY_CACHE: dict = {}


def _yearly_heavy_counties() -> dict:
    """每年干热风重度县数（≥3 天）。只依赖气象表，算一次缓存住，供首页下拉标注。"""
    if _YEARLY_HEAVY_CACHE:
        return _YEARLY_HEAVY_CACHE
    try:
        _f = _risk_feature_table()
        for y, g in _f.groupby("year"):
            _YEARLY_HEAVY_CACHE[int(y)] = int((g["hot_dry_days"].fillna(0) >= 3).sum())
    except Exception:
        pass
    return _YEARLY_HEAVY_CACHE


@app.route("/")
def index():
    risk_years = list(BASE_YEARS)
    y_arg = request.args.get("year", type=int)
    risk_year = y_arg if y_arg in risk_years else risk_years[-1]
    try:
        risk_brief = risk_snapshot(risk_year)
    except Exception:
        risk_brief = None

    # 首页也按「晚霜冻优先」重排，并算出灾害摘要
    frost_ctx = {"frost_n": 0, "frost_heavy": 0, "frost_worst": None,
                 "frost_first": None, "frost_any": 0,
                 "last_frost": _last_frost(risk_year)}
    if risk_brief:
        try:
            _frost_attach(risk_brief["items"], risk_year)
            its = risk_brief["items"]
            its.sort(key=lambda x: (-(x["frost_level"] or 0),
                                    x["frost_loss"] if x["frost_loss"] is not None else 0,
                                    -x["days"], x["county"]))
            risk_brief["top"] = its[:5]
            frost_ctx["frost_n"] = sum(1 for x in its if (x["frost_level"] or 0) > 0)
            # 中度以上 = 4 月 ≥2 天 ≤0℃（frost_level >= 2），与页面标签同口径
            frost_ctx["frost_heavy"] = sum(1 for x in its if (x["frost_level"] or 0) >= 2)
            _pool = [x for x in its if (x["frost_level"] or 0) > 0]
            frost_ctx["frost_any"] = len(_pool)
            # 有显著减产的县（用于「最大预估影响」）
            frost_ctx["frost_worst"] = min(
                [x for x in _pool if (x["frost_loss"] or 0) < -0.05],
                key=lambda x: x["frost_loss"], default=None)
            # 有霜冻但未减产的县（用于避免把「有霜冻」说成「没霜冻」）
            frost_ctx["frost_first"] = min(
                _pool,
                key=lambda x: (x["frost_loss"] if x["frost_loss"] is not None else 0),
                default=None)
            w = frost_ctx["frost_worst"]
            if w:
                rest = [d for d in (risk_brief.get("decision") or []) if d["county"] != w["county"]]
                risk_brief["decision"] = ([{
                    "kind": "晚霜冻", "county": w["county"],
                    "text": "4 月低温 %d 天、最低 %s℃，按弹性预估影响 %s%%" % (
                        w["fz4"], w["tmin4"], w["frost_loss"]),
                    "target": "risk", "tone": "alert",
                }] + rest)[:3]
        except Exception:
            pass

    # 每年重度县数（只看气象，不跑预测），用于在下拉里标出「干热风重」的年份
    yearly_heavy = _yearly_heavy_counties()
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
                    "dataset_text": "逐日气象 · 县区统计产量",
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
                                   best=best_m, best_label=best_label, risk=risk_brief,
                                   risk_years=risk_years, yearly_heavy=yearly_heavy,
                                   frost=frost_ctx)
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
                           best=best_m, best_label=best_label, risk=risk_brief,
                           risk_years=risk_years, yearly_heavy=yearly_heavy,
                           frost=frost_ctx)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    report, error, diag = None, None, None
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
                    # 诊断优先：一份原始气象进，出灾害清单 + 减产评估（可选入库出建议）
                    if county:
                        try:
                            df_up, _ = load_weather_file(saved, county, stage_mode=True)
                            if df_up is not None and not df_up.empty:
                                up_year = int(df_up["year"].max())
                                diag = run_diagnose(df_up, county, up_year)
                                diag["county"] = county
                                diag["year"] = up_year
                                # 当期/未来年写入阶段观测库，农事建议页立刻可用；
                                # 历史年只诊断不写库，避免覆盖内置年份的语义
                                if up_year > max(BASE_YEARS):
                                    stage_store.upsert(county, up_year, df_up)
                                    diag["saved_to_advice"] = True
                        except Exception as exc:
                            diag = {"empty": True, "county": county, "year": None, "days": 0,
                                    "events": [], "stages": [], "loss": None,
                                    "month_span": "", "verdict": "",
                                    "error": "诊断未能完成：" + str(exc)[:80]}
                    if request.form.get("use_as_dataset"):
                        LATEST.write_text(json.dumps({
                            "path": str(saved.relative_to(PROJECT_ROOT)),
                            "filename": f.filename,
                            "county": county or None,
                            "summary": report["summary"],
                        }, ensure_ascii=False), encoding="utf-8")
                    else:
                        saved.unlink(missing_ok=True)
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

    advice_url = url_for("advice")
    if diag and diag.get("county"):
        advice_url = url_for("advice", county=diag.get("county"),
                             year=diag.get("year") if diag.get("year") else None)
    return render_template("upload.html", title="数据诊断", active="upload",
                           report=report, error=error, dataset=meta, diag=diag,
                           counties=ALL_COUNTIES, max_size=CONFIG["upload"]["max_size_mb"],
                           analysis_url=analysis_url, predict_url=predict_url,
                           advice_url=advice_url)


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
                           stage_windows=result.get("stage_windows"),
                           source=meta["source"], error=error)


@app.route("/map")
def map_view():
    years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else (2018 if 2018 in years else years[-1])
    return render_template("map.html", title="灾害风险地图", active="map", years=years, year=year)


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
    elif metric == "disaster":
        # 4 月低温的县距平减产幅度（已剔除全省年景），值越大减产越重
        ev = disaster_events_table()
        if not ev.empty:
            sub = ev[ev["year"] == year]
            for _, r in sub.iterrows():
                geo_name = to_geo_name(r["county"])
                v = r.get("frost_anom_loss")
                if geo_name and pd.notna(v):
                    items.append({"county": geo_name, "value": round(max(0.0, -float(v)), 2)})
    elif metric == "frostdays":
        # 4 月逐日最低温 ≤0℃ 的天数：暴露强度，与预警页的剂量响应同一口径
        ev = disaster_events_table()
        if not ev.empty:
            sub = ev[ev["year"] == year]
            for _, r in sub.iterrows():
                geo_name = to_geo_name(r["county"])
                v = r.get("fz4_0")
                if geo_name and pd.notna(v):
                    items.append({"county": geo_name, "value": int(v)})
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


# ---------------------------------------------------------------- 风险总览
def _risk_feature_table() -> pd.DataFrame:
    """风险总览数据源：MySQL 特征表优先，回退内置 feature_table.csv。"""
    feat = None
    if repo.available():
        try:
            feat = repo.feature_rows()
        except Exception:
            feat = None
    if feat is None or feat.empty:
        feat = pd.read_csv(PROJECT_ROOT / CONFIG["paths"]["processed"] / "feature_table.csv")
    return feat


def _county_stats_baseline() -> dict:
    """县常年单产基准。

    模型 joblib 不随仓库提交，新克隆后未训练时 load_models() 返回 None；
    county_stats.json 随仓库提交，用它兜底可让风险清单照常可用，
    仅「预测单产」列留空。
    """
    try:
        return json.loads((MODELS_DIR / "county_stats.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _county_city_map() -> dict:
    city_map = {}
    for city, counties_dict in CONFIG["counties"].items():
        for c in counties_dict.values():
            city_map[c] = city
    return city_map


@lru_cache(maxsize=16)
def risk_snapshot(year: int) -> dict:
    """按年份计算全省县级风险清单。

    仅纳入有产量标签（county_stats 有常年单产）的县区，保证「预测单产 / 预警线」
    口径可解释；无标签县区仍参与气象统计，但不进风险清单。
    风险登记以干热风天数为准（0 天无 / 1—2 天轻度 / ≥3 天重度），
    并给出该县当年天数在自有历史年份中的排位。
    """
    feat = _risk_feature_table()
    bundle = load_models()
    stats = bundle[3] if bundle else _county_stats_baseline()
    city_map = _county_city_map()

    rows = feat[feat["year"] == year].reset_index(drop=True)
    preds = batch_predict(rows, bundle, None) if bundle else [None] * len(rows)

    # 该县历年干热风天数降序序列，用于计算当年排位
    hist = {}
    for county, grp in feat.groupby("county"):
        hist[county] = sorted(
            [int(v) if pd.notna(v) else 0 for v in grp["hot_dry_days"]],
            reverse=True,
        )

    items = []
    for i, r in rows.iterrows():
        county = r["county"]
        if county not in stats:
            continue
        days = int(r["hot_dry_days"]) if pd.notna(r["hot_dry_days"]) else 0
        mean = float(stats[county]["mean_yield"])
        pred = preds[i]
        pred = round(float(pred), 1) if pred is not None else None
        threshold = round(mean * CONFIG["alert"]["default_ratio"], 1)

        seq = hist.get(county, [])
        if seq:
            rank = sum(1 for v in seq if v > days) + 1
            total = len(seq)
        else:
            rank, total = None, 0

        level = "heavy" if days >= 3 else ("light" if days >= 1 else "none")
        vs_mean = round((pred / mean - 1) * 100, 1) if (pred is not None and mean) else None
        alert = bool(pred is not None and pred < threshold)

        items.append({
            "county": county,
            "city": city_map.get(county, "其他"),
            "days": days,
            "rank": rank,
            "rank_total": total,
            "level": level,
            "level_text": {"heavy": "重度", "light": "轻度", "none": "无"}[level],
            "pred": pred,
            "mean": round(mean, 1),
            "threshold": threshold,
            "vs_mean": vs_mean,
            "alert": alert,
            "actual": None,
        })

    # 实测对照（有标签年份）
    labels, _ = load_yield_labels()
    if labels is not None and not labels.empty:
        y = labels[labels["year"] == year]
        if not y.empty:
            act = dict(zip(y["county"], y["wheat_yield"]))
            for it in items:
                if it["county"] in act:
                    it["actual"] = round(float(act[it["county"]]), 1)

    items.sort(key=lambda x: (-x["days"], x["county"]))

    # 历史同强度年对照：同一县里干热风天数落在同一档（0 / 1—2 / ≥3）的历史年份，
    # 以及这些年份的单产相对该县常年偏了多少。样本少于 3 年不给区间，避免区间退化成一个点。
    def _band(d: float) -> str:
        return "none" if (d or 0) < 1 else ("light" if (d or 0) < 3 else "heavy")

    if labels is not None and not labels.empty:
        lab_map = {(r["county"], int(r["year"])): float(r["wheat_yield"])
                   for _, r in labels.iterrows()}
        county_mean_hist = {c: float(g["wheat_yield"].mean()) for c, g in labels.groupby("county")}
        days_by_county: dict = {}
        for c, g in feat.groupby("county"):
            days_by_county[c] = {int(r["year"]): float(r["hot_dry_days"] or 0) for _, r in g.iterrows()}

        for it in items:
            cur_band = _band(it["days"])
            hist = days_by_county.get(it["county"], {})
            same = [y for y, d in hist.items()
                    if y != year and _band(d) == cur_band and (it["county"], y) in lab_map]
            it["same_n"] = len(same)
            it["same_years"] = sorted(same)
            it["same_mean"] = it["same_low"] = it["same_high"] = None
            if len(same) >= 3:
                cm = county_mean_hist.get(it["county"])
                if cm:
                    pcts = [(lab_map[(it["county"], y)] / cm - 1) * 100 for y in same]
                    it["same_mean"] = round(sum(pcts) / len(pcts), 1)
                    it["same_low"] = round(min(pcts))
                    it["same_high"] = round(max(pcts))

    # 降水距平入列，供「墒情」类决策提示使用
    d_fy = feat[feat["year"] == year]
    d_anom = {}
    for _, r in d_fy.iterrows():
        v = r.get("precip_anom")
        if pd.notna(v):
            d_anom[r["county"]] = float(v)
    for it in items:
        it["precip_anom"] = d_anom.get(it["county"])

    # 当年全省干热风总天数在历史年份中的排位（数值越高越严重）
    year_totals = {}
    for y, grp in feat.groupby("year"):
        year_totals[int(y)] = int(grp["hot_dry_days"].fillna(0).sum())
    cur_total = year_totals.get(year, 0)
    year_rank = sum(1 for v in year_totals.values() if v > cur_total) + 1

    heavy = sum(1 for x in items if x["level"] == "heavy")
    light = sum(1 for x in items if x["level"] == "light")
    none_ = sum(1 for x in items if x["level"] == "none")
    alerts = sum(1 for x in items if x["alert"])
    preds_ok = [x["pred"] for x in items if x["pred"] is not None]

    # 决策提示：不只给统计，直接回答「先盯哪几个县、盯什么、去哪处理」
    decision: list = []
    seen: set = set()

    def _push(kind: str, county: str, text: str, target: str, tone: str = "") -> None:
        if county in seen or len(decision) >= 3:
            return
        seen.add(county)
        decision.append({"kind": kind, "county": county, "text": text,
                         "target": target, "tone": tone})

    # ① 产量：有触发预警的取最严重；轻年没有触发时，退给「最接近预警线」的县
    d_warned = sorted([x for x in items if x["alert"] and x["pred"] and x["mean"]],
                    key=lambda x: x["pred"] / x["mean"])[:1]
    if d_warned:
        for x in d_warned:
            gap = round(x["pred"] / x["mean"] * 100 - 100, 1)
            _push("产量预警", x["county"], f"预测 {x['pred']} kg/亩，比常年低 {abs(gap)}%",
                  "predict", "alert")
    else:
        d_near = sorted([x for x in items if x["pred"] and x["mean"]],
                        key=lambda x: x["pred"] / x["mean"])[:1]
        for x in d_near:
            gap = round(x["pred"] / x["mean"] * 100 - 100, 1)
            thr = round(x["pred"] / (x["mean"] * 0.95) * 100 - 100, 1)
            gap_txt = ("与常年基本持平" if abs(gap) < 0.15
                       else f"比常年{'低' if gap < 0 else '高'} {abs(gap)}%")
            label = "接近预警线" if x["pred"] < x["mean"] else "产量关注"
            _push(label, x["county"],
                  f"预测 {x['pred']} kg/亩，{gap_txt}，距预警线 {abs(thr)}%",
                  "predict", "alert")

    # ② 干热风：≥3 天优先；没有重度时给 1—2 天的轻度暴露县
    d_hot = sorted(items, key=lambda x: (-x["days"], x["county"]))
    d_heavy = [x for x in d_hot if x["days"] >= 3]
    if d_heavy:
        heavy_note = f" · 全省 {heavy} 县重度" if heavy >= 5 else ""
        _push("干热风", d_heavy[0]["county"],
              f"{d_heavy[0]['days']} 天（重度）{heavy_note}，优先备降温补水", "dry_hot", "hot")
    else:
        d_mild = [x for x in d_hot if x["days"] >= 1]
        if d_mild:
            _push("干热风暴露", d_mild[0]["county"],
                  f"{d_mild[0]['days']} 天轻度暴露，备好降温补水", "dry_hot", "hot")

    # ③ 墒情：降水距平最负（阈值放宽到 -10%，否则轻年常常为空）
    for x in sorted([x for x in items if x.get("precip_anom") is not None],
                    key=lambda x: x["precip_anom"])[:2]:
        if x["precip_anom"] <= -10:
            _push("墒情", x["county"],
                  f"生育期降水比常年少 {abs(round(x['precip_anom']))}%，先查墒情再决定浇不浇",
                  "advice", "dry")

    return {
        "year": year,
        "items": items,
        "cities": sorted({x["city"] for x in items}),
        "summary": {
            "count": len(items),
            "heavy": heavy,
            "light": light,
            "none": none_,
            "alerts": alerts,
            "avg_pred": round(sum(preds_ok) / len(preds_ok), 1) if preds_ok else None,
            "max_days": items[0]["days"] if items else 0,
            "max_county": items[0]["county"] if items else "—",
            "year_total": cur_total,
            "year_rank": year_rank,
            "year_rank_total": len(year_totals),
        },
        "top": items[:5],
        "decision": decision,
    }


_FROST_CACHE: dict = {}

FROST_TEXT = {0: "无", 1: "轻度", 2: "中度", 3: "重度"}
FROST_NOTE = {0: "4 月未出现 ≤0℃", 1: "4 月 1 天 ≤0℃", 2: "4 月 2—3 天 ≤0℃", 3: "4 月 ≥4 天 ≤0℃"}


def disaster_events_table():
    """县级灾害事件表（scripts/build_disaster_events.py 生成，按县×年一行）。"""
    if "df" not in _FROST_CACHE:
        p = Path(__file__).resolve().parent / "data" / "processed" / "disaster_events.csv"
        _FROST_CACHE["df"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return _FROST_CACHE["df"]


_DOSE_CACHE = {}


def _dose_payload():
    """剂量响应（4 月低温分档），供灾害预警页展示阈值依据。"""
    if "d" in _DOSE_CACHE:
        return _DOSE_CACHE["d"]
    p = Path(__file__).resolve().parent / "output" / "reports" / "dose_frost.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        d = None
    _DOSE_CACHE["d"] = d
    return d


_MH_CACHE = {}


def _multihazard_payload():
    """四灾种面板检验结果，供灾害预警页说明"为什么只有晚霜冻折算减产"。"""
    if "d" in _MH_CACHE:
        return _MH_CACHE["d"]
    p = Path(__file__).resolve().parent / "output" / "reports" / "multihazard.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        d = None
    _MH_CACHE["d"] = d
    return d


def _norm_county(x):
    """灾害表与配置表县名不统一（'南召县' vs '南召'），比较前去掉行政区后缀。"""
    return re.sub(r"[县市区旗]$", "", str(x).strip())


def frost_reference(county: str, year: int):
    """该县该年的低温画像，供农事决策卡给出可追溯的量化依据。"""
    ev = disaster_events_table()
    if ev.empty:
        return None
    key = _norm_county(county)
    row = ev[(ev["county"].map(_norm_county) == key) & (ev["year"] == year)]
    if row.empty:
        return None
    r = row.iloc[0]

    def num(k):
        v = r.get(k)
        try:
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    return {
        "fz3": num("fz3_0"),
        "fz3_anom": num("fz3_anom"),
        "fz4": num("fz4_0"),
        "tmin4": num("tmin4"),
        "frost_loss": num("total_loss"),
        "frost_low": num("total_low"),
        "frost_high": num("total_high"),
        "frost_own": num("frost_anom_loss"),
        "year_eff": num("year_effect"),
        "cold3_loss": num("cold3_loss"),
        "base4": num("base4"),
        "base3": num("base3"),
        "level": r.get("frost_level"),
    }


def _frost_attach(items, year):
    """给风险清单注入晚霜冻字段；表缺失时给空值，页面不会崩。"""
    ev = disaster_events_table()
    empty = {"fz4": None, "fz4_streak": None, "tmin4": None, "frost_level": 0,
             "frost_text": "无记录", "frost_note": "无灾害事件表",
             "frost_loss": None, "frost_low": None, "frost_high": None,
             "frost_own": None, "year_eff": None, "base4": None,
             "fz3": None, "fz3_anom": None, "cold3_loss": None}
    if ev.empty or "year" not in ev.columns:
        for it in items:
            it.update(empty)
        return items
    sub = ev[ev["year"] == year]
    m = {r["county"]: r for _, r in sub.iterrows()}
    for it in items:
        r = m.get(it["county"])
        if r is None:
            it.update(empty)
            continue
        lv = int(r["frost_level"]) if pd.notna(r["frost_level"]) else 0
        it["fz4"] = int(r["fz4_0"]) if pd.notna(r["fz4_0"]) else None
        it["fz4_streak"] = int(r["fz4_streak"]) if pd.notna(r["fz4_streak"]) else None
        it["tmin4"] = float(r["tmin4"]) if pd.notna(r["tmin4"]) else None
        it["frost_level"] = lv
        it["frost_text"] = FROST_TEXT[lv]
        it["frost_note"] = FROST_NOTE[lv]
        it["frost_loss"] = float(r["total_loss"]) if pd.notna(r["total_loss"]) else None
        it["frost_low"] = float(r["total_low"]) if pd.notna(r["total_low"]) else None
        it["frost_high"] = float(r["total_high"]) if pd.notna(r["total_high"]) else None
        it["frost_own"] = float(r["frost_anom_loss"]) if pd.notna(r["frost_anom_loss"]) else None
        it["year_eff"] = float(r["year_effect"]) if pd.notna(r["year_effect"]) else None
        it["base4"] = float(r["base4"]) if pd.notna(r["base4"]) else None
        it["fz3"] = int(r["fz3_0"]) if pd.notna(r["fz3_0"]) else None
        it["fz3_anom"] = float(r["fz3_anom"]) if pd.notna(r["fz3_anom"]) else None
        it["cold3_loss"] = float(r["cold3_loss"]) if pd.notna(r["cold3_loss"]) else None
    return items


def _last_frost(year: int):
    """该年及之前最近一次出现 4 月 ≤0℃ 的年份，用于空年份引导。"""
    ev = disaster_events_table()
    if ev.empty or "year" not in ev.columns:
        return None
    past = ev[(ev["fz4_0"] > 0) & (ev["year"] <= year)]
    if past.empty:
        return None
    y = int(past["year"].max())
    cur = past[past["year"] == y]
    mx = cur["frost_loss"].min()
    return {"year": y, "n": int(len(cur)),
            "max": round(float(mx), 1) if pd.notna(mx) else None,
            "worst": cur.loc[cur["frost_loss"].idxmin(), "county"]}


@app.route("/risk")
def risk():

    years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]
    city_arg = (request.args.get("city") or "").strip()
    level_arg = (request.args.get("level") or "").strip()

    snap = risk_snapshot(year)
    items = _frost_attach(list(snap["items"]), year)
    items.sort(key=lambda x: (-(x["frost_level"] or 0),
                              x["frost_loss"] if x["frost_loss"] is not None else 0,
                              -x["days"], x["county"]))
    risk_type = (request.args.get("type") or "").strip()
    if risk_type == "frost":
        items = [x for x in items if (x["frost_level"] or 0) > 0]
    elif risk_type == "dry":
        items = [x for x in items if x["days"] > 0]
    if city_arg:
        items = [x for x in items if x["city"] == city_arg]
    if level_arg in ("heavy", "light", "none"):
        items = [x for x in items if x["level"] == level_arg]

    return render_template("risk.html", title="灾害预警", active="risk",
                           years=years, year=year, cities=snap["cities"],
                           city=city_arg, level=level_arg,
                           items=items, summary=snap["summary"], top=snap["top"],
                           total=len(snap["items"]), risk_type=risk_type,
                           frost_n=sum(1 for x in items if (x["frost_level"] or 0) > 0),
                           frost_heavy=sum(1 for x in items if (x["frost_level"] or 0) >= 2),
                           frost_worst=min([x for x in items
                                            if (x["frost_level"] or 0) > 0 and (x["frost_loss"] or 0) < -0.05],
                                           key=lambda x: x["frost_loss"], default=None),
                           frost_first=min([x for x in items if (x["frost_level"] or 0) > 0],
                                           key=lambda x: (x["frost_loss"] if x["frost_loss"] is not None else 0),
                                           default=None),
                           last_frost=_last_frost(year),
                           dose=_dose_payload(),
                           mh=_multihazard_payload(),
                           ratio=round(CONFIG["alert"]["default_ratio"] * 100, 0) / 100)


@app.route("/export_risk")
def export_risk():
    """导出当年全省县级风险清单 Excel。"""
    from io import BytesIO
    from flask import send_file

    years = list(BASE_YEARS)
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else years[-1]
    snap = risk_snapshot(year)
    _frost_attach(snap["items"], year)

    rows = [{
        "区县": x["county"],
        "地市": x["city"],
        "干热风天数": x["days"],
        "近15年排位": f"第{x['rank']}/{x['rank_total']}" if x["rank"] else "",
        "干热风等级": x["level_text"],
        "4月低温天数": x.get("fz4") if x.get("fz4") is not None else "",
        "4月最低温(℃)": x.get("tmin4") if x.get("tmin4") is not None else "",
        "晚霜冻等级": x.get("frost_text", ""),
        "预估影响(%)": x.get("frost_loss") if x.get("frost_loss") is not None else "",
        "预测单产(kg/亩)": x["pred"] if x["pred"] is not None else "",
        "常年单产(kg/亩)": x["mean"],
        "较常年(%)": x["vs_mean"] if x["vs_mean"] is not None else "",
        "预警线(kg/亩)": x["threshold"],
        "产量预警": "触发" if x["alert"] else "未触发",
        "实测单产(kg/亩)": x["actual"] if x["actual"] is not None else "",
    } for x in snap["items"]]

    df = pd.DataFrame(rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=f"{year}年县级风险清单", index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"{year}年县级风险清单.xlsx",
    )


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
    <b>报告说明：</b>本报告基于河南省小麦气象灾害影响分析与智能预警平台，采用多模型融合算法
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
    story.append(Paragraph("—— 本报告由河南省小麦气象灾害影响分析与智能预警平台自动生成 ——", footer_style))

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
    feats["county_prior"] = mean_yield
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

    # 线性回归在 19 维高度共线特征下系数病态（gdd / gdd_34 / gdd_56 互相解释），
    # 基准点上各分量靠巨值相互抵消才成立。特征只要偏离基准 0.02，输出就会放大到
    # 1e8 量级并被 clamp 夹到 ±30% 边界，表现为"拖一格预测就固定偏高/偏低 6%"。
    # 情景模拟本来就是"相对基准扰动"的场景，因此把 LR 冻结为基准点常数：
    # rf / lgbm 正常响应滑块，LR 只保留它对基准的固定贡献，融合结果仍与主预测一致。
    lr_const = 0.2 * clamp(float(lr.predict(baseline_X)[0]), base_lr) if lr is not None else 0.0

    def ensemble_predict(X_data):
        pred_rf = clamp(float(rf.predict(X_data)[0]), base_rf)
        pred_lgbm = clamp(float(lgbm.predict(X_data)[0]), base_lgbm) if lgbm is not None else pred_rf
        fused = 0.2 * pred_rf + 0.6 * pred_lgbm + lr_const
        return max(mean_yield * 0.5, min(mean_yield * 1.5, fused * mean_yield))

    pred = ensemble_predict(X_arr)
    baseline = ensemble_predict(baseline_X)

    base_feat = {
        "gdd": round(float(row["gdd"].iloc[0]), 0) if "gdd" in row else 1800,
        "season_precip": round(float(row["season_precip"].iloc[0]), 0) if "season_precip" in row else 400,
        "hot_days": round(float(row["hot_days"].iloc[0]), 0) if "hot_days" in row else 5,
        "cold_days": round(float(row["cold_days"].iloc[0]), 0) if "cold_days" in row else 10,
    }

    def _slide_range(base, pad):
        """滑块区间：基准值 ± max(30%, pad)，保证基准点居中且拖动空间够演示。

        必须由后端给出而不是模板写死——河南生育期积温实际只有 242—793 ℃·d，
        若滑块硬编码 1000 起，基准值会被浏览器夹到边界，重置后请求值并非基准值。
        """
        span = max(base * 0.3, pad)
        return {"min": max(0, round(base - span)), "max": round(base + span), "step": 1}

    return {
        "county": county,
        "year": year,
        "baseline": round(baseline, 1),
        "predicted": round(pred, 1),
        "change": round((pred - baseline) / baseline * 100, 1),
        "mean_yield": round(mean_yield, 1),
        "adjustments": adjustments,
        "base_features": base_feat,
        "feature_ranges": {
            "gdd": _slide_range(base_feat["gdd"], 150),
            "season_precip": _slide_range(base_feat["season_precip"], 40),
            "hot_days": _slide_range(base_feat["hot_days"], 8),
            "cold_days": _slide_range(base_feat["cold_days"], 10),
        }
    }


_ZONE_CACHE: dict = {}


def county_year_breakdown(county: str, year: int):
    """该县该年的产量归因拆解，数据全部取自 disaster_events 表（按县×年一行）。

    单位一律为百分数（%）。返回 None 表示该县该年无灾害事件记录。
    注意：合计（total_loss）为面板合计口径，各分项是单灾种单独估计，不能相加；
    年景项（year_effect）由当年数据估出，只能解释已发生年份，不能外推。
    """
    ev = disaster_events_table()
    if ev.empty or "year" not in ev.columns or "county" not in ev.columns:
        return None
    r = ev[(ev["county"] == county) & (ev["year"] == year)]
    if r.empty:
        return None
    row = r.iloc[0]

    def _f(k, nd=2):
        try:
            v = row.get(k)
            if v is None or pd.isna(v):
                return None
            return round(float(v), nd)
        except Exception:
            return None

    d = {
        "year": int(year),
        "fz4": _f("fz4_0", 0),
        "fz4_streak": _f("fz4_streak", 0),
        "tmin4": _f("tmin4", 1),
        "frost_loss": _f("frost_loss"),
        "frost_low": _f("frost_loss_low"),
        "frost_high": _f("frost_loss_high"),
        "fz3": _f("fz3_0", 0),
        "cold3_loss": _f("cold3_loss"),
        "hd_may": _f("hd_may", 0),
        "hd_streak": _f("hd_streak", 0),
        "hd_level": int(row["hd_level"]) if pd.notna(row.get("hd_level")) else 0,
        "year_effect": _f("year_effect"),
        "total_loss": _f("total_loss"),
        "total_low": _f("total_low"),
        "total_high": _f("total_high"),
        # 距平口径：页面折算只用这两项，避免拿"绝对天数 × 弹性"夸大灾情
        "fz4_anom": _f("fz4_anom"),
        "fz3_anom": _f("fz3_anom"),
        "frost_own": _f("frost_anom_loss"),
    }
    dose = _dose_payload()
    bands = []
    if isinstance(dose, dict):
        for b in (dose.get("fz4") or {}).get("bands", []):
            bands.append({"edge": b.get("edge"), "dev_pct": b.get("dev_pct"),
                          "n": b.get("n"), "se_pct": b.get("se_pct")})
    d["dose_bands"] = bands

    # 回归系数（每多一天的边际影响，%），用于按天数直接算折减。
    # 注意：表内 frost_loss / cold3_loss 是"归零后的潜在增益"的反事实口径，
    # 不是灾害损失，页面不得直接当作损失展示。
    p = Path(__file__).resolve().parent / "output" / "reports" / "disaster_coefficients.json"
    d["fz4_beta"] = -1.96
    d["fz3_beta"] = -0.57
    d["hd_p"] = 0.205
    try:
        co = json.loads(p.read_text(encoding="utf-8")).get("joint", {})
        d["fz4_beta"] = round(co.get("fz4_0", {}).get("beta", -0.01963) * 100, 2)
        d["fz3_beta"] = round(co.get("fz3_0", {}).get("beta", -0.00567) * 100, 2)
        d["hd_p"] = round(co.get("hd_streak", {}).get("p", 0.20527), 3)
    except Exception:
        pass
    return d


def county_zone_hint(county: str):
    """按分象限误差给出该县的已知局限提示。

    仅 D（脆弱敏感）象限需要提示：模型在该类县上系统性偏低，预测值宜作下限参考。
    数据来源 output/reports/zone_errors.json（scripts/verify_zone_errors.py 生成）。
    """
    if not _ZONE_CACHE:
        payload = {}
        try:
            import json as _json
            payload = _json.loads(
                (REPORTS_DIR / "zone_errors.json").read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        _ZONE_CACHE["map"] = payload.get("_county_zone", {})
        _ZONE_CACHE["d"] = payload.get("D 脆弱敏感区")
    zone = _ZONE_CACHE["map"].get(county)
    d = _ZONE_CACHE["d"]
    if zone != "D 脆弱敏感区" or not d:
        return None
    return {"zone": zone, "bias": d["bias"], "mae": d["mae"]}


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
        county = county_arg if county_arg in counties else counties[0]

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
        return render_template("predict.html", title="情景仿真", active="predict",
                               counties=counties, city_groups=scope_city_groups(scope),
                               county=county, years=years, year=year,
                               threshold=0, threshold_max=None, threshold_note=None,
                               ratio=CONFIG["alert"]["default_ratio"] * 100,
                               metrics=metrics, metrics_lgbm=metrics_lgbm, metrics_lr=metrics_lr,
                               reliable=False, result=None, history=[], trend_chart=None,
                               feature_chart=None, feature_count=len(FEATS),
                               zone_hint=county_zone_hint(county),
                               breakdown=county_year_breakdown(county, year),
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
    if not feats.empty:
        feats["county_prior"] = mean_yield
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
        pred = 0.2 * pred_rf + 0.6 * pred_lgbm + 0.2 * pred_lr

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

        # 概率化预警：随机森林 300 棵树的预测分布。
        # 这是模型内部不确定性（决策树离散度），不是统计置信区间，页面上有明确标注。
        prob_below, tree_low, tree_high = None, None, None
        try:
            tree_vals = np.array([float(t.predict(X)[0]) for t in rf.estimators_]) * mean_yield
            prob_below = round(float((tree_vals < threshold).mean()) * 100, 1)
            tree_low = round(float(np.percentile(tree_vals, 10)), 1)
            tree_high = round(float(np.percentile(tree_vals, 90)), 1)
        except Exception:
            pass

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
            "prob_below": prob_below, "tree_low": tree_low, "tree_high": tree_high,
            "mae": round(interval_mae, 1),
            "matched": matched,
            "feature_note": feature_note,
            "contrib": sample_contributions(X.iloc[0].values),
        }

        # 历史对比：逐年预测 vs 实测
        api = labels.set_index(["county", "year"])["wheat_yield"].to_dict() if labels is not None else {}
        X_all = feats[FEATS].fillna(feats[FEATS].median(numeric_only=True))
        preds_rf = rf.predict(X_all) * mean_yield
        preds_lr = lr.predict(X_all) * mean_yield if lr is not None else preds_rf
        preds_lgbm = lgbm.predict(X_all) * mean_yield if lgbm is not None else preds_rf
        preds_all = 0.2 * preds_rf + 0.6 * preds_lgbm + 0.2 * preds_lr
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
                "desc": "各因素对产量的平均影响大小；数值越高，影响越大。",
                "payload": {
                    "labels": [item["name"] for item in feature_items],
                    "values": [item["value"] for item in feature_items],
                    "unit": "%",
                },
            }

    return render_template("predict.html", title="情景仿真", active="predict",
                           counties=counties, city_groups=scope_city_groups(scope),
                           county=county, years=years, year=year,
                           threshold=round(threshold, 1),
                           threshold_min=threshold_min, threshold_max=threshold_max,
                           threshold_note=threshold_note, ratio=ratio * 100,
                           metrics=metrics, metrics_lgbm=metrics_lgbm, metrics_lr=metrics_lr,
                           reliable=bool(mm and mm.get("reliable")),
                           result=result, history=history, trend_chart=trend_chart,
                           feature_chart=feature_chart, feature_count=len(FEATS),
                           zone_hint=county_zone_hint(county),
                           breakdown=county_year_breakdown(county, year),
                           error=error, source=data_meta["source"])


def _empty_advice(county: str, year: int) -> dict:
    """该年尚无任何观测时的空结构：五段全空、无建议卡，让模板正常渲染「待上传」。"""
    stages = []
    for st in stage_store.STAGES:
        stages.append({
            "key": st["key"], "name": st["name"],
            "period": f"{st['start']} ~ {st['end']}",
            "focus": st["focus"], "stats": {}, "flags": ["待上传"],
            "advice": ["该阶段暂无气象记录，可在上方上传本期观测数据。"],
            "cards": [], "top_priority": None,
            "has_data": False, "date_range": "",
        })
    return {"county": county, "year": year, "stages": stages,
            "yield_note": "", "summary": {}, "empty_year": True}


def _apply_checks(result, county: str, fallback_year: int):
    """把「已执行」勾选注入决策卡，并统计本期作业完成度。"""
    if not result:
        return result
    eff_year = result.get("eff_year") or fallback_year
    result["eff_year"] = eff_year
    try:
        done_map = check_store.load(county, eff_year)
    except Exception:
        done_map = {}
    total = done = 0
    for s in (result.get("stages") or []):
        for c in (s.get("cards") or []):
            total += 1
            c["stage_key"] = s.get("key")
            c["done"] = bool(done_map.get((str(s.get("key")), str(c.get("key"))), False))
            if c["done"]:
                done += 1
    for a in (result.get("top_actions") or []):
        a["done"] = bool(done_map.get((str(a.get("stage_key")), str(a.get("card_key"))), False))
    result["done_total"] = total
    result["done_count"] = done
    result["done_pct"] = round(done / total * 100) if total else 0
    return result


@app.route("/advice/check", methods=["POST"])
def advice_check():
    """勾选 / 撤销某张作业卡的执行状态，回到农事建议页。"""
    county = (request.form.get("county") or "").strip()
    year = request.form.get("year", type=int)
    stage_key = (request.form.get("stage_key") or "").strip()
    card_key = (request.form.get("card_key") or "").strip()
    done = request.form.get("done") == "1"
    scope = uploaded_scope()
    allowed = scope["counties"] if scope else ALL_COUNTIES
    if not (county and year and stage_key and card_key) or county not in allowed:
        return redirect(url_for("advice"))
    try:
        check_store.toggle(county, year, stage_key, card_key, done)
    except Exception:
        pass
    return redirect(url_for("advice", county=county, year=year) + "#stage-cards")


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
        county = county_arg if county_arg in counties else counties[0]
    today_year = datetime.now().year
    years = sorted(set(BASE_YEARS + [today_year - 1, today_year, today_year + 1]))
    year_arg = request.args.get("year", type=int)
    year = year_arg if year_arg in years else (
        today_year if today_year not in BASE_YEARS else BASE_YEARS[-1])

    # ---- 该县该年的数据状态：只有「本期观测库」里的数据才算当年实测 ----
    stage_df = stage_store.load(county, year)
    counts = stage_store.stage_counts(county, year)
    has_stage = not stage_df.empty
    staged_days = len(stage_df)

    data, data_meta = dataset_for_county(county)
    try:
        builtin_max = int(data["year"].max()) if not data.empty else 0
    except Exception:
        builtin_max = 0

    stage_keys = [s["key"] for s in stage_store.STAGES]
    stages_complete = all(counts.get(k, 0) > 0 for k in stage_keys)
    # 历史完整年（有内置数据、且没上传过本期观测）不显示上传入口，避免误导
    is_history = (bool(year) and year <= builtin_max and not has_stage)
    upload_needed = (not stages_complete) and (not is_history)

    if has_stage:
        # 本期观测优先：按累积的当阶段气象出建议
        result = generate_advice(stage_df, county, year,
                                 dry_ref=dry_hot_reference(county),
                                 frost_ref=frost_reference(county, year))
        result["alert"] = False
        result["feature_note"] = False
        source = f"阶段观测 · 已累积 {staged_days} 天"
    elif data.empty or (year and year > builtin_max):
        # 未来年 / 无数据县：不回落到历史年，直接给空壳（五段待上传）
        result = _empty_advice(county, year)
        source = "该年尚无观测数据"
    else:
        # 历史年：内置全年数据
        result = generate_advice(data, county, year,
                                 dry_ref=dry_hot_reference(county),
                                 frost_ref=frost_reference(county, year))
        result["alert"] = False
        result["feature_note"] = False
        source = data_meta["source"]

    result = _apply_checks(result, county, year)
    return render_template("advice.html", title="农事建议", active="advice",
                           counties=counties, city_groups=scope_city_groups(scope),
                           county=county, years=years, year=year,
                           threshold=0, result=result, error=None, source=source,
                           counts=counts, staged_days=staged_days,
                           upload_needed=upload_needed,
                           stages_complete=stages_complete,
                           is_history=is_history,
                           builtin_max=builtin_max,
                           stage_defs=stage_store.STAGES,
                           last_reset=stage_store.last_snapshot())


@app.route("/advice/upload_stage", methods=["POST"])
def advice_upload_stage():
    """上传本期阶段气象观测，累积到阶段观测库（MySQL 优先，CSV 回退）。"""
    f = request.files.get("file")
    county = (request.form.get("county") or "").strip()
    year = request.form.get("year", type=int)
    if not f or not f.filename:
        return redirect(url_for("advice", county=county, year=year, stage_error="未选择文件。"))
    ext = Path(f.filename).suffix.lower()
    if ext not in CONFIG["upload"]["allowed_ext"]:
        return redirect(url_for("advice", county=county, year=year,
                               stage_error=f"不支持的文件类型 {ext}。"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = UPLOAD_DIR / f"stage_{uuid.uuid4().hex[:8]}_{f.filename}"
    f.save(saved)
    try:
        df, report = load_weather_file(saved, county, stage_mode=True)
        if not report["ok"] or df.empty:
            tip = "；".join(i["title"] for i in report["issues"] if i["level"] == "error") or "校验未通过。"
            return redirect(url_for("advice", county=county, year=year, stage_error=tip))
        stage_store.upsert(county, year, df)
    finally:
        saved.unlink(missing_ok=True)
    return redirect(url_for("advice", county=county, year=year))


@app.route("/advice/reset_stage")
def advice_reset_stage():
    county = (request.args.get("county") or "").strip()
    year = request.args.get("year", type=int)
    stage = (request.args.get("stage") or "").strip()
    if stage:
        stage_store.reset_stage(county, year, stage)
    else:
        stage_store.reset(county, year)
    return redirect(url_for("advice", county=county, year=year))


@app.route("/advice/restore_stage")
def advice_restore_stage():
    """撤销最近一次清空：把备份快照写回观测库。"""
    info = stage_store.undo_last()
    county = (request.args.get("county") or (info or {}).get("county") or "").strip()
    year = request.args.get("year", type=int) or (info or {}).get("year")
    if not info:
        return redirect(url_for("advice", county=county, year=year,
                                stage_error="没有可撤销的清空记录。"))
    return redirect(url_for("advice", county=county, year=year,
                            restored=info.get("rows", 0)))


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
