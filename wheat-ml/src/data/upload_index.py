"""上传数据集的清单。

一个文件一条记录，不再互相顶替。字段包括县名、时间范围、行数、
类型（年度 / 当期观测）和启用状态，供各业务页按县取数。

类型是按日期跨度自动判的，不靠用户选：
  收尾日期距今不远、跨度也不长 → 当期观测（进预警和农事建议）
  否则 → 年度数据集（进分析和预测）

存储先走 web/uploads/index.json，读写失败就当没有上传数据。
"""

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"
INDEX_PATH = UPLOAD_DIR / "index.json"
LEGACY_PATH = UPLOAD_DIR / "latest.json"

# 当期观测的判定阈值
RECENT_GAP_DAYS = 30      # 收尾日期距今不超过 30 天
RECENT_SPAN_DAYS = 100    # 覆盖跨度不超过 100 天

KIND_ANNUAL = "annual"
KIND_RECENT = "recent"
KIND_LABEL = {KIND_ANNUAL: "年度数据集", KIND_RECENT: "当期观测"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read() -> dict:
    if not INDEX_PATH.exists():
        return {"datasets": []}
    try:
        obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or not isinstance(obj.get("datasets"), list):
            return {"datasets": []}
        return obj
    except Exception:
        return {"datasets": []}


def _write(obj: dict) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def classify(start: date | None, end: date | None, today: date | None = None) -> str:
    """按时间跨度判定数据集类型。日期缺失时按年度处理。"""
    if not start or not end:
        return KIND_ANNUAL
    today = today or date.today()
    span = (end - start).days + 1
    gap = (today - end).days
    if gap <= RECENT_GAP_DAYS and span <= RECENT_SPAN_DAYS:
        return KIND_RECENT
    return KIND_ANNUAL


def add(*, rel_path: str, filename: str, county: str, df,
        force_kind: str | None = None) -> dict:
    """登记一份上传数据；df 需带 date 列。

    force_kind 给定时按它定类型：数据管理员知道这批是留档还是当前在盯的，
    自动判定只是省事，不该覆盖人的判断。
    """
    dates = None
    if df is not None and not df.empty and "date" in df.columns:
        try:
            dates = list(df["date"].dropna())
        except Exception:
            dates = None
    start = min(dates).date() if dates else None
    end = max(dates).date() if dates else None
    years = sorted({int(y) for y in df["year"].dropna().unique()}) if (
        df is not None and not df.empty and "year" in df.columns) else []

    rec = {
        "id": uuid.uuid4().hex[:12],
        "path": rel_path,
        "filename": filename,
        "county": county,
        "kind": force_kind if force_kind in (KIND_ANNUAL, KIND_RECENT) else classify(start, end),
        "years": years,
        "rows": int(len(df)) if df is not None else 0,
        "start": start.strftime("%Y-%m-%d") if start else "",
        "end": end.strftime("%Y-%m-%d") if end else "",
        "uploaded_at": _now(),
        "active": True,
    }
    obj = _read()
    obj["datasets"].append(rec)
    _write(obj)
    return rec


def migrate_legacy() -> None:
    """把旧的单槽 latest.json 升级成一条清单记录，升级后删掉旧文件。"""
    if not LEGACY_PATH.exists():
        return
    try:
        meta = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    except Exception:
        LEGACY_PATH.unlink(missing_ok=True)
        return
    county = (meta.get("county") or "").strip()
    path = PROJECT_ROOT / meta.get("path", "")
    if county and path.exists():
        obj = _read()
        if not any(d.get("path") == meta.get("path") for d in obj["datasets"]):
            obj["datasets"].append({
                "id": uuid.uuid4().hex[:12],
                "path": meta.get("path"),
                "filename": meta.get("filename", path.name),
                "county": county,
                "kind": KIND_ANNUAL,
                "years": [],
                "rows": int((meta.get("summary") or {}).get("rows") or 0),
                "start": "",
                "end": "",
                "uploaded_at": _now(),
                "active": True,
            })
            _write(obj)
    LEGACY_PATH.unlink(missing_ok=True)


def all(active_only: bool = True) -> list[dict]:
    """清单里的记录，缺文件的顺手跳过。"""
    migrate_legacy()
    out = []
    for rec in _read()["datasets"]:
        if active_only and not rec.get("active", True):
            continue
        if not (PROJECT_ROOT / rec.get("path", "")).exists():
            continue
        out.append(rec)
    return out


def counties(active_only: bool = True) -> list[str]:
    return sorted({r["county"] for r in all(active_only) if r.get("county")})


def by_county(county: str, active_only: bool = True) -> list[dict]:
    return [r for r in all(active_only) if r.get("county") == county]


def latest_record() -> dict | None:
    """最后登记的那一条。

    不按 uploaded_at 排序：同一批多份文件是同一秒入库的，秒级时间戳分不出先后，
    只有清单里的追加顺序能表达「这批里谁最后传的」。
    """
    recs = [r for r in all() if r.get("county")]
    return recs[-1] if recs else None


def latest_county() -> str:
    rec = latest_record()
    return rec["county"] if rec else ""


def years_for(county: str) -> list[int]:
    ys = set()
    for r in by_county(county):
        ys.update(int(y) for y in (r.get("years") or []))
    return sorted(ys)


def total_rows(active_only: bool = True) -> int:
    return sum(int(r.get("rows") or 0) for r in all(active_only))


def find(ds_id: str) -> dict | None:
    for r in _read()["datasets"]:
        if r["id"] == ds_id:
            return r
    return None


def remove(ds_id: str) -> dict | None:
    """删除一条记录并清掉对应文件；返回被删的记录。"""
    obj = _read()
    keep, hit = [], None
    for r in obj["datasets"]:
        if r["id"] == ds_id:
            hit = r
        else:
            keep.append(r)
    if hit:
        obj["datasets"] = keep
        _write(obj)
        p = PROJECT_ROOT / hit.get("path", "")
        if p.exists() and p.parent == UPLOAD_DIR:
            p.unlink(missing_ok=True)
    return hit


def toggle(ds_id: str) -> dict | None:
    obj = _read()
    hit = None
    for r in obj["datasets"]:
        if r["id"] == ds_id:
            r["active"] = not r.get("active", True)
            hit = r
    if hit:
        _write(obj)
    return hit


def deactivate_all() -> int:
    """全部停用，等价于回到内置库，但记录和文件都留着。"""
    obj = _read()
    n = 0
    for r in obj["datasets"]:
        if r.get("active", True):
            r["active"] = False
            n += 1
    _write(obj)
    return n


def prune_missing() -> int:
    """清掉指向已不存在文件的记录。"""
    obj = _read()
    keep = [r for r in obj["datasets"] if (PROJECT_ROOT / r.get("path", "")).exists()]
    n = len(obj["datasets"]) - len(keep)
    if n:
        obj["datasets"] = keep
        _write(obj)
    return n
