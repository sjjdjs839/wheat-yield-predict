"""
模型导出模块
==============
把训练好的随机森林转成 JSON 树结构，供 HTML 大屏在浏览器端直接推理。
这样答辩现场不需要起服务、不需要联网。

输出: output/dashboard/model.json
"""

import json
import joblib
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MODELS = PROJECT_ROOT / CONFIG["paths"]["models"]
DASH = PROJECT_ROOT / CONFIG["paths"]["dashboard"]


def export() -> Path:
    model_path = MODELS / "model_rf.joblib"
    stats_path = MODELS / "county_stats.json"
    if not model_path.exists():
        raise FileNotFoundError(f"缺少模型 {model_path}，请先运行 src.models.train_model")

    model = joblib.load(model_path)
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "cl": t.children_left.tolist(),
            "cr": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "t": t.threshold.round(6).tolist(),
            "v": t.value.ravel().round(6).tolist(),
        })

    payload = {
        "features": CONFIG["feature_cols"],
        "county_stats": json.loads(stats_path.read_text(encoding="utf-8")),
        "trees": trees,
    }

    DASH.mkdir(parents=True, exist_ok=True)
    out = DASH / "model.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"模型已导出 -> {out}  ({out.stat().st_size/1024:.0f} KB, {len(trees)} 棵树)")
    return out


if __name__ == "__main__":
    export()
