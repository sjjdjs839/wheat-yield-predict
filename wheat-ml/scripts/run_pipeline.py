"""
一键跑全流程脚本
==================
顺序：数据清洗 → 特征工程 → 模型训练 → 导出浏览器端模型 → 启动 Web 应用

用法:
  cd wheat-ml
  python -m scripts.run_pipeline          # 只跑数据处理与训练
  python -m scripts.run_pipeline --serve  # 跑完后直接启动 Web 应用
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, module_path: str, fn_name: str) -> bool:
    t0 = time.time()
    print(f"\n{'='*50}\n▶ [{name}]\n{'='*50}")
    try:
        import importlib
        mod = importlib.import_module(module_path)
        getattr(mod, fn_name)()
        print(f"✓ [{name}] 完成，耗时 {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"✗ [{name}] 失败，耗时 {time.time() - t0:.1f}s")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    steps = [
        ("数据清洗",       "src.data.clean",            "run_clean"),
        ("特征工程",       "src.data.build_features",   "run_build"),
        ("模型训练",       "src.models.train_model",    "run_train"),
        ("导出模型",       "src.models.export_model",   "export"),
    ]

    print("🌾 河南省小麦气象数据分析与产量预测系统 —— 全流程")
    print(f"项目根目录: {PROJECT_ROOT}")

    results = [(name, run_step(name, mod, fn)) for name, mod, fn in steps]

    print(f"\n{'='*50}\n📋 全流程总结\n{'='*50}")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(steps)} 步完成")

    if passed == len(steps):
        if "--serve" in sys.argv:
            print("\n启动 Web 应用 ...")
            import app
            app.app.run(host=app.CONFIG["web"]["host"], port=app.CONFIG["web"]["port"])
        else:
            print("\n启动 Web 应用：python app.py  →  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
