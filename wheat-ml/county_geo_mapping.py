# -*- coding: utf-8 -*-
"""区县名映射：数据里的名字 -> GeoJSON里的标准名"""

# 数据名 -> GeoJSON名
COUNTY_TO_GEO = {
    # 郑州市
    "中牟": "中牟县",
    "巩义": "巩义市",
    "荥阳": "荥阳市",
    "新密": "新密市",
    "新郑": "新郑市",
    "登封": "登封市",
    # 开封市
    "兰考": "兰考县",
    "尉氏": "尉氏县",
    "通许": "通许县",
    # 洛阳市
    "孟津": "孟津区",
    "新安": "新安县",
    "栾川": "栾川县",
    "汝阳": "汝阳县",
    "宜阳": "宜阳县",
    "洛宁": "洛宁县",
    "伊川": "伊川县",
    "偃师": "偃师区",
    "瀍河区": "瀍河回族区",
    # 三门峡市
    "渑池": "渑池县",
    "卢氏": "卢氏县",
    # 周口市
    "淮阳": "淮阳区",
    "淮阳县": "淮阳区",
    "扶沟": "扶沟县",
    "扶沟县": "扶沟县",
}

# 没有GeoJSON对应的（跳过，不显示）
NO_GEO = {"示范区"}


def to_geo_name(county: str) -> str | None:
    """把数据里的区县名转成GeoJSON里的标准名，返回None表示不显示"""
    if county in NO_GEO:
        return None
    return COUNTY_TO_GEO.get(county, county)


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    y = pd.read_csv(Path(__file__).resolve().parent / "data" / "raw" / "county_yield.csv")
    names = sorted(y["county"].unique())
    matched = 0
    skipped = 0
    for n in names:
        geo = to_geo_name(n)
        if geo:
            matched += 1
        else:
            skipped += 1
    print(f"能显示: {matched} 个区县")
    print(f"跳过: {skipped} 个区县（无GeoJSON对应）")
