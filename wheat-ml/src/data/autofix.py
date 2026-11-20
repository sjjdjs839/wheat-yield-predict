"""上传文件的宽容解析层。

用户导出的气象表常常不规范：列名带单位、日期格式混、编码乱、表头前面有
标题行、数值带着 ℃ 和百分号。这里尽量把它们修回标准口径，同时保住一条底线
——只有确实像逐日气象数据的表才放行，别的表不能硬修成气象表。
"""

import re

import pandas as pd

# ---------------- 文本与编码 ----------------

ENC_ORDER = ("utf-8-sig", "utf-8", "gb18030", "gbk")

_TRAD = {"溫": "温", "氣": "气", "濕": "湿", "風": "风", "輻": "辐", "陽": "阳",
         "點": "点", "時": "时", "間": "间", "數": "数", "據": "据", "區": "区",
         "縣": "县", "測": "测", "記": "记", "錄": "录", "範": "范", "圍": "围",
         "雲": "云", "壓": "压", "濃": "浓", "霧": "雾", "陰": "阴"}

_FULLWIDTH = str.maketrans("０１２３４５６７８９．，％（）：；", "0123456789.,%():;")

_NOISE_UNIT = re.compile(r"单位[:：].*$")

# 表头行识别用的常见词（autofix 单独使用时的兜底，validate 会传更全的表）
HEADER_WORDS = ("日期", "年份", "年", "日", "月", "最高", "最低", "平均", "气温",
                "温度", "降水", "降雨", "湿度", "风速", "辐射", "露点", "date",
                "year", "day", "tmax", "tmin", "temp", "precip", "rain", "rh",
                "wind", "solar", "dew")


def decode_text(raw: bytes) -> tuple:
    """返回 (文本, 实际编码)。"""

    for enc in ENC_ORDER:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8 或 GBK 编码的 CSV")


def norm_col(col) -> str:
    """列名归一化：繁转简、全角转半角、去括号单位、去空白与符号。"""

    s = str(col).replace("\ufeff", "").strip()
    s = s.translate(_FULLWIDTH)
    s = "".join(_TRAD.get(ch, ch) for ch in s)
    s = _NOISE_UNIT.sub("", s)
    s = re.sub(r"[（(][^）)]*[）)]", "", s)          # 去掉 (℃) (mm) 之类的单位
    s = re.sub(r"[℃%‰·\s\"'`]", "", s)
    s = s.replace("*", "").replace("＊", "")
    return s.lower()


# ---------------- 分隔符与表头 ----------------

_CANDIDATE_SEPS = [",", "\t", ";", "|"]


def sniff_sep(lines: list) -> str:
    """按列数一致性挑分隔符，挑不出就退回逗号。"""

    best, best_cols = ",", 0
    for sep in _CANDIDATE_SEPS:
        counts = [len(l.split(sep)) for l in lines[:6] if l.strip()]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        consistent = counts.count(mode) >= max(1, len(counts) - 1)
        if consistent and mode >= 2 and mode > best_cols:
            best, best_cols = sep, mode
    return best


def find_header_row(lines: list, sep: str, is_alias=None) -> int:
    """在前 30 行里挑最像表头的一行；都不像就退回第一行。"""

    if is_alias is None:
        def is_alias(cell):
            n = norm_col(cell)
            return any(w in n for w in HEADER_WORDS)

    best_idx, best_score = 0, -1
    for i, line in enumerate(lines[:30]):
        if not line.strip():
            continue
        cells = line.split(sep)
        if len(cells) < 2:
            continue
        score = sum(1 for c in cells if is_alias(c))
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx if best_score >= 2 else 0


def read_table(path, is_alias=None):
    """读 CSV / Excel，返回 (DataFrame, 说明列表)。表头前的标题行会被跳过。"""

    import io

    notes = []
    ext = str(path.suffix).lower()

    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        for name, one in sheets.items():
            if one is not None and len(one.dropna(how="all")) > 0:
                if one.columns.duplicated().any():
                    notes.append("Excel 有重复列名，已自动加后缀")
                if len(sheets) > 1:
                    notes.append("工作簿有 %d 个表，取第一个有数据的「%s」"
                                 % (len(sheets), name))
                return one, notes
        raise ValueError("Excel 里没有找到任何有数据的工作表")

    text, enc = decode_text(path.read_bytes())
    if enc not in ("utf-8-sig", "utf-8"):
        notes.append("文件编码是 %s，已自动按该编码读取" % enc.upper())

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("文件是空的，没有任何内容")

    sep = sniff_sep(lines)
    header_idx = find_header_row(lines, sep, is_alias)
    if header_idx > 0:
        notes.append("表头在第 %d 行，前面 %d 行标题已跳过"
                     % (header_idx + 1, header_idx))

    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), sep=sep,
                     skipinitialspace=True, on_bad_lines="skip")
    df.columns = [c for c in df.columns]
    return df, notes


# ---------------- 数值 ----------------

_MISSING = {"", "na", "n/a", "nan", "null", "none", "-", "—", "‐", "--",
            "-999", "-9999", "999", "9999", "缺测", "无", "空"}

_NEG_CHARS = {"−": "-", "–": "-", "—": "-", "﹣": "-", "－": "-"}


def clean_numeric(series) -> pd.Series:
    """把 «−1.8℃» «3.2 mm» «45%» «1,234» 这类值修成纯数值。"""

    if series is None:
        return None
    s = series.astype("string").str.strip().str.translate(_FULLWIDTH)
    for src, dst in _NEG_CHARS.items():
        s = s.str.replace(src, dst, regex=False)
    low = s.str.lower()
    s = s.where(~low.isin(_MISSING))
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d.\-+eE]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


# ---------------- 日期 ----------------

_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
                 "%Y年%m月%d日", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                 "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]

_MONTH_DAY = re.compile(r"^(\d{1,2})[-/月](\d{1,2})日?$")


def _excel_serial(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return pd.NaT
    if 20000 <= f <= 80000:          # 1954—2119 之间的序列号
        return pd.to_datetime(f, unit="D", origin="1899-12-30")
    return pd.NaT


def parse_dates(series, year_series=None) -> tuple:
    """返回 (日期序列, 说明)。

    认标准格式、多种分隔符、八位数字、中文年月日、带时间戳、Excel 序列号，
    以及「4/17」「4月17日」这种只有月日的写法——年份优先取表里的年份列，
    没有就取本表其它行已解析出的年份，且只在年份唯一时才补。
    """

    s = series.astype("string").str.strip()
    d = pd.to_datetime(s, errors="coerce")
    if float(d.notna().mean()) >= 0.9:
        return d, ""

    out, md_pos, md_val = [], [], []
    for pos, v in enumerate(s.tolist()):
        if v is None or str(v).strip() == "":
            out.append(pd.NaT)
            continue
        v = str(v).strip()
        parsed = pd.NaT
        for fmt in _DATE_FORMATS:
            try:
                parsed = pd.to_datetime(v, format=fmt)
                break
            except (ValueError, TypeError):
                continue
        if pd.isna(parsed):
            parsed = _excel_serial(v)
        if pd.isna(parsed):
            md = _MONTH_DAY.match(v)
            if md:
                md_pos.append(pos)
                md_val.append((int(md.group(1)), int(md.group(2))))
        out.append(parsed)

    result = pd.Series(out, index=series.index).astype("datetime64[ns]")
    note = ""

    if md_pos:
        years = None
        if year_series is not None:
            try:
                got = {int(float(y)) for y in year_series.dropna().unique()}
                years = got or None
            except (TypeError, ValueError):
                years = None
        if not years:
            got = {int(x.year) for x in result.dropna()}
            years = got or None
        if years and len(years) == 1:
            y = years.pop()
            for pos, (m, dd) in zip(md_pos, md_val):
                try:
                    result.iloc[pos] = pd.to_datetime(
                        "%d-%02d-%02d" % (y, m, dd), errors="coerce")
                except (ValueError, TypeError):
                    continue
            note = "%d 行只写了月日，已按本表年份 %d 补齐" % (len(md_pos), y)
        else:
            note = "%d 行只写了月日，表里推不出唯一年份，未补" % len(md_pos)

    return result, note


# ---------------- 底线：这到底是不是气象数据 ----------------

TEMP_MIN_PLAUSIBLE = -45.0
TEMP_MAX_PLAUSIBLE = 50.0


def guard(df: pd.DataFrame) -> tuple:
    """判是不是逐日气象数据。返回 (是否放行, 拒收原因)。

    修得再狠，这条底线不放：日期成不了、温度不成数、温度大面积超出物理范围，
    都说明这不是气象表。
    """

    if "date" not in df.columns:
        return False, "表里找不到日期列，无法按日对齐。"
    n = len(df)
    if n == 0:
        return False, "文件里没有任何数据行。"
    date_ok = float(df["date"].notna().mean())
    if date_ok < 0.8:
        return False, ("日期只有 %.0f%% 能解析成日期，看起来不是逐日气象数据。"
                       % (date_ok * 100))

    temps = [c for c in ("t2m_max", "t2m_min", "t2m") if c in df.columns]
    if not temps:
        return False, "找不到最高温/最低温/平均温任何一列，不像是气象数据。"
    if max(float(df[c].notna().mean()) for c in temps) < 0.8:
        return False, "温度列几乎都读不成数字，不像是气象数据。"

    names = {"t2m_max": "最高温", "t2m_min": "最低温", "t2m": "平均温"}
    for c in temps:
        v = df[c].dropna()
        if not len(v):
            continue
        out = float(((v < TEMP_MIN_PLAUSIBLE) | (v > TEMP_MAX_PLAUSIBLE)).mean())
        if out > 0.3:
            return False, ("%s 有 %.0f%% 的值超出 %d—%d℃，多半不是气温列或单位不对。"
                           % (names[c], out * 100,
                              TEMP_MIN_PLAUSIBLE, TEMP_MAX_PLAUSIBLE))
    return True, ""
