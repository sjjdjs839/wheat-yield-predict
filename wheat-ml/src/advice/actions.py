"""
农事决策卡（作业清单）
======================
把「生育期气象统计」翻译成可以直接排产的动作清单，回答四个问题：
  1. 依据是什么（客观气象数据，带数字）
  2. 还有多少时间（阶段剩余天数）
  3. 先做哪一件（P0 立即 / P1 本周 / P2 常规）
  4. 不做会怎样（方向性后果）

边界（刻意的克制）：
  - 不给施肥量、灌溉量、药剂浓度 —— 缺少土壤、品种、地块尺度数据，编出来就是假处方
  - 不承诺「能挽回 X% 产量」 —— 县域尺度气象与单产的统计关联较弱，见 reports/_legacy_note.md
  - 历史参考只给样本量与区间，并标注仅供参考
"""

from datetime import date

P0, P1, P2 = "P0", "P1", "P2"
PRIORITY_LABEL = {P0: "立即处理", P1: "本周安排", P2: "常规关注"}
_ORDER = {P0: 0, P1: 1, P2: 2}


def _card(priority, title, basis, actions, skip, window, evidence=None):
    # 阶段已经结束的（历史年份或当年已过窗口），不要再喊“立即处理/本周安排”
    label = PRIORITY_LABEL[priority]
    if "结束" in window and priority != P2:
        label = "复盘 · 阶段已结束"
    return {
        "priority": priority,
        "priority_label": label,
        "title": title,
        "basis": basis,
        "actions": actions,
        "skip": skip,
        "window": window,
        "evidence": evidence,
    }


def _window(period_end: str, today_md: str, year, cur_year: int, name: str) -> str:
    """阶段剩余时间；历史年份只做复盘提示，不给紧迫度。"""
    if year and year < cur_year:
        return f"{year} 年历史复盘，该阶段已于 {period_end} 结束"
    if year and year > cur_year:
        return f"计划年份，{name}自 {period_end} 前推，尚未开始"
    if today_md > period_end:
        return f"该阶段已于 {period_end} 结束"
    mm, dd = int(period_end[:2]), int(period_end[3:])
    try:
        end_d = date(cur_year, mm, dd)
        today_d = date(cur_year, int(today_md[:2]), int(today_md[3:]))
        left = (end_d - today_d).days
    except (ValueError, TypeError):
        return f"本阶段持续到 {period_end}"
    if left <= 0:
        return f"已在 {period_end} 收尾，进入下一阶段"
    return f"距本阶段结束还有 {left} 天，建议在 {min(left, 7)} 天内安排"


def _dry_basis(stats: dict, climate: dict) -> list[str]:
    out = []
    if stats.get("干热风日数"):
        out.append(f"灌浆期干热风 {stats['干热风日数']} 天（日最高温 ≥35℃ 且相对湿度 ≤40%）")
    if stats.get("高温日数", 0):
        out.append(f"35℃ 以上高温 {stats['高温日数']} 天")
    cp = climate.get("precip")
    if cp and stats.get("累计降水") is not None:
        diff = (stats["累计降水"] - cp) / cp * 100
        word = "少" if diff < 0 else "多"
        out.append(f"阶段累计降水 {stats['累计降水']} mm，比常年同期{word} {abs(diff):.0f}%")
    return out


def build_cards(key: str, stats: dict, climate: dict, period_end: str,
                today_md: str, year, cur_year: int, view: str = "",
                dry_ref: dict | None = None, frost_ref: dict | None = None) -> list[dict]:
    """按生育期生成决策卡，按优先级排序；无风险时给一张 P2 常规卡。"""
    if not stats:
        return []
    win = _window(period_end, today_md, year, cur_year, view)
    cards: list[dict] = []
    precip = stats.get("累计降水", 0)
    cp = climate.get("precip") or 0
    short = (precip < cp * 0.6) if cp else False
    fr = frost_ref or {}
    fz3a = fr.get("fz3_anom")
    fz4 = fr.get("fz4")

    if key == "greening":
        if short:
            cards.append(_card(
                P1, "先查墒情，再决定浇不浇",
                [f"阶段累计降水 {precip} mm，比常年同期少 {(1 - precip / cp) * 100:.0f}%"
                 if cp else f"阶段累计降水 {precip} mm",
                 f"下雨天数仅 {stats['降水日数']} 天（阶段共 {stats['days']} 天）"],
                ["挖开 5—10 cm 土层看干湿，干土层超过 3 cm 再浇",
                 "小水慢浇、不漫灌，日均气温稳定过 3℃ 后再浇",
                 "返青期根系浅，一次浇不透就分两次"],
                "返青水不足会减少有效分蘖，后期成穗数偏低，且错过难以补回",
                win))
        if (fz3a is not None and fz3a >= 1.5) or (frost_ref is None and stats["霜冻日数"] >= 8):
            basis = []
            if fr:
                basis.append(f"3 月最低温 ≤0℃ 共 {int(fr.get('fz3') or 0)} 天，"
                             f"比本县常年多 {fz3a:.1f} 天")
                basis.append("县级面板回归：3 月低温每多 1 天，单产相对常年下降 0.57%（p<0.001）")
                if fr.get("cold3_loss") is not None:
                    basis.append(f"按本县距平折算，预计影响 {fr['cold3_loss']:.1f}%")
            else:
                basis.append(f"最低气温低于 0℃ 共 {stats['霜冻日数']} 天")
                basis.append(f"阶段平均最低温 {stats['平均最低温']}℃")
            cards.append(_card(
                P0, "3 月低温偏多，防倒春寒伤苗",
                basis,
                ["寒潮来临前 1—2 天小水补墒提温，别等冻了再浇",
                 "有条件的覆盖秸秆或无纺布，重点护弱苗地块",
                 "寒后 3—5 天查苗，缺苗断垄的及时补种"],
                "返青期分蘖正在形成，低温会减少有效分蘖，后期成穗数补不回来",
                win))

    elif key == "jointing":
        if precip < 30:
            cards.append(_card(
                P1, "追肥和浇水一起安排",
                [f"阶段累计降水 {precip} mm，墒情偏干",
                 f"35℃ 以上高温 {stats['高温日数']} 天，蒸发快"],
                ["结合苗情追肥，让肥跟着水走，提高利用率",
                 "浇透不积水，低洼地块注意排涝",
                 "弱苗、脱肥地块优先安排"],
                "拔节期缺水缺肥直接减少小穗数，是决定穗粒数的关键窗口",
                win))
        if (fz4 is not None and fz4 >= 1) or (frost_ref is None and stats["霜冻日数"] >= 2):
            basis = []
            ev = None
            if fr:
                t = fr.get("tmin4")
                basis.append(f"4 月最低温 ≤0℃ 共 {int(fz4)} 天"
                             + (f"，最低 {t:.1f}℃" if t is not None else ""))
                basis.append("双向固定效应面板回归：4 月低温每多 1 天，"
                             "单产相对常年下降 1.96%（95%CI −2.55%~−1.38%，p<0.001）")
                fl = fr.get("frost_loss")
                if fl is not None and fl < 0:
                    lo, hi = fr.get("frost_low"), fr.get("frost_high")
                    own, ye = fr.get("frost_own"), fr.get("year_eff")
                    line = (f"相对该县常年预计影响 {fl:.2f}%"
                            + (f"（区间 {lo:.2f}%~{hi:.2f}%）" if lo is not None and hi is not None else ""))
                    if own is not None and ye is not None:
                        line += f"，其中全省年景 {ye:.2f}%、本县偏冷再叠加 {own:.2f}%"
                    basis.append(line)
                    b4 = fr.get("base4")
                    if b4 is not None:
                        gap = max(0.0, float(fz4) - float(b4))
                        if gap > 0.01:
                            basis.append(
                                f"防灾空间：本县比常年多 {gap:.1f} 天低温暴露，"
                                f"若用覆盖、冬灌、熏烟这类措施把超出常年的部分压回去，"
                                f"按同一弹性折算可少损失约 {gap * 1.96:.1f} 个百分点"
                            )
                        else:
                            basis.append(
                                "防灾空间：本县 4 月低温没有超出常年水平，"
                                "减产主要来自全省年景，本地可腾挪的空间有限"
                            )
                    basis.append(
                        "折算是「暴露天数—单产损失」的模型换算，不是措施的实测挽回率；"
                        "具体能抢回多少要以当地品种与管理条件下的试验为准"
                    )
                ev = ("4 月低温是本项目面板回归中唯一稳定的减产因子（干热风六种口径均不显著）。"
                      "该系数为县域平均弹性，地力、品种与灌溉条件不同的地块实际损失会有差异。")
            else:
                basis.append(f"最低气温低于 0℃ 共 {stats['霜冻日数']} 天")
            cards.append(_card(
                P0, "防晚霜冻伤穗",
                basis,
                ["盯 4 月寒潮预警，寒前 1—2 天浇水提温，别等冻了再浇",
                 "受冻地块及时补肥促恢复，别急着毁种",
                 "霜后 3—5 天剥穗查看幼穗受害程度，分级判断是否补种"],
                "拔节后幼穗离地面近，4 月霜冻每多 1 天对应约 2% 的单产损失，小穗一旦退化后期无法补救",
                win, evidence=ev))
        if stats["高温日数"] >= 3:
            cards.append(_card(
                P2, "高温天保持土壤湿润",
                [f"35℃ 以上高温 {stats['高温日数']} 天"],
                ["避开中午高温时段浇水，减少蒸发损失"],
                "高温叠加干旱会加速土壤失墒，加重后期水分压力",
                win))

    elif key == "heading":
        # 抽穗扬花期遇低温没有独立回归系数，沿用 4 月低温那条弹性作方向参考
        if stats.get("霜冻日数", 0) >= 1:
            basis = [f"本段最低温 ≤0℃ 共 {stats['霜冻日数']} 天",
                     f"阶段平均最低温 {stats['平均最低温']}℃"]
            if fz4 is not None:
                basis.append(f"4 月最低温 ≤0℃ 共 {int(fz4)} 天（与拔节期同一条面板回归口径，每多 1 天约 −1.96%）")
            cards.append(_card(
                P0 if stats["霜冻日数"] >= 2 else P1,
                "抽穗期再遇低温，护住花粉和幼穗",
                basis,
                ["寒前 1—2 天小水补墒提温，别等冻了再浇",
                 "霜后 3—5 天剥穗查看，幼穗变褐的按受害程度分级评估",
                 "叶面喷施磷酸二氢钾等抗逆调节剂帮助恢复，不追施速效氮肥"],
                "抽穗扬花期低温会降低花粉活力、增加不孕小穗，这段受损直接落在穗粒数上",
                win))
        if stats.get("平均相对湿度", 0) >= 80 or precip >= 40:
            cards.append(_card(
                P0, "花期高湿，优先防赤霉病",
                [f"阶段平均相对湿度 {stats.get('平均相对湿度', 0)}%",
                 f"下雨天数 {stats['降水日数']} 天，累计降水 {precip} mm"],
                ["雨前或雨隙按当地植保部门推荐的药剂预防，扬花期避开上午开花高峰喷药",
                 "雨后及时排水，降低田间湿度",
                 "收割前按安全间隔期停药"],
                "花期遇连阴高湿是赤霉病的主要诱因，一旦流行既减产又降品质，籽粒毒素超标会影响收购",
                win))
        if precip < 10:
            cards.append(_card(
                P1, "防干旱影响授粉",
                [f"阶段累计降水仅 {precip} mm"],
                ["小水补墒，保持田间湿润", "避开中午高温时段灌溉"],
                "抽穗扬花期缺水会造成花粉活力下降、结实率降低",
                win))

    elif key == "filling":
        hd = stats.get("干热风日数", 0)
        if hd >= 2 or stats["高温日数"] >= 3:
            ev = None
            if dry_ref:
                ev = (f"历史参考：本县 {dry_ref['range']} 年中，灌浆期干热风 ≥3 天的年份 "
                      f"{dry_ref['n']} 个，这些年份单产相对本县常年平均 {dry_ref['mean']:.1f}%"
                      f"（区间 {dry_ref['low']:.0f}%—{dry_ref['high']:.0f}%）。"
                      "县域尺度上气象与单产的统计关联较弱，此处只作方向参考，不作为减产承诺。")
            cards.append(_card(
                P0, "防干热风逼熟，保住灌浆期",
                _dry_basis(stats, climate),
                ["小水勤浇或喷灌降温，避开中午 11—15 点",
                 "叶面喷施磷酸二氢钾等抗逆调节剂，增强抗高温能力",
                 "保持田间通风，不要大水漫灌后积水"],
                "高温低湿会加速叶片衰老、缩短灌浆时间，麦粒饱满度下降，千粒重直接受影响",
                win, evidence=ev))
        if precip < 20:
            cards.append(_card(
                P1, "补水保灌浆",
                [f"阶段累计降水 {precip} mm，低于常年同期",
                 f"下雨天数 {stats['降水日数']} 天"],
                ["盯土壤墒情，干土层超过 3 cm 就补浇",
                 "小水勤浇，避免大水漫灌引起倒伏"],
                "灌浆期缺水直接降低千粒重，是产量形成的最后窗口",
                win))
        if stats.get("平均辐射", 99) < 14:
            cards.append(_card(
                P2, "光照偏弱，保住上部叶片",
                [f"阶段平均辐射 {stats['平均辐射']} MJ/m²/day，低于常年"],
                ["重点防病防早衰，保上部功能叶",
                 "有条件的叶面补肥延缓衰老"],
                "灌浆期主要靠上部叶片光合产物，叶片早枯会减少粒重",
                win))

    elif key == "maturity":
        if stats["降水日数"] >= 5:
            cards.append(_card(
                P0, "抢晴收获，防穗发芽",
                [f"下雨天数 {stats['降水日数']} 天",
                 f"阶段累计降水 {precip} mm"],
                ["盯未来 3 天预报，抓住晴好窗口机械收获",
                 "收后及时晾晒或烘干，湿度大的麦堆勤翻动",
                 "先收成熟度高的地块，避免过熟落粒"],
                "成熟期连阴雨会造成穗发芽、霉变，产量和品质同时降级，且不可逆",
                win))
        if stats.get("大风日数", 0) >= 3:
            cards.append(_card(
                P1, "防倒伏和落粒",
                [f"风速 ≥5 m/s 的大风 {stats['大风日数']} 天"],
                ["大风前不浇水，减少倒伏风险",
                 "倒伏地块单独安排收割方向和割台高度"],
                "倒伏后机械收割损失率明显上升，落粒也直接减损",
                win))
        if not cards:
            cards.append(_card(
                P2, "天气平稳，按计划收获",
                [f"阶段降水 {precip} mm、下雨 {stats['降水日数']} 天，无大风"],
                ["按计划安排机械与人力", "关注后熟与晾晒，控制入库水分"],
                "正常年份按农时收获即可",
                win))

    if not cards:
        cards.append(_card(
            P2, "本期气象条件平稳",
            [f"阶段累计降水 {precip} mm，下雨 {stats['降水日数']} 天",
             f"高温 {stats['高温日数']} 天、霜冻 {stats['霜冻日数']} 天"],
            ["按常规农时安排田间管理", "保持关注下一阶段天气变化"],
            "无明显气象风险，按当地常规做法执行",
            win))

    # 给每张卡一个稳定标识（阶段 key + 生成顺序），供执行勾选持久化。
    # 注意：规则按现有顺序排列，新增规则请追加到对应分段的末尾，避免已有勾选错位。
    for i, c in enumerate(cards):
        c["key"] = f"{key}_{i}"

    cards.sort(key=lambda c: _ORDER[c["priority"]])
    return cards[:3]


def top_actions(stages: list[dict], limit: int = 5) -> list[dict]:
    """跨生育期汇总优先处理项，用于页面顶部的作业清单。"""
    pool = []
    for s in stages:
        for c in (s.get("cards") or []):
            if c["priority"] in (P0, P1):
                pool.append({"priority": c["priority"], "priority_label": c["priority_label"],
                             "stage": s["name"], "title": c["title"], "window": c["window"],
                             "stage_key": s["key"], "card_key": c.get("key")})
    pool.sort(key=lambda x: _ORDER[x["priority"]])
    return pool[:limit]
