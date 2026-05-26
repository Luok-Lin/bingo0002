from __future__ import annotations

import math
from io import BytesIO
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

from .report_narrative import build_report_narrative


def _clean(value: Any, default: str = "-") -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text or default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return default


def _fmt_number(value: Any, digits: int = 2, default: str = "-") -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if number is not None else default


def _fmt_percent(value: Any, digits: int = 2, default: str = "-") -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}%" if number is not None else default


def _ratio(value: Any, *, percent_scale: bool = False, default: float = 0.0) -> float:
    number = _safe_float(value, default)
    if number is None:
        number = default
    if percent_scale:
        number = number / 100.0
    elif number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _clip(value: Any, limit: int = 900) -> str:
    text = _clean(value, "")
    if not text:
        return "-"
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _join_items(items: Any, limit: int = 6) -> str:
    if not isinstance(items, list):
        return _clean(items)
    values = [_clean(item, "") for item in items if _clean(item, "")]
    return "；".join(values[:limit]) or "-"


def _stringify_structured(value: Any, limit: int = 700) -> str:
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                continue
            clean_val = _clean(val, "")
            if clean_val:
                parts.append(f"{key}: {clean_val}")
        return _clip("；".join(parts), limit)
    if isinstance(value, list):
        return _clip(_join_items(value), limit)
    return _clip(value, limit)


def _paragraph(text: Any, style):
    from reportlab.platypus import Paragraph

    return Paragraph(escape(_clean(text)).replace("\n", "<br/>"), style)


def _html_paragraph(html: str, style):
    from reportlab.platypus import Paragraph

    return Paragraph(html, style)


def _action_color(action: Any, colors):
    action_text = str(action or "").upper()
    if "BUY" in action_text:
        return colors.HexColor("#137a4b")
    if "SELL" in action_text:
        return colors.HexColor("#b42318")
    return colors.HexColor("#9a6500")


SOURCE_TYPE_LABELS = {
    "news": "个股新闻",
    "market_news": "市场要闻",
    "report": "券商研报",
    "announcement": "公司公告",
    "financial_abstract": "财务摘要",
    "financial_indicator": "财务指标",
    "income_statement": "利润表",
    "balance_sheet": "资产负债表",
    "cash_flow": "现金流量表",
    "fallback": "兜底文本",
}

SOURCE_LINK_DEFS = {
    "news": ("东方财富个股新闻", "https://so.eastmoney.com/news/s?keyword={ticker}"),
    "market_news": ("财新市场要闻", "https://www.caixin.com/search/{ticker}.html"),
    "report": ("东方财富研报", "https://data.eastmoney.com/report/{ticker}.html"),
    "announcement": ("巨潮资讯公告", "https://www.cninfo.com.cn/new/fulltextSearch?notautosubmit=&keyWord={ticker}"),
    "financial_abstract": ("东方财富财务摘要", "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx"),
    "financial_indicator": ("东方财富财务指标", "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx"),
    "income_statement": ("东方财富利润表", "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx"),
    "balance_sheet": ("东方财富资产负债表", "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx"),
    "cash_flow": ("东方财富现金流量表", "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx"),
}


def _exchange_ticker_for_link(ticker: Any) -> str:
    code = _clean(ticker, "").zfill(6)
    return ("SH" if code.startswith("6") else "SZ") + code


def _derive_source_links(source_types: dict, ticker: Any) -> list[dict]:
    code = _clean(ticker, "").zfill(6)
    exchange_ticker = _exchange_ticker_for_link(code)
    links = []
    for source, count in sorted(source_types.items(), key=lambda item: (-int(item[1] or 0), str(item[0]))):
        try:
            safe_count = int(count or 0)
        except Exception:
            safe_count = 0
        if safe_count <= 0 or source not in SOURCE_LINK_DEFS:
            continue
        label, template = SOURCE_LINK_DEFS[source]
        links.append(
            {
                "source": source,
                "label": label,
                "url": template.format(ticker=quote(code), exchange_ticker=quote(exchange_ticker)),
                "count": safe_count,
            }
        )
    return links


def build_advice_pdf_bytes(advice: dict, *, use_llm_narrative: bool | None = None) -> bytes:
    try:
        from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import KeepTogether, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:  # pragma: no cover - depends on deployment packages
        raise RuntimeError("PDF导出依赖未安装，请先安装 requirements.txt 中的 reportlab。") from exc

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception:
        pass

    base_font = "STSong-Light"
    palette = {
        "ink": colors.HexColor("#202124"),
        "muted": colors.HexColor("#667085"),
        "line": colors.HexColor("#ddd5cf"),
        "line_soft": colors.HexColor("#eee6e1"),
        "paper": colors.HexColor("#fffdfa"),
        "wash": colors.HexColor("#f8f3ef"),
        "accent": colors.HexColor("#8f1d1b"),
        "accent_dark": colors.HexColor("#641615"),
        "accent_soft": colors.HexColor("#f7dfda"),
        "green": colors.HexColor("#137a4b"),
        "green_soft": colors.HexColor("#dff4e8"),
        "amber": colors.HexColor("#9a6500"),
        "amber_soft": colors.HexColor("#fff1cc"),
        "red": colors.HexColor("#b42318"),
        "red_soft": colors.HexColor("#ffe2de"),
        "blue": colors.HexColor("#246b9f"),
        "blue_soft": colors.HexColor("#e2f0fa"),
    }

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=base_font,
        fontSize=23,
        leading=30,
        textColor=palette["ink"],
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        leading=13,
        textColor=palette["muted"],
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=13,
        leading=18,
        textColor=palette["accent_dark"],
        spaceBefore=13,
        spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9.5,
        leading=14.5,
        textColor=palette["ink"],
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=8.2,
        leading=11.5,
        textColor=palette["muted"],
    )
    card_title_style = ParagraphStyle(
        "CardTitle",
        parent=body_style,
        fontSize=8,
        leading=10,
        textColor=palette["muted"],
        alignment=TA_CENTER,
    )
    card_value_style = ParagraphStyle(
        "CardValue",
        parent=body_style,
        fontSize=14,
        leading=17,
        textColor=palette["ink"],
        alignment=TA_CENTER,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=small_style,
        backColor=palette["wash"],
        borderColor=palette["line_soft"],
        borderWidth=0.4,
        borderPadding=7,
        leftIndent=0,
        rightIndent=0,
    )

    rec = advice.get("recommendation", {}) if isinstance(advice.get("recommendation"), dict) else {}
    risk = advice.get("risk", {}) if isinstance(advice.get("risk"), dict) else {}
    referee = advice.get("referee", {}) if isinstance(advice.get("referee"), dict) else {}
    data_quality = advice.get("data_quality", {}) if isinstance(advice.get("data_quality"), dict) else {}
    latest = advice.get("latest_market", {}) if isinstance(advice.get("latest_market"), dict) else {}
    multi_period = advice.get("multi_period_advice", {}) if isinstance(advice.get("multi_period_advice"), dict) else {}
    technical_features = advice.get("technical_features", {}) if isinstance(advice.get("technical_features"), dict) else {}
    analysts = advice.get("analyst_cases", {}) if isinstance(advice.get("analyst_cases"), dict) else {}
    stability = advice.get("stability_diagnostics", {}) if isinstance(advice.get("stability_diagnostics"), dict) else {}
    user_profile = advice.get("user_profile", {}) if isinstance(advice.get("user_profile"), dict) else {}
    personalization = advice.get("personalization", {}) if isinstance(advice.get("personalization"), dict) else {}
    warnings = advice.get("warnings", []) if isinstance(advice.get("warnings"), list) else []
    narrative = build_report_narrative(advice, use_llm=use_llm_narrative)

    ticker = _clean(advice.get("ticker"), "UNKNOWN")
    action = _clean(rec.get("action"), "HOLD").upper()
    execution_action = _clean(rec.get("execution_action") or risk.get("final_action") or action, action).upper()
    action_fill = _action_color(action, colors)
    position = _safe_float(rec.get("position_percent"), 0.0) or 0.0
    confidence = _safe_float(rec.get("confidence"), 0.0) or 0.0
    quality_score = _safe_float(data_quality.get("score"), 0.0) or 0.0
    stability_score = _safe_float(advice.get("stability_score"), None)
    if stability_score is None:
        failures = (stability.get("parse_fail_count") or 0) + (stability.get("rule_fallback_count") or 0) + (stability.get("empty_reason_count") or 0)
        stability_score = max(0.0, 1.0 - min(1.0, float(failures) * 0.18))

    short_term = multi_period.get("short_term") or advice.get("short_term") or {}
    swing_term = multi_period.get("swing_term") or advice.get("swing_term") or {}
    risk_plan = multi_period.get("risk_plan") or advice.get("risk_plan") or {}

    trend = technical_features.get("trend", {}) if isinstance(technical_features.get("trend"), dict) else {}
    momentum = technical_features.get("momentum", {}) if isinstance(technical_features.get("momentum"), dict) else {}
    tech_risk = technical_features.get("risk", {}) if isinstance(technical_features.get("risk"), dict) else {}
    volume = technical_features.get("volume", {}) if isinstance(technical_features.get("volume"), dict) else {}
    signals = technical_features.get("signals", []) if isinstance(technical_features.get("signals"), list) else []

    rag = data_quality.get("rag", {}) if isinstance(data_quality.get("rag"), dict) else {}
    market_quality = data_quality.get("market", {}) if isinstance(data_quality.get("market"), dict) else {}
    components = data_quality.get("components", {}) if isinstance(data_quality.get("components"), dict) else {}
    source_types = rag.get("source_types", {}) if isinstance(rag.get("source_types"), dict) else {}
    source_links = rag.get("source_links", []) if isinstance(rag.get("source_links"), list) else []
    if not source_links:
        source_links = _derive_source_links(source_types, ticker)

    def page_chrome(canvas, doc) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(palette["accent"])
        canvas.rect(0, height - 10 * mm, width, 10 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont(base_font, 8)
        canvas.drawString(18 * mm, height - 6.5 * mm, "TradingAgents 智能投研平台")
        canvas.drawRightString(width - 18 * mm, height - 6.5 * mm, f"{ticker} 投资建议报告")
        canvas.setFillColor(palette["muted"])
        canvas.setFont(base_font, 8)
        canvas.drawString(18 * mm, 10 * mm, "自动生成报告，仅用于投研复盘与交易计划讨论，不构成投资承诺。")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title=f"{ticker} 投资建议报告",
        author="TradingAgents",
    )
    content_width = doc.width

    def action_badge(text: Any) -> Table:
        fill = _action_color(text, colors)
        badge = Table([[_paragraph(_clean(text).upper(), ParagraphStyle("Badge", parent=body_style, fontSize=10, leading=12, textColor=colors.white, alignment=TA_CENTER))]], colWidths=[27 * mm])
        badge.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), fill),
                    ("BOX", (0, 0), (-1, -1), 0.2, fill),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return badge

    def metric_card(title: str, value: str, fill=None) -> Table:
        fill = fill or colors.white
        card = Table(
            [[_paragraph(title, card_title_style)], [_paragraph(value, card_value_style)]],
            colWidths=[content_width / 4 - 3],
            rowHeights=[12 * mm, 15 * mm],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), fill),
                    ("BOX", (0, 0), (-1, -1), 0.5, palette["line"]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return card

    def section_table(rows: list[tuple[str, Any]], *, label_width: float = 33 * mm) -> Table:
        table_data = [[_paragraph(label, body_style), _paragraph(_stringify_structured(value), body_style)] for label, value in rows]
        table = Table(table_data, colWidths=[label_width, content_width - label_width])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), base_font),
                    ("GRID", (0, 0), (-1, -1), 0.25, palette["line"]),
                    ("BACKGROUND", (0, 0), (0, -1), palette["wash"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def section_html_table(rows: list[tuple[str, str]], *, label_width: float = 33 * mm) -> Table:
        table_data = [[_paragraph(label, body_style), _html_paragraph(value, body_style)] for label, value in rows]
        table = Table(table_data, colWidths=[label_width, content_width - label_width])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), base_font),
                    ("GRID", (0, 0), (-1, -1), 0.25, palette["line"]),
                    ("BACKGROUND", (0, 0), (0, -1), palette["wash"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def source_link_rows(limit: int = 8) -> list[tuple[str, str]]:
        rows = []
        for item in source_links[:limit]:
            if not isinstance(item, dict):
                continue
            url = _clean(item.get("url"), "")
            if not url:
                continue
            label = _clean(item.get("label") or SOURCE_TYPE_LABELS.get(str(item.get("source")), "") or item.get("source"), "数据来源")
            count = _safe_float(item.get("count"), None)
            count_text = f"（{int(count)}条）" if count is not None and count > 0 else ""
            rows.append((label, f'<a href={quoteattr(url)} color="#246b9f">{escape(label)}{escape(count_text)}</a><br/><font color="#667085">{escape(url)}</font>'))
        return rows

    def profile_summary() -> str:
        if not user_profile:
            return ""
        risk_map = {"conservative": "稳健型", "balanced": "均衡型", "aggressive": "进取型"}
        period_map = {"intraday": "日内", "swing": "波段", "mid_term": "中期"}
        parts = [
            risk_map.get(user_profile.get("risk_profile"), _clean(user_profile.get("risk_profile"), "")),
            period_map.get(user_profile.get("holding_period"), _clean(user_profile.get("holding_period"), "")),
            f"单票上限 {_fmt_percent(user_profile.get('max_position_per_stock'))}",
        ]
        if user_profile.get("already_holding"):
            current = f"当前持仓 {_fmt_percent(user_profile.get('current_position_percent'))}"
            cost = _safe_float(user_profile.get("cost_price"), None)
            if cost is not None and cost > 0:
                current += f"，成本约 {_fmt_number(cost)}"
            parts.append(current)
        else:
            parts.append("当前未持仓")
        parts.append("偏好明确止损" if user_profile.get("prefer_stop_loss", True) else "止损约束偏弹性")
        return "；".join(part for part in parts if part)

    def personalization_summary() -> str:
        if not personalization or not personalization.get("applied"):
            return ""
        original = _clean(personalization.get("original_action"), "")
        adjusted = _clean(personalization.get("adjusted_action"), "")
        if original and adjusted and original != adjusted:
            return f"已根据用户画像将策略动作从 {original} 校准为 {adjusted}，并同步约束仓位与风控条件。"
        return "已根据用户画像完成仓位与风控校准，当前执行动作保持不变。"

    def stability_summary() -> str:
        if not stability:
            return _clean(advice.get("stability_note"), "暂无稳定度说明。")
        parse_fail_count = int(_safe_float(stability.get("parse_fail_count"), 0.0) or 0)
        rule_fallback_count = int(_safe_float(stability.get("rule_fallback_count"), 0.0) or 0)
        empty_reason_count = int(_safe_float(stability.get("empty_reason_count"), 0.0) or 0)
        consensus = stability.get("consensus_tech_fund")
        referee_action = _clean(stability.get("referee_action"), "")
        risk_action = _clean(stability.get("risk_action"), "")
        recommendation_action = _clean(stability.get("recommendation_action"), "")
        level = _clean(advice.get("stability_level"), "")
        note = _clean(advice.get("stability_note"), "")

        parts = []
        if level:
            parts.append(f"建议稳定度为 {level}，评分 {_fmt_number(stability_score)}。")
        else:
            parts.append(f"建议稳定度评分 {_fmt_number(stability_score)}。")
        if parse_fail_count == 0 and empty_reason_count == 0:
            parts.append("分析链路解析完整，主要观点均有文字依据。")
        else:
            quality_notes = []
            if parse_fail_count:
                quality_notes.append("部分分析输出未能完整解析")
            if empty_reason_count:
                quality_notes.append("部分观点缺少充分说明")
            if quality_notes:
                parts.append("；".join(quality_notes) + "，已在最终建议中降低权重。")
        if rule_fallback_count:
            parts.append("本次存在规则兜底或保护性降级，执行时应优先遵守风控约束。")
        if consensus is True:
            parts.append("技术面与基本面方向一致。")
        elif consensus is False:
            parts.append("技术面与基本面存在分歧，建议等待更多确认信号。")
        actions = [x for x in [referee_action, risk_action, recommendation_action] if x]
        if len(set(actions)) > 1:
            parts.append("裁判、风控与最终建议不完全一致，报告已按风控优先原则给出执行动作。")
        elif actions:
            parts.append(f"裁判、风控与最终建议方向一致，当前核心动作为 {actions[0]}。")
        if note:
            parts.append(note)
        return "".join(parts)

    def metric_bar_chart(rows: list[tuple[str, float, str, Any]], width: float | None = None) -> Drawing:
        width = width or content_width
        height = 63 * mm
        left = 28 * mm
        bar_width = width - left - 16 * mm
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=palette["paper"], strokeColor=palette["line"], strokeWidth=0.45))
        drawing.add(String(8, height - 14, "核心指标仪表盘", fontName=base_font, fontSize=10.5, fillColor=palette["accent_dark"]))
        y = height - 28
        for label, ratio, value_label, fill in rows:
            ratio = max(0.0, min(1.0, ratio))
            drawing.add(String(8, y + 2, label, fontName=base_font, fontSize=8.5, fillColor=palette["ink"]))
            drawing.add(Rect(left, y, bar_width, 8, fillColor=palette["line_soft"], strokeColor=None))
            drawing.add(Rect(left, y, bar_width * ratio, 8, fillColor=fill, strokeColor=None))
            drawing.add(String(left + bar_width + 5, y + 1, value_label, fontName=base_font, fontSize=8, fillColor=palette["muted"]))
            y -= 12
        return drawing

    def source_distribution_chart() -> Drawing:
        width = content_width
        height = 48 * mm
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=palette["paper"], strokeColor=palette["line"], strokeWidth=0.45))
        drawing.add(String(8, height - 14, "证据来源分布", fontName=base_font, fontSize=10.5, fillColor=palette["accent_dark"]))
        if not source_types:
            drawing.add(String(8, height / 2, "暂无来源分类数据", fontName=base_font, fontSize=9, fillColor=palette["muted"]))
            return drawing
        items = sorted(((str(k), _safe_float(v, 0.0) or 0.0) for k, v in source_types.items()), key=lambda x: x[1], reverse=True)
        top_items = items[:7]
        total = sum(v for _, v in items) or 1.0
        x = 8
        y = height - 29
        max_width = width - 16
        colors_seq = [palette["accent"], palette["blue"], palette["green"], palette["amber"], palette["red"], colors.HexColor("#7a5fa0"), colors.HexColor("#5f7f84")]
        cursor = x
        for idx, (label, value) in enumerate(top_items):
            segment = max_width * (value / total)
            drawing.add(Rect(cursor, y, segment, 10, fillColor=colors_seq[idx % len(colors_seq)], strokeColor=None))
            cursor += segment
        legend_x = 8
        legend_y = 12
        for idx, (label, value) in enumerate(top_items):
            if legend_x > width - 54 * mm:
                legend_x = 8
                legend_y -= 11
            drawing.add(Rect(legend_x, legend_y, 6, 6, fillColor=colors_seq[idx % len(colors_seq)], strokeColor=None))
            drawing.add(String(legend_x + 8, legend_y - 0.5, f"{label} {int(value)}", fontName=base_font, fontSize=7.2, fillColor=palette["muted"]))
            legend_x += 38 * mm
        return drawing

    def radar_chart(width: float | None = None) -> Drawing:
        width = width or content_width
        height = 67 * mm
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=palette["paper"], strokeColor=palette["line"], strokeWidth=0.45))
        drawing.add(String(8, height - 14, "技术画像雷达", fontName=base_font, fontSize=10.5, fillColor=palette["accent_dark"]))
        cx = width / 2
        cy = height / 2 - 3
        radius = min(22 * mm, width * 0.24)
        labels = ["趋势", "动量", "量能", "风险韧性", "数据质量"]
        trend_score = max(0.0, min(1.0, ((_safe_float(trend.get("price_vs_ma20_pct"), 0.0) or 0.0) + 12) / 24))
        momentum_score = max(0.0, min(1.0, ((_safe_float(momentum.get("return_20d_pct"), 0.0) or 0.0) + 20) / 45))
        volume_score = max(0.0, min(1.0, (_safe_float(volume.get("volume_ratio_5_20"), 0.0) or 0.0) / 2.5))
        resilience_score = 1.0 - max(0.0, min(1.0, (_safe_float(tech_risk.get("atr14_pct"), 0.0) or 0.0) / 12.0))
        values = [trend_score, momentum_score, volume_score, resilience_score, _ratio(quality_score)]
        for level in [0.25, 0.5, 0.75, 1.0]:
            points = []
            for i in range(len(labels)):
                angle = math.pi / 2 + i * 2 * math.pi / len(labels)
                points.extend([cx + math.cos(angle) * radius * level, cy + math.sin(angle) * radius * level])
            drawing.add(Polygon(points, strokeColor=palette["line_soft"], fillColor=None, strokeWidth=0.45))
        value_points = []
        for i, value in enumerate(values):
            angle = math.pi / 2 + i * 2 * math.pi / len(labels)
            outer_x = cx + math.cos(angle) * radius
            outer_y = cy + math.sin(angle) * radius
            drawing.add(Line(cx, cy, outer_x, outer_y, strokeColor=palette["line_soft"], strokeWidth=0.45))
            label_x = cx + math.cos(angle) * (radius + 11)
            label_y = cy + math.sin(angle) * (radius + 7)
            drawing.add(String(label_x - 11, label_y - 3, labels[i], fontName=base_font, fontSize=7.0, fillColor=palette["muted"]))
            value_points.extend([cx + math.cos(angle) * radius * value, cy + math.sin(angle) * radius * value])
        drawing.add(Polygon(value_points, strokeColor=palette["accent"], fillColor=colors.Color(0.56, 0.11, 0.10, alpha=0.18), strokeWidth=1.2))
        drawing.add(Circle(cx, cy, 2.2, fillColor=palette["accent"], strokeColor=None))
        return drawing

    def period_timeline() -> Drawing:
        width = content_width
        height = 38 * mm
        drawing = Drawing(width, height)
        drawing.add(Rect(0, 0, width, height, fillColor=palette["paper"], strokeColor=palette["line"], strokeWidth=0.45))
        items = [
            ("短线", short_term.get("horizon") or "1-3 trading days", short_term.get("action") or action),
            ("中线", swing_term.get("horizon") or "2-4 weeks", swing_term.get("action") or action),
            ("执行", "交易计划", execution_action),
        ]
        box_w = (width - 24) / 3
        for idx, (title, horizon, item_action) in enumerate(items):
            x = 8 + idx * (box_w + 4)
            fill = _action_color(item_action, colors)
            drawing.add(Rect(x, 11, box_w, 18, fillColor=colors.white, strokeColor=palette["line"], strokeWidth=0.5))
            drawing.add(Rect(x, 27, box_w, 8, fillColor=fill, strokeColor=None))
            drawing.add(String(x + 5, 29, title, fontName=base_font, fontSize=7.6, fillColor=colors.white))
            drawing.add(String(x + 5, 20, _clean(item_action).upper()[:24], fontName=base_font, fontSize=8.5, fillColor=fill))
            drawing.add(String(x + 5, 13, _clean(horizon)[:34], fontName=base_font, fontSize=7.2, fillColor=palette["muted"]))
            if idx < 2:
                arrow_x = x + box_w + 1
                drawing.add(Line(arrow_x, 20, arrow_x + 3, 20, strokeColor=palette["muted"], strokeWidth=0.8))
        return drawing

    def explanation_text() -> str:
        parts = []
        if action == execution_action or execution_action.startswith(action):
            parts.append(f"策略层与执行层当前保持一致，核心动作为 {action}，建议仓位约 {_fmt_percent(position)}。")
        else:
            parts.append(f"策略层给出 {action}，执行层根据风险约束调整为 {execution_action}，建议仓位约 {_fmt_percent(position)}。")
        if confidence:
            parts.append(f"置信度为 {_fmt_number(confidence)}，属于{'较高' if confidence >= 0.7 else '中等' if confidence >= 0.45 else '偏低'}信号。")
        if quality_score:
            parts.append(f"数据质量评分 {_fmt_number(quality_score)}，质量等级为 {_clean(data_quality.get('level'))}。")
        if personalization.get("applied"):
            parts.append("本报告已结合用户画像进行个性化仓位和执行动作校准。")
        return "".join(parts)

    def risk_text() -> str:
        invalid = _join_items(risk_plan.get("invalid_conditions") if isinstance(risk_plan, dict) else [], 5)
        return (
            f"风控重点是避免在证据分歧或波动放大的阶段追价。止损参考：{_clean(risk_plan.get('stop_loss') if isinstance(risk_plan, dict) else '')}；"
            f"止盈参考：{_clean(risk_plan.get('take_profit') if isinstance(risk_plan, dict) else '')}；"
            f"计划失效条件：{invalid}。"
        )

    def narrative_block(title: str, text: Any) -> KeepTogether:
        return KeepTogether(
            [
                _paragraph(title, ParagraphStyle("NarrativeTitle", parent=body_style, fontSize=10.2, leading=13, textColor=palette["accent_dark"])),
                _paragraph(text, note_style),
                Spacer(1, 4),
            ]
        )

    def watchlist_text(items: Any) -> str:
        if not isinstance(items, list):
            return _clean(items, "暂无观察清单。")
        values = [_clean(item, "") for item in items if _clean(item, "")]
        return "；".join(values[:5]) or "暂无观察清单。"

    story = []
    title_row = Table(
        [
            [
                _paragraph(f"{ticker} 结构化投资建议报告", title_style),
                action_badge(action),
            ]
        ],
        colWidths=[content_width - 34 * mm, 34 * mm],
    )
    title_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    story.append(title_row)
    story.append(
        _paragraph(
            f"建议日期：{_clean(advice.get('as_of_date'))}   生成时间：{_clean(advice.get('generated_at'))}   行情日期：{_clean(latest.get('as_of_date') or market_quality.get('as_of_date'))}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 7))

    metric_fill = palette["green_soft"] if action == "BUY" else palette["red_soft"] if action == "SELL" else palette["amber_soft"]
    metrics = Table(
        [
            [
                metric_card("建议动作", action, metric_fill),
                metric_card("执行动作", execution_action, palette["blue_soft"]),
                metric_card("建议仓位", _fmt_percent(position), colors.white),
                metric_card("置信度", _fmt_number(confidence), colors.white),
            ],
            [
                metric_card("数据质量", f"{_fmt_number(quality_score)} / {_clean(data_quality.get('level'))}", colors.white),
                metric_card("收盘价", _fmt_number(latest.get("close")), colors.white),
                metric_card("涨跌幅", _fmt_percent(latest.get("pct_change")), colors.white),
                metric_card("换手率", _fmt_percent(latest.get("turnover")), colors.white),
            ],
        ],
        colWidths=[content_width / 4] * 4,
    )
    metrics.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(metrics)
    story.append(Spacer(1, 8))
    story.append(_paragraph(explanation_text(), note_style))

    story.append(_paragraph("一、报告总览", heading_style))
    story.append(narrative_block("执行摘要", narrative.get("executive_summary")))
    chart_col_width = content_width * 0.50 - 2
    story.append(
        Table(
            [
                [
                    metric_bar_chart(
                        [
                            ("置信度", _ratio(confidence), _fmt_number(confidence), palette["blue"]),
                            ("建议仓位", _ratio(position, percent_scale=True), _fmt_percent(position), action_fill),
                            ("数据质量", _ratio(quality_score), _fmt_number(quality_score), palette["green"]),
                            ("稳定度", _ratio(stability_score), _fmt_number(stability_score), palette["amber"]),
                        ],
                        chart_col_width,
                    ),
                    radar_chart(chart_col_width),
                ]
            ],
            colWidths=[content_width * 0.50, content_width * 0.50],
        )
    )
    story.append(Spacer(1, 6))
    story.append(source_distribution_chart())
    link_rows = source_link_rows()
    if link_rows:
        story.append(Spacer(1, 6))
        story.append(_paragraph("数据来源链接", heading_style))
        story.append(section_html_table(link_rows))

    story.append(_paragraph("二、核心结论与理由", heading_style))
    story.append(narrative_block("核心逻辑", narrative.get("investment_thesis")))
    story.append(
        section_table(
            [
                ("最终建议", rec.get("reason") or rec.get("thesis") or rec.get("rationale")),
                ("执行说明", f"建议动作 {action}，执行动作 {execution_action}，建议仓位 {_fmt_percent(position)}。"),
                ("风控意见", risk.get("reason") or risk.get("decision") or risk.get("action")),
                ("裁判结论", referee.get("reason") or referee.get("decision") or referee.get("sentiment")),
                ("质量校准", rec.get("data_quality_note") or data_quality.get("note")),
            ]
        )
    )

    story.append(_paragraph("三、多周期交易计划", heading_style))
    story.append(narrative_block("执行计划", narrative.get("execution_plan")))
    story.append(period_timeline())
    story.append(Spacer(1, 6))
    story.append(
        section_table(
            [
                ("短线计划", f"{_clean(short_term.get('action') or action)} / 置信度 {_fmt_number(short_term.get('confidence'))} / {_clip(short_term.get('reason'), 650)}"),
                ("中线计划", f"{_clean(swing_term.get('action') or action)} / 置信度 {_fmt_number(swing_term.get('confidence'))} / {_clip(swing_term.get('reason'), 650)}"),
                ("止损参考", risk_plan.get("stop_loss") if isinstance(risk_plan, dict) else ""),
                ("止盈参考", risk_plan.get("take_profit") if isinstance(risk_plan, dict) else ""),
                ("失效条件", _join_items(risk_plan.get("invalid_conditions") if isinstance(risk_plan, dict) else [], 6)),
            ]
        )
    )
    story.append(_paragraph(risk_text(), note_style))

    story.append(_paragraph("四、行情、技术特征与证据质量", heading_style))
    story.append(narrative_block("证据链说明", narrative.get("evidence_review")))
    story.append(
        section_table(
            [
                ("行情覆盖", f"来源 {_clean(market_quality.get('source') or latest.get('source'))}，样本 {_clean(market_quality.get('rows'))} 行，最近日期 {_clean(market_quality.get('as_of_date') or latest.get('as_of_date'))}。"),
                ("趋势结构", trend),
                ("动量状态", momentum),
                ("风险波动", tech_risk),
                ("量能状态", volume),
                ("技术信号", _join_items(signals, 8)),
                ("RAG证据", f"文档 {_clean(rag.get('documents'))} 份，最近日期 {_clean(rag.get('latest_date'))}，状态 {_clean(rag.get('status'))}。"),
                ("质量分项", components),
            ]
        )
    )

    analyst_rows = []
    for name, case in analysts.items():
        if not isinstance(case, dict):
            continue
        analyst_rows.append(
            (
                str(case.get("agent") or name),
                f"倾向：{_clean(case.get('sentiment') or case.get('decision'))}；置信度：{_fmt_number(case.get('confidence'))}；{_clip(case.get('reasoning') or case.get('thought_process'), 850)}",
            )
        )
    if analyst_rows:
        story.append(_paragraph("五、分析师观点摘要", heading_style))
        story.append(section_table(analyst_rows[:8]))

    story.append(PageBreak())
    story.append(_paragraph("六、执行清单与复盘要点", heading_style))
    story.append(narrative_block("风险与稳定度", f"{_clean(narrative.get('risk_review'))} {_clean(narrative.get('stability_review'))}"))
    checklist_rows = [
        ("开仓/调仓前", f"确认价格未触发失效条件；若执行 {execution_action}，单票仓位不超过 {_fmt_percent(position)}。"),
        ("盘中观察", "关注量能是否延续、关键均线是否有效、新闻或公告是否改变基本面假设。"),
        ("风险处理", _clean(risk_plan.get("stop_loss") if isinstance(risk_plan, dict) else risk.get("reason"))),
        ("止盈处理", _clean(risk_plan.get("take_profit") if isinstance(risk_plan, dict) else "")),
        ("复盘记录", watchlist_text(narrative.get("watchlist"))),
    ]
    if user_profile:
        checklist_rows.append(("用户画像", profile_summary()))
    personal_note = personalization_summary()
    if personal_note:
        checklist_rows.append(("个性化校准", personal_note))
    story.append(section_table(checklist_rows))

    evidence_rows = []
    for name, case in analysts.items():
        if not isinstance(case, dict):
            continue
        reports = case.get("source_reports", []) if isinstance(case.get("source_reports"), list) else []
        for report in reports:
            if not isinstance(report, dict):
                continue
            evidence_rows.append(
                (
                    report.get("agent") or name,
                    report.get("reasoning") or report.get("thought_process") or report.get("summary"),
                )
            )
    if evidence_rows:
        story.append(_paragraph("七、证据附录", heading_style))
        story.append(section_table(evidence_rows[:12]))

    diagnostics = data_quality.get("diagnostics", []) if isinstance(data_quality.get("diagnostics"), list) else []
    system_rows = [
        ("稳定度诊断", stability_summary()),
        ("数据质量提示", _join_items(diagnostics, 8)),
    ]
    if warnings:
        system_rows.extend((f"系统提示 {idx + 1}", item) for idx, item in enumerate(warnings[:8]))
    story.append(_paragraph("八、系统说明", heading_style))
    story.append(section_table(system_rows))
    story.append(Spacer(1, 8))
    story.append(
        _paragraph(
            "免责声明：本报告由 TradingAgents 根据行情、检索证据、多 Agent 分析和用户画像自动生成。报告内容仅用于研究、复盘和交易计划讨论，不构成任何收益承诺或投资顾问服务。",
            note_style,
        )
    )

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return buffer.getvalue()
