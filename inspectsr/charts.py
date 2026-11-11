"""Altair charts. Judgement colours use a fixed status palette and always come with an icon + label."""
from __future__ import annotations

import altair as alt
import pandas as pd


# categorical slots (validated reference palette), fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE_COLORS = {"Retraction Watch": CAT[0], "Crossref (from RW)": "#86b6ef", "Crossref": CAT[1],
                 "PubMed": CAT[2], "OpenAlex": CAT[6]}
NOTICE_COLORS = {"Publication": "#898781", "Retraction": CAT[7], "Expression of concern": CAT[1],
                 "Correction": CAT[2], "Reinstatement": CAT[0]}
INK_MUTED = "#898781"


def funnel(steps: list[tuple[str, int]]) -> alt.Chart:
    df = pd.DataFrame(steps, columns=["Step", "Studies"])
    bars = alt.Chart(df).mark_bar(color=CAT[0], cornerRadiusEnd=4).encode(
        x=alt.X("Studies:Q", title=None, axis=alt.Axis(grid=False, labels=False, ticks=False, domain=False)),
        y=alt.Y("Step:N", sort=list(df["Step"]), title=None, axis=alt.Axis(labelLimit=220)),
        tooltip=["Step", "Studies"],
    )
    labels = bars.mark_text(align="left", dx=4, color=INK_MUTED).encode(text="Studies:Q")
    return (bars + labels).properties(height=alt.Step(28), width="container")


def source_agreement(evidence: pd.DataFrame) -> alt.Chart | None:
    """Which sources flagged each retracted / EoC study: shows why one source is not enough."""
    ev = evidence
    if ev.empty:
        return None
    g = ev.groupby(["ID", "Study", "Flag", "Source"]).size().reset_index(name="n")
    g["Row"] = g["ID"] + "  " + g["Study"] + " · " + g["Flag"]
    sources = ["Retraction Watch", "Crossref (from RW)", "Crossref", "PubMed", "OpenAlex"]
    base = alt.Chart(g).encode(
        x=alt.X("Source:N", sort=sources, axis=alt.Axis(orient="top", labelAngle=0, title=None, domain=False, ticks=False)),
        y=alt.Y("Row:N", title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=320)),
        tooltip=["Study", "Flag", "Source", alt.Tooltip("n", title="Records")],
    )
    pts = base.mark_square(size=260).encode(color=alt.Color("Source:N", scale=alt.Scale(
        domain=sources, range=[SOURCE_COLORS[s] for s in sources]), legend=None))
    txt = base.mark_text(color="white", fontSize=11, fontWeight="bold").encode(text=alt.value("✓"))
    return (pts + txt).properties(height=alt.Step(30), width=alt.Step(110))


def notice_timeline(rows: pd.DataFrame, search_date: str | None = None) -> alt.Chart | None:
    """rows: Study, Event, Date (one row per publication / notice)."""
    if rows.empty:
        return None
    rows = rows.copy()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows = rows.dropna(subset=["Date"])
    if rows.empty:
        return None
    kinds = [k for k in NOTICE_COLORS if k in set(rows["Event"])]
    span = rows.groupby("Study")["Date"].agg(["min", "max"]).reset_index()
    order = list(span.sort_values("min")["Study"])
    line = alt.Chart(span).mark_rule(color="#c3c2b7", strokeWidth=2).encode(
        y=alt.Y("Study:N", sort=order, title=None), x="min:T", x2="max:T")
    pts = alt.Chart(rows).mark_point(size=120, filled=True, opacity=1).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y", grid=False)),
        y=alt.Y("Study:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("Event:N", scale=alt.Scale(domain=kinds, range=[NOTICE_COLORS[k] for k in kinds]),
                        legend=alt.Legend(title=None, orient="bottom", symbolSize=140)),
        shape=alt.Shape("Event:N", scale=alt.Scale(domain=kinds, range=[
                            {"Publication": "circle", "Retraction": "cross", "Expression of concern": "triangle-up",
                             "Correction": "diamond", "Reinstatement": "square"}[k] for k in kinds]),
                        legend=alt.Legend(title=None, orient="bottom", symbolSize=140)),
        tooltip=["Study", "Event", alt.Tooltip("Date:T", format="%Y-%m-%d"), "Detail"],
    )
    layers = [line, pts]
    if search_date:
        sd = pd.DataFrame({"Date": [pd.to_datetime(search_date)], "label": ["Review search date"]})
        layers.append(alt.Chart(sd).mark_rule(strokeDash=[4, 3], color=INK_MUTED).encode(x="Date:T", tooltip=["label"]))
        layers.append(alt.Chart(sd).mark_text(align="left", dx=4, dy=-6, color=INK_MUTED, baseline="bottom")
                      .encode(x="Date:T", y=alt.value(0), text="label"))
    return alt.layer(*layers).properties(height=alt.Step(34), width="container")


def reasons_bar(reasons: pd.DataFrame) -> alt.Chart | None:
    """reasons: Reason, Studies"""
    if reasons.empty:
        return None
    bars = alt.Chart(reasons).mark_bar(color=CAT[0], cornerRadiusEnd=4).encode(
        x=alt.X("Studies:Q", title="Studies", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("Reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["Reason", "Studies"],
    )
    return (bars + bars.mark_text(align="left", dx=4, color=INK_MUTED).encode(text="Studies:Q")) \
        .properties(height=alt.Step(24), width="container")


def registration_lag(df: pd.DataFrame) -> alt.Chart | None:
    """df: Study, Trial, Lag (days between study start and first registration)."""
    if df.empty:
        return None
    df = df.copy()
    df["Lag (months)"] = df["Lag"] / 30.44
    df["Timing"] = pd.cut(df["Lag"], [-1e9, 30, 180, 1e9],
                          labels=["Prospective / ≤30 days", "31–180 days", ">180 days"]).astype(str)
    timing = ["Prospective / ≤30 days", "31–180 days", ">180 days"]
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    pts = alt.Chart(df).mark_point(size=130, filled=True, opacity=1).encode(
        x=alt.X("Lag (months):Q", title="Registration relative to study start (months)",
                axis=alt.Axis(tickCount=8, grid=False)),
        y=alt.Y("Study:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
        color=alt.Color("Timing:N", scale=alt.Scale(domain=timing, range=["#0ca30c", "#fab219", "#d03b3b"]),
                        legend=alt.Legend(title=None, orient="bottom", symbolSize=140)),
        shape=alt.Shape("Timing:N", scale=alt.Scale(domain=timing, range=["circle", "triangle-up", "cross"]),
                        legend=alt.Legend(title=None, orient="bottom", symbolSize=140)),
        tooltip=["Study", "Trial", alt.Tooltip("Lag:Q", title="Days after start"), "Timing"],
    )
    return (zero + pts).properties(height=alt.Step(30), width="container")
