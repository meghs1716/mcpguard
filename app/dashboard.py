"""
app/dashboard.py - Streamlit security dashboard for MCPGuard.
Reads events.jsonl (or sample_events.jsonl in replay mode).
Displays:
  - Metric summary cards (counts per action: allow, ask, block)
  - Interactive probability chart per tool
  - Filterable events table with colored action badges
  - Sidebar filters for tool and stage, plus replay mode toggle
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_EVENTS_PATH = os.path.join(REPO_ROOT, "events.jsonl")
SAMPLE_EVENTS_PATH = os.path.join(REPO_ROOT, "sample_events.jsonl")


def load_events(file_path: str) -> pd.DataFrame:
    """Load events from a JSONL file into a standardized pandas DataFrame."""
    cols = ["timestamp", "stage", "tool", "probability", "action", "reason"]
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=cols)

    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    # Normalize fields to support both GuardedSession and victim agent logs
                    tool = ev.get("tool") or ev.get("tool_name") or "unknown"
                    stage = ev.get("stage") or "call"
                    prob = float(ev.get("probability", 0.0))
                    action = ev.get("action") or ("allow" if not ev.get("guarded", False) else "unknown")
                    reason = ev.get("reason") or ev.get("arguments") or ""
                    if isinstance(reason, (dict, list)):
                        reason = json.dumps(reason)

                    records.append({
                        "timestamp": str(ev.get("timestamp", "")),
                        "stage": str(stage),
                        "tool": str(tool),
                        "probability": round(prob, 4),
                        "action": str(action).lower(),
                        "reason": str(reason),
                    })
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame(columns=cols)

    if not records:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(records)


def filter_events(
    df: pd.DataFrame,
    selected_tools: Optional[List[str]] = None,
    selected_stages: Optional[List[str]] = None,
    selected_actions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter DataFrame by selected tools, stages, and actions."""
    filtered = df.copy()
    if filtered.empty:
        return filtered

    if selected_tools is not None:
        filtered = filtered[filtered["tool"].isin(selected_tools)]
    if selected_stages is not None:
        filtered = filtered[filtered["stage"].isin(selected_stages)]
    if selected_actions is not None:
        filtered = filtered[filtered["action"].isin(selected_actions)]

    return filtered


def style_action_cells(val: Any) -> str:
    """Return CSS styling for action cells based on allow/ask/block status."""
    v = str(val).lower()
    if v == "allow":
        return "background-color: rgba(40, 167, 69, 0.2); color: #28a745; font-weight: bold;"
    elif v == "ask":
        return "background-color: rgba(255, 193, 7, 0.25); color: #e65100; font-weight: bold;"
    elif v == "block":
        return "background-color: rgba(220, 53, 69, 0.25); color: #dc3545; font-weight: bold;"
    return ""


def render_dashboard() -> None:
    """Main Streamlit dashboard renderer."""
    st.set_page_config(
        page_title="MCPGuard Security Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ MCPGuard Security Monitoring Dashboard")
    st.caption("Real-time interception monitoring and defense metrics for Model Context Protocol (MCP) clients.")

    # 1. Sidebar Configuration & Replay Mode Toggle
    st.sidebar.header("⚙️ Configuration")
    replay_mode = st.sidebar.toggle(
        "Replay Mode",
        value=False,
        help="Toggle replay mode to load sample_events.jsonl without requiring live model calls.",
    )

    active_path = SAMPLE_EVENTS_PATH if replay_mode else DEFAULT_EVENTS_PATH
    source_name = "sample_events.jsonl (Replay Mode)" if replay_mode else "events.jsonl (Live Events)"

    st.sidebar.info(f"📁 **Data Source:** `{os.path.basename(active_path)}`")

    # Load data
    df_raw = load_events(active_path)

    if df_raw.empty:
        st.warning(
            f"No events found in `{active_path}`. "
            "Run a guarded session or enable **Replay Mode** in the sidebar to view sample events."
        )
        return

    # 2. Sidebar Filters
    st.sidebar.subheader("🔍 Filters")
    available_tools = sorted(df_raw["tool"].unique().tolist())
    selected_tools = st.sidebar.multiselect(
        "Filter by Tool",
        options=available_tools,
        default=available_tools,
    )

    available_stages = sorted(df_raw["stage"].unique().tolist())
    selected_stages = st.sidebar.multiselect(
        "Filter by Stage",
        options=available_stages,
        default=available_stages,
    )

    available_actions = sorted(df_raw["action"].unique().tolist())
    selected_actions = st.sidebar.multiselect(
        "Filter by Action",
        options=available_actions,
        default=available_actions,
    )

    # Apply filters
    df = filter_events(df_raw, selected_tools, selected_stages, selected_actions)

    # 3. Action Counts (Summary Metrics)
    total_events = len(df)
    allow_count = len(df[df["action"] == "allow"])
    ask_count = len(df[df["action"] == "ask"])
    block_count = len(df[df["action"] == "block"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", total_events)
    c2.metric("✅ Allowed", allow_count, delta=f"{allow_count/total_events*100:.1f}%" if total_events else "0%")
    c3.metric("⚠️ Asked (Confirmation)", ask_count, delta=f"{ask_count/total_events*100:.1f}%" if total_events else "0%")
    c4.metric(
        "🛑 Blocked (Threats)",
        block_count,
        delta=f"-{block_count/total_events*100:.1f}%" if total_events else "0%",
        delta_color="inverse",
    )

    st.divider()

    # 4. Probability per Tool Chart
    st.subheader("📊 Threat & Risk Probability per Tool")
    if not df.empty:
        chart = (
            alt.Chart(df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("tool:N", title="Tool Name", sort=available_tools),
                y=alt.Y("probability:Q", title="Evaluated Probability", scale=alt.Scale(domain=[-0.05, 1.05])),
                color=alt.Color(
                    "action:N",
                    title="Action",
                    scale=alt.Scale(
                        domain=["allow", "ask", "block"],
                        range=["#28a745", "#ff9800", "#dc3545"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:N", title="Timestamp"),
                    alt.Tooltip("stage:N", title="Stage"),
                    alt.Tooltip("tool:N", title="Tool"),
                    alt.Tooltip("probability:Q", title="Probability", format=".2f"),
                    alt.Tooltip("action:N", title="Action"),
                    alt.Tooltip("reason:N", title="Reason"),
                ],
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No events match the selected filters.")

    st.divider()

    # 5. Event Table with Colored Actions
    st.subheader("📋 Security Event Log")
    st.caption("Detailed interception events across tool definition scanning, argument validation, and result verification.")

    if not df.empty:
        display_df = df.copy()
        if "timestamp" in display_df.columns:
            display_df = display_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

        styled_table = display_df.style.map(
            style_action_cells,
            subset=["action"],
        ).format({
            "probability": "{:.2f}",
        })

        st.dataframe(
            styled_table,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "stage": st.column_config.TextColumn("Stage", width="small"),
                "tool": st.column_config.TextColumn("Tool", width="medium"),
                "probability": st.column_config.NumberColumn("Probability", format="%.2f", width="small"),
                "action": st.column_config.TextColumn("Action", width="small"),
                "reason": st.column_config.TextColumn("Reason / Policy Check", width="large"),
            },
            hide_index=True,
        )
    else:
        st.info("No records found.")


if __name__ == "__main__":
    render_dashboard()
