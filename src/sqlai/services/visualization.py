"""
Convert chart specifications into Plotly figures.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TYPE_MAP = {
    "bar": px.bar,
    "line": px.line,
    "scatter": px.scatter,
    "pie": px.pie,
    "histogram": px.histogram,
}


def build_chart(chart_spec: Optional[Dict[str, Any]], dataframe: pd.DataFrame) -> Optional[go.Figure]:
    if not chart_spec:
        return None

    def _normalize_layout_options(options: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(options)
        legend_value = normalized.get("legend")
        if isinstance(legend_value, bool):
            normalized.pop("legend")
            normalized["showlegend"] = legend_value
        return normalized

    chart_type = chart_spec.get("type", "bar").lower()
    builder = CHART_TYPE_MAP.get(chart_type)
    if not builder:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    x_field = chart_spec.get("x")
    y_field = chart_spec.get("y")
    required = [field for field in (x_field, y_field) if field]
    if required and any(field not in dataframe.columns for field in required):
        raise ValueError(
            f"Chart specification refers to missing columns: "
            f"{', '.join(field for field in required if field not in dataframe.columns)}"
        )
    fig = builder(
        dataframe,
        x=x_field,
        y=y_field,
        color=chart_spec.get("color"),
        title=chart_spec.get("title"),
    )
    options = chart_spec.get("options") or {}
    fig.update_layout(**_normalize_layout_options(options))
    return fig

