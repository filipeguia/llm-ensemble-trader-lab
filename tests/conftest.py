from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from llm_ensemble_lab.bars import Bar
from llm_ensemble_lab.config import RunConfig, config_from_dict


def make_bars(rows, start="2024-01-01T00:00:00", minutes=60):
    """Build bars from (open, high, low, close) tuples"""
    t0 = datetime.fromisoformat(start)
    return [Bar(t0 + timedelta(minutes=minutes * i), o, h, l, c) for i, (o, h, l, c) in enumerate(rows)]


@pytest.fixture
def small_cfg() -> RunConfig:
    """Default config on a small synthetic series so tests stay fast"""
    data = RunConfig().to_dict()
    data["data"]["synthetic"]["n_bars"] = 500
    data["ensemble"]["n_agents"] = 5
    return config_from_dict(data)
