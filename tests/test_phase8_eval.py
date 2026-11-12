"""FASE 8: event matching, event detection, questionnaire analysis, robustness grouping, scenario script."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.replay import make_sample, read_features
from src.core.config import load_config
from src.core.events import FrameFeatures
from tests.conftest import load_script
from training import evaluate_robustness as er

CFG = load_config()


@pytest.fixture(scope="module")
def ev():
    return load_script("eval_events")


@pytest.fixture(scope="module")
def aq():
    return load_script("analyze_questionnaire")


def test_match_with_tolerance(ev) -> None:
    assert ev.match([10.0, 31.5, 70.0], [11.0, 30.0, 50.0], 2.0) == (2, 1, 1)
    assert ev.match([], [1.0], 2.0) == (0, 0, 1)
    assert ev.match([5.0, 5.5], [5.2], 2.0) == (1, 1, 0)  # one label matches one detection only


def test_detect_events_on_synthetic_sample(ev, tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    make_sample(path)
    det = ev.detect_events(list(read_features(path)), CFG, 0.21)
    # sample: 3 yawns (+4 in the fatigue segment, debounced), drowsy closures every 8 s, hand episodes
    assert len(det["yawn"]) >= 3 and len(det["eye_closed_long"]) >= 7 and len(det["hand_to_forehead"]) >= 4
    assert all(b >= a for a, b in zip(det["yawn"], det["yawn"][1:]))


def test_short_closures_are_not_long_events(ev) -> None:
    feats = [FrameFeatures(ts=i * 0.1, user_present=True, ear_mean=0.08 if i % 40 < 2 else 0.31) for i in range(400)]
    assert ev.detect_events(feats, CFG, 0.21)["eye_closed_long"] == []  # 200 ms blinks only


def test_evaluate_and_report(ev, tmp_path: Path) -> None:
    labels = tmp_path / "l.csv"
    labels.write_text("type,start_sec\nyawn,10\nyawn,40\nhand_to_forehead,20\n")
    res = {r.type: r for r in ev.evaluate({"yawn": [11.0], "hand_to_forehead": [20.5, 60.0]}, ev.read_labels(labels), 2.0)}
    assert (res["yawn"].precision, res["yawn"].recall) == (1.0, 0.5)
    assert (res["hand_to_forehead"].precision, res["hand_to_forehead"].recall) == (0.5, 1.0)
    assert res["eye_closed_long"].precision is None
    ev.write_report(list(res.values()), "uji", 2.0, tmp_path)
    assert "| yawn | 2 | 1 | 1 | 0 | 1 | 1.00 | 0.50 |" in (tmp_path / "events.md").read_text()
    bad = tmp_path / "b.csv"
    bad.write_text("type,start_sec\nsneeze,1\n")
    with pytest.raises(ValueError):
        ev.read_labels(bad)


def test_scenario_script_labels() -> None:
    ss = load_script("scenario_session")
    labels = ss.script_labels(ss.DEFAULT_SCRIPT)
    kinds = [k for k, _ in labels]
    assert kinds.count("yawn") == 3 and kinds.count("eye_closed_long") == 3 and kinds.count("hand_to_forehead") == 3
    assert labels[0] == ("yawn", 20.0)
    step, left = ss.step_at(ss.DEFAULT_SCRIPT, 25.0)
    assert step.event == "yawn" and left == pytest.approx(5.0)
    assert ss.step_at(ss.DEFAULT_SCRIPT, 10_000)[0] is None


# --- questionnaire ----------------------------------------------------------------------

def _db_with_scores(path: Path, rows: list[tuple[str, float]]) -> None:
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE stress_log (ts REAL, score REAL)")
        c.execute("CREATE TABLE advice_log (ts REAL, message_key TEXT, expression TEXT, response TEXT)")
        for when, score in rows:
            c.execute("INSERT INTO stress_log VALUES (?, ?)", (datetime.strptime(when, "%Y-%m-%d %H:%M").timestamp(), score))
        c.executemany("INSERT INTO advice_log VALUES (0, 'k', 'e', ?)", [("oke",), ("nanti",), ("oke",), (None,)])


def test_stress_correlation(aq, tmp_path: Path) -> None:
    db = tmp_path / "p01.db"
    scores = [(f"2026-11-20 {h:02d}:15", s) for h, s in zip(range(9, 15), (10, 20, 35, 50, 60, 80), strict=True)]
    _db_with_scores(db, scores)
    ratings = pd.DataFrame({"peserta": ["p01"] * 6, "waktu": [f"2026-11-20 {h:02d}:30" for h in range(9, 15)],
                            "skor_lelah": [1, 2, 2, 3, 4, 5]})
    df, stats = aq.stress_analysis(ratings, {"p01": db}, 30)
    assert df["skor_sistem"].tolist() == [10, 20, 35, 50, 60, 80]
    pooled = stats[-1]
    assert pooled["n"] == 6 and pooled["spearman"] > 0.9 and pooled["pearson"] > 0.9
    aq.write_stress_report(stats, 30, tmp_path / "s.md")
    assert "sangat kuat (positif)" in (tmp_path / "s.md").read_text()


def test_correlation_needs_pairs_and_variance(aq) -> None:
    assert aq.correlations(pd.DataFrame({"skor_sistem": [1, 2], "skor_lelah": [1, 2]}))["pearson"] is None
    assert aq.correlations(pd.DataFrame({"skor_sistem": [5, 5, 5], "skor_lelah": [1, 2, 3]}))["spearman"] is None
    assert aq.strength(-0.1) == "sangat lemah (negatif)"


def test_ux_summary_and_follow_ratio(aq, tmp_path: Path) -> None:
    likert = pd.DataFrame({"peserta": ["a", "b", "c"], "gangguan": [2, 3, 1], "kelucuan": [5, 4, 5]})
    s = aq.ux_analysis(likert).set_index("item")
    assert s.loc["gangguan", "mean"] == pytest.approx(2.0) and s.loc["kelucuan", "median"] == 5
    db = tmp_path / "x.db"
    _db_with_scores(db, [])
    follow = aq.advice_follow({"a": db})
    assert follow[0]["saran"] == 4 and follow[0]["diikuti"] == 2 and follow[0]["rasio"] == 0.5
    aq.write_ux_report(aq.ux_analysis(likert), follow, tmp_path / "ux.md")
    assert "| a | 4 | 2 | 0.50 |" in (tmp_path / "ux.md").read_text()
    with pytest.raises(SystemExit):
        aq.parse_dbs(["tanpa-path"])


def test_templates(aq, tmp_path: Path) -> None:
    paths = aq.write_templates(tmp_path)
    assert {p.name for p in paths} == {"ratings.csv", "likert.csv", "event_labels.csv"}
    assert pd.read_csv(tmp_path / "likert.csv").columns.tolist()[1:] == list(aq.UX_ITEMS)


# --- robustness ---------------------------------------------------------------------------

def test_group_by_condition_and_truth(tmp_path: Path) -> None:
    imgs, labels = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    labels.mkdir()
    for name in ("a__x__redup__1.jpg", "a__x__redup__2.jpg", "b__x__terang__1.jpg", "b__x__terang__1__aug1.jpg", "odd.jpg"):
        (imgs / name).touch()
    groups = er.group_by_condition(imgs)
    assert {k: len(v) for k, v in groups.items()} == {"redup": 2, "terang": 1, "unknown": 1}
    (labels / "l.txt").write_text("0 0.5 0.5 0.2 0.1\n2 0.5 0.7 0.3 0.3\n")
    assert er.truth_of(labels / "l.txt") == (True, True)
    (labels / "m.txt").write_text("1 0.5 0.5 0.2 0.1\n")
    assert er.truth_of(labels / "m.txt") == (False, False)
    assert er.truth_of(labels / "none.txt") == (None, False)


def test_binary_score() -> None:
    s = er.BinaryScore()
    for truth, pred in ((True, True), (True, False), (False, True), (False, False), (True, True)):
        s.add(truth, pred)
    m = s.metrics()
    assert (m["precision"], m["recall"], m["accuracy"], m["n"]) == (2 / 3, 2 / 3, 0.6, 5)
    assert "Label draf" in er.render("landmark", "test", {}, "draf")


def test_pipeline_benchmark_cli_flags() -> None:
    bench = load_script("benchmark_fps")
    args = bench.parse_args(["--pipeline", "--max-speed", "--duration", "3"])
    assert args.pipeline and args.max_speed and args.duration == 3.0
