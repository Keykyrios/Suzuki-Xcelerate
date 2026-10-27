"""Repository: parquet round-trips, DuckDB queries, document storage."""

from __future__ import annotations

import json

import pytest

from app.data_access.repository import Repository, demand_record_rows
from app.demand.demand_model import generate_demand_records


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository(tmp_path)


def test_record_round_trip_preserves_values(repo):
    records = [{"id": 1, "value": 2.5, "label": "a"}, {"id": 2, "value": 3.5, "label": "b"}]
    path = repo.write_records("sample", records)
    assert path.name == "sample.parquet"
    frame = repo.read_table("sample")
    assert list(frame.columns) == ["id", "value", "label"]
    assert len(frame) == 2
    assert frame["value"].tolist() == pytest.approx([2.5, 3.5])


def test_synthetic_and_processed_are_separate_stores(repo):
    repo.write_records("both", [{"x": 1}], synthetic=True)
    repo.write_records("both", [{"x": 2}], synthetic=False)
    assert repo.read_table("both", synthetic=True)["x"].tolist() == [1]
    assert repo.read_table("both", synthetic=False)["x"].tolist() == [2]
    assert set(repo.list_tables()) >= {"both"}


def test_missing_table_reports_the_expected_path(repo):
    with pytest.raises(FileNotFoundError) as excinfo:
        repo.read_table("does_not_exist")
    assert "does_not_exist" in str(excinfo.value)


def test_json_document_round_trip(repo):
    document = {"scenario_id": "demo", "nested": {"a": [1, 2]}, "flag": True}
    repo.write_json_document("scenario_demo", document)
    assert repo.read_json_document("scenario_demo") == document


def test_json_document_requires_a_document_table(repo):
    repo.write_records("plain", [{"x": 1}])
    with pytest.raises(ValueError):
        repo.read_json_document("plain")
    with pytest.raises(FileNotFoundError):
        repo.read_json_document("absent")


def test_duckdb_queries_the_stored_parquet(repo, demo_scenario, cfg):
    records = generate_demand_records(demo_scenario.graph.zones, 60, seed=3, cfg=cfg)
    repo.write_records("demand_records", demand_record_rows(records), synthetic=True)
    table_path = str(repo.synthetic / "demand_records.parquet")
    frame = repo.query(
        "SELECT service_class, SUM(request_count) AS requests "
        f"FROM read_parquet('{table_path}') GROUP BY 1 ORDER BY requests DESC"
    )
    assert not frame.empty
    assert {"service_class", "requests"} <= set(frame.columns)
    assert frame["requests"].sum() > 0


def test_query_examples_point_at_real_tables(repo, demo_scenario, cfg):
    records = generate_demand_records(demo_scenario.graph.zones, 40, seed=5, cfg=cfg)
    repo.write_records("demand_records", demand_record_rows(records), synthetic=True)
    examples = repo.query_examples()
    assert set(examples) == {"demand_records", "benchmark_metrics"}
    frame = repo.query(examples["demand_records"])
    assert not frame.empty


def test_rowsAreFlattenedFromDemandRecords(repo, demo_scenario, cfg):
    records = generate_demand_records(demo_scenario.graph.zones, 10, seed=9, cfg=cfg)
    rows = demand_record_rows(records)
    assert rows
    for row in rows:
        assert set(row) == {
            "record_id",
            "hour_of_day",
            "day_type",
            "service_class",
            "origin_zone_id",
            "destination_zone_id",
            "request_count",
            "source_status",
        }
        assert 0 <= row["hour_of_day"] <= 23
        assert row["source_status"] == "SYNTHETIC"


def test_stored_document_is_valid_json_text(repo):
    repo.write_json_document("doc", {"value": 1})
    path = repo.processed / "doc.parquet"
    assert path.is_file()
    frame = repo.read_table("doc")
    parsed = json.loads(frame["document_json"].iloc[0])
    assert parsed == {"value": 1}