"""Parquet-backed repository with DuckDB query access.

Repeatable experiment data lands here: demand histories, scenario snapshots,
decisions, and benchmark metrics. Files are deterministic given their seed and
run ID, and DuckDB reads them directly for analysis without a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.config.loader import find_project_root


class Repository:
    """Filesystem parquet store under data/processed with DuckDB queries."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()
        self.processed = self.root / "data" / "processed"
        self.synthetic = self.root / "data" / "synthetic"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.synthetic.mkdir(parents=True, exist_ok=True)

    # -- writes -----------------------------------------------------------
    def write_records(self, name: str, records: list[dict[str, Any]], *, synthetic: bool = False) -> Path:
        """Write a list of flat records as a parquet table."""
        directory = self.synthetic if synthetic else self.processed
        path = directory / f"{name}.parquet"
        table = pa.Table.from_pylist(records)
        pq.write_table(table, path)
        return path

    def write_json_document(self, name: str, document: dict[str, Any]) -> Path:
        """Store a nested document (scenario/decision) as a single-row table."""
        path = self.processed / f"{name}.parquet"
        payload = json.dumps(document, sort_keys=True, default=str)
        table = pa.Table.from_pylist([{"name": name, "document_json": payload}])
        pq.write_table(table, path)
        return path

    # -- reads ------------------------------------------------------------
    def read_table(self, name: str, *, synthetic: bool = False) -> pd.DataFrame:
        directory = self.synthetic if synthetic else self.processed
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"no stored table named '{name}' at {path}")
        return pq.read_table(path).to_pandas()

    def read_json_document(self, name: str) -> dict[str, Any]:
        frame = self.read_table(name)
        if "document_json" not in frame.columns or frame.empty:
            raise ValueError(f"table '{name}' does not contain a stored document")
        return json.loads(frame["document_json"].iloc[0])

    def list_tables(self) -> list[str]:
        names = [p.stem for p in self.processed.glob("*.parquet")]
        names += [p.stem for p in self.synthetic.glob("*.parquet")]
        return sorted(names)

    # -- duckdb -----------------------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        """Run a read-only DuckDB query against the stored parquet files."""
        import duckdb

        connection = duckdb.connect(database=":memory:")
        try:
            return connection.execute(sql).df()
        finally:
            connection.close()

    def query_examples(self) -> dict[str, str]:
        """Ready-made queries used by the evidence views."""
        return {
            "demand_records": (
                "SELECT origin_zone_id, destination_zone_id, service_class, "
                "SUM(request_count) AS requests "
                f"FROM READ_PARQUET('{self.synthetic / 'demand_records.parquet'}') "
                "GROUP BY 1, 2, 3 ORDER BY requests DESC LIMIT 20"
            ),
            "benchmark_metrics": (
                "SELECT scenario_id, baseline_loss_inr, optimized_loss_inr, "
                "optimized_loss_inr - baseline_loss_inr AS delta_inr "
                f"FROM READ_PARQUET('{self.processed / 'benchmark_metrics.parquet'}') "
                "ORDER BY scenario_id"
            ),
        }


def demand_record_rows(records) -> list[dict[str, Any]]:
    """Flatten demand records for parquet storage."""
    return [
        {
            "record_id": r.record_id,
            "hour_of_day": r.hour_of_day,
            "day_type": r.day_type.value,
            "service_class": r.service_class.value,
            "origin_zone_id": r.origin_zone_id,
            "destination_zone_id": r.destination_zone_id,
            "request_count": r.request_count,
            "source_status": r.source_status.value,
        }
        for r in records
    ]
