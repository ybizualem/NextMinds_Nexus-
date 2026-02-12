"""Tests for database operations (all DB calls mocked)."""

import pytest
from unittest.mock import patch, MagicMock, call
from contextlib import contextmanager

from src.db.operations import (
    get_connection,
    init_schema,
    upsert_activity,
    mark_missing_inactive,
    create_crawl_log,
    complete_crawl_log,
    search_activities,
    get_activity_stats,
    update_health_status,
)


# ── Helper: mock connection context manager ───────────────────────

def make_mock_conn():
    """Create a mock psycopg connection with common methods."""
    conn = MagicMock()
    conn.execute.return_value = MagicMock()
    conn.commit = MagicMock()
    conn.close = MagicMock()
    return conn


@contextmanager
def mock_get_connection(mock_conn):
    """Context manager that yields a mock connection."""
    yield mock_conn


# ── init_schema ───────────────────────────────────────────────────

class TestInitSchema:
    @patch("src.db.operations.get_connection")
    def test_executes_schema_sql(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        init_schema()

        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


# ── upsert_activity ──────────────────────────────────────────────

class TestUpsertActivity:
    def test_insert_returns_inserted(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (True,)  # xmax = 0 → is_insert
        mock_conn.execute.return_value = mock_result

        result = upsert_activity(
            conn=mock_conn,
            activity_name="Test Activity",
            grade_band="K-2",
            stage="Intro",
            resource_url="https://drive.google.com/drive/folders/1A",
            resource_type="drive_folder",
            drive_id="1A",
            embedding=[0.1] * 384,
        )

        assert result == "inserted"
        mock_conn.execute.assert_called_once()

    def test_update_returns_updated(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (False,)  # existing row
        mock_conn.execute.return_value = mock_result

        result = upsert_activity(
            conn=mock_conn,
            activity_name="Test Activity",
            grade_band="K-2",
            stage="Intro",
            resource_url="https://drive.google.com/drive/folders/1A",
            resource_type="drive_folder",
            drive_id="1A",
            embedding=[0.1] * 384,
        )

        assert result == "updated"

    def test_passes_optional_fields(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (True,)
        mock_conn.execute.return_value = mock_result

        upsert_activity(
            conn=mock_conn,
            activity_name="Full Activity",
            grade_band="3-5",
            stage="Design",
            resource_url="https://drive.google.com/file/d/1B/view",
            resource_type="drive_file",
            drive_id="1B",
            embedding=[0.2] * 384,
            description="A test description",
            estimated_time=30,
            keywords=["test", "design"],
        )

        # Verify the SQL params include optional fields
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["desc"] == "A test description"
        assert params["time"] == 30
        assert params["keywords"] == ["test", "design"]


# ── mark_missing_inactive ────────────────────────────────────────

class TestMarkMissingInactive:
    def test_marks_activities_not_in_set(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("id1",), ("id2",)]
        mock_conn.execute.return_value = mock_result

        count = mark_missing_inactive(mock_conn, {"url1", "url2"})

        assert count == 2
        mock_conn.execute.assert_called_once()

    def test_empty_urls_returns_zero(self):
        mock_conn = make_mock_conn()
        count = mark_missing_inactive(mock_conn, set())
        assert count == 0
        mock_conn.execute.assert_not_called()


# ── create_crawl_log ─────────────────────────────────────────────

class TestCreateCrawlLog:
    def test_returns_log_id(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("uuid-123",)
        mock_conn.execute.return_value = mock_result

        log_id = create_crawl_log(mock_conn, triggered_by="cli")

        assert log_id == "uuid-123"

    def test_default_triggered_by(self):
        mock_conn = make_mock_conn()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("uuid-456",)
        mock_conn.execute.return_value = mock_result

        create_crawl_log(mock_conn)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["by"] == "manual"


# ── complete_crawl_log ───────────────────────────────────────────

class TestCompleteCrawlLog:
    def test_updates_log(self):
        mock_conn = make_mock_conn()

        complete_crawl_log(
            conn=mock_conn,
            log_id="uuid-123",
            added=10,
            updated=5,
            removed=2,
            errors=["error1"],
            status="completed",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["added"] == 10
        assert params["updated"] == 5
        assert params["removed"] == 2
        assert params["status"] == "completed"

    def test_no_errors(self):
        mock_conn = make_mock_conn()

        complete_crawl_log(
            conn=mock_conn,
            log_id="uuid-123",
            added=0,
            updated=0,
            removed=0,
        )

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["errors"] is None


# ── search_activities ─────────────────────────────────────────────

class TestSearchActivities:
    @patch("src.db.operations.get_connection")
    def test_basic_search(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        mock_result = MagicMock()
        mock_result.description = [
            MagicMock(name="id"),
            MagicMock(name="activity_name"),
            MagicMock(name="similarity"),
        ]
        # Override .name attribute on description mocks
        mock_result.description[0].name = "id"
        mock_result.description[1].name = "activity_name"
        mock_result.description[2].name = "similarity"
        mock_result.fetchall.return_value = [
            ("uuid-1", "Brainstorming 101", 0.95),
            ("uuid-2", "Prototype Testing", 0.80),
        ]
        mock_conn.execute.return_value = mock_result

        results = search_activities(query_embedding=[0.1] * 384, limit=5)

        assert len(results) == 2
        assert results[0]["activity_name"] == "Brainstorming 101"
        assert results[0]["similarity"] == 0.95

    @patch("src.db.operations.get_connection")
    def test_with_grade_filter(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        mock_result = MagicMock()
        mock_result.description = [MagicMock(name="id")]
        mock_result.description[0].name = "id"
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        search_activities(
            query_embedding=[0.1] * 384,
            grade_band="K-2",
        )

        # Verify the SQL contains grade_band filter
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "grade_band" in sql
        assert params["grade"] == "K-2"

    @patch("src.db.operations.get_connection")
    def test_with_all_filters(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        mock_result = MagicMock()
        mock_result.description = [MagicMock(name="id")]
        mock_result.description[0].name = "id"
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        search_activities(
            query_embedding=[0.1] * 384,
            grade_band="3-5",
            stage="Design",
            max_time=30,
            limit=3,
        )

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["grade"] == "3-5"
        assert params["stage"] == "%Design%"
        assert params["max_time"] == 30
        assert params["limit"] == 3


# ── get_activity_stats ────────────────────────────────────────────

class TestGetActivityStats:
    @patch("src.db.operations.get_connection")
    def test_returns_stats_dict(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        # Three sequential execute calls: main stats, by_grade_band, by_stage
        main_result = MagicMock()
        main_result.description = [
            MagicMock(name="total"),
            MagicMock(name="active"),
            MagicMock(name="grade_bands"),
            MagicMock(name="stages"),
            MagicMock(name="oldest_crawl"),
            MagicMock(name="newest_crawl"),
        ]
        for i, name in enumerate(["total", "active", "grade_bands", "stages", "oldest_crawl", "newest_crawl"]):
            main_result.description[i].name = name
        main_result.fetchone.return_value = (95, 93, 4, 6, "2025-01-01", "2025-01-02")

        grade_result = MagicMock()
        grade_result.fetchall.return_value = [("K-2", 20), ("3-5", 25)]

        stage_result = MagicMock()
        stage_result.fetchall.return_value = [("Intro", 15), ("Design", 20)]

        mock_conn.execute.side_effect = [main_result, grade_result, stage_result]

        stats = get_activity_stats()

        assert stats["total"] == 95
        assert stats["active"] == 93
        assert stats["by_grade_band"]["K-2"] == 20
        assert stats["by_stage"]["Design"] == 20


# ── update_health_status ─────────────────────────────────────────

class TestUpdateHealthStatus:
    @patch("src.db.operations.get_connection")
    def test_updates_accessible(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        update_health_status("https://drive.google.com/drive/folders/1A", True)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["accessible"] is True
        assert params["url"] == "https://drive.google.com/drive/folders/1A"
        mock_conn.commit.assert_called_once()

    @patch("src.db.operations.get_connection")
    def test_updates_inaccessible(self, mock_gc):
        mock_conn = make_mock_conn()
        mock_gc.return_value = mock_get_connection(mock_conn)

        update_health_status("https://drive.google.com/drive/folders/1A", False)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params["accessible"] is False
