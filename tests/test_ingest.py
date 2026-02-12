"""Tests for the ingestion orchestrator."""

import pytest
from unittest.mock import patch, MagicMock, call
from contextlib import contextmanager

from src.crawler.site_crawler import CrawledActivity
from src.ingest import run_full_ingestion, run_health_check


# ── Helpers ───────────────────────────────────────────────────────

def make_sample_activities():
    return [
        CrawledActivity("Act1", "K-2", "Intro", "https://drive.google.com/drive/folders/1A", "drive_folder"),
        CrawledActivity("Act2", "3-5", "Design", "https://drive.google.com/file/d/1B/view", "drive_file"),
    ]


@contextmanager
def mock_conn_cm(mock_conn):
    yield mock_conn


# ── run_full_ingestion ────────────────────────────────────────────

class TestRunFullIngestion:
    @patch("src.ingest.complete_crawl_log")
    @patch("src.ingest.create_crawl_log")
    @patch("src.ingest.mark_missing_inactive")
    @patch("src.ingest.upsert_activity")
    @patch("src.ingest.get_connection")
    @patch("src.ingest.embed_batch")
    @patch("src.ingest.crawl_all")
    @patch("src.ingest.init_schema")
    def test_full_pipeline_success(
        self, mock_init, mock_crawl, mock_embed, mock_gc,
        mock_upsert, mock_mark, mock_create_log, mock_complete_log,
    ):
        # Setup
        activities = make_sample_activities()
        mock_crawl.return_value = activities
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        mock_conn = MagicMock()
        mock_gc.return_value = mock_conn_cm(mock_conn)
        mock_create_log.return_value = "log-uuid"
        mock_upsert.return_value = "inserted"
        mock_mark.return_value = 0

        # Execute
        summary = run_full_ingestion(triggered_by="test")

        # Verify
        mock_init.assert_called_once()
        mock_crawl.assert_called_once()
        mock_embed.assert_called_once()
        assert mock_upsert.call_count == 2
        mock_mark.assert_called_once()
        mock_complete_log.assert_called_once()

        assert summary["total_crawled"] == 2
        assert summary["added"] == 2
        assert summary["updated"] == 0
        assert summary["errors"] == 0

    @patch("src.ingest.complete_crawl_log")
    @patch("src.ingest.create_crawl_log")
    @patch("src.ingest.mark_missing_inactive")
    @patch("src.ingest.upsert_activity")
    @patch("src.ingest.get_connection")
    @patch("src.ingest.embed_batch")
    @patch("src.ingest.crawl_all")
    @patch("src.ingest.init_schema")
    def test_mixed_insert_and_update(
        self, mock_init, mock_crawl, mock_embed, mock_gc,
        mock_upsert, mock_mark, mock_create_log, mock_complete_log,
    ):
        activities = make_sample_activities()
        mock_crawl.return_value = activities
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        mock_conn = MagicMock()
        mock_gc.return_value = mock_conn_cm(mock_conn)
        mock_create_log.return_value = "log-uuid"
        mock_upsert.side_effect = ["inserted", "updated"]
        mock_mark.return_value = 1

        summary = run_full_ingestion()

        assert summary["added"] == 1
        assert summary["updated"] == 1
        assert summary["removed"] == 1

    @patch("src.ingest.complete_crawl_log")
    @patch("src.ingest.create_crawl_log")
    @patch("src.ingest.mark_missing_inactive")
    @patch("src.ingest.upsert_activity")
    @patch("src.ingest.get_connection")
    @patch("src.ingest.embed_batch")
    @patch("src.ingest.crawl_all")
    @patch("src.ingest.init_schema")
    def test_upsert_error_handling(
        self, mock_init, mock_crawl, mock_embed, mock_gc,
        mock_upsert, mock_mark, mock_create_log, mock_complete_log,
    ):
        activities = make_sample_activities()
        mock_crawl.return_value = activities
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        mock_conn = MagicMock()
        mock_gc.return_value = mock_conn_cm(mock_conn)
        mock_create_log.return_value = "log-uuid"
        mock_upsert.side_effect = ["inserted", Exception("DB error")]
        mock_mark.return_value = 0

        summary = run_full_ingestion()

        assert summary["added"] == 1
        assert summary["errors"] == 1

    @patch("src.ingest.crawl_all")
    @patch("src.ingest.init_schema")
    def test_no_activities_crawled(self, mock_init, mock_crawl):
        mock_crawl.return_value = []

        summary = run_full_ingestion()

        assert "error" in summary


# ── run_health_check ──────────────────────────────────────────────

class TestRunHealthCheck:
    @patch("src.ingest.update_health_status")
    @patch("src.ingest.verify_drive_links")
    @patch("src.ingest.get_connection")
    def test_checks_active_links(self, mock_gc, mock_verify, mock_update):
        mock_conn = MagicMock()
        mock_gc.return_value = mock_conn_cm(mock_conn)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("https://drive.google.com/drive/folders/1A", "Act1", "drive_folder"),
            ("https://docs.google.com/document/d/1B/edit", "Act2", "google_doc"),
        ]
        mock_conn.execute.return_value = mock_result

        mock_verify.return_value = {
            "https://drive.google.com/drive/folders/1A": True,
            "https://docs.google.com/document/d/1B/edit": False,
        }

        result = run_health_check()

        assert result["checked"] == 2
        assert result["broken"] == 1
        assert mock_update.call_count == 2

    @patch("src.ingest.get_connection")
    def test_no_links_to_check(self, mock_gc):
        mock_conn = MagicMock()
        mock_gc.return_value = mock_conn_cm(mock_conn)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result

        result = run_health_check()
        assert result["checked"] == 0
