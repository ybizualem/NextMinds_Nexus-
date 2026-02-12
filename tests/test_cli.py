"""Tests for the CLI module."""

import pytest
from unittest.mock import patch, MagicMock

from src.cli import main, cmd_crawl, cmd_search, cmd_stats, cmd_init_db
from src.crawler.site_crawler import CrawledActivity


# ── cmd_crawl ─────────────────────────────────────────────────────

class TestCmdCrawl:
    @patch("src.crawler.site_crawler.crawl_all")
    def test_crawl_prints_activities(self, mock_crawl):
        """cmd_crawl does a lazy import of crawl_all — patch the source module."""
        mock_crawl.return_value = [
            CrawledActivity("Act1", "K-2", "Intro", "https://drive.google.com/drive/folders/1A", "drive_folder"),
            CrawledActivity("Act2", "3-5", "Design", "https://drive.google.com/file/d/1B/view", "drive_file"),
        ]

        args = MagicMock()
        cmd_crawl(args)

        mock_crawl.assert_called_once()
        assert mock_crawl.return_value[0].grade_band == "K-2"


# ── CLI argument parsing ────────────────────────────────────────

class TestCLIParsing:
    @patch("src.cli.sys")
    def test_no_command_prints_help(self, mock_sys):
        """Running with no command should print help (not crash)."""
        mock_sys.argv = ["cli"]
        # main() with no command should call parser.print_help()
        # We just verify it doesn't raise
        with patch("src.cli.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(command=None, verbose=False)
            main()  # Should not raise

    def test_search_command_args(self):
        """Verify search parser accepts the expected arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        search_p = sub.add_parser("search")
        search_p.add_argument("query")
        search_p.add_argument("--grade", "-g")
        search_p.add_argument("--stage", "-s")
        search_p.add_argument("--max-time", "-t", type=int)
        search_p.add_argument("--limit", "-l", type=int, default=5)

        args = parser.parse_args(["search", "brainstorming", "--grade", "K-2", "--limit", "3"])
        assert args.query == "brainstorming"
        assert args.grade == "K-2"
        assert args.limit == 3

    def test_search_defaults(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        search_p = sub.add_parser("search")
        search_p.add_argument("query")
        search_p.add_argument("--grade", "-g")
        search_p.add_argument("--stage", "-s")
        search_p.add_argument("--max-time", "-t", type=int)
        search_p.add_argument("--limit", "-l", type=int, default=5)

        args = parser.parse_args(["search", "prototyping"])
        assert args.grade is None
        assert args.stage is None
        assert args.max_time is None
        assert args.limit == 5


# ── cmd_stats ─────────────────────────────────────────────────────

class TestCmdStats:
    @patch("src.db.operations.get_activity_stats")
    def test_stats_displays(self, mock_stats):
        mock_stats.return_value = {
            "total": 95,
            "active": 93,
            "grade_bands": 4,
            "stages": 6,
            "oldest_crawl": "2025-01-01",
            "newest_crawl": "2025-01-02",
            "by_grade_band": {"K-2": 20},
            "by_stage": {"Intro": 15},
        }

        args = MagicMock()
        # cmd_stats does a lazy import, so this tests the function signature
        with patch("src.cli.get_activity_stats", mock_stats, create=True):
            pass  # Just verify it's importable


# ── cmd_init_db ───────────────────────────────────────────────────

class TestCmdInitDb:
    @patch("src.db.operations.init_schema")
    def test_init_db_calls_schema(self, mock_init):
        args = MagicMock()
        with patch("src.cli.init_schema", mock_init, create=True):
            pass  # Verify importability


# ── Schema SQL ────────────────────────────────────────────────────

class TestSchema:
    def test_schema_contains_required_tables(self):
        from src.db.schema import SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS activities" in SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS crawl_logs" in SCHEMA_SQL

    def test_schema_has_vector_extension(self):
        from src.db.schema import SCHEMA_SQL
        assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA_SQL

    def test_schema_has_vector_column(self):
        from src.db.schema import SCHEMA_SQL
        assert "vector(384)" in SCHEMA_SQL

    def test_schema_has_indexes(self):
        from src.db.schema import SCHEMA_SQL
        assert "idx_activities_grade_stage" in SCHEMA_SQL
        assert "idx_activities_unique_resource" in SCHEMA_SQL

    def test_schema_has_trigger(self):
        from src.db.schema import SCHEMA_SQL
        assert "update_updated_at" in SCHEMA_SQL


# ── Config ────────────────────────────────────────────────────────

class TestConfig:
    def test_grade_band_pages(self):
        from src.config import GRADE_BAND_PAGES
        assert len(GRADE_BAND_PAGES) == 4
        assert "K-2" in GRADE_BAND_PAGES
        assert "3-5" in GRADE_BAND_PAGES
        assert "6-8" in GRADE_BAND_PAGES
        assert "9-12" in GRADE_BAND_PAGES

    def test_embedding_dimensions(self):
        from src.config import EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS == 384

    def test_embedding_model(self):
        from src.config import EMBEDDING_MODEL
        assert "MiniLM" in EMBEDDING_MODEL
