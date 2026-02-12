"""Tests for the CTIC website crawler."""

import pytest
from unittest.mock import patch, MagicMock

from src.crawler.site_crawler import (
    extract_drive_id,
    classify_url,
    normalize_stage_name,
    CTICSectionParser,
    CrawledActivity,
    crawl_grade_band,
    crawl_all,
    verify_drive_links,
)
from tests.conftest import SAMPLE_CTIC_HTML


# ── extract_drive_id ──────────────────────────────────────────────

class TestExtractDriveId:
    def test_drive_folder_url(self):
        url = "https://drive.google.com/drive/folders/1ABCdef_GHI?usp=sharing"
        assert extract_drive_id(url) == "1ABCdef_GHI"

    def test_drive_file_url(self):
        url = "https://drive.google.com/file/d/1XYZabc-123/view?usp=sharing"
        assert extract_drive_id(url) == "1XYZabc-123"

    def test_google_doc_url(self):
        url = "https://docs.google.com/document/d/1DOCid_456/edit"
        assert extract_drive_id(url) == "1DOCid_456"

    def test_google_slides_url(self):
        url = "https://docs.google.com/presentation/d/1SLIDEid/edit"
        assert extract_drive_id(url) == "1SLIDEid"

    def test_youtube_url_returns_none(self):
        url = "https://www.youtube.com/watch?v=abc123"
        assert extract_drive_id(url) is None

    def test_unrecognized_url_returns_none(self):
        url = "https://example.com/something"
        assert extract_drive_id(url) is None

    def test_empty_string(self):
        assert extract_drive_id("") is None


# ── classify_url ──────────────────────────────────────────────────

class TestClassifyUrl:
    def test_drive_folder(self):
        assert classify_url("https://drive.google.com/drive/folders/abc") == "drive_folder"

    def test_drive_file(self):
        assert classify_url("https://drive.google.com/file/d/abc/view") == "drive_file"

    def test_google_doc(self):
        assert classify_url("https://docs.google.com/document/d/abc/edit") == "google_doc"

    def test_youtube_full(self):
        assert classify_url("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtube_short(self):
        assert classify_url("https://youtu.be/abc") == "youtube"

    def test_other(self):
        assert classify_url("https://example.com/resource") == "other"


# ── normalize_stage_name ─────────────────────────────────────────

class TestNormalizeStageName:
    def test_stage_prefix(self):
        assert normalize_stage_name("Stage 1: Introduction To Inventing") == "Introduction To Inventing"

    def test_step_prefix(self):
        assert normalize_stage_name("Step 4: Engineering Design Process") == "Engineering Design Process"

    def test_no_prefix(self):
        assert normalize_stage_name("Supporting Materials") == "Supporting Materials"

    def test_whitespace(self):
        # Leading whitespace is stripped, but normalize_stage_name strips outer whitespace
        # then applies regex. Input with leading spaces: regex won't match the prefix.
        assert normalize_stage_name("Stage 2:  Identifying and Ideating  ") == "Identifying and Ideating"

    def test_leading_whitespace_no_prefix_match(self):
        # Leading spaces prevent the Stage/Step prefix regex from matching
        result = normalize_stage_name("  Stage 2: Identifying")
        assert result == "Stage 2: Identifying"  # prefix not stripped due to leading spaces

    def test_case_insensitive(self):
        assert normalize_stage_name("STAGE 3: Research") == "Research"


# ── CrawledActivity dataclass ────────────────────────────────────

class TestCrawledActivity:
    def test_drive_id_auto_extracted(self):
        a = CrawledActivity(
            activity_name="Test",
            grade_band="K-2",
            stage="Intro",
            resource_url="https://drive.google.com/drive/folders/1MY_FOLDER",
            resource_type="drive_folder",
        )
        assert a.drive_id == "1MY_FOLDER"

    def test_drive_id_explicit(self):
        a = CrawledActivity(
            activity_name="Test",
            grade_band="K-2",
            stage="Intro",
            resource_url="https://drive.google.com/drive/folders/1MY_FOLDER",
            resource_type="drive_folder",
            drive_id="OVERRIDE",
        )
        assert a.drive_id == "OVERRIDE"

    def test_youtube_no_drive_id(self):
        a = CrawledActivity(
            activity_name="Video",
            grade_band="3-5",
            stage="Intro",
            resource_url="https://www.youtube.com/watch?v=abc",
            resource_type="youtube",
        )
        assert a.drive_id is None


# ── CTICSectionParser ─────────────────────────────────────────────

class TestCTICSectionParser:
    def test_parses_activities_from_html(self):
        parser = CTICSectionParser()
        parser.feed(SAMPLE_CTIC_HTML)
        parser.close()

        # 4 raw activities: the non-RENDERED card heading is skipped,
        # but the link under it still attaches to the previous activity_name
        assert len(parser.activities) == 4

    def test_stage_extraction(self):
        parser = CTICSectionParser()
        parser.feed(SAMPLE_CTIC_HTML)
        parser.close()

        stages = [a["stage"] for a in parser.activities]
        assert "Introduction To Inventing" in stages
        assert "Identifying and Ideating" in stages

    def test_activity_names(self):
        parser = CTICSectionParser()
        parser.feed(SAMPLE_CTIC_HTML)
        parser.close()

        names = [a["activity_name"] for a in parser.activities]
        assert "Intro to Inventing" in names
        assert "What is an Inventor?" in names
        assert "Brainstorming 101" in names

    def test_resource_urls(self):
        parser = CTICSectionParser()
        parser.feed(SAMPLE_CTIC_HTML)
        parser.close()

        urls = [a["resource_url"] for a in parser.activities]
        assert "https://drive.google.com/drive/folders/1AAA_folder_id" in urls
        assert "https://docs.google.com/document/d/1BBB_doc_id/edit" in urls

    def test_non_rendered_heading_skipped(self):
        """The non-RENDERED card heading is not captured as a new activity name,
        but links underneath still attach to the previous activity_name.
        Deduplication happens in crawl_grade_band, not the parser."""
        parser = CTICSectionParser()
        parser.feed(SAMPLE_CTIC_HTML)
        parser.close()

        # The parser captures the duplicate link (same name+url), but
        # crawl_grade_band deduplicates by (name, url)
        brainstorm = [a for a in parser.activities if a["activity_name"] == "Brainstorming 101"]
        assert len(brainstorm) == 2  # parser sees both, dedup is downstream

    def test_empty_html(self):
        parser = CTICSectionParser()
        parser.feed("<html><body></body></html>")
        parser.close()
        assert parser.activities == []


# ── crawl_grade_band ──────────────────────────────────────────────

class TestCrawlGradeBand:
    @patch("src.crawler.site_crawler.requests.get")
    def test_returns_crawled_activities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CTIC_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        activities = crawl_grade_band("K-2", "/k-2-curriculum")

        assert len(activities) == 3
        assert all(isinstance(a, CrawledActivity) for a in activities)
        assert all(a.grade_band == "K-2" for a in activities)

    @patch("src.crawler.site_crawler.requests.get")
    def test_deduplicates_by_name_and_url(self, mock_get):
        # HTML with duplicate (name, url) pair
        html = """
        <div data-aid="SECTION__TITLE__1"><h2>Stage 1: Intro</h2></div>
        <h4 data-ux="ContentCardHeading" data-aid="CARD1_RENDERED">Same Activity</h4>
        <a href="https://drive.google.com/drive/folders/1SAME">Link</a>
        <a href="https://drive.google.com/drive/folders/1SAME">Duplicate Link</a>
        """
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        activities = crawl_grade_band("K-2", "/k-2-curriculum")
        # Same (name, url) should be deduplicated
        assert len(activities) == 1

    @patch("src.crawler.site_crawler.requests.get")
    def test_http_error_raises(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(requests.RequestException):
            crawl_grade_band("K-2", "/k-2-curriculum")


# ── crawl_all ─────────────────────────────────────────────────────

class TestCrawlAll:
    @patch("src.crawler.site_crawler.time.sleep")  # skip sleeps
    @patch("src.crawler.site_crawler.crawl_grade_band")
    def test_aggregates_all_grade_bands(self, mock_crawl, mock_sleep):
        mock_crawl.return_value = [
            CrawledActivity("Act1", "K-2", "Intro", "https://drive.google.com/drive/folders/1A", "drive_folder"),
        ]

        activities = crawl_all()
        assert len(activities) == 4  # one per grade band (4 bands)
        assert mock_crawl.call_count == 4

    @patch("src.crawler.site_crawler.time.sleep")
    @patch("src.crawler.site_crawler.crawl_grade_band")
    def test_handles_grade_band_failure(self, mock_crawl, mock_sleep):
        import requests
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise requests.RequestException("Server error")
            return [CrawledActivity("Act", args[0], "Intro", "https://drive.google.com/drive/folders/1A", "drive_folder")]

        mock_crawl.side_effect = side_effect

        activities = crawl_all()
        assert len(activities) == 3  # 4 bands, 1 failed = 3 succeed


# ── verify_drive_links ────────────────────────────────────────────

class TestVerifyDriveLinks:
    @patch("src.crawler.site_crawler.time.sleep")
    @patch("src.crawler.site_crawler.requests.head")
    def test_accessible_links(self, mock_head, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_head.return_value = mock_resp

        activity = CrawledActivity(
            "Test", "K-2", "Intro",
            "https://drive.google.com/drive/folders/1A",
            "drive_folder",
        )
        results = verify_drive_links([activity])
        assert results[activity.resource_url] is True

    @patch("src.crawler.site_crawler.time.sleep")
    @patch("src.crawler.site_crawler.requests.head")
    def test_broken_links(self, mock_head, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_head.return_value = mock_resp

        activity = CrawledActivity(
            "Test", "K-2", "Intro",
            "https://drive.google.com/drive/folders/1A",
            "drive_folder",
        )
        results = verify_drive_links([activity])
        assert results[activity.resource_url] is False

    @patch("src.crawler.site_crawler.time.sleep")
    @patch("src.crawler.site_crawler.requests.head")
    def test_skips_youtube(self, mock_head, mock_sleep):
        activity = CrawledActivity(
            "Video", "K-2", "Intro",
            "https://www.youtube.com/watch?v=abc",
            "youtube",
        )
        results = verify_drive_links([activity])
        assert results[activity.resource_url] is True
        mock_head.assert_not_called()
