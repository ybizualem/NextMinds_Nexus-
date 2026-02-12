"""Shared fixtures for CTIC curriculum engine tests."""

import pytest
from src.crawler.site_crawler import CrawledActivity


@pytest.fixture
def sample_activity():
    """A single sample CrawledActivity."""
    return CrawledActivity(
        activity_name="Brainstorming 101",
        grade_band="K-2",
        stage="Identifying and Ideating",
        resource_url="https://drive.google.com/drive/folders/1ABCdef",
        resource_type="drive_folder",
    )


@pytest.fixture
def sample_activities():
    """A list of sample CrawledActivities across grade bands."""
    return [
        CrawledActivity(
            activity_name="Brainstorming 101",
            grade_band="K-2",
            stage="Identifying and Ideating",
            resource_url="https://drive.google.com/drive/folders/1ABCdef",
            resource_type="drive_folder",
        ),
        CrawledActivity(
            activity_name="Prototype Testing",
            grade_band="3-5",
            stage="Engineering Design Process",
            resource_url="https://drive.google.com/file/d/1XYZabc/view",
            resource_type="drive_file",
        ),
        CrawledActivity(
            activity_name="Patent Basics",
            grade_band="6-8",
            stage="Introduction To Inventing",
            resource_url="https://docs.google.com/document/d/1DOCid/edit",
            resource_type="google_doc",
        ),
        CrawledActivity(
            activity_name="Inventor Spotlight",
            grade_band="9-12",
            stage="Introduction To Inventing",
            resource_url="https://www.youtube.com/watch?v=abc123",
            resource_type="youtube",
        ),
    ]


# Realistic HTML snippet from a CTIC grade-band page
SAMPLE_CTIC_HTML = """
<html>
<body>
<div data-aid="SECTION__TITLE__1">
  <h2>Stage 1: Introduction To Inventing</h2>
</div>
<div data-ux="ContentCard">
  <h4 data-ux="ContentCardHeading" data-aid="SECTION1_CARD1_RENDERED">Intro to Inventing</h4>
  <a href="https://drive.google.com/drive/folders/1AAA_folder_id">View Resources</a>
</div>
<div data-ux="ContentCard">
  <h4 data-ux="ContentCardHeading" data-aid="SECTION1_CARD2_RENDERED">What is an Inventor?</h4>
  <a href="https://docs.google.com/document/d/1BBB_doc_id/edit">Lesson Plan</a>
</div>

<div data-aid="SECTION__TITLE__2">
  <h2>Stage 2: Identifying and Ideating</h2>
</div>
<div data-ux="ContentCard">
  <h4 data-ux="ContentCardHeading" data-aid="SECTION2_CARD1_RENDERED">Brainstorming 101</h4>
  <a href="https://drive.google.com/drive/folders/1CCC_brainstorm_id">Activities</a>
</div>

<!-- Duplicate rendering (non-RENDERED) — should be ignored -->
<div data-ux="ContentCard">
  <h4 data-ux="ContentCardHeading" data-aid="SECTION2_CARD1_OTHER">Brainstorming 101</h4>
  <a href="https://drive.google.com/drive/folders/1CCC_brainstorm_id">Activities</a>
</div>
</body>
</html>
"""
