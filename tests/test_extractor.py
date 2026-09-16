"""
test_extractor.py
Basic unit tests for skill/experience/education extraction.
Run with: python -m pytest tests/ (or python -m unittest tests.test_extractor)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extractor import (
    extract_skills,
    extract_years_experience,
    extract_education_level,
)


class TestExtractSkills(unittest.TestCase):
    def test_finds_known_skills(self):
        text = "Experienced in Python, Django, and PostgreSQL development."
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("django", skills)
        self.assertIn("postgresql", skills)

    def test_does_not_false_match_substring(self):
        # "go" should not match inside "google" or "mango"
        text = "Worked with Google Cloud and ate a mango."
        skills = extract_skills(text, vocab=["go"])
        self.assertEqual(skills, set())

    def test_case_insensitive(self):
        text = "PYTHON and django"
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("django", skills)


class TestExtractYearsExperience(unittest.TestCase):
    def test_explicit_years_phrase(self):
        self.assertEqual(extract_years_experience("5 years of experience in backend."), 5.0)

    def test_no_match_returns_zero(self):
        self.assertEqual(extract_years_experience("Recent graduate with strong Python skills."), 0.0)

    def test_decimal_years(self):
        self.assertEqual(extract_years_experience("2.5 years of experience"), 2.5)


class TestExtractEducationLevel(unittest.TestCase):
    def test_detects_bachelor(self):
        self.assertEqual(extract_education_level("B.Tech in Computer Science"), "bachelor")

    def test_detects_master(self):
        self.assertEqual(extract_education_level("M.Sc in Computer Science"), "master")

    def test_detects_phd(self):
        self.assertEqual(extract_education_level("PhD in Computer Science"), "phd")

    def test_unknown_when_absent(self):
        self.assertEqual(extract_education_level("Self-taught developer, no formal degree listed."), "unknown")


if __name__ == "__main__":
    unittest.main()
