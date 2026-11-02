"""Repository hygiene: no dash characters, no absolute drive paths, no secrets in tracked text files"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", "runs", "exchange", ".git"}
SKIP_FILES = {"SANITIZATION_REPORT.md"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".csv", ".cfg", ".ini", ""}

# Built from code points so this file does not contain the characters it forbids
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)
DRIVE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")
KEY_LIKE = re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}")


def text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS or path.name in SKIP_FILES or path.name.endswith(".egg-info"):
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            yield path


def test_there_are_files_to_check():
    assert len(list(text_files())) > 20


def test_no_em_or_en_dashes():
    offenders = [
        str(p.relative_to(ROOT))
        for p in text_files()
        if EM_DASH in p.read_text(encoding="utf-8", errors="ignore")
        or EN_DASH in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_no_absolute_drive_paths():
    offenders = [
        str(p.relative_to(ROOT))
        for p in text_files()
        if p.suffix != ".csv" and DRIVE_PATH.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def test_no_api_key_like_strings():
    offenders = [
        str(p.relative_to(ROOT))
        for p in text_files()
        if KEY_LIKE.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def test_license_file_is_mit():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Filipe Fonseca da Guia" in text
