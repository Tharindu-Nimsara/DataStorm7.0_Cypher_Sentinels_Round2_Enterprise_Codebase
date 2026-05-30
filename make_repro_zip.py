"""Build the 'Reproducible Codebase' zip for the Data Storm submission (form slot 3).

Bundles everything a judge needs to run the pipeline end-to-end in minutes — code + the raw
bronze CSVs + the 16k-file POI cache + the precomputed silver/gold/model/outputs — so they
never have to wait ~5 hours on the Overpass scrape. Deliberately EXCLUDES secrets and the
AI-collaboration scaffolding, and KEEPS the GENAI_LOG transparency deliverable.

Usage:  python make_repro_zip.py        ->  dist/cypher_sentinels_reproducible_codebase.zip

The zip is self-contained: unzip, `pip install -r requirements.txt`, and the run order in the
README reproduces every output. We verified the uncompressed payload is ~305 MB, comfortably
under the form's 1 GB limit (and zips smaller).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "dist"
OUT_ZIP = OUT_DIR / "cypher_sentinels_reproducible_codebase.zip"

# What goes in. Directories are walked recursively; files added as-is.
INCLUDE_DIRS = ["src", "app", "data", "models", "reports", "notebooks"]
INCLUDE_FILES = ["README.md", "requirements.txt", "GENAI_LOG.md", ".env.example", ".gitignore"]

# What never goes in — secrets, agent scaffolding, VCS, caches, build artifacts. Matched as
# path *parts* (any directory level) or by suffix.
EXCLUDE_PARTS = {".git", ".claude", "__pycache__", ".ipynb_checkpoints", "dist", ".dist",
                 ".vscode", ".idea"}
EXCLUDE_NAMES = {".env", "CLAUDE.md", "SOLO_PLAYBOOK.md", "tasks.md",
                 "paper_tasks.md", "deck_tasks.md"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in rel.parts):
        return True
    if rel.name in EXCLUDE_NAMES:
        return True
    if rel.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def iter_files():
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(ROOT)
                if not _excluded(rel):
                    yield p, rel
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.is_file() and not _excluded(p.relative_to(ROOT)):
            yield p, p.relative_to(ROOT)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    files = list(iter_files())
    if not files:
        print("Nothing to zip — run the pipeline first.", file=sys.stderr)
        return 1

    # Safety net: assert no secret/scaffolding slipped through before we write the zip.
    bad = [str(rel) for _, rel in files
           if rel.name in EXCLUDE_NAMES or any(p in EXCLUDE_PARTS for p in rel.parts)]
    assert not bad, f"refusing to zip excluded files: {bad}"

    total_bytes = sum(p.stat().st_size for p, _ in files)
    print(f"Zipping {len(files)} files, {total_bytes/1e6:.0f} MB uncompressed -> {OUT_ZIP.name}")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()                      # idempotent: rebuild fresh
    root_name = "cypher_sentinels_codebase"   # everything nests under one clean folder
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p, rel in files:
            zf.write(p, arcname=str(Path(root_name) / rel))

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print(f"Wrote {OUT_ZIP} — {size_mb:.0f} MB compressed.")
    if size_mb > 1000:
        print("WARNING: zip exceeds 1 GB submission limit!", file=sys.stderr)
        return 1
    # Final guard: list anything sensitive that (shouldn't) be inside
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        leaked = [n for n in names if Path(n).name in EXCLUDE_NAMES]
        print(f"Secret/scaffolding files in zip: {leaked if leaked else 'none (OK)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
