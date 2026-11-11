"""Shared CSV loader for all Streamlit domain pages.

Place this file in the repository root, beside Home.py. CSV files belong in
<data root>/data/ and should include the domain key in the filename, e.g.
healthcare_queue.csv or call_center_part_01.csv.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"


def _normalize(value: str) -> str:
    """Case-insensitive filename matching; ignore spaces, '_' and '-'."""
    return re.sub(r"[\s_-]+", "", str(value).casefold())


def find_domain_files(domain: str, data_dir: Path | str | None = None) -> list[Path]:
    """Find CSVs whose filename contains the normalized domain key."""
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    if not folder.is_dir():
        return []

    key = _normalize(domain)
    return sorted(
        (
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".csv"
            and key in _normalize(path.stem)
        ),
        key=lambda path: path.name.casefold(),
    )


def _select_files(files: list[Path]) -> list[Path]:
    """Prefer one complete CSV; otherwise combine matching part CSVs."""
    complete = [path for path in files if "part" not in _normalize(path.stem)]
    if complete:
        # Prefer the shortest matching filename to avoid choosing backup copies.
        return [min(complete, key=lambda path: (len(path.name), path.name.casefold()))]
    return [path for path in files if "part" in _normalize(path.stem)]


def _read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Read UTF-8 CSVs, falling back to Latin-1 for older exports."""
    try:
        return pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(contents), encoding="latin-1")


@st.cache_data(show_spinner=False)
def _read_paths(paths: tuple[str, ...], file_mtimes: tuple[float, ...]) -> pd.DataFrame:
    frames = []
    for filename in paths:
        path = Path(filename)
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="latin-1")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    data.columns = data.columns.astype(str).str.strip()
    return data


def load_domain_data(
    domain: str,
    uploaded: BinaryIO | bytes | None = None,
    *,
    data_dir: Path | str | None = None,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    """Load an uploaded CSV or the matching file(s) from ``data/``.

    Returns None only when no matching file exists. ``required_columns`` is
    optional because column schemas differ across domains; if supplied, a
    ValueError describes any missing column names.
    """
    if uploaded is not None:
        contents = uploaded if isinstance(uploaded, bytes) else uploaded.getvalue()
        data = _read_csv_bytes(contents)
        data.columns = data.columns.astype(str).str.strip()
    else:
        files = _select_files(find_domain_files(domain, data_dir))
        if not files:
            return None
        data = _read_paths(
            tuple(str(path) for path in files),
            tuple(path.stat().st_mtime for path in files),
        )

    if required_columns is not None:
        missing = sorted(set(required_columns) - set(data.columns))
        if missing:
            raise ValueError(
                f"The {domain!r} CSV is missing required columns: {', '.join(missing)}. "
                f"Columns found: {', '.join(data.columns)}"
            )
    return data


def explain_missing(domain: str, data_dir: Path | str | None = None) -> None:
    """Show actionable Streamlit diagnostics when no matching CSV is found."""
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    st.error(f"No CSV file matched the domain key {domain!r}.")
    st.code(f"Searched folder: {folder}\nFolder exists: {folder.is_dir()}")
    if folder.is_dir():
        names = sorted(path.name for path in folder.iterdir())
        st.code("Files found: " + (", ".join(names) if names else "(none)"))
    st.info(
        f"Upload a CSV in the sidebar or place one in {folder} with "
        f"{domain!r} in its filename."
    )

