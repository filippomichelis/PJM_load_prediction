"""
Utility to download PJM HRL load data either from the PJM API or an OSF ZIP archive.

Examples:
    export PJM_API_KEY=...
    python pjm_download.py --source api --year 2025 --month 11

    python pjm_download.py --source osf                 # downloads 2018–2025 archives
    python pjm_download.py --source osf --year 2024     # single OSF year
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path
import zipfile

import pandas as pd
import requests

BASE_URL = "https://api.pjm.com/api/v1"
DEFAULT_DATASET = "hrl_load_metered"
DEFAULT_ENV_KEY = "PJM_API_KEY"
OSF_ZIP_URL = "https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip="
DEFAULT_OSF_FILENAME = "hrl_load_metered_2025_nov.csv"
OSF_MEMBER_BY_YEAR = {
    2018: "hrl_load_metered_2018.csv",
    2019: "hrl_load_metered_2019.csv",
    2020: "hrl_load_metered_2020.csv",
    2021: "hrl_load_metered_2021.csv",
    2022: "hrl_load_metered_2022.csv",
    2023: "hrl_load_metered_2023.csv",
    2024: "hrl_load_metered_2024.csv",
    2025: "hrl_load_metered_2025.csv",
}
MIN_YEAR = 2018

DEFAULT_API_YEAR = 2025
DATETIME_COLUMNS = ["datetime_beginning_utc", "datetime_beginning_ept"]
DATETIME_FORMAT = "%-m/%-d/%Y %-I:%M:%S %p"


def month_bounds(year: int, month: int) -> tuple[str, str]:
    """Return inclusive start and exclusive end YYYY-MM-DD strings."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end_excl = start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)  # first day of next month
    return start.strftime("%Y-%m-%d"), end_excl.strftime("%Y-%m-%d")


def default_output_path(dataset: str, year: int, month: int) -> Path:
    month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%b").lower()
    fname = f"{dataset}_{year}_{month_name}.csv"
    return Path(__file__).resolve().parent / "data" / "raw" / "pjm" / fname


def default_osf_output_path(filename: str, target_year: int | None = None) -> Path:
    path = Path(__file__).resolve().parent / "data" / "raw" / "pjm" / filename
    if target_year == 2025 and filename == OSF_MEMBER_BY_YEAR.get(2025):
        path = path.with_name("hrl_load_metered_2025_jan_oct.csv")
    return path


def _format_datetime_series(dt_series: pd.Series, pattern: str) -> pd.Series:
    """Format datetimes with UNIX and Windows compatible directives."""
    try:
        return dt_series.dt.strftime(pattern)
    except ValueError:
        alt_pattern = pattern.replace("%-", "%#")
        return dt_series.dt.strftime(alt_pattern)


def normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATETIME_COLUMNS:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        mask = parsed.notna()
        if mask.any():
            localized = parsed[mask]
            try:
                localized = localized.dt.tz_localize(None)
            except TypeError:
                pass
            formatted = _format_datetime_series(localized, DATETIME_FORMAT)
            df.loc[mask, col] = formatted.values
    return df


def normalize_csv_datetimes(path: Path) -> Path:
    df = pd.read_csv(path)
    normalize_datetime_columns(df)
    df.to_csv(path, index=False)
    return path


def fetch_chunk(
    dataset: str,
    api_key: str,
    start_date: str,
    end_date: str,
    start_row: int,
    row_count: int,
) -> pd.DataFrame:
    """Fetch a single chunk as JSON items."""
    url = f"{BASE_URL}/{dataset}"
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    date_filter = f"{start_ts.strftime('%m/%d/%Y %H:%M')}to{end_ts.strftime('%m/%d/%Y %H:%M')}"
    params = [
        ("startRow", start_row),
        ("rowCount", row_count),
        ("sort", "datetime_beginning_ept"),
        ("order", "Asc"),
        ("datetime_beginning_ept", date_filter),
    ]
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=90)
    resp.raise_for_status()

    payload = resp.json()
    if "errors" in payload:
        message = payload.get("message") or payload["errors"]
        raise RuntimeError(f"PJM API error: {message}")

    items = payload.get("items")
    if items is None:
        return pd.DataFrame()

    if isinstance(items, list):
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items)
    if isinstance(items, dict):
        return pd.DataFrame([items])
    return pd.DataFrame()


def download_month(dataset: str, api_key: str, year: int, month: int, out_path: Path, row_count: int) -> Path:
    start_date, end_date = month_bounds(year, month)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    start_row = 1
    total_rows = 0

    print(f"[PJM] Downloading {dataset} {start_date} → {end_date} into {out_path}")

    while True:
        chunk = fetch_chunk(dataset, api_key, start_date, end_date, start_row, row_count)
        if chunk.empty:
            break
        frames.append(chunk)
        fetched = len(chunk)
        total_rows += fetched
        print(f"  • Rows {start_row:,}–{start_row + fetched - 1:,} ({fetched:,} rows)")
        start_row += fetched
        if fetched < row_count:
            break

    if not frames:
        raise RuntimeError("No data received from PJM; check date range or API key.")

    combined = pd.concat(frames, ignore_index=True)
    normalize_datetime_columns(combined)
    combined.to_csv(out_path, index=False)
    print(f"[PJM] Saved {len(combined):,} rows to {out_path}")
    return out_path


def download_zip(url: str) -> io.BytesIO:
    print(f"[OSF] Downloading ZIP archive from {url}")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    buffer = io.BytesIO(resp.content)
    buffer.seek(0)
    return buffer


def _read_csv_from_tar(
    data: bytes,
    member: str | None = None,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if member:
            target = next((m for m in members if m.name == member), None)
            if target is None:
                available = ", ".join(m.name for m in members) or "<empty>"
                raise ValueError(f"Member '{member}' not found in tar archive; available entries: {available}")
            if target.name.lower().endswith(".tar") or target.name.lower().endswith(".tar.gz"):
                nested = tf.extractfile(target).read()
                return _read_csv_from_tar(nested)
            with tf.extractfile(target) as fh:
                return pd.read_csv(fh)

        csv_members = [m for m in members if m.name.lower().endswith(".csv")]
        if file_pattern:
            csv_members = [m for m in csv_members if file_pattern in m.name]
        if csv_members:
            if len(csv_members) > 1:
                names = ", ".join(m.name for m in csv_members)
                raise ValueError(
                    f"Multiple CSV files found in tar archive. Specify one with --osf-member. Options: {names}"
                )
            with tf.extractfile(csv_members[0]) as fh:
                return pd.read_csv(fh)

        nested_tars = [m for m in members if m.name.lower().endswith((".tar", ".tar.gz", ".tgz"))]
        if nested_tars:
            if len(nested_tars) > 1:
                names = ", ".join(m.name for m in nested_tars)
                raise ValueError(f"Multiple tar files found in archive. Specify one with --osf-member. Options: {names}")
            nested = tf.extractfile(nested_tars[0]).read()
            return _read_csv_from_tar(nested, file_pattern=file_pattern)

        available = ", ".join(m.name for m in members) or "<empty>"
        raise ValueError(f"No CSV files found in tar archive. Available entries: {available}")


def _read_csv_from_zip(zf: zipfile.ZipFile, member: str | None, file_pattern: str | None = None) -> pd.DataFrame:
    if member:
        try:
            info = zf.getinfo(member)
        except KeyError as exc:
            available = ", ".join(info.filename for info in zf.infolist()) or "<empty>"
            raise ValueError(f"Member '{member}' not found in archive; available entries: {available}") from exc
        if member.lower().endswith(".zip"):
            nested_data = zf.read(info)
            with zipfile.ZipFile(io.BytesIO(nested_data)) as nested:
                return _read_csv_from_zip(nested, None, file_pattern)
        if member.lower().endswith((".tar", ".tar.gz", ".tgz")):
            nested_data = zf.read(info)
            return _read_csv_from_tar(nested_data, file_pattern=file_pattern)
        with zf.open(info, "r") as src:
            return pd.read_csv(src)

    csv_members = [info for info in zf.infolist() if info.filename.lower().endswith(".csv")]
    if file_pattern:
        csv_members = [info for info in csv_members if info.filename == file_pattern]
    if csv_members:
        if len(csv_members) > 1:
            names = ", ".join(info.filename for info in csv_members)
            raise ValueError(f"Multiple CSV files found in archive. Specify one with --osf-member. Options: {names}")
        with zf.open(csv_members[0], "r") as src:
            return pd.read_csv(src)

    zip_members = [info for info in zf.infolist() if info.filename.lower().endswith(".zip")]
    if zip_members:
        if len(zip_members) > 1:
            names = ", ".join(info.filename for info in zip_members)
            raise ValueError(f"Multiple ZIP files found in archive. Specify one with --osf-member. Options: {names}")
        nested_data = zf.read(zip_members[0])
        with zipfile.ZipFile(io.BytesIO(nested_data)) as nested:
            return _read_csv_from_zip(nested, None, file_pattern)

    tar_members = [info for info in zf.infolist() if info.filename.lower().endswith((".tar", ".tar.gz", ".tgz"))]
    if tar_members:
        if len(tar_members) > 1:
            names = ", ".join(info.filename for info in tar_members)
            raise ValueError(f"Multiple tar files found in archive. Specify one with --osf-member. Options: {names}")
        nested_data = zf.read(tar_members[0])
        return _read_csv_from_tar(nested_data, file_pattern=file_pattern)

    available = ", ".join(info.filename for info in zf.infolist()) or "<empty>"
    raise ValueError(f"No CSV files found in the OSF archive. Available entries: {available}")


def download_osf_archive(url: str, member: str | None, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = download_zip(url)
    csv_pattern = None
    zip_member = member
    if member and member.lower().endswith(".csv"):
        csv_pattern = member
        zip_member = None
    with zipfile.ZipFile(buffer) as zf:
        df = _read_csv_from_zip(zf, zip_member, csv_pattern)
    normalize_datetime_columns(df)
    df.to_csv(out_path, index=False)
    print(f"[OSF] Saved {len(df):,} rows to {out_path}")
    return out_path


def osf_download_targets(args: argparse.Namespace) -> list[tuple[int | None, str, str]]:
    if args.osf_member:
        filename = args.osf_filename or Path(args.osf_member).name
        return [(args.year, args.osf_member, filename)]

    years = [args.year] if args.year else sorted(OSF_MEMBER_BY_YEAR)
    targets: list[tuple[int | None, str, str]] = []
    for year in years:
        member = OSF_MEMBER_BY_YEAR.get(year)
        if not member:
            continue
        filename = args.osf_filename or member
        targets.append((year, member, filename))
    return targets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download PJM HRL load data from the API or an OSF archive.")
    ap.add_argument(
        "--source",
        choices=["api", "osf"],
        default="api",
        help="Data source to use: 'api' for PJM API, 'osf' for archived ZIP (default: api)",
    )
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="PJM dataset name (default: hrl_load_metered)")
    ap.add_argument("--year", type=int, default=None, help="Target year (API default 2025; OSF defaults to all years)")
    ap.add_argument("--month", type=int, default=11, help="Target month 1-12 (default 11)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/raw/pjm/{dataset}_YYYY_mon.csv)",
    )
    ap.add_argument("--api-key-env", default=DEFAULT_ENV_KEY, help="Environment variable containing PJM API key")
    ap.add_argument(
        "--row-count",
        type=int,
        default=50000,
        help="Rows per request (PJM max is ~50k; lower if you hit errors)",
    )
    ap.add_argument("--osf-url", default=OSF_ZIP_URL, help="OSF ZIP download URL")
    ap.add_argument("--osf-filename", default=None, help="Filename used when saving OSF data (defaults per year)")
    ap.add_argument(
        "--osf-member",
        default=None,
        help="Name of the CSV file inside the OSF ZIP (defaults to the only CSV present)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        if args.source == "api":
            year = args.year or DEFAULT_API_YEAR
            if year < MIN_YEAR:
                raise RuntimeError(f"PJM downloads are only supported for {MIN_YEAR} and later (requested {year}).")
            api_key = '3a5536a5d8cf446eb436a0e6ebaf19de'
            if not api_key:
                raise RuntimeError(f"Missing API key in ${args.api_key_env}. Set it before running.")
            out_path = args.out or default_output_path(args.dataset, year, args.month)
            download_month(args.dataset, api_key, year, args.month, out_path, args.row_count)
        else:
            if args.year is not None and args.year < MIN_YEAR:
                raise RuntimeError(f"PJM downloads are only supported for {MIN_YEAR} and later (requested {args.year}).")
            targets = osf_download_targets(args)
            if not targets:
                raise RuntimeError("No OSF targets determined; specify --osf-member or a supported year.")
            if len(targets) > 1 and args.out:
                raise RuntimeError("--out can only be used when downloading a single OSF file.")
            if len(targets) > 1 and args.osf_filename:
                raise RuntimeError("--osf-filename can only be used when downloading a single OSF file.")
            for year, member, filename in targets:
                out_path = args.out or default_osf_output_path(filename, year)
                download_osf_archive(args.osf_url, member, out_path)
    except Exception as exc:
        print(f"Failed to download PJM data: {exc}", file=sys.stderr)
        sys.exit(2)

    # Attach data to existing file
    RAW_PATH = Path("data/raw/pjm/hrl_load_metered_2025_nov.csv")
    MASTER_PATH = Path("data/hrl_load_metered_2025.csv")

    # Windows-specific format: 1/1/2025 5:00:00 AM
    OUT_FMT = "%#m/%#d/%Y %#I:%M:%S %p"

    # --- 1. Read BOTH files as proper datetimes ---

    raw = pd.read_csv(
        RAW_PATH,
        parse_dates=["datetime_beginning_ept", "datetime_beginning_utc"],
    )

    raw = raw.dropna(subset=["datetime_beginning_ept"])
    raw = raw.sort_values(["datetime_beginning_ept", "load_area"])

    master = pd.read_csv(
        MASTER_PATH,
        parse_dates=["datetime_beginning_ept", "datetime_beginning_utc"],
    )

    master = master.dropna(subset=["datetime_beginning_ept"])

    # --- 2. Decide what to append (still in datetime) ---

    last_dt = master["datetime_beginning_ept"].max()

    new_rows = raw[raw["datetime_beginning_ept"] > last_dt].copy()

    if new_rows.empty:
        print("Master file already has all rows in the raw download.")
    else:
        # --- 3. Append while everything is still datetime ---

        combined = (
            pd.concat([master, new_rows], ignore_index=True)
            .sort_values(["datetime_beginning_ept", "load_area"])
        )

        # Optional: drop duplicates if you want to be extra safe
        # combined = combined.drop_duplicates(
        #     subset=["datetime_beginning_ept", "load_area"],
        #     keep="last"
        # )

        # --- 4. NOW convert datetimes to final string format ---

        combined["datetime_beginning_ept"] = combined["datetime_beginning_ept"].dt.strftime(OUT_FMT)
        combined["datetime_beginning_utc"] = combined["datetime_beginning_utc"].dt.strftime(OUT_FMT)

        # Everything else stays as-is; only these two columns become strings
        combined.to_csv(MASTER_PATH, index=False)



if __name__ == "__main__":
    main()
