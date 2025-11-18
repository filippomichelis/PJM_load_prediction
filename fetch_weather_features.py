#!/usr/bin/env python3
"""
Selective downloader for PJM load-area weather points.

Given a list of years and months, fetch Meteostat hourly history for just those
windows and write the combined rows to a CSV.
"""

import argparse
import calendar
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import tz
from meteostat import Hourly, Point, Stations

OUTPUT_PATH = "data/weather_points_oct_nov_dec.csv"

def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Meteostat weather for selected months/years."
    )
    p.add_argument(
        "--points-csv",
        type=str,
        default="pjm_load_area_cities(1).csv",
        help="CSV with columns load_area, zone_area, state, city, latitude, longitude",
    )
    p.add_argument(
        "--tz",
        type=str,
        default="America/New_York",
        help="Timezone for timestamp_local",
    )
    p.add_argument(
        "--out",
        type=str,
        default=OUTPUT_PATH,
        help="Output CSV path",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of load-area points (for testing)",
    )
    return p.parse_args()


def build_recent_window():
    """
    Build weather windows for:
      - years: 2019, 2020, 2022, 2023, 2024, 2025
      - months: October, November, December
    For past years: full months.
    For the current year: up to tomorrow's date (inclusive) for the current month;
    future months are skipped.
    """
    YEARS = [2019, 2020, 2022, 2023, 2024, 2025]
    MONTHS = [10, 11, 12]  # Oct, Nov, Dec

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    current_year = today.year

    windows = []

    for year in YEARS:
        for month in MONTHS:
            # Skip years in the future
            if year > current_year:
                continue

            start_dt = datetime(year, month, 1)

            if year < current_year:
                # Full month for past years: [month-01, next_month-01)
                last_day = calendar.monthrange(year, month)[1]
                end_dt = datetime(year, month, last_day) + timedelta(days=1)
            else:
                # year == current_year
                if month < tomorrow.month:
                    # Full past months in current year
                    last_day = calendar.monthrange(year, month)[1]
                    end_dt = datetime(year, month, last_day) + timedelta(days=1)
                elif month == tomorrow.month:
                    # Current month in current year: up to *tomorrow* inclusive
                    upto = tomorrow  # inclusive date
                    end_dt = datetime(upto.year, upto.month, upto.day) + timedelta(days=1)
                else:
                    # Future months this year -> skip
                    continue

            # Label uses the inclusive end date (end_dt - 1 day)
            label_start = start_dt.date()
            label_end = (end_dt - timedelta(days=1)).date()
            label = f"{label_start:%Y-%m-%d}:{label_end:%Y-%m-%d}"

            windows.append((label, start_dt, end_dt))

    return windows


def fetch_history(lat, lon, start_dt, end_dt, tz_name):
    point = Point(lat, lon)
    df = Hourly(point, start_dt, end_dt, model=True).fetch()
    station_id = None
    if df.empty:
        station_row = Stations().nearby(lat, lon).inventory("hourly").fetch(1)
        if station_row.empty:
            raise RuntimeError("No nearby Meteostat station with hourly data.")
        station_id = station_row.index[0]
        df = Hourly(station_id, start_dt, end_dt, model=True).fetch()
        if df.empty:
            raise RuntimeError("Meteostat returned no data even after station fallback.")

    rename_map = {
        "temp": "temp_c",
        "dwpt": "dewpoint_c",
        "rhum": "humidity_pct",
        "wspd": "wind_speed_ms",
        "pres": "pressure_hpa",
        "prcp": "precip_mm",
        "snow": "snow_mm",
        "coco": "weather_code",
        "tsun": "sunshine_minutes",
    }
    df = df.rename(columns=rename_map)
    for col in rename_map.values():
        if col not in df.columns:
            df[col] = np.nan

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    local = tz.gettz(tz_name)
    df["timestamp_utc"] = df.index.tz_convert("UTC")
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(local)
    df = df.reset_index(drop=True)

    df["temp_f"] = df["temp_c"] * 9.0 / 5.0 + 32.0
    base_f = 65.0
    df["cooling_deg_f"] = np.clip(df["temp_f"] - base_f, 0, None)
    df["heating_deg_f"] = np.clip(base_f - df["temp_f"], 0, None)
    df["source"] = "history"
    return df, station_id


def main():
    args = parse_args()
    points_path = Path(args.points_csv)
    if not points_path.exists():
        raise FileNotFoundError(f"Points CSV not found: {points_path}")

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    points = pd.read_csv(points_path)
    required_cols = {"load_area", "zone_area", "state", "city", "latitude", "longitude"}
    missing = required_cols.difference(points.columns)
    if missing:
        raise ValueError(f"Points CSV missing required columns: {missing}")

    if args.limit is not None:
        points = points.head(args.limit)

    windows = build_recent_window()
    total = len(points)
    header_written = False

    for idx, row in enumerate(points.itertuples(index=False), start=1):
        lat = float(row.latitude)
        lon = float(row.longitude)
        point_id = f"{row.load_area}_{row.city}".replace(" ", "_")
        print(f"[{idx}/{total}] Fetching {row.load_area} - {row.city} across {len(windows)} windows")

        point_frames = []
        for label, start_dt, end_dt in windows:
            try:
                weather, station_id = fetch_history(lat, lon, start_dt, end_dt, args.tz)
                if station_id:
                    print(f"    used fallback station {station_id} for window {label}")
            except Exception as exc:
                print(f"  !! Failed window {label} for {row.load_area} - {row.city}: {exc}")
                continue

            weather = weather.assign(
                load_area=row.load_area,
                zone_area=row.zone_area,
                state=row.state,
                city=row.city,
                latitude=lat,
                longitude=lon,
                point_id=point_id,
            )
            point_frames.append(weather)

        if not point_frames:
            print(f"    Skipping {row.load_area} - {row.city}: no successful downloads.")
            continue

        combined = pd.concat(point_frames, ignore_index=True)
        combined.to_csv(
            out_path,
            mode="a",
            header=not header_written,
            index=False,
        )
        header_written = True

    print(f"Done. Wrote weather rows to {out_path}")


if __name__ == "__main__":
    main()
