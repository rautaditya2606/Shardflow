#!/usr/bin/env python3
"""
Archive GitHub Traffic Metrics (Views, Clones, Referrers, Paths)
Preserves 14-day rolling statistics indefinitely in CSV format.
"""

import os
import sys
import csv
import json
import requests
from datetime import datetime

REPO = os.environ.get("GITHUB_REPOSITORY", "rautaditya2606/Shardflow")
TOKEN = os.environ.get("GH_STATS_TOKEN") or os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    print("Error: GH_STATS_TOKEN or GITHUB_TOKEN environment variable is required.")
    sys.exit(1)

OUTPUT_DIR = os.environ.get("TRAFFIC_DIR", ".github/traffic_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "traffic-archiver"
}

def fetch_traffic_data(endpoint):
    url = f"https://api.github.com/repos/{REPO}/traffic/{endpoint}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Warning: Failed to fetch {endpoint} (HTTP {resp.status_code}): {resp.text}")
        return None
    return resp.json()

def merge_time_series(filename, new_entries, key_field="timestamp", fields=("timestamp", "count", "uniques")):
    filepath = os.path.join(OUTPUT_DIR, filename)
    data = {}

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row[key_field]
                data[ts] = {k: int(row.get(k, 0)) if k != key_field else ts for k in fields}

    for item in new_entries:
        ts = item[key_field]
        data[ts] = {
            key_field: ts,
            "count": int(item.get("count", 0)),
            "uniques": int(item.get("uniques", 0))
        }

    sorted_entries = sorted(data.values(), key=lambda x: x[key_field])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted_entries)

    print(f"Saved {len(sorted_entries)} records to {filepath}")
    return sorted_entries

def merge_snapshot_data(filename, new_entries, key_field, fields):
    filepath = os.path.join(OUTPUT_DIR, filename)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    data = {}

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (row.get("date", ""), row.get(key_field, ""))
                data[k] = row

    for item in new_entries:
        k = (today, item.get(key_field, ""))
        row = {"date": today}
        for f in fields:
            if f != "date":
                row[f] = item.get(f, "")
        data[k] = row

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        all_fields = ["date"] + [f for f in fields if f != "date"]
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(data.values())

    print(f"Saved {len(data)} records to {filepath}")

def generate_summary(views_data, clones_data):
    total_views = sum(int(x.get("count", 0)) for x in views_data)
    total_unique_views = sum(int(x.get("uniques", 0)) for x in views_data)
    total_clones = sum(int(x.get("count", 0)) for x in clones_data)
    total_unique_cloners = sum(int(x.get("uniques", 0)) for x in clones_data)

    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 Shardflow Traffic & Growth Analytics\n\n")
        f.write(f"This directory stores historical GitHub traffic statistics for **[{REPO}](https://github.com/{REPO})**.\n\n")
        f.write(f"**Last Updated**: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`\n\n")
        f.write(f"### 🚀 Lifetime Totals\n\n")
        f.write(f"| Metric | Total Count | Total Uniques |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| **Git Clones** | **{total_clones:,}** | **{total_unique_cloners:,}** |\n")
        f.write(f"| **Repository Views** | **{total_views:,}** | **{total_unique_views:,}** |\n\n")
        f.write(f"### 📁 Data Files\n\n")
        f.write(f"- [`clones.csv`](./clones.csv) – Daily clone volume and unique cloners.\n")
        f.write(f"- [`views.csv`](./views.csv) – Daily page views and unique visitors.\n")
        f.write(f"- [`referrers.csv`](./referrers.csv) – Referral sources history.\n")
        f.write(f"- [`popular_paths.csv`](./popular_paths.csv) – Top visited repository paths.\n\n")
        f.write(f"### 📈 Recent Clones (Last 14 Recorded Days)\n\n")
        f.write(f"| Date | Clones | Unique Cloners |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        for row in reversed(clones_data[-14:]):
            ts = row.get("timestamp", "").split("T")[0]
            f.write(f"| {ts} | {row.get('count', 0)} | {row.get('uniques', 0)} |\n")
        f.write(f"\n")

    print(f"Generated summary report at {readme_path}")

def main():
    print(f"Fetching traffic data for {REPO}...")

    views_resp = fetch_traffic_data("views")
    clones_resp = fetch_traffic_data("clones")
    referrers_resp = fetch_traffic_data("popular/referrers")
    paths_resp = fetch_traffic_data("popular/paths")

    views_data = []
    clones_data = []

    if views_resp and "views" in views_resp:
        views_data = merge_time_series("views.csv", views_resp["views"])
    
    if clones_resp and "clones" in clones_resp:
        clones_data = merge_time_series("clones.csv", clones_resp["clones"])

    if referrers_resp:
        merge_snapshot_data("referrers.csv", referrers_resp, "referrer", ["date", "referrer", "count", "uniques"])

    if paths_resp:
        merge_snapshot_data("popular_paths.csv", paths_resp, "path", ["date", "path", "title", "count", "uniques"])

    generate_summary(views_data, clones_data)
    print("Traffic archival complete.")

if __name__ == "__main__":
    main()
