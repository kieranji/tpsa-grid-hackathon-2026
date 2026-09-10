"""Reproduce public ROI day-ahead prices with explicit gaps, hashes and durations.

Raw reports stay in --cache-dir, outside committed data. No interpolation occurs.
The auction delivery day is auction_date_UTC + 1 day, as stated by SEM-DA D+1.
It is not the Irish local calendar day or the UTC date of each period.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pathlib
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zipfile import ZipFile

API = 'https://reports.semopx.com/api/v1/documents/static-reports'
DOCUMENTS = 'https://reports.semopx.com/documents/'
ARCHIVE_URL = 'https://www.semopx.com/sites/semo/files/documents/general-publications/DAM-IDM-Market-Results.zip'
SCHEMA = ['timestamp_utc', 'duration_hours', 'price_eur_mwh', 'market_date', 'source_file']


def utc(stamp: str) -> datetime:
    value = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    if value.utcoffset() != timedelta(0):
        raise ValueError('Expected explicit UTC timestamp: ' + stamp)
    return value


def parse_prices(data: bytes, source: str) -> list[dict]:
    # Some archive files contain trailing comma padding on metadata/timestamp
    # lines. Decimal commas inside price values are preserved.
    text = '\n'.join(line.rstrip(',') for line in data.decode('utf-8-sig').splitlines())
    lines = [row for row in csv.reader(io.StringIO(text), delimiter=';') if row]
    auction = next(row[1] for row in lines if row[0].strip().lower() == 'auction date time')
    market_date = (utc(auction).date() + timedelta(days=1)).isoformat()
    area, out = None, []
    for i, row in enumerate(lines):
        if row[0].strip() in {'Market Area', 'Market'}:
            area = row[1].strip()
        if area == 'ROI-DA' and row[0].strip() == 'Index prices' and row[2].strip() == 'EUR':
            minutes = int(row[1])
            if minutes not in {15, 30, 60}:
                raise ValueError(('Unsupported period duration', minutes, source))
            times, values = lines[i + 1], lines[i + 2]
            if len(times) != len(values):
                raise ValueError(('Unequal timestamps and values', source))
            for stamp, value in zip(times, values):
                if not stamp.strip():
                    continue
                utc(stamp)
                price = float(value.replace(',', '.'))
                if not math.isfinite(price):
                    raise ValueError(('Nonfinite price', source, stamp))
                out.append(dict(zip(SCHEMA, [stamp, minutes / 60, price, market_date, source])))
    if not out:
        raise ValueError(('No ROI-DA EUR index prices', source))
    if coverage(out)['gaps']:
        raise ValueError(('Noncontiguous report', source))
    return out


def coverage(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda row: row['timestamp_utc'])
    if not rows:
        return {'rows': 0, 'covered_hours': 0, 'gaps': []}
    gaps = []
    for left, right in zip(rows, rows[1:]):
        end = utc(left['timestamp_utc']) + timedelta(hours=float(left['duration_hours']))
        start = utc(right['timestamp_utc'])
        if end != start:
            gaps.append({'after': left['timestamp_utc'], 'before': right['timestamp_utc'],
                         'missing_hours': (start - end).total_seconds() / 3600})
    hours = sum(float(row['duration_hours']) for row in rows)
    last_end = utc(rows[-1]['timestamp_utc']) + timedelta(hours=float(rows[-1]['duration_hours']))
    dates = sorted({row['market_date'] for row in rows})
    daily_hours = {date: sum(float(row['duration_hours']) for row in rows if row['market_date'] == date)
                   for date in dates}
    return {'rows': len(rows), 'covered_hours': hours, 'first_utc': rows[0]['timestamp_utc'],
            'end_exclusive_utc': last_end.isoformat().replace('+00:00', 'Z'),
            'span_hours': (last_end - utc(rows[0]['timestamp_utc'])).total_seconds() / 3600,
            'first_market_date': dates[0], 'last_market_date': dates[-1], 'market_days': len(dates),
            'duration_hours_values': sorted({float(row['duration_hours']) for row in rows}),
            'non_24_hour_market_days': {date: count for date, count in daily_hours.items() if count != 24},
            'gaps': gaps,
            'mean_eur_mwh': sum(float(row['price_eur_mwh'])*float(row['duration_hours']) for row in rows)/hours,
            'min_eur_mwh': min(float(row['price_eur_mwh']) for row in rows),
            'max_eur_mwh': max(float(row['price_eur_mwh']) for row in rows)}


def download(url: str, path: pathlib.Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path.read_bytes()


def write_dataset(path: pathlib.Path, rows: list[dict]) -> dict:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **coverage(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('data/market'))
    parser.add_argument('--cache-dir', type=pathlib.Path, default=pathlib.Path('.codex_work/market_data'))
    parser.add_argument('--archive-path', type=pathlib.Path)
    parser.add_argument('--refresh-listing', action='store_true')
    parser.add_argument('--recent-start', default='2025-09-12')
    parser.add_argument('--recent-end-exclusive', default='2026-09-10')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive_path or args.cache_dir / 'DAM-IDM-Market-Results.zip'
    archive_bytes = download(ARCHIVE_URL, archive)
    archive_rows = []
    with ZipFile(io.BytesIO(archive_bytes)) as bundle:
        for name in sorted(bundle.namelist()):
            if 'MarketResult_SEM-DA_' in name and name.endswith('.csv'):
                archive_rows.extend(parse_prices(bundle.read(name), name.rsplit('/', 1)[-1]))
    params = {'Group': 'Market Data', 'DPuG_ID': 'EA-001', 'ResourceName': 'MarketResult_SEM-DA',
              'page_size': 500, 'sort_by': 'Date', 'order_by': 'ASC'}
    query = API + '?' + urllib.parse.urlencode(params)
    listing_path = args.cache_dir / 'recent_api_listing.json'
    if args.refresh_listing or not listing_path.exists():
        with urllib.request.urlopen(query, timeout=60) as response:
            listing = json.load(response)
        listing_path.write_text(json.dumps(listing, indent=2), encoding='utf-8')
    listing = json.loads(listing_path.read_text(encoding='utf-8'))
    if listing['pagination']['totalPages'] != 1:
        raise ValueError('Report listing requires pagination; refusing silent truncation')
    def get_item(item):
        filename = item['ResourceName']
        url = DOCUMENTS + urllib.parse.quote(filename)
        data = download(url, args.cache_dir / 'raw_recent' / filename)
        return parse_prices(data, filename), {'resource_name': filename, 'url': url,
            'publish_time': item['PublishTime'], 'sha256': hashlib.sha256(data).hexdigest()}
    recent_rows, reports = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows, meta in pool.map(get_item, listing['items']):
            recent_rows.extend(rows)
            reports.append(meta)
    # Deduplicate equivalent copies; retain latest publication for corrections.
    def deduplicate(rows):
        seen, revisions = {}, []
        for row in sorted(rows, key=lambda value: value['source_file']):
            old = seen.get(row['timestamp_utc'])
            if old and old['price_eur_mwh'] != row['price_eur_mwh']:
                revisions.append({'old': old, 'new': row})
            seen[row['timestamp_utc']] = row
        return [seen[stamp] for stamp in sorted(seen)], revisions
    archive_rows, archive_revisions = deduplicate(archive_rows)
    recent_rows, recent_revisions = deduplicate(recent_rows)
    recent_rows = [row for row in recent_rows if args.recent_start <= row['market_date'] < args.recent_end_exclusive]
    manifest = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'market': 'ROI-DA',
        'unit': 'EUR/MWh', 'currency_basis': 'Observed nominal EUR at each delivery date',
        'archive_url': ARCHIVE_URL, 'archive_sha256': hashlib.sha256(archive_bytes).hexdigest(),
        'api_query': query, 'raw_reports': reports, 'datasets': [],
        'revisions': archive_revisions + recent_revisions,
        'method': 'UTC delivery starts and source durations preserved. D+1 market day from auction timestamp. '
                  'No forward filling/interpolation. Contemporary capacity and service rules are separate scenarios.'}
    for year in [2019, 2020, 2021, 2022, 2023]:
        rows = [row for row in archive_rows if row['market_date'].startswith(str(year))]
        info = coverage(rows)
        expected = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).total_seconds() / 3600
        complete = info['covered_hours'] == expected and not info['gaps'] and info['market_days'] in {365, 366}
        suffix = '' if complete else '_partial'
        dataset = write_dataset(args.output_dir / f'semopx_dam_{year}{suffix}.csv', rows)
        dataset.update(expected_year_hours=expected, complete_market_year=complete)
        manifest['datasets'].append(dataset)
    recent_info = write_dataset(args.output_dir / 'semopx_dam_recent.csv', recent_rows)
    recent_info.update(complete_market_year=False, requested_start=args.recent_start,
                       requested_end_exclusive=args.recent_end_exclusive,
                       annualization_warning='363 observed market days; not a full calendar year or full trailing year')
    manifest['datasets'].append(recent_info)
    (args.output_dir / 'semopx_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for item in manifest['datasets']:
        print(item['file'], item['covered_hours'], 'hours', len(item['gaps']), 'gaps')


if __name__ == '__main__':
    main()
