"""Fetch CSO all-items monthly CPI for consistent real-2024-EUR cash flows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL = 'https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/CPM20/JSON-stat/2.0/en'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/market'))
    parser.add_argument('--raw-input', type=Path)
    args = parser.parse_args()
    raw = args.raw_input.read_bytes() if args.raw_input else urlopen(URL, timeout=60).read()
    doc = json.loads(raw)
    dims = doc['dimension']
    stats = dims['STATISTIC']['category']['index']
    months = dims['TLIST(M1)']['category']['index']
    groups = dims['C04624V05409']['category']['index']
    statistic_index = stats.index('CPM20C06')
    commodity_index = groups.index('CP00')
    rows = []
    for month_index, month in enumerate(months):
        if month < '201901':
            continue
        index = ((statistic_index * len(months)) + month_index) * len(groups) + commodity_index
        value = doc['value'][index]
        if value is not None:
            rows.append({'month': month[:4] + '-' + month[4:], 'cpi_dec2023_100': value,
                         'to_real_2024_eur_factor': 100.7 / value})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'ireland_cpi_monthly.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'source_url': URL, 'source_updated_utc': doc['updated'],
                'raw_sha256': hashlib.sha256(raw).hexdigest(),
                'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'statistic': 'CPM20C06', 'commodity': 'CP00',
                'base_2024_annual_average_index': 100.7,
                'annual_reference_url': 'https://www.cso.ie/en/releasesandpublications/ep/p-cpi/consumerpriceindexdecember2025/',
                'formula': 'real_2024_EUR = nominal_EUR_in_month * 100.7 / monthly_index',
                'last_observed_month': rows[-1]['month'],
                'unpublished_months_policy': 'Not included; any carry-forward is an explicit modelling assumption',
                'caveat': 'Consumer-price deflator normalizes purchasing-power units; it is not a BESS equipment cost index.'}
    (args.output_dir / 'ireland_cpi_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(rows[-8:], indent=2))


if __name__ == '__main__':
    main()
