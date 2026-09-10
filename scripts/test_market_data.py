"""Parser and recorded-data integrity checks; never download during tests."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

from fetch_semopx_prices import coverage, parse_prices


def report(market='Market Area', minutes=60, timestamps=None, prices=None):
    stamps = timestamps or ['2025-09-11T22:00:00Z', '2025-09-11T23:00:00Z']
    prices = prices or ['-10,50', '20,25']
    return ('Auction date time;2025-09-11T10:00:00Z\n' + market + ';ROI-DA\n'
            f'Index prices;{minutes};EUR\n' + ';'.join(stamps) + '\n' + ';'.join(prices) + '\n').encode()


class ParseTests(unittest.TestCase):
    def test_legacy_decimal_and_delivery_day(self):
        rows = parse_prices(report(), 'legacy.csv')
        self.assertEqual(rows[0]['price_eur_mwh'], -10.5)
        self.assertEqual(rows[0]['market_date'], '2025-09-12')
        self.assertEqual(rows[0]['duration_hours'], 1)

    def test_new_format_native_half_hour(self):
        rows = parse_prices(report('Market', 30, ['2025-09-11T22:00:00Z', '2025-09-11T22:30:00Z']), 'new.csv')
        self.assertEqual(coverage(rows)['covered_hours'], 1)
        self.assertEqual(rows[1]['timestamp_utc'], '2025-09-11T22:30:00Z')

    def test_missing_price_is_not_filled(self):
        with self.assertRaises(ValueError):
            parse_prices(report(prices=['1,00']), 'missing.csv')
        with self.assertRaises(ValueError):
            parse_prices(report(timestamps=['2025-09-11T22:00:00Z', '2025-09-12T00:00:00Z']), 'gap.csv')

    def test_datasets_match_hashes_and_reported_coverage(self):
        root = Path(__file__).resolve().parents[1] / 'data' / 'market'
        metadata = json.loads((root / 'semopx_provenance.json').read_text())
        for item in metadata['datasets']:
            with self.subTest(file=item['file']):
                path = root / item['file']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])
                with path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                info = coverage(rows)
                self.assertEqual(info['covered_hours'], item['covered_hours'])
                self.assertEqual(info['gaps'], item['gaps'])
                self.assertEqual(len({row['timestamp_utc'] for row in rows}), len(rows))
        recent = next(item for item in metadata['datasets'] if item['file'] == 'semopx_dam_recent.csv')
        self.assertEqual(recent['covered_hours'], 8712)
        self.assertEqual(recent['non_24_hour_market_days'], {'2025-10-26': 25, '2026-03-29': 23})
        self.assertFalse(recent['complete_market_year'])


if __name__ == '__main__':
    unittest.main()
