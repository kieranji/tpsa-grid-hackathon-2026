# Observed SEMOpx prices and Irish CPI

These files contain public market observations, not forward prices or predicted BESS revenue. Monetary prices are nominal EUR/MWh until the model explicitly deflates them.

| File | Market days / hours | Use |
|---|---:|---|
| `semopx_dam_2019.csv` | 365 / 8,760 | Historical comparison |
| `semopx_dam_2020.csv` | 366 / 8,784 | Pandemic regime sensitivity |
| `semopx_dam_2021.csv` | 365 / 8,760 | Historical price stress |
| `semopx_dam_2022_partial.csv` | 361 / 8,664 | Incomplete; 96 missing hours, never treat as a full year |
| `semopx_dam_2023.csv` | 365 / 8,760 | Most recent complete calendar market year retrieved from archive |
| `semopx_dam_recent.csv` | 363 / 8,712 | 2025-09-12 to 2026-09-09 inclusive, contiguous but not a complete year |

`timestamp_utc` is the delivery-period start. `duration_hours` comes from each official report (0.5 and 1 in the recent data; parser also accepts 0.25). `market_date` is the D+1 auction delivery day, inferred from the explicit auction timestamp. It differs from the UTC date for the first periods of a market day. Source periods and daylight-saving days are preserved: recent 2025-10-26 has 25 hours and 2026-03-29 has 23 hours.

`source_file` identifies each original report. `semopx_provenance.json` records original URLs, raw-report hashes, output hashes, coverage and revisions. No missing prices were filled. The latest retrieved official API window does not cover 1–11 September 2025, so a September 2025–August 2026 full-year claim is not supported.

## Reproduce

From the repository root:

```sh
python scripts/fetch_semopx_prices.py
python scripts/fetch_cso_cpi.py
python scripts/test_market_data.py
```

Raw downloads stay in `.codex_work/market_data`; they are not part of committed data. The market archive is about 115 MB. API retention means a later download can have less historical coverage; retain local raw caches for exact replay. The normalized files and recorded hashes are the study's immutable input snapshot.

The [SEMOpx API guide](https://www.semopx.com/sites/semo/files/documents/general-publications/SEMOpx-Website-Report-API.pdf) documents the public, unauthenticated report-list and document endpoints. The [data publication guide](https://www.semopx.com/sites/semo/files/documents/general-publications/SEMOpx-Data-Publication-Guide-Issue-7.5.pdf) defines report dates, duration and price sections. The parser also handles the observed newer `Market` header, replacing the older `Market Area` header, and surplus trailing-comma padding in some 2022 archive files.

## Consistent real prices

`ireland_cpi_monthly.csv` contains CSO all-items CPI, December 2023=100, through August 2026. `to_real_2024_eur_factor = 100.7 / monthly_CPI` uses the published 2024 annual-average index. The [CSO monthly table](https://data.cso.ie/table/CPM20) was updated 10 September 2026. The [December 2025 release](https://www.cso.ie/en/releasesandpublications/ep/p-cpi/consumerpriceindexdecember2025/) supplies the annual reference.

September 2026 CPI and future tariff-period CPI are not observed. Holding August's index at 107.4 is a modelling assumption if used; it is not an official September or October observation. Consumer CPI normalizes purchasing-power units and is not a battery equipment cost index.

## Interpretation

Use full chronological dispatch or daily rolling decisions with continuous inter-day SOC. A run using all realised future daily prices is an upper-bound backtest unless bid timing and price uncertainty are addressed. Revenue streams must share one physical MW/MWh/SOC budget. Do not add independently maximised arbitrage and reserve revenue. Historical prices do not establish the 2033 price distribution, and repeated synthetic hackathon snapshots do not establish annual profit.
