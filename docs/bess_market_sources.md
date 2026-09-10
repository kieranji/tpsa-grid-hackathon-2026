# BESS market economics: evidence, units and assumptions

Research cut-off: **10 September 2026**. The configured case screens an **85 MW / 310 MWh DC-nameplate** project commissioned in 2033. It reports pretax, unlevered cash in **constant 2024 EUR**, discounted at an assumed 7% real rate over 15 operating years. Historical backtests and conditional contract sensitivities are not a 2033 merchant forecast or evidence of awarded income. Configuration: `configs/bess_market_public_benchmarks.json`; implementation: `src/bess_market/`.

## Observed energy prices and consistent money units

The inputs are the official **ROI-DA, EUR/MWh** index sections of SEMOpx auction reports. Native timestamps and durations are preserved, including negative prices and daylight-saving days. The recent data contain hourly and half-hourly periods; no 15-minute observations are claimed. `market_date` identifies the auction's D+1 delivery day, which can differ from a period's UTC calendar date. Source: [SEMOpx publication guide](https://www.semopx.com/sites/semo/files/documents/general-publications/SEMOpx-Data-Publication-Guide-Issue-7.5.pdf).

| Input snapshot | Observed market days | Observed hours | Interpretation |
|---|---:|---:|---|
| 2019 | 365 | 8,760 | Complete historical comparison |
| 2020 | 366 | 8,784 | Complete leap-year comparison |
| 2021 | 365 | 8,760 | Complete historical comparison |
| 2022 partial | 361 | 8,664 | Missing 96 hours; not a complete year |
| 2023 | 365 | 8,760 | Complete market calendar year |
| 2025-09-12 to 2026-09-09 | 363 | 8,712 | Contiguous recent observations, not a complete year |

The requested September-2025-to-August-2026 full year could not be retrieved from the public API's current retention window. Missing days were not estimated. Data, raw-report URLs, publication revisions and SHA256 hashes are in `data/market/`; reproduce with `scripts/fetch_semopx_prices.py`. The [official report API guide](https://www.semopx.com/sites/semo/files/documents/general-publications/SEMOpx-Website-Report-API.pdf) describes unauthenticated report listing and downloads; the older years came from the [official historical ZIP](https://www.semopx.com/sites/semo/files/documents/general-publications/DAM-IDM-Market-Results.zip). The large raw archive is excluded from deliverable data.

Nominal electricity prices are converted month by month using `100.7 / monthly_CPI`. CSO's published 2024 annual CPI is **100.7**, December 2023=100. Latest observed CPI is **August 2026: 107.4**, published on the research date. Historical CPI gaps cause an error. September 2026 is explicitly flagged as using the last observation. October 2026 network tariffs and observed 2026 capacity-auction quotations use that same August proxy; this is a current-quotation normalization convention, not observed future inflation or a forecast of the purchasing power of a 2029/30 delivery-year payment. Sources: [CSO CPM20](https://data.cso.ie/table/CPM20), [annual reference](https://www.cso.ie/en/releasesandpublications/ep/p-cpi/consumerpriceindexdecember2025/). CPI is a purchasing-power deflator, not an Irish battery procurement index.

The causal policy uses only older price observations, withholding the previous delivery day, with a flat forecast at cold start. The daily oracle uses realized delivery-day prices and omits the noncash wear regularizer so it is a cash upper benchmark under the same constraints. Each day begins and ends at the same inventory; multi-day energy shifting is excluded. Results retain observed-period cash and disclose any `8760 / observed_hours` annualization. Repeating a historical pattern through the project life is an explicit sensitivity.

## Capital cost, efficiency and maintenance

[NREL's 2025 cost report](https://docs.nlr.gov/docs/fy25osti/93281.pdf), Table 2, printed page 16 (PDF page 23), provides mainland-US, four-hour overnight installed costs in **real 2024 USD per usable AC-output kWh**:

| Commissioning year | Low | Mid | High |
|---|---:|---:|---:|
| 2024 | 334 | 334 | 334 |
| 2026 | 255 | 308 | 366 |
| 2033 | 171 | 257 | 345 |

The report's 2024 duration split is **USD241/usable AC-kWh + USD372/kW**. The model scales both coefficients by the relevant 2033 four-hour ratio, `(171, 257, 345) / 334`. Holding those component shares is an analyst inference; NREL did not publish this project's 2033 duration-specific quote. The source has US cost, tax and supply-chain assumptions. Dividing by **1.0820 USD/EUR**, the [Federal Reserve's 2024 annual mean](https://www.federalreserve.gov/releases/g5a/20250106/), retains a real-2024 currency convention.

The configured 85% AC round-trip efficiency, symmetric conversion losses and 10%-90% SOC window imply:

```text
usable AC MWh = 310 * (0.9 - 0.1) * sqrt(0.85) = 228.644703
deliverable duration = 228.644703 / 85 = 2.689938 hours
```

The temporary 97% capability fraction reduces dispatch capability; it does not shrink the plant being purchased. **310 MWh is not 310 MWh of usable AC output, and this case is not accredited as four-hour storage.** These values supersede any earlier illustration using 90% efficiency.

| Configured cost case | Irish installation multiplier (assumed) | Connection allowance (assumed) | Total upfront, EUR2024 m | Annual maintained-plant FOM, EUR2024 m |
|---|---:|---:|---:|---:|
| Low | 1.00 | 2m | 43.0354 | 1.6414 |
| Reference | 1.15 | 5m | 75.9240 | 2.8370 |
| High | 1.30 | 10m | 117.6279 | 4.3051 |

Irish multipliers and connection allowances are sensitivity inputs, not quotations. Check future EPC scope against separately allowed connection works, land, owner costs and insurance before claiming complete project pricing. The assumed year-15 decommissioning charge is 3% of total upfront cost, with zero residual value.

[NREL ATB 2025](https://atb.nlr.gov/electricity/2025/utility-scale_battery_storage) assumes **15-year life, about one cycle/day, 85% round-trip efficiency and annual FOM of 4% of capital cost, including augmentation to maintain rated capacity**. The model applies 4% to the installed plant, separately from the connection allowance, and does not add replacement/augmentation CAPEX or cash cycling wear. Its EUR2/MWh throughput wear bid is only a causal-dispatch regularizer. Maintaining capacity at the quoted FOM is a benchmark assumption, not a supplier performance guarantee. [IRENA's global 2024 USD192/kWh observation](https://www.irena.org/News/pressreleases/2025/Jul/91-Percent-of-New-Renewable-Projects-Now-Cheaper-Than-Fossil-Fuels-Alternatives) is a separate project-mix cross-check, not the Irish cost basis.

## Ireland network charges

The binding [CRU/2026/96 decision](https://cruie-live-96ca64acab2247eca8a850a7e54b-5b34f62.divio-media.com/documents/CRU202696_Interim_Transmission_Network_Charges_for_Energy_Storage_-_Decision.pdf), effective **1 October 2026**, applies G-TUoS to standalone storage and storage co-located with generation, with D-TUoS on house load. DUoS is unchanged; demand-colocated/autoproducer exceptions remain. Zero network costs are therefore unsupported.

The [EirGrid 2026/27 Statement of Charges](https://cms.eirgrid.ie/sites/default/files/publications/Statement_of_Charges_2026-2027.pdf), published 9 September 2026, gives these nearby regional proxies. No exact Croaghonagh line item was found:

| Location | Nominal EUR/kW/year | Nominal EUR/year at 85 MW | EUR2024/year using August CPI proxy |
|---|---:|---:|---:|
| Golagh BESS | 10.2533 | 871,530.50 | 817,161.28 |
| Meentycat | 11.2269 | 954,286.50 | 894,754.66 |
| Binbane | 11.8005 | 1,003,042.50 | 940,469.09 |

These are transmission-connected location proxies, not known 2033 tariffs or this project's approved charge. Distribution-connected GTS-D uses MEC above 5 MW with separate DUoS. House-load GTS-T capacity is EUR2,502.8933/MW/month; the transfer component is EUR4.6316/MWh, with system-service charges additional. The model's **EUR100,000/year** house-load and incremental-fee allowance is assumed. The **EUR1/MWh** charge-plus-discharge execution fee and **EUR3/MWh** import adder are assumed commercial sensitivities, not official tariff observations; check their supplier coverage against house-load and fixed O&M scope to prevent duplicate invoicing.

## Revenue eligibility and conditional prices

### Capacity

Only successful capacity-auction units earn payments; awards use de-rated MW and carry difference-charge/non-delivery exposure. Gross scarcity energy proceeds cannot be added to capacity payments without the associated clawback. Source: [SEMO capacity-market overview](https://www.sem-o.com/markets/capacity-market-overview).

| Observed auction | Published | Clearing price, EUR/de-rated MW/year |
|---|---|---:|
| 2026/27 T-1 | 26 August 2026 | 53,978.87 |
| 2029/30 T-4 | 5 May 2026 | 135,499.99 |

Sources: [2026/27 T-1 final results](https://www.sem-o.com/sites/semo/files/2026-08/Final%20Capacity%20Auction%20Results%20report%20FCAR2627T-1.pdf), [2029/30 T-4 final results](https://www.sem-o.com/sites/semo/files/2026-05/2029_2030%20T-4%20Final%20Capacity%20Auction%20Results%20Report%20FCAR2930T-4.pdf). They concern different auction horizons and delivery years; they are price references, not probabilities or a 2033 price range with statistical confidence.

The [2029/30 initial auction pack v1.1, Table 3](https://www.sem-o.com/sites/semo/files/2025-08/IAIP2930T-4.pdf) supplies the **80 < installed MW <= 90** storage curve: factors 0.142 at two hours and 0.210 at three hours. Interpolating the configured 2.689938-hour AC duration gives **0.188915762**. Applying the model's separate 97% capability haircut yields **15.576105 conditional awarded MW**. Using this one initial T-4 curve for both price references is a comparison assumption, not T-1 accreditation or final 2033 qualification. Gross reference payments before difference charges are approximately **EUR0.7883m / EUR1.9789m per year in 2024 money** under the stated CPI convention.

The model conservatively reserves separate inverter headroom and four hours of energy for the capacity obligation at every interval. It deducts `awarded_MW * duration * max(DA_price - EUR500/MWh, 0)`. The **EUR500/MWh strike is a sensitivity**, and DA is only a proxy for the actual reference-price settlement rules. Actual scarcity events, nonperformance, secondary trading and collateral are not reconstructed.

### System services

Historical DS3 rates for FFR/POR/SOR/TOR1/TOR2 are **1.94/2.92/1.76/1.40/1.12 EUR per MW-hour of service availability**, before scalars. They are not prices per MWh of activated energy. Source: [DS3 Statement of Payments, effective January 2022](https://cms.eirgrid.ie/sites/default/files/publications/EirGrid-DS3-System-Services-Statement-of-Payments-December-2021.pdf). [SEM-26-012](https://www.semcommittee.com/files/semcommittee/2026-03/sem-26-012-ds3-system-services-tariff-review-decision.pdf) kept underlying tariffs but reduced the specified fast-reserve scarcity scalars in April 2026. Technical qualification, recharge restrictions, state-of-charge reporting and performance discounts apply under [Protocol v4.2](https://cms.eirgrid.ie/sites/default/files/publications/DS3-SS-Protocol-v4.2.pdf).

**The 2033 model uses zero legacy DS3 tariff revenue.** [SEM-25-031](https://www.semcommittee.com/files/semcommittee/2025-06/SEM-25-031%20System%20Services%20Regulated%20Arrangements%20to%20FASS%20-%20The%20Gap%20-%20Decision%20Paper.pdf) sets product-by-product transition to competitive procurement and a September 2027 long-stop. The [26 August 2026 industry workshop](https://cms.eirgrid.ie/sites/default/files/publications/FM-Industry-Workshop-Presentation-August-2026.pdf) reports schedule uncertainty; FASS/LDES revenue is not already contracted to this project.

Future generic availability prices **EUR0/2/4/6 per MW-hour in 2024 money** are analyst sensitivity inputs. They are **gross** before the model's 90% payment capture and activation/execution/import costs. The assumed symmetric block needs one hour of endurance and shares power and SOC with energy and capacity. Assumed activation is 1% each way, with physical losses and energy purchases/sales included once. This is not a product-specific FASS qualification model.

One-year cases stop conditional services and capacity after year one and revert to energy-only dispatch. Fifteen-year cases explicitly repeat assumed awards and service prices for the full study; they do not assert an actual 15-year service contract or automatic capacity renewal. **Secured capacity and services revenue is zero in every case.**

## Cash boundaries and remaining evidence

The ledger adds gross energy sales, subtracts all charging purchases, and then adds eligible services and capacity once. It separately deducts execution/import charges, performance deductions, capacity difference charges, fixed O&M, G-TUoS, house load and decommissioning. Energy and reserve activation use one physical MW/MWh budget. No standalone DA, intraday and balancing maximum-profit estimates are added together. Balancing participation has evolved since the November 2025 ESPS implementation, but actual registrations and settlements would be needed before adding a balancing uplift. Sources: [SEMO registration](https://www.sem-o.com/markets/balancing-market-overview/joining-the-balancing-market), [December 2025 market development update](https://www.sem-o.com/sites/semo/files/2025-12/Market%20Development%20Plan%20Update.pdf).

Avoided curtailment, lower losses, carbon savings and deferred grid investment stay outside developer cash unless an explicit payment mechanism transfers them. Tolling or a floor agreement must allocate merchant rights and opportunity cost; full toll and full merchant income cannot both be credited for the same rights. Local flexibility, inertia, voltage, black start and LDES support receive no assumed payment without an eligible priced contract.

An EPC/connection quotation, site MIC/MEC and firm-access rights, operating guarantees, actual service/CRM awards, an execution contract and forward electricity scenarios remain necessary for an investable return estimate. The hackathon's synthetic network chronology does not verify annual grid feasibility. The useful result is the conditional feasibility range, operating cash ledger and additional **net** annual contracted payment needed to break even, with its required term stated separately.

Independent checks in `tests/bess_market/test_pipeline_review.py` cover published CPI gaps, explicitly flagged CPI carry, delivery-month treatment, AC/DC cost conversion, four-hour source identity, annualization, contract expiry, continued-contract sensitivities and maintenance-inclusive cash accounting.
