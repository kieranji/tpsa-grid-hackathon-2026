# Q4 Techno-Economic Siting Analysis

**This document is scenario modelling, not investment advice or a connection commitment.**

- Backend: `pypsa`
- Cases completed: 12 / 12; failed: 0
- Financial horizon: 20 years; real discount rate: 7.0%
- Currency: real EUR, 2026 purchasing power; pre-tax unlevered
- Assumptions status: illustrative_distributed_bess_architecture_comparison_mip_gap_0p5pct_not_market_quotes
- Annualisation: {'mode': 'repeat_block', 'hours_per_year': 8760.0}


## Comparison basis

Technical optimum: the evaluated candidate with the greatest lifetime proxy for additional renewable energy used, net of project battery losses.
This is not source-by-source MWh tracing and is not a certification of total system energy including existing storage or controllable-link losses.
Economic optimum: the evaluated candidate with the greatest lifecycle NPV under the same system-operator dispatch rule; choose no build when every NPV is non-positive.
The NPV is not a global optimum for autonomous developer bidding or arbitrage. Both optima are limited to the discrete candidates evaluated.
System dispatch-cost savings are reported separately and are never counted as developer revenue, project cash flow, or NPV.
`04_technical_revenue_cost_npv_payback.csv` separates technical improvement, project revenue, electricity purchases, O&M, maintenance capital, system savings, NPV, and payback.

## Results

**bess / technical_max_net_RE_gain**: `bess_f11231f128` @ `Croaghonagh`, NPV €-121,923,701.

**bess / economic_max_conditional_NPV**: `NO_BUILD` @ `-`, NPV €0.

## Limitations that must be retained

1. The official network's DC flow representation, boundaries, and parallel circuits are preserved. Q4 does not replace 15/16-node or all-island validation.
2. Repeating a synthetic representative week to a year is a scenario extrapolation. Replacing the price series with actual prices does not turn synthetic wind and load into an actual joint time series.
3. Capacity-constrained dispatch is rerun for each battery year. Capacity is held at its beginning-of-year value within a year; degradation and maintenance occur at year end.
4. The model excludes tax, debt, subsidies, actual system-service bidding, endogenous price feedback, AC voltage, and N-1 validation. Network topology and background generation remain unchanged over the project life.
5. Added wind uses a common weather profile by default to isolate grid-location effects; this is not a site-specific wind-resource assessment. Co-located wind farms share available power proportionally and no dispatch priority is modeled.
6. LCOS is reported only for stand-alone batteries. Hybrid projects use net settlement at a common meter and do not count internal charging as a sale.
7. O&M excludes cell augmentation or replacement, which is listed separately. The dispatch wear penalty is not deducted again as a cash cost.
8. Financial sensitivity revalues a fixed physical operating path. Changes to degradation, efficiency, or life require a new configured run.
9. The first-year constraint split is a counterfactual that relaxes Line and Transformer thermal limits; it is not SNSP curtailment.

Annual cash flows, first-year hourly power and state of charge, and full-network branch metrics are in `cases/<case_id>/`.
Inputs, versions, and data fingerprints are in `manifest.json`; failed cases do not enter the ranking.