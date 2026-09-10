# Problem 3.2 — Nationwide Constraint Group Generation

## Direct answer

The official all-island network was solved for **168 hours**. It produced **16 directional near-congestion modes**, retained as overlapping constraint groups with **2168 bus memberships**, and condensed into **16 non-exclusive merged mode groups** only where both membership and relief-vector tests passed.

A separate, mutually exclusive electrical-response zoning layer contains **5 zones**. Constraint groups and response zones are not interchangeable: a bus may belong to many directional constraint groups, but exactly one response zone inside its passive AC component.

The explicit RES crosswalk covers **612 wind, solar, hydro, and biomass generators**; **612** have at least one active directional group in this 168-hour scenario. File 17 retains each official generator-to-bus mapping. File 18 is the compressed full bus-by-bus electrical response similarity matrix; cross-component and featureless comparisons are intentionally blank.

## Key safeguards

- Positive flow always follows branch bus0-to-bus1 orientation; each opposite direction is a different mode.
- Relief is MW of directional branch relief per MW of curtailment or BESS charging.
- Load-weighted and uniform balancing results are both published. Their raw relief and memberships may differ.
- Response zones use centered pairwise response differences. The maximum reference-change residual after centering is **1.110e-16**.
- Sampled independent LPF validation has maximum absolute error **6.036e-13**.
- DLR changes event weights and headroom, not the PTDF. Q3 DLR coverage is limited to the supplied target line.
- Q6 priority status changes allocation and is reported separately from physical constraint similarity.

## Interpretation

The national model identifies electrically related resources that need not be geographically adjacent. File 07_nonlocal_similar_bus_pairs.csv gives explicit examples. These are model-derived planning signals, not a substitute for protection, voltage, stability, outage, or market studies.

## Evidence map

Every headline metric is in manifest.json; row-level evidence is in files 01–16. File 00_input_audit.json records official inputs and hashes. PRESENTATION_CALCULATIONS.md gives worked formulas and exact table locators.
