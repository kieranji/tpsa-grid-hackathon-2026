# Problem 3.2 Presentation Calculations

## 1. Directional congestion event

For each branch and direction:

loading = abs(flow_MW) / effective_rating_MVA

event_weight = snapshot_hours times clip((loading - 0.9) / (1 - 0.9), 0, 1)

Example top mode: 3581-89516-1::+1. Its maximum loading is 1.0000000000000393 pu and its event-weighted severity is 21.988046812186205 hours. Source: 01_directional_constraint_modes.csv, key mode_id=3581-89516-1::+1.

## 2. Directional relief and overlapping membership

relief(mode, bus) = flow_sign(mode) times PTDF(branch, bus)

normalised_relief = max(relief, 0) / max_bus(max(relief, 0))

A bus is a member when normalised relief is at least 0.05. Example: mode 3581-89516-1::+1, bus 1061, raw relief 0.10002300251415745, normalised relief 0.19481315854174996. Source: 02_overlapping_constraint_memberships.csv with the exact mode and bus keys.

## 3. Mode merging

Two modes merge only when they are in the same passive AC component and both membership Jaccard is at least 0.8 and positive-relief cosine is at least 0.98. Pair-level audit: 04_mode_similarity_audit.csv. Final mapping: 03_merged_mode_map.csv.

## 4. Electrical response zones

For each component, the bus feature for each active mode is the centered directional PTDF multiplied by the square root of that mode's normalized congestion-event weight. Centering removes the reference-dependent row constant. Average-linkage clustering chooses the number of zones by maximum silhouette, with a minimum useful score of 0.05. Sources: 05_bus_response_zones.csv and 06_response_zone_summary.csv.

## 5. Reference test

The raw load-weighted and uniform factors can change because the balancing injection changes. For every mode, subtracting the within-component mean leaves pairwise bus-response differences. The resulting maximum discrepancy is 1.110e-16. Source: 08_reference_sensitivity.csv.

## 6. Independent electrical validation

A 1 MW source injection is balanced over load in the same passive AC component, and two PyPSA LPFs measure the branch-flow delta. This is compared against the analytical PTDF in 09_lpf_sample_validation.csv. Maximum absolute error: 5.228e-13.

## 7. Q5, DLR, and Q6 links

- File 10_north_west_q5_consistency.csv recomputes the Q5 target-line factors on the North-West network and separately shows the all-island factor where the identifier exists.
- File 11_static_vs_dlr.csv changes the Q3 target rating while holding the solved-flow trace fixed; this isolates headroom/event classification, not redispatch.
- File 12_q6_priority_crosswalk.csv imports Q6 policy allocation metrics. Priority status is not an electrical-similarity input and is not developer BESS revenue.

## 8. Explicit RES correlation map

File 17_renewable_resource_constraint_memberships.csv maps every configured renewable generator through its official model connection bus to zero or more overlapping directional groups and exactly one response zone. File 18_bus_electrical_similarity.csv.gz gives the full centered, event-weighted cosine matrix for reproducible national correlation analysis.
