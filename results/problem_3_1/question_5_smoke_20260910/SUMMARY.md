# Q5 Wind-Farm Shift Factors

**This is a DC-network study result, not an operational dispatch instruction or a connection offer.**

- Network: WP2033 north-west (15 buses, 18 passive branches, 2 passive AC components, 24 hourly snapshots)
- Wind farms assessed: 3
- Primary balancing reference: load_weighted
- Single-bus comparison reference: Letterkenny
- Constraint-group threshold: 5.0%
- Run mode: SMOKE TEST

**Smoke-test outputs use a shortened snapshot set and a limited farm set. They must not be used as final study results.**

## Definition and sign convention

A shift factor is the change in monitored branch flow, in MW, caused by a +1 MW injection at the wind farm's model-assigned substation and an equal withdrawal at the declared balancing reference.

Load-weighted and uniform balancing are normalized separately inside the source bus's passive AC component. A farm outside the monitored line's component therefore has zero effect on that line. A single-slack transfer across disconnected components is undefined and is left blank in the CSV.

Positive branch flow follows bus0 to bus1. A negative shift factor therefore does not mean low impact; it means the incremental flow is in the opposite direction. The directional relief column combines the factor with the observed binding-flow direction. A positive relief value means that reducing the farm by 1 MW would reduce absolute loading on the active constraint.

## Monitored circuit 5041-17010-2

- Orientation: Srananagh 220 to Cathaleen's Fall is positive.
- Observed binding direction: Cathaleen's Fall to Srananagh 220.
- Binding hours: 13; positive-direction hours: 0; negative-direction hours: 13.
- Maximum absolute flow: 210.000 MW against 210.000 MVA at the peak.

| Rank | Wind farm | Substation | Shift factor | Relief per MW curtailed | Group |
|---:|---|---|---:|---:|:---:|
| 1 | Corderry wind | Corderry | +0.578799 | -0.578799 | no |
| 2 | Clogher wind | Clogher | -0.238396 | +0.238396 | yes |
| 3 | Moy wind | Moy | +0.000000 | -0.000000 | no |

## Validation

- Independent PyPSA LPF finite-difference checks passed: True.
- Maximum absolute analytical-versus-LPF error: 2.331e-15.
- Maximum PTDF change after a rating-only DLR test: 0.000e+00.
- DLR changes thermal headroom. It does not change PTDF unless topology or reactance also changes.

## Link to the Q4 battery comparison

All evaluated Q4 battery sites have the same primary shift factor (-0.238396) on the monitored circuit. One MW of charging provides +0.238396 MW of directional relief under the observed constraint direction.

This explains why splitting the same total MW/MWh across those sites did not improve the strict Q4 technical result: their aggregate PTDF is unchanged, while extra sites add fixed connection and site costs.

The site and allocation arithmetic is recorded in 07_q4_site_and_portfolio_crosswalk.csv.

## Interpretation limits

1. The official model already assigns each aggregated wind generator to a bus. No independent wind-farm coordinates are provided, so 'nearest substation' means that official model connection bus; it is not a new geospatial nearest-neighbour study.
2. Shift factors are linear DC sensitivities. They exclude voltage, reactive power, losses, dynamic stability, fault levels, and N-1 contingencies.
3. Values depend on the balancing reference. The load-weighted reference is primary; uniform and single-bus values are retained so that the convention remains auditable. Balancing is component-local; cross-component single-slack entries are undefined.
4. A high shift factor shows network leverage, not project profitability or a right to dispatch. Q4 economics remain separate.
5. The synthetic 168-hour profiles provide operational context only. The PTDF itself is set by topology and reactance, not by the hourly profile.

See manifest.json for versions, hashes, references, and input provenance.
