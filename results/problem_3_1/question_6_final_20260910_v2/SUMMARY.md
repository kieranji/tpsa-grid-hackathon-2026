# Q6 Priority and Non-Priority Wind Dispatch

**The official project network contains no verified legal priority-status field. All status labels in this run are explicit scenario choices, not Irish operational facts.**

- Network: WP2033 north-west
- Run horizon: 168 snapshots / 168.0 hours
- Priority farms in the illustrative assignment: 4
- Non-priority farms: 10
- Solver rule: exact lexicographic load-shedding, priority-energy, total-wind-energy, then physical-cost stages

## Direct answer

The illustrative priority rule changes total wind dispatch-down by +0.000000 MWh over the model block relative to neutral maximum-wind dispatch.

After subtracting the corresponding copperplate allocation effect, the network-specific difference-in-differences is +0.000000 MWh.

Priority status protects 2,603.625666 MWh across farms with reduced dispatch-down and transfers 2,603.625666 MWh of additional burden to farms with increased dispatch-down. Protection and burden need not be equal when the priority rule changes total technically feasible wind energy.

## Why allocation and inefficiency are different

A status rule can move dispatch-down from one owner to another while leaving total wind energy unchanged. That is a distributional effect. It is a technical inefficiency only when the same network could accept more total wind under the neutral maximum-wind solution.

## Largest farm-level changes

- Largest additional burden: Letterkenny wind at Letterkenny, +1,165.747319 MWh.
- Largest protected amount: Drumkeen wind at Drumkeen, -1,378.793405 MWh (negative means less dispatch-down).

## Where the inefficiency is located

The largest tested non-additive line-relaxation attribution is 5041-17010-2: relaxing only this branch changes the status inefficiency by +0.000000 MWh.

Line attributions are marginal counterfactuals. They are not additive when several constraints bind together.

## DLR interaction

Under the linked Q3 rating series, the priority-minus-neutral total dispatch-down difference is +0.000000 MWh.

DLR changes hourly thermal headroom, not the PTDF. It can therefore change how often a priority allocation conflicts with a network limit without changing the underlying linear sensitivity.

## Q4 economic bridge

The greatest conditional project NPV in the linked Q4 comparison is NO_BUILD at EUR 0.

Q6 wind-revenue transfers are reported separately. They are not booked as stand-alone BESS revenue, system savings, or a change in battery NPV. The official input supplies neither contractual compensation rules nor a status right for stored energy.

## Interpretation limits

1. Status assignments are scenario choices because the supplied network has no verified priority field.
2. The priority rule is a transparent lexicographic counterfactual, not a reproduction of SEM settlement or EirGrid control-room rules.
3. The 168-hour profiles are synthetic. Repeat-block annualisation is an illustrative extrapolation.
4. The network model is lossless DC power flow and excludes voltage, reactive power, inertia, SNSP and transient security.
5. Multiple simultaneous constraints make single-line causal attribution non-additive.
6. Physical generator cost uses original marginal costs only; status objectives are never reported as cost.

See 12_conclusion_provenance.csv and manifest.json for exact source and formula lineage.
