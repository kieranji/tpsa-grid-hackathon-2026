# Q4 presentation calculations

Status: the network and Q3 DLR series are project data; prices, costs, repeated-week
annualisation and all investment conclusions remain illustrative scenario assumptions.

Q2 source correction: the repository final Q2 recommendation is 85 MW / 310 MWh.
The 90 MW / 440 MWh point is retained only as an independently requested envelope case.

## Step-by-step calculation

1. Annualisation factor = 8,760 / 168 = 52.142857.
2. Annual technical gain = factor × [(renewable dispatch with design − scenario-matched
   no-build counterfactual renewable dispatch) − project battery losses].
3. Lifetime technical gain is the undiscounted sum of the annual technical gain
   after the battery degradation and maintenance trajectory is re-solved each year.
4. Project revenue = common-meter external energy sales + configured grid-service
   revenue. External charging imports are paid; internal hybrid transfers are not
   counted as a sale.
5. Net project cash flow = sales + grid services + residual value − imports − fixed
   and variable O&M − replacement/augmentation/overhaul − decommissioning − year-0 CAPEX.
6. NPV = sum from year 0 to year N of net cash flow divided by (1 + real discount
   rate)^year. System dispatch-cost savings are reported separately and never enter
   developer revenue, project cash flow or NPV.
7. Simple and discounted payback are blank when cumulative cash flow never crosses zero.

## Capacity: fixed lines versus Q3 DLR

| MW | MWh | Fixed gain MWh | BESS gain with DLR MWh | DLR+BESS gain vs fixed MWh | Fixed NPV EUR | Conditional DLR NPV EUR |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | 110 | 281,013 | 274,780 | 874,965 | -41,129,703 | -40,452,297 |
| 45 | 220 | 553,004 | 515,984 | 1,116,169 | -79,856,992 | -79,580,950 |
| 62 | 227 | 579,761 | 537,786 | 1,137,971 | -92,444,917 | -90,907,514 |
| 68 | 330 | 795,546 | 745,741 | 1,345,926 | -118,941,651 | -117,046,342 |
| 85 | 310 | 759,383 | 707,221 | 1,307,406 | -124,119,704 | -123,543,915 |
| 90 | 440 | 1,027,014 | 975,932 | 1,576,116 | -157,909,624 | -155,218,773 |

DLR standalone gain = (3660.590
− 3085.071) × 52.142857
× 20 = 600,185 MWh over the modeled lifetime.
The conditional DLR NPV columns contain battery-project cash flows only. No DLR
CAPEX, OPEX or DLR-owner revenue is available in the source and none is invented.

Technical maximum among the fixed-line battery candidates: 90
MW / 440 MWh. Best battery-project NPV among built
candidates: EUR -41,129,703; because this is below zero, the
economic choice under the stated scenario is NO_BUILD with NPV EUR 0.

## Strict 45 MW / 220 MWh distributed tie-break

| Sites | Placement | Lifetime gain MWh | Gain delta vs single MWh | Initial CAPEX EUR | NPV EUR | NPV delta vs single EUR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Croaghonagh | 553,004 | 0 | 53,790,000 | -79,856,992 | 0 |
| 2 | Croaghonagh;Binbane | 553,004 | 0 | 54,395,000 | -80,798,962 | -941,971 |
| 3 | Croaghonagh;Binbane;Ardnagappary | 553,004 | 0 | 55,000,000 | -81,740,933 | -1,883,941 |
| 4 | Croaghonagh;Binbane;Ardnagappary;Drumkeen | 553,004 | 0 | 55,605,000 | -82,682,904 | -2,825,912 |

The strict run uses a 0.0001% MILP relative gap. Equal technical gains within
floating-point precision mean the extra sites do not improve this modeled network
case; they only add site, connection and O&M costs.

## Decision boundary

These results do not establish a merchant business case: the model has no real SEM
price series, DS3/FASS bid stack, taxes, debt, grants, endogenous price response,
AC voltage or N-1 security. Replacing only prices does not make the synthetic wind
and load week a historical joint time series.
