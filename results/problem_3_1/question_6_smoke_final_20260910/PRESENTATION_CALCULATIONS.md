# Q6 Presentation Calculations

## Step 1 - Define dispatch-down

For wind farm g and hour t:

$$c_{g,t}=a_{g,t}-w_{g,t}$$

where a is available wind and w is dispatched wind.

## Step 2 - Apply exact priority stages

Neutral dispatch first minimises unserved energy, then maximises total wind, then minimises original physical operating cost. Priority dispatch inserts a stage that maximises priority-wind energy before maximising total wind.

The optimum of every earlier stage is fixed within the configured tolerance, so later stages cannot trade it away.

## Step 3 - Measure technical inefficiency

- Neutral dispatch-down: 696.373353571 MWh
- Priority dispatch-down: 696.373353571 MWh
- Priority minus neutral: 696.373353571 - 696.373353571 = -0.000000000 MWh

A positive difference is lost feasible wind energy. A zero difference with non-zero farm changes is redistribution rather than technical inefficiency.

## Step 4 - Keep project and system money separate

The illustrative annualisation factor is 8,760 / 24.0 = 365.000000000. With an illustrative energy value of EUR 70.00/MWh:

$$Annual\ wind\ value\ change = -(-0.000000000) 	imes 365.000000000 	imes 70.00 = EUR\ 0.00/year$$

Using the Q4 real discount rate and horizon, the annuity present-value factor is 10.594014246, giving an aggregate wind-value transfer proxy of EUR 0.00.

This value belongs only in the wind-owner allocation scenario. It is not BESS revenue and is not the model's system dispatch-cost saving.

## Step 5 - Farm-level example

Drumkeen wind changes by -333.612139945 MWh of dispatch-down. Positive means extra burden; negative means protected energy.

Its annualised energy-value transfer at the stated assumption is EUR +8,523,790.18/year, with present value EUR +90,301,154.55.

