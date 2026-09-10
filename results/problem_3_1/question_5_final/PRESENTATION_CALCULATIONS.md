# Q5 Presentation Calculations

## Step 1 — Fix the monitored circuit and direction

The Q1/Q3 monitored circuit is 5041-17010-2, oriented Srananagh 220 to Cathaleen's Fall. Its fixed rating is 210.0 MVA.

The solved 168-hour case reaches the rating for 52 hours. All binding hours have sign -1, so the active direction is Cathaleen's Fall to Srananagh 220.

Q1 cross-check:

- Q1 binding hours: 52
- Q1 energy recovered by a 25% rating uplift: 588.614 MWh

## Step 2 — Build the DC PTDF

Let K be the bus-by-branch incidence matrix and B the diagonal branch susceptance matrix, with branch susceptance equal to 1/x_pu_eff.

$$L = K B K^T$$

$$PTDF = B K^T L^+$$

L+ is the Moore-Penrose pseudoinverse. Positive flow follows each branch's bus0-to-bus1 orientation.

## Step 3 — Apply the balancing reference

For a farm g connected at bus b(g), the primary component-local load-weighted shift factor is:

$$SF_{line,g} = PTDF_{line,b(g)} - \sum_{j \in component(g)} w_j PTDF_{line,j}$$

where the mean-load weights w sum to one inside the farm's passive AC component. A farm in another disconnected component has zero effect on this line.

## Step 4 — Convert the signed factor into directional relief

For a 1 MW curtailment, the injection change is -1 MW. The first-order reduction in absolute loading is:

$$Relief_{curtail} = s_{binding} \times SF_{line,g}$$

Worked example — Croaghonagh wind:

- Primary shift factor: -0.238395702
- Binding-flow sign: -1
- Relief per MW curtailed: (-1) × (-0.238395702) = +0.238395702 MW/MW
- Installed capacity: 139.200 MW
- Maximum linear relief if reduced from full output to zero: 139.200 × 0.238395702 = 33.185 MW

Positive relief means curtailment unloads the active direction. Negative relief means curtailing that farm would move the circuit the wrong way.

## Step 5 — Explain the Q4 distributed-battery result

Every evaluated Q4 site has primary factor approximately -0.238395702. Charging is a negative injection, so its directional relief is:

$$Relief_{charge} = -1 \times (-0.238395702) = +0.238395702 MW/MW$$

For the 45.0 MW Q4 comparison, simultaneous charging at full power therefore provides about 45.0 × 0.238395702 = 10.728 MW of first-order relief.

The 1–4-site portfolio factors range only from -0.238395702387 to -0.238395702387; the differences are numerical roundoff.

Therefore, splitting the same total MW/MWh among these buses does not change aggregate PTDF leverage. The strict Q4 dispatch result stays the same while additional sites add fixed costs.

## Step 6 — Connect the result to DLR

A rating-only DLR multiplier changes available headroom but not x_pu_eff, K, B, or the PTDF. The computed maximum shift-factor change is 0.000e+00.

The linked Q3 series has mean multiplier 1.095551 and maximum 1.279501. Applied to 210.0 MVA, these correspond to 230.066 MVA mean dynamic capacity and 268.695 MVA maximum dynamic capacity.

## Step 7 — Numerical validation

- 14 farm transfers were checked against independent PyPSA LPF perturbations.
- Maximum analytical-versus-LPF error: 4.885e-15.
- Maximum component-local reference residual: 1.735e-17.

These calculations quantify network sensitivity only. They do not assign market revenue, dispatch priority, or project NPV.
