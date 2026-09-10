"""Chronological daily BESS energy / generic reserve co-optimization.

This is a price-taking screening model, not an executable market bid or a FASS
product simulator. `causal` uses only prices older than the previous delivery
day; `daily_oracle` is explicitly an ex-post price-information upper benchmark.
Each day returns to the same inventory, so no unpriced daily energy is created.
Actual price gaps are rejected, never interpolated. Only forecasts interpolate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Battery:
    power_mw: float = 85.0
    energy_dc_mwh: float = 310.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    round_trip_efficiency: float = 0.85
    capability_fraction: float = 0.97
    max_efc_per_day: float = 1.5
    # Dispatch regularizer only; not a cash expense when maintenance includes augmentation.
    wear_bid_eur_per_mwh_throughput: float = 2.0
    execution_fee_eur_per_mwh: float = 1.0
    import_adder_eur_per_mwh: float = 3.0
    reserve_duration_hours: float = 1.0
    # Stylized equal up/down activated fractions, not an observed FASS trajectory.
    reserve_activation_fraction_each_way: float = 0.01
    reserve_payment_capture: float = 0.90
    crm_delivery_hours: float = 4.0

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Battery parameters must be finite")
        if min(self.power_mw, self.energy_dc_mwh, self.max_efc_per_day,
               self.reserve_duration_hours, self.crm_delivery_hours) <= 0:
            raise ValueError("Power, energy, endurance and cycle budget must be positive")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("Invalid SOC window")
        if not 0 < self.round_trip_efficiency <= 1 or not 0 < self.capability_fraction <= 1:
            raise ValueError("Invalid efficiency or capability")
        if not 0 <= self.reserve_payment_capture <= 1:
            raise ValueError("Invalid reserve payment capture")
        if not 0 <= self.reserve_activation_fraction_each_way <= .5:
            raise ValueError("Invalid activation fraction")
        if min(self.wear_bid_eur_per_mwh_throughput, self.execution_fee_eur_per_mwh,
               self.import_adder_eur_per_mwh) < 0:
            raise ValueError("Costs cannot be negative")

    @property
    def eta(self):
        return math.sqrt(self.round_trip_efficiency)

    @property
    def deliverable_ac_mwh(self):
        return self.energy_dc_mwh * (self.soc_max - self.soc_min) * self.eta


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_utc", "duration_hours", "price_eur_mwh", "market_date"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    df = frame.copy()
    df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    for c in ("duration_hours", "price_eur_mwh"):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"Nonfinite {c}")
    if df.empty or df.timestamp_utc.duplicated().any() or (df.duration_hours <= 0).any():
        raise ValueError("Empty, duplicate or nonpositive price intervals")
    ends = df.timestamp_utc + pd.to_timedelta(df.duration_hours, unit="h")
    if not np.all(ends.iloc[:-1].to_numpy() == df.timestamp_utc.iloc[1:].to_numpy()):
        raise ValueError("Actual price intervals have gaps or overlaps; no filling permitted")
    df["market_date"] = pd.to_datetime(df.market_date).dt.date.astype(str)
    if not df.market_date.is_monotonic_increasing:
        raise ValueError("Delivery dates must be chronological")
    return df


def forecast_day(history: pd.DataFrame, day: pd.DataFrame) -> np.ndarray:
    """Lagged local-hour median, fixed before gate closure; no future outcomes.

    Withhold the previous delivery day completely. Use at most seven older
    delivery days. A cold start uses a flat zero forecast (no arbitrage signal).
    Hour-of-day median is intentionally simple and is NOT a commercial forecast.
    """
    cutoff = pd.Timestamp(day.market_date.iloc[0]) - pd.Timedelta(days=1)
    h = history.loc[pd.to_datetime(history.market_date) < cutoff].copy()
    h = h.loc[pd.to_datetime(h.market_date) >= cutoff - pd.Timedelta(days=7)]
    if h.empty:
        return np.zeros(len(day))
    local = h.timestamp_utc.dt.tz_convert("Europe/Dublin")
    h["local_hour"] = local.dt.hour
    means = h.groupby("local_hour").price_eur_mwh.median().reindex(range(24))
    means = means.interpolate(limit_direction="both")
    hour = day.timestamp_utc.dt.tz_convert("Europe/Dublin").dt.hour
    return means.loc[hour].to_numpy(float)


def solve_day(day: pd.DataFrame, forecast: np.ndarray, battery: Battery,
              reserve_price_eur_mw_h: float = 0., crm_obligation_mw: float = 0.,
              crm_strike_price_eur_mwh: float | None = None) -> pd.DataFrame:
    """MILP, with shared power/SOC between scheduled energy and one reserve block.

    Capacity award is conservatively protected at every interval for the chosen
    delivery duration, in addition to reserve endurance. Difference charges use
    DA as an explicit proxy reference, so no balancing settlement is implied.
    A capacity award requires a strike-price input; no silent zero clawback.
    """
    n = len(day)
    dt = day.duration_hours.to_numpy(float)
    f = np.asarray(forecast, float)
    if f.shape != (n,) or not np.isfinite(f).all():
        raise ValueError("Invalid forecast")
    if not np.isfinite(reserve_price_eur_mw_h) or reserve_price_eur_mw_h < 0:
        raise ValueError("Reserve sensitivity price must be finite and nonnegative")
    if not 0 <= crm_obligation_mw <= battery.power_mw * battery.capability_fraction:
        raise ValueError("Invalid capacity obligation")
    if crm_obligation_mw and (crm_strike_price_eur_mwh is None or
                             not np.isfinite(crm_strike_price_eur_mwh)):
        raise ValueError("Capacity obligation requires explicit strike-price sensitivity")
    p = battery.power_mw * battery.capability_fraction
    eta = battery.eta
    e_lo = battery.soc_min * battery.energy_dc_mwh * battery.capability_fraction
    e_hi = battery.soc_max * battery.energy_dc_mwh * battery.capability_fraction
    crm_energy = crm_obligation_mw * battery.crm_delivery_hours / eta
    if e_lo + crm_energy > e_hi + 1e-8:
        raise ValueError("Capacity obligation cannot be sustained within usable SOC")
    initial = max((e_lo + e_hi) / 2, e_lo + crm_energy)
    a = battery.reserve_activation_fraction_each_way
    # Variables per interval: scheduled charge c, discharge d, end SOC e,
    # symmetric reserve r, and charge mode z (binary).
    c, d, e, r, z = [np.arange(i*n, (i+1)*n) for i in range(5)]
    cost = np.zeros(5*n)
    variable = battery.execution_fee_eur_per_mwh + battery.wear_bid_eur_per_mwh_throughput
    cost[c] = (f + variable + battery.import_adder_eur_per_mwh) * dt
    cost[d] = (-f + variable) * dt
    cost[r] = (-reserve_price_eur_mw_h * battery.reserve_payment_capture
               + a * (2*variable + battery.import_adder_eur_per_mwh)) * dt
    lb = np.zeros(5*n)
    ub = np.full(5*n, np.inf)
    ub[c] = ub[d] = p
    lb[e], ub[e] = e_lo, e_hi
    ub[r] = p if reserve_price_eur_mw_h > 0 else 0
    ub[z] = 1
    integrality = np.zeros(5*n, int)
    integrality[z] = 1
    row_ids, col_ids, vals, lower, upper = [], [], [], [], []
    def constraint(items, low=-np.inf, high=np.inf):
        ix = len(lower)
        for j, v in items:
            row_ids.append(ix); col_ids.append(int(j)); vals.append(v)
        lower.append(low); upper.append(high)
    for t in range(n):
        dyn = [(e[t], 1), (c[t], -eta*dt[t]), (d[t], dt[t]/eta),
               (r[t], a*dt[t]*(1/eta-eta))]
        if t:
            dyn.append((e[t-1], -1))
        constraint(dyn, initial if t == 0 else 0, initial if t == 0 else 0)
        constraint([(c[t], 1), (z[t], -p)], high=0)
        constraint([(d[t], 1), (z[t], p)], high=p)
        # Conservative capacity and reserves cannot reuse the same headroom.
        constraint([(d[t], 1), (r[t], 1)], high=p-crm_obligation_mw)
        constraint([(c[t], 1), (r[t], 1)], high=p)
        for idx in ([e[t-1], e[t]] if t else [None, e[t]]):
            soc = initial if idx is None else None
            terms = [(r[t], battery.reserve_duration_hours/eta)]
            if idx is not None:
                terms.append((idx, -1))
            constraint(terms, high=(soc if soc is not None else 0)-e_lo-crm_energy)
            terms = [(r[t], battery.reserve_duration_hours*eta)]
            if idx is not None:
                terms.append((idx, 1))
            constraint(terms, high=e_hi-(soc if soc is not None else 0))
    constraint([(e[-1], 1)], initial, initial)
    cycle_budget = (e_hi-e_lo)*eta*battery.max_efc_per_day*dt.sum()/24
    constraint([(d[t], dt[t]) for t in range(n)] +
               [(r[t], a*dt[t]) for t in range(n)], high=cycle_budget)
    mat = coo_matrix((vals, (row_ids, col_ids)), shape=(len(lower), 5*n)).tocsc()
    result = milp(cost, integrality=integrality, bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(mat, lower, upper),
                  options={"mip_rel_gap": 1e-7, "time_limit": 30})
    if not result.success or result.x is None:
        raise RuntimeError(f"Dispatch MILP failed: {result.message}")
    x = result.x
    actual = day.price_eur_mwh.to_numpy(float)
    out = day.copy()
    out["forecast_eur_mwh"] = f
    out["charge_mw"], out["discharge_mw"] = x[c], x[d]
    out["soc_end_mwh"], out["reserve_mw"] = x[e], x[r]
    out["soc_start_mwh"] = np.r_[initial, x[e][:-1]]
    out["activation_up_mwh"] = out["activation_down_mwh"] = a*x[r]*dt
    out["charge_mwh"] = (x[c]+a*x[r])*dt
    out["discharge_mwh"] = (x[d]+a*x[r])*dt
    out["energy_sales_eur"] = out.discharge_mwh * actual
    out["energy_purchases_eur"] = out.charge_mwh * actual
    out["execution_fees_eur"] = (out.charge_mwh+out.discharge_mwh)*battery.execution_fee_eur_per_mwh
    out["import_adder_eur"] = out.charge_mwh*battery.import_adder_eur_per_mwh
    out["reserve_mw_hours"] = x[r]*dt
    out["service_gross_eur"] = out.reserve_mw_hours*reserve_price_eur_mw_h
    out["service_deductions_eur"] = out.service_gross_eur*(1-battery.reserve_payment_capture)
    out["capacity_difference_charge_eur"] = (crm_obligation_mw * dt *
        np.maximum(actual-float(crm_strike_price_eur_mwh), 0)
        if crm_obligation_mw else np.zeros(n))
    out["net_trading_and_services_eur"] = (out.energy_sales_eur-out.energy_purchases_eur
        -out.execution_fees_eur-out.import_adder_eur+out.service_gross_eur
        -out.service_deductions_eur-out.capacity_difference_charge_eur)
    out["soc_balance_error_mwh"] = (out.soc_end_mwh-out.soc_start_mwh
        -eta*out.charge_mwh+out.discharge_mwh/eta)
    if out.soc_balance_error_mwh.abs().max() > 1e-5 or np.minimum(x[c], x[d]).max() > 1e-6:
        raise AssertionError("Physical dispatch validation failed")
    return out


def backtest(frame: pd.DataFrame, battery: Battery, policy="causal",
             reserve_price_eur_mw_h=0., crm_obligation_mw=0.,
             crm_strike_price_eur_mwh=None) -> tuple[pd.DataFrame, dict]:
    df = validate_prices(frame)
    if policy not in {"causal", "daily_oracle"}:
        raise ValueError("Unknown policy")
    # Oracle maximizes the reported cash objective, so omit the non-cash wear
    # regularizer. Physical cycle budget still applies to both policies.
    dispatch_battery = (replace(battery, wear_bid_eur_per_mwh_throughput=0.)
                        if policy == "daily_oracle" else battery)
    parts = []
    for _, day in df.groupby("market_date", sort=False):
        forecast = (day.price_eur_mwh.to_numpy() if policy == "daily_oracle"
                    else forecast_day(df, day))
        parts.append(solve_day(day, forecast, dispatch_battery, reserve_price_eur_mw_h,
                               crm_obligation_mw, crm_strike_price_eur_mwh))
    dispatch = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in dispatch if c.endswith("_eur") or c.endswith("_mw_hours")]
    summary = {c: float(dispatch[c].sum()) for c in sum_cols}
    summary.update({"policy": policy, "hours": float(df.duration_hours.sum()),
                    "days": int(df.market_date.nunique()),
                    "price_information": "ex_post_daily_upper_benchmark" if policy == "daily_oracle" else "lagged_prices_only",
                    "dispatch_wear_regularizer_eur_mwh": dispatch_battery.wear_bid_eur_per_mwh_throughput,
                    "reserve_price_eur_mw_h": reserve_price_eur_mw_h,
                    "crm_obligation_mw": crm_obligation_mw,
                    "charge_mwh": float(dispatch.charge_mwh.sum()),
                    "discharge_mwh": float(dispatch.discharge_mwh.sum()),
                    "efc": float(dispatch.discharge_mwh.sum()/(battery.deliverable_ac_mwh*battery.capability_fraction)),
                    "max_soc_balance_error_mwh": float(dispatch.soc_balance_error_mwh.abs().max()),
                    "max_simultaneous_scheduled_mw": float(np.minimum(dispatch.charge_mw, dispatch.discharge_mw).max()),
                    "annual_equivalent_multiplier": 8760/float(df.duration_hours.sum()),
                    "grid_feasibility_verified": False,
                    "service_price_is_hypothetical": True})
    return dispatch, summary
