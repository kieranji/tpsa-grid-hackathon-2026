"""Adversarial dispatch checks independent of the implementation author's suite."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bess_market.dispatch import Battery, backtest, forecast_day, solve_day, validate_prices


def intervals(prices, durations=None, start="2025-01-01T00:00:00Z", market_date="2025-01-01"):
    durations = np.ones(len(prices)) if durations is None else np.asarray(durations, float)
    starts = pd.Timestamp(start) + pd.to_timedelta(np.r_[0, np.cumsum(durations)[:-1]], unit="h")
    return pd.DataFrame({
        "timestamp_utc": starts, "duration_hours": durations,
        "price_eur_mwh": prices, "market_date": market_date,
    })


class DispatchAdversarialTests(unittest.TestCase):
    def test_negative_prices_do_not_create_unbalanced_energy_or_simultaneous_scheduled_modes(self):
        battery = Battery(power_mw=1, energy_dc_mwh=4, soc_min=0, soc_max=1,
                          capability_fraction=1, round_trip_efficiency=.81,
                          max_efc_per_day=10, wear_bid_eur_per_mwh_throughput=0,
                          execution_fee_eur_per_mwh=0, import_adder_eur_per_mwh=0)
        day = intervals([-100, -100, -100, -100])
        output = solve_day(day, day.price_eur_mwh.to_numpy(), battery)
        self.assertGreater(output.charge_mwh.sum(), 0)
        self.assertGreater(output.net_trading_and_services_eur.sum(), 0)
        self.assertLess(np.minimum(output.charge_mw, output.discharge_mw).max(), 1e-7)
        self.assertAlmostEqual(output.soc_end_mwh.iloc[-1], output.soc_start_mwh.iloc[0], places=7)
        self.assertAlmostEqual(output.discharge_mwh.sum(), .81 * output.charge_mwh.sum(), places=7)
        self.assertLess(output.soc_balance_error_mwh.abs().max(), 1e-7)

    def test_reserve_and_capacity_have_separate_power_and_begin_end_soc(self):
        battery = Battery(power_mw=2, energy_dc_mwh=10, soc_min=.1, soc_max=.9,
                          capability_fraction=.8, round_trip_efficiency=.81,
                          max_efc_per_day=10, reserve_duration_hours=1,
                          reserve_activation_fraction_each_way=.01)
        day = intervals([0, 30, 180, 20, -40, 70])
        obligation = .4
        output = solve_day(day, day.price_eur_mwh.to_numpy(), battery,
                           reserve_price_eur_mw_h=100, crm_obligation_mw=obligation,
                           crm_strike_price_eur_mwh=100)
        self.assertGreater(output.reserve_mw_hours.sum(), 0)
        power = battery.power_mw * battery.capability_fraction
        lo = battery.soc_min * battery.energy_dc_mwh * battery.capability_fraction
        hi = battery.soc_max * battery.energy_dc_mwh * battery.capability_fraction
        crm_energy = obligation * battery.crm_delivery_hours / battery.eta
        self.assertLessEqual((output.discharge_mw + output.reserve_mw + obligation).max(), power + 1e-7)
        self.assertLessEqual((output.charge_mw + output.reserve_mw).max(), power + 1e-7)
        for state in (output.soc_start_mwh, output.soc_end_mwh):
            self.assertTrue(np.all(state - lo >= output.reserve_mw / battery.eta + crm_energy - 1e-7))
            self.assertTrue(np.all(hi - state >= output.reserve_mw * battery.eta - 1e-7))
        np.testing.assert_allclose(output.activation_up_mwh, output.activation_down_mwh)
        self.assertGreater(output.charge_mwh.sum(), output.discharge_mwh.sum())
        self.assertLess(output.soc_balance_error_mwh.abs().max(), 1e-7)

    def test_forecast_and_dispatch_ignore_previous_day_current_and_future_outcomes(self):
        frames = []
        for index in range(10):
            date = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index)
            prices = np.arange(24) * 4.0 + index
            frames.append(intervals(prices, start=date.isoformat(), market_date=str(date.date())))
        original = validate_prices(pd.concat(frames, ignore_index=True))
        target = original[original.market_date == "2025-01-08"].copy()
        changed = original.copy()
        # January 7 is withheld in addition to delivery-day/future outcomes.
        changed.loc[changed.market_date >= "2025-01-07", "price_eur_mwh"] = -9876
        target_changed = changed[changed.market_date == "2025-01-08"].copy()
        forecast = forecast_day(original, target)
        alternative = forecast_day(changed, target_changed)
        np.testing.assert_array_equal(forecast, alternative)
        battery = Battery(power_mw=1, energy_dc_mwh=4, capability_fraction=1)
        schedule = solve_day(target, forecast, battery)
        changed_schedule = solve_day(target_changed, alternative, battery)
        for column in ("charge_mw", "discharge_mw", "reserve_mw", "soc_end_mwh"):
            np.testing.assert_allclose(schedule[column], changed_schedule[column], atol=1e-7)
        self.assertNotEqual(schedule.net_trading_and_services_eur.sum(),
                            changed_schedule.net_trading_and_services_eur.sum())

    def test_capacity_clawback_is_duration_weighted_and_requires_strike(self):
        day = intervals([50, 100, 150, 200], durations=[.5, 1, .25, .75])
        battery = Battery(power_mw=2, energy_dc_mwh=20, soc_min=0, soc_max=1,
                          capability_fraction=1, round_trip_efficiency=1)
        with self.assertRaisesRegex(ValueError, "strike"):
            solve_day(day, np.zeros(4), battery, crm_obligation_mw=1)
        output = solve_day(day, np.zeros(4), battery, crm_obligation_mw=1,
                           crm_strike_price_eur_mwh=100)
        np.testing.assert_allclose(output.capacity_difference_charge_eur, [0, 0, 12.5, 75])
        self.assertAlmostEqual(output.capacity_difference_charge_eur.sum(), 87.5)
        self.assertAlmostEqual(output.net_trading_and_services_eur.sum(), -87.5)

    def test_dst_cycle_budget_counts_23_and_25_actual_hours(self):
        battery = Battery(power_mw=1, energy_dc_mwh=4, soc_min=0, soc_max=1,
                          capability_fraction=1, round_trip_efficiency=.81,
                          max_efc_per_day=.03, reserve_activation_fraction_each_way=.1,
                          wear_bid_eur_per_mwh_throughput=0,
                          execution_fee_eur_per_mwh=0, import_adder_eur_per_mwh=0)
        for date, hours in [("2025-03-30", 23), ("2025-10-26", 25)]:
            with self.subTest(date=date):
                local_start = pd.Timestamp(date, tz="Europe/Dublin")
                local_end = local_start + pd.DateOffset(days=1)
                stamps = pd.date_range(local_start, local_end, freq="h", inclusive="left").tz_convert("UTC")
                day = pd.DataFrame({
                    "timestamp_utc": stamps, "duration_hours": 1.0,
                    "price_eur_mwh": 0.0, "market_date": date,
                })
                output, summary = backtest(day, battery, "daily_oracle", reserve_price_eur_mw_h=100)
                self.assertEqual(summary["hours"], hours)
                budget = battery.deliverable_ac_mwh * battery.max_efc_per_day * hours / 24
                self.assertAlmostEqual(output.discharge_mwh.sum(), budget, places=6)
                self.assertAlmostEqual(summary["efc"], .03 * hours / 24, places=6)
                self.assertAlmostEqual(output.soc_start_mwh.iloc[0], output.soc_end_mwh.iloc[-1], places=7)
                self.assertTrue(summary["service_price_is_hypothetical"])
                self.assertFalse(summary["grid_feasibility_verified"])

    def test_cash_ledger_charges_activation_fees_once_and_excludes_wear_regularizer(self):
        battery = Battery(power_mw=1, energy_dc_mwh=5, capability_fraction=1,
                          execution_fee_eur_per_mwh=3, import_adder_eur_per_mwh=4,
                          wear_bid_eur_per_mwh_throughput=8,
                          reserve_activation_fraction_each_way=.01)
        day = intervals([0, 200, -100, 150])
        output = solve_day(day, day.price_eur_mwh.to_numpy(), battery, reserve_price_eur_mw_h=50)
        expected = (
            ((output.discharge_mwh - output.charge_mwh) * day.price_eur_mwh).sum()
            - (output.charge_mwh + output.discharge_mwh).sum() * 3
            - output.charge_mwh.sum() * 4
            + output.reserve_mw_hours.sum() * 50 * battery.reserve_payment_capture
        )
        self.assertAlmostEqual(output.net_trading_and_services_eur.sum(), expected, places=7)

    def test_cash_upper_bound_requires_zero_wear_regularization(self):
        # A low positive cash margin is rejected when an additional noncash wear
        # bid is in the objective. Thus that oracle is not a strict CASH upper bound.
        battery = Battery(power_mw=1, energy_dc_mwh=2, soc_min=0, soc_max=1,
                          capability_fraction=1, round_trip_efficiency=1,
                          max_efc_per_day=20, wear_bid_eur_per_mwh_throughput=20,
                          execution_fee_eur_per_mwh=0, import_adder_eur_per_mwh=0)
        day = intervals([0, 10])
        regularized = solve_day(day, np.array([0, 10]), battery)
        cash_oracle = solve_day(day, np.array([0, 10]), replace(battery, wear_bid_eur_per_mwh_throughput=0))
        self.assertAlmostEqual(regularized.net_trading_and_services_eur.sum(), 0)
        self.assertAlmostEqual(cash_oracle.net_trading_and_services_eur.sum(), 10)
        # The public upper-benchmark entry point must remove the noncash bid.
        _, summary = backtest(day, battery, policy="daily_oracle")
        self.assertAlmostEqual(summary["net_trading_and_services_eur"], 10)
        self.assertEqual(summary["dispatch_wear_regularizer_eur_mwh"], 0)

    def test_actual_price_gaps_overlaps_duplicates_and_nonfinite_values_fail(self):
        valid = intervals([1, 2, 3])
        cases = []
        gap = valid.copy()
        gap.loc[1, "timestamp_utc"] += pd.Timedelta(minutes=1)
        cases.append(gap)
        duplicate = valid.copy()
        duplicate.loc[1, "timestamp_utc"] = duplicate.loc[0, "timestamp_utc"]
        cases.append(duplicate)
        nonfinite = valid.copy()
        nonfinite.loc[1, "price_eur_mwh"] = np.nan
        cases.append(nonfinite)
        overlap = valid.copy()
        overlap.loc[0, "duration_hours"] = 2
        cases.append(overlap)
        for case in cases:
            with self.subTest(case=case.to_dict()), self.assertRaises(ValueError):
                validate_prices(case)


if __name__ == "__main__":
    unittest.main()
