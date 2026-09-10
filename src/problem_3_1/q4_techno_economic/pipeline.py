"""Command-line workflow with preflight, short smoke runs, checkpoints and audit trail."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import pandas as pd
from . import __version__
from .config import load_config, validate_config, annual_factor
from .types import Design
from .grid import PyPSABackend, aligned_csv
from .lifecycle import simulate_lifetime
from .sensitivity import financial_sensitivity
from .reporting import write_report


def _json_clean(obj):
    if isinstance(obj, dict): return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None: return obj
    return str(obj)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def designs_for(sites, cfg, only=None):
    designs = []
    for bus in sites:
        if only in (None, "wind"):
            designs.extend(Design(bus, wind_mw=float(p)) for p in cfg["designs"]["wind_mw"])
        for policy in cfg["designs"]["maintenance_policies"]:
            if only in (None, "hybrid"):
                designs.extend(Design(bus, wind_mw=float(w), battery_mw=float(p), battery_mwh=float(e), maintenance=policy)
                               for w, p, e in cfg["designs"]["hybrid_wind_bess_mw_mwh"])
    if only in (None, "bess"):
        specs = cfg["designs"]["bess_portfolios"] or [
            {"sites": [bus], "allocations": [1.0], "label": f"single:{bus}"} for bus in sites]
        selected = set(sites)
        for spec in specs:
            placement_sites = [str(site) for site in spec["sites"]]
            if not set(placement_sites).issubset(selected):
                continue
            placements = tuple(zip(placement_sites, map(float, spec["allocations"])))
            for policy in cfg["designs"]["maintenance_policies"]:
                designs.extend(Design(placement_sites[0], battery_mw=float(p), battery_mwh=float(e),
                                      maintenance=policy, battery_placements=placements)
                               for p, e in cfg["designs"]["bess_mw_mwh"])
    return list(dict.fromkeys(designs))

def smoke_designs_for(sites, cfg, only=None):
    """Choose a small but representative smoke set without invalidating portfolios."""
    if not cfg["designs"]["bess_portfolios"]:
        smoke_sites = sites[:1]
        return smoke_sites, designs_for(smoke_sites, cfg, only)
    candidates = designs_for(sites, cfg, only)
    chosen = []
    for technology in ("wind", "hybrid"):
        group = [d for d in candidates if d.technology == technology]
        if group:
            chosen.append(group[0])
    bess = [d for d in candidates if d.technology == "bess"]
    if bess:
        chosen.append(max(bess, key=lambda d: d.site_count))
    used_sites = []
    for design in chosen:
        for site, _share in design.placements:
            if site not in used_sites:
                used_sites.append(site)
    return used_sites, chosen


def first_year_metrics(result, baseline, design, factor, relaxed=None, relaxed_base=None):
    metrics = result.metrics
    available = metrics["project_wind_available_mwh"]
    delivered = float(result.project_wind_mw @ result.duration_h)
    reduction = baseline.metrics["renewable_dispatch_down_mwh"] - metrics["renewable_dispatch_down_mwh"]
    keys = ["component", "branch"]
    old = baseline.branches.set_index(keys).binding_hours if len(baseline.branches) else pd.Series(dtype=float)
    new = result.branches.set_index(keys).binding_hours if len(result.branches) else pd.Series(dtype=float)
    introduced = new.index[(new > 1e-8) & (old.reindex(new.index).fillna(0) <= 1e-8)]
    record = {
        "block_hours": result.hours, "annualisation_factor": factor,
        "year_1_project_wind_acceptance_ratio": delivered / available if available > 1e-9 else np.nan,
        "year_1_added_wind_available_mwh": available * factor,
        "year_1_total_dispatch_down_change_mwh": -reduction * factor,
        "year_1_net_renewable_gain_proxy_mwh": (metrics["renewable_dispatched_mwh"] - baseline.metrics["renewable_dispatched_mwh"] - metrics["battery_loss_mwh"]) * factor,
        "block_binding_branch_hours": metrics["binding_branch_hours"],
        "block_unserved_mwh": metrics["unserved_mwh"],
        "new_binding_branches": ";".join(": ".join(map(str, b)) for b in introduced),
        "year_1_passive_constraint_dd_mwh": np.nan,
        "year_1_passive_constraint_relief_mwh": np.nan,
    }
    if relaxed is not None:
        this_split = metrics["renewable_dispatch_down_mwh"] - relaxed.metrics["renewable_dispatch_down_mwh"]
        base_split = baseline.metrics["renewable_dispatch_down_mwh"] - relaxed_base.metrics["renewable_dispatch_down_mwh"]
        record["year_1_passive_constraint_dd_mwh"] = this_split * factor
        record["year_1_passive_constraint_relief_mwh"] = (base_split - this_split) * factor
    return record


def parse_args():
    p = argparse.ArgumentParser(description="TPSA 3.1 Q4 techno-economic design screening")
    p.add_argument("--config", type=Path)
    p.add_argument("--scenario")
    p.add_argument("--scope")
    p.add_argument("--network", type=Path, help="Use a validated exported PyPSA .nc; no forced node count")
    p.add_argument("--kit-dir", type=Path)
    p.add_argument("--buses", help="Comma-separated EXACT bus IDs; quote names containing spaces")
    p.add_argument("--limit-sites", type=int)
    p.add_argument("--only", choices=["wind", "bess", "hybrid"])
    p.add_argument("--years", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--prices", type=Path, help="CSV: snapshot,price_eur_mwh; exact timestamp alignment")
    p.add_argument("--dlr-multipliers", type=Path, help="Q3 CSV: snapshot plus exact line-ID multiplier columns")
    p.add_argument("--dlr-label", help="Audit label for the DLR series, e.g. q3_selected_wind_proxy_40pct")
    p.add_argument("--allow-illustrative-economics", action="store_true")
    p.add_argument("--demo", action="store_true", help="Synthetic TWO-BUS demo, not the Irish network")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Preflight only; default")
    mode.add_argument("--smoke", action="store_true", help="1 site, 24 hours, 3 project years, one size per type")
    mode.add_argument("--run", action="store_true", help="Run the requested design/lifetime grid")
    return p.parse_args()


def main(root: Path) -> int:
    args = parse_args()
    path = args.config or root / "configs" / "problem_3_1" / "question_4.json"
    config = load_config(path if path.exists() else None)
    if args.config is not None and not args.config.exists(): raise FileNotFoundError(args.config)
    if args.scenario: config["grid"]["scenario"] = args.scenario
    if args.scope: config["grid"]["scope"] = args.scope
    if args.prices:
        config["prices"].update(mode="csv", csv=str(args.prices.resolve()))
    if args.dlr_multipliers:
        config["grid"]["dlr_multiplier_csv"] = str(args.dlr_multipliers.resolve())
    if args.dlr_label:
        config["grid"]["dlr_label"] = args.dlr_label
    if args.years is not None: config["years"] = args.years
    if args.smoke:
        config["years"] = 3
        for k in ("wind_mw", "bess_mw_mwh", "hybrid_wind_bess_mw_mwh", "maintenance_policies"):
            config["designs"][k] = config["designs"][k][:1]
    validate_config(config)
    if args.demo:
        from .demo import DemoBackend
        backend = DemoBackend(config, 24 if args.smoke else None)
    else:
        backend = PyPSABackend(root, config, args.network, args.kit_dir, 24 if args.smoke else None)
    sites = backend.sites.loc[backend.sites.eligible, "bus"].astype(str).tolist()
    configured_portfolio_sites = {str(site) for spec in config["designs"]["bess_portfolios"] for site in spec["sites"]}
    missing_portfolio_sites = configured_portfolio_sites - set(sites)
    if missing_portfolio_sites:
        raise ValueError(f"Configured BESS portfolio buses are not eligible exact IDs: {sorted(missing_portfolio_sites)}")
    if args.buses:
        selected = [s.strip() for s in args.buses.split(",") if s.strip()]
        invalid = set(selected) - set(sites)
        if invalid: raise ValueError(f"Not eligible exact bus IDs: {sorted(invalid)}. Run --check for the list")
        sites = list(dict.fromkeys(selected))
    limit = args.limit_sites
    if limit is not None:
        if limit < 1: raise ValueError("limit-sites must be positive")
        sites = sites[:limit]
    if not sites: raise ValueError("No selected sites")
    if args.smoke:
        sites, designs = smoke_designs_for(sites, config, args.only)
    else:
        designs = designs_for(sites, config, args.only)
    if not designs: raise ValueError("No designs selected")
    factor = annual_factor(config, float(backend.duration_h.sum()))
    print("=== Q4 INPUT CHECK ===", flush=True)
    print(f"Version {__version__} | Backend: {backend.metadata['backend']}")
    print(f"Nodes: {len(backend.sites)} | snapshots: {len(backend.snapshots)} | selected sites: {len(sites)}")
    print(f"Sites: {', '.join(sites)}")
    print(f"Designs: {len(designs)} | project horizon: {config['years']} years | annual block multiplier: {factor:.6g}")
    print(f"Maximum approximate solve count: {len(designs) * (config['years'] + int(config['grid']['counterfactual_first_year'])) + 2}")
    print("No topology certification. No SEM merchant-bidding/ancillary-revenue model.")
    print(f"Assumptions: {config['assumptions_status']}", flush=True)
    print(backend.sites.loc[sites, ["bus", "display_name", "eligible"]].to_string(index=False), flush=True)
    if not args.run and not args.smoke:
        print("CHECK ONLY. Nothing solved. Use --smoke first; --run starts the chosen grid.")
        return 0
    illustrative = (config["annualisation"]["mode"] == "repeat_block" or config["prices"]["mode"] == "flat"
                    or config["assumptions_status"].startswith("illustrative"))
    if illustrative and not args.allow_illustrative_economics and not args.demo:
        raise ValueError("Illustrative input/annualisation detected. Add --allow-illustrative-economics; results will remain explicitly labelled scenarios")
    if config["prices"]["mode"] == "csv":
        price_path = Path(config["prices"]["csv"])
        if not price_path.is_absolute(): price_path = root / price_path
        prices = aligned_csv(price_path, backend.snapshots, ["price_eur_mwh"])["price_eur_mwh"].to_numpy()
    else:
        prices = np.full(len(backend.snapshots), config["prices"]["flat_eur_mwh"], float)
    tag = ("question_4_demo" if args.demo else "question_4_smoke" if args.smoke
           else "question_4_dlr" if config["grid"]["dlr_multiplier_csv"] else "question_4")
    out = (args.out or root / "results" / "problem_3_1" / tag).resolve()
    out.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        h.update(source.name.encode() + source.read_bytes())
    inputs = {"config": config, "network": backend.metadata, "designs": [asdict(d) for d in designs],
              "code_sha256": h.hexdigest(), "prices_sha256": hashlib.sha256(prices.tobytes()).hexdigest(),
              "smoke": args.smoke, "version": __version__, "python": platform.python_version()}
    # Include custom wind resource data: backend network hash alone is insufficient.
    profile_csv = config["grid"]["wind_profile_csv"]
    if profile_csv:
        f = Path(profile_csv)
        if not f.is_absolute(): f = root / f
        inputs["wind_profile_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(_json_clean(inputs), sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError(f"{out} contains a different run. Choose a NEW --out path; old results will not be overwritten")
    elif any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no Q4 manifest. Use a new directory")
    write_json(manifest_path, {"signature": signature, **inputs})
    backend.sites.to_csv(out / "00_bus_inventory.csv", index=False)
    pd.DataFrame({"snapshot": backend.snapshots, "price_eur_mwh": prices}).to_csv(out / "00_prices_used.csv", index=False)
    no_build = Design(sites[0])
    print("[baseline] Solving original fixed network...", flush=True)
    baseline = backend.solve(no_build, 0.0)
    if baseline.metrics["unserved_mwh"] > config["grid"]["max_unserved_mwh"]:
        raise RuntimeError("Baseline has unserved energy. Finish the network/boundary audit before economics")
    baseline.branches.to_csv(out / "00_baseline_branches.csv", index=False)
    write_json(out / "00_baseline_metrics.json", baseline.metrics)
    relaxed_base = backend.solve(no_build, 0.0, relax_passive_limits=True) if config["grid"]["counterfactual_first_year"] else None
    summaries, failures = [], []
    for i, design in enumerate(designs, 1):
        case_dir = out / "cases" / design.id
        done = case_dir / "completed.json"
        if done.exists():
            summaries.append(json.loads(done.read_text(encoding="utf-8")))
            print(f"[{i}/{len(designs)}] RESUME {design.id}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(designs)}] {design.id} @ {design.bus}: wind {design.wind_mw:g} MW; battery {design.battery_mw:g}/{design.battery_mwh:g}; {design.maintenance}", flush=True)
        try:
            def progress(year, result):
                print(f"  year {year:02d}/{config['years']} | DD(block) {result.metrics['renewable_dispatch_down_mwh']:.2f} MWh", flush=True)
            cash, summary, first = simulate_lifetime(design, backend, config, baseline, prices, progress)
            relaxed = backend.solve(design, design.battery_mwh, relax_passive_limits=True) if relaxed_base is not None else None
            summary.update(first_year_metrics(first, baseline, design, factor, relaxed, relaxed_base))
            summary.update(asdict(design), case_id=design.id, technology=design.technology,
                           site_count=design.site_count, sites=design.sites_label,
                           allocations=design.allocations_label, dlr_label=backend.metadata.get("dlr_label", "none"),
                           dlr_mean_multiplier=backend.metadata.get("dlr_mean_multiplier", 1.0),
                           dlr_max_multiplier=backend.metadata.get("dlr_max_multiplier", 1.0),
                           years=config["years"], assumptions_status=config["assumptions_status"],
                           result_scope="SMOKE_ONLY" if args.smoke else "conditional_design_screen")
            cash.to_csv(case_dir / "annual_cashflows.csv", index=False)
            first.hourly_frame().to_csv(case_dir / "year_1_hourly_dispatch.csv", index=False)
            first.branches.to_csv(case_dir / "year_1_all_branches.csv", index=False)
            if config["sensitivity"]["enabled"]:
                financial_sensitivity(cash, config).to_csv(case_dir / "financial_sensitivity.csv", index=False)
            write_json(case_dir / "design.json", asdict(design))
            write_json(done, summary)  # written last: only complete cases can be resumed
            summaries.append(summary)
        except Exception as exc:
            failure = {"case_id": design.id, "error": repr(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            write_json(case_dir / "failure.json", failure)
            print(f"  FAILED: {exc}", flush=True)
        write_json(out / "failures.json", failures)
        if summaries:
            pd.DataFrame(summaries).to_csv(out / "checkpoint_ranking.csv", index=False)
    frame = pd.DataFrame(summaries)
    write_json(out / "failures.json", failures)
    write_report(out, frame, config, backend.metadata, failures, len(designs), args.smoke)
    print(f"=== Q4 FINISHED: {len(summaries)}/{len(designs)} completed, {len(failures)} failed ===")
    print(f"Output: {out}")
    print("Read SUMMARY.md and 02_technical_vs_economic.csv before quoting any number.")
    return 0 if not failures else 2
