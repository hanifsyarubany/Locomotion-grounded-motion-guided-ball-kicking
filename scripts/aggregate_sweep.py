#!/usr/bin/env python3
"""Aggregate checkpoint-sweep results into per-policy medians for the paper.

Reads every out/sim2sim_eval/*-sweep-*/sim2sim_eval.json produced by
scripts/run_checkpoint_sweep.sh and reports, per skill, the MEDIAN across the 8
swept checkpoints plus the min/max spread, alongside the single checkpoint that
was previously reported. Large spread = the earlier single-checkpoint number was
a lottery draw; small spread = it was representative.
"""
import glob, json, os, statistics as st

# paper naming: video id -> skill_N
NAME = {"012": 1, "016": 2, "017": 3, "018": 4, "019": 5, "020": 6, "011": 7}
PREV = {  # previously reported checkpoint per policy
    "s012": 393000, "s016": 366000, "s017": 396000, "s018": 387000,
    "s019": 336000, "s020": 366000, "s011": 275000,
    "u7": 440000, "u6": 474000,
    # 2026-09-12: the re-anchored re-run sweeps the SAME 16 checkpoints, so the same
    # previously-reported checkpoint is the right anchor for its prev-ckpt column.
    # Key is the prefix after the replace() below, i.e. "sweep-unified-7-reanchor" -> "u7-reanchor".
    "u7-reanchor": 440000,
}

# Checkpoints the operator chose by hand, per config. Everything else in that
# config's sweep was added blind (neighbours / evenly-spaced bridge points).
# Comparing the two medians is the selection-bias check: if the picks median
# sits well above the blind median, the picks were performance-informed and the
# picks-inclusive median is optimistic.
OPERATOR_PICKS = {
    "sweep-unified-7": {389000, 390000, 430000, 436000, 438000, 441000, 442000, 444000},
    # Same checkpoint list, same picks -- the re-anchored re-run differs from sweep-unified-7 in
    # exactly one thing (reanchor_ball_at_trigger), so the selection-bias check applies unchanged.
    "sweep-unified-7-reanchor": {389000, 390000, 430000, 436000, 438000, 441000, 442000, 444000},
}

METRICS = ["success_rate_0.5", "success_rate_1", "hit_rate", "fall_rate", "shot_error_mean"]


def load_runs():
    """Newest run dir per config name -> {target_name: {skill_idx: {metric: mean}}}"""
    runs = {}
    for p in sorted(glob.glob("out/sim2sim_eval/*/sim2sim_eval.json")):
        d = json.load(open(p))
        cfg = os.path.basename(d.get("config_path", "") or "")
        if not cfg.startswith("sweep-"):
            continue
        runs[cfg.replace(".yaml", "")] = d  # later timestamps overwrite earlier
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No sweep results found under out/sim2sim_eval/. Run scripts/run_checkpoint_sweep.sh first.")
        return

    for cfg in sorted(runs):
        d = runs[cfg]
        res = d["results"]
        prefix = cfg.replace("sweep-skill-", "s").replace("sweep-unified-", "u")
        prev = PREV.get(prefix)
        print(f"\n{'='*78}\n{cfg}   ({len(res)} checkpoints, {d['iterations']} iters x 10 trials each)\n{'='*78}")

        skills = sorted({s for t in res.values() for s in t})
        for sk in skills:
            label = f"skill idx {sk}"
            if cfg.startswith("sweep-skill-"):
                vid = cfg.split("-")[-1]
                label = f"skill_{NAME[vid]} (video_{vid})"
            print(f"\n  {label}")
            print(f"    {'metric':<20} {'median':>9} {'min':>9} {'max':>9} {'spread':>9} {'prev-ckpt':>10}")
            for m in METRICS:
                vals, prev_val = [], None
                for tname, tdata in res.items():
                    if sk not in tdata:
                        continue
                    # .get("kick", {}): a sweep whose evaluations include non-kick types (e.g.
                    # sweep-ablation-a1b1's locomotion metrics) has skill entries with no "kick"
                    # group at all. Indexing it raised KeyError and killed the whole aggregation
                    # part-way through, silently hiding every config sorted after it.
                    v = tdata[sk].get("kick", {}).get(m, {}).get("mean")
                    if v is None:
                        continue
                    scale = 100 if m.endswith(("_0.5", "_1", "_rate")) else 1
                    v *= scale
                    vals.append(v)
                    if prev is not None and tname.endswith(f"_{prev}"):
                        prev_val = v
                if not vals:
                    continue
                med, lo, hi = st.median(vals), min(vals), max(vals)
                pv = f"{prev_val:9.2f}" if prev_val is not None else "        -"
                print(f"    {m:<20} {med:9.2f} {lo:9.2f} {hi:9.2f} {hi-lo:9.2f} {pv}")

            picks = OPERATOR_PICKS.get(cfg)
            if picks:
                print(f"    {'-'*66}")
                print(f"    selection-bias check   {'picks':>10} {'blind':>10} {'delta':>10}")
                for m in ("success_rate_0.5", "success_rate_1"):
                    pv_, bv_ = [], []
                    for tname, tdata in res.items():
                        if sk not in tdata:
                            continue
                        v = tdata[sk].get("kick", {}).get(m, {}).get("mean")  # see note above
                        if v is None:
                            continue
                        step = int(tname.rsplit("_", 1)[-1])
                        (pv_ if step in picks else bv_).append(v * 100)
                    if pv_ and bv_:
                        a, b = st.median(pv_), st.median(bv_)
                        print(f"    {m:<20} {a:10.2f} {b:10.2f} {a-b:+10.2f}")


if __name__ == "__main__":
    main()
