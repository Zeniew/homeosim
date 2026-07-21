"""
Fast repro of the GOGO_PLAST=1 500Hz-runaway, cheap enough to run in minutes
instead of ~8 hours.

Why the real run takes 8 hours: numGR = 1,048,576. That population (and its
MFGR / GOGR connectivity, which is sized off of it) dominates runtime, but
it has nothing to do with the GOGO-specific hypothesis we're testing. So:

  - numGO, numMF, and the GOGO_connect_arr are REAL and UNCHANGED (this is
    the thing under test).
  - numBins, numTrial, CSon/CSoff, useCS, target_spikes, plast_mult_constant
    are REAL and UNCHANGED (these control the timing of the divergence, so
    changing them would change when/whether it shows up).
  - numGR is shrunk (default 50,000, ~21x smaller) and MFGR/GOGR
    connectivity is regenerated synthetically at matching density so GR
    cells still receive roughly the same per-cell input rate as in
    production. GRGO out-degree (30, targets among the unchanged 4096 GO
    cells) is left as-is since it doesn't depend on numGR.

This is an approximation on the GR side, so treat any Hz numbers as
qualitative, not a quantitative match to your real run. What matters is
whether the SAME divergence pattern (low in-degree GOGO cells saturating
gogoW, then a synchronized firing-rate blowup) shows up here too, and on
the same relative timescale (trial ~700-900 for the real 75% file).

Usage:
    python fast_repro_test.py --recip 75
    python fast_repro_test.py --recip 0
    python fast_repro_test.py --recip 100
    python fast_repro_test.py --recip 75 --numTrial 300   # even faster smoke test
"""
import argparse
import os
import time
import numpy as np
import cupy as cp

import MFGOGrFunctions_synaptic_scaling as mfgogr
import importConnect as connect

GOGO_FILES = {
    "0":   "/home/data/einez/connect_arr/R00_C12_PRE.gogo",
    "75":  "/home/data/einez/connect_arr/R75_C12_PRE.gogo",
    "100": "/home/data/einez/connect_arr/connect_arr_PRE.gogo",
}

REAL_NUMGR = 1_048_576
REAL_MFGR_OUTDEGREE = 1289   # out of numGR, per MF cell
REAL_GOGR_OUTDEGREE = 975    # out of numGR, per GO cell
REAL_GRGO_OUTDEGREE = 30     # out of numGO, per GR cell -- unaffected by numGR shrink


def make_synthetic_arr(num_rows, num_pool, out_degree, seed):
    """Dense (no -1 padding needed) random connectivity: each row picks
    `out_degree` distinct targets uniformly from [0, num_pool)."""
    rng = np.random.default_rng(seed)
    out_degree = max(1, min(out_degree, num_pool))
    arr = np.empty((num_rows, out_degree), dtype=np.int32)
    for i in range(num_rows):
        arr[i] = rng.choice(num_pool, size=out_degree, replace=False)
    return arr


def compute_in_degree(connect_arr, num_targets):
    valid = (connect_arr >= 0) & (connect_arr < num_targets)
    return np.bincount(connect_arr[valid], minlength=num_targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recip", choices=["0", "75", "100"], required=True)
    ap.add_argument("--numGO", type=int, default=4096)
    ap.add_argument("--numMF", type=int, default=4096)
    ap.add_argument("--numGR", type=int, default=50_000,
                     help="Shrunk Granule population size (real=1,048,576)")
    ap.add_argument("--numBins", type=int, default=5000)
    ap.add_argument("--numTrial", type=int, default=1000)
    ap.add_argument("--CSon", type=int, default=500)
    ap.add_argument("--CSoff", type=int, default=3500)
    ap.add_argument("--useCS", type=int, default=0)
    ap.add_argument("--explode_hz", type=float, default=200.0,
                     help="Early-stop once mean GO firing rate crosses this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./fast_repro_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"recip{args.recip}_numGR{args.numGR}_seed{args.seed}"

    numGO, numMF, numGR = args.numGO, args.numMF, args.numGR
    numBins, numTrial = args.numBins, args.numTrial

    print(f"Config: {tag}, numBins={numBins}, numTrial={numTrial}")
    print("Loading REAL MFGO and GOGO connectivity (unchanged)...")

    MFGO_arr = connect.read_connect(
        "/home/data/einez/connect_arr/connect_arr_PRE.mfgo", numMF, 20
    )[:, :16]

    GOGO_arr = connect.read_connect(GOGO_FILES[args.recip], numGO, 12)

    print("Building SYNTHETIC (density-matched, shrunk) MFGR / GOGR / GRGO...")
    scale = numGR / REAL_NUMGR
    mfgr_deg = max(1, round(REAL_MFGR_OUTDEGREE * scale))
    gogr_deg = max(1, round(REAL_GOGR_OUTDEGREE * scale))
    print(f"  numGR scale factor={scale:.4f} -> MFGR out-degree={mfgr_deg}, "
          f"GOGR out-degree={gogr_deg}, GRGO out-degree={REAL_GRGO_OUTDEGREE} (unchanged)")

    MFGR_arr = make_synthetic_arr(numMF, numGR, mfgr_deg, seed=args.seed + 1)
    GOGR_arr = make_synthetic_arr(numGO, numGR, gogr_deg, seed=args.seed + 2)
    GRGO_arr = make_synthetic_arr(numGR, numGO, REAL_GRGO_OUTDEGREE, seed=args.seed + 3)

    # ---- precompute GOGO in-degree once: this is the independent variable
    # we're correlating against gogoW / firing rate divergence ----
    go_in_degree = compute_in_degree(GOGO_arr, numGO)
    np.save(os.path.join(args.outdir, f"{tag}_go_in_degree.npy"), go_in_degree)
    print(f"GOGO in-degree: mean={go_in_degree.mean():.2f} std={go_in_degree.std():.2f} "
          f"min={go_in_degree.min()} max={go_in_degree.max()} "
          f"isolated={(go_in_degree == 0).sum()}")

    print("Initializing objects...")
    MF = mfgogr.Mossy(numMF, args.CSon, args.CSoff)
    GO = mfgogr.Golgi(numGO, args.CSon, args.CSoff, args.useCS, numBins, plast_ratio=1.0)
    GR = mfgogr.Granule(numGR, args.CSon, args.CSoff, args.useCS, numBins)

    # ---- instrumentation ----
    gogoW_history = np.zeros((numTrial, numGO), dtype=np.float64)
    mean_hz_history = np.full(numTrial, np.nan)
    max_hz_history = np.full(numTrial, np.nan)
    trial_seconds = numBins / 1000.0  # matches how the real analysis presumably converts bins->Hz if 1ms bins

    saturated_trial = np.full(numGO, -1, dtype=np.int64)  # first trial gogoW crosses 0.99

    print("Running...")
    stop_trial = numTrial
    for trial in range(numTrial):
        t0 = time.time()
        for t in range(numBins):
            MFact = MF.do_MF_dist(t, args.useCS)
            GR.update_input_activity(MFGR_arr, 1, mfAct=MFact)
            GR.do_Granule(t)
            GRact = GR.get_act()
            GO.update_input_activity(GRGO_arr, 3, grAct=GRact[t])
            GO.update_input_activity(MFGO_arr, 1, mfAct=MFact)
            GO.do_Golgi(t)
            GO.update_input_activity(GOGO_arr, 2, t=t)
            GOact = GO.get_act()
            GR.update_input_activity(GOGR_arr, 2, goAct=GOact[t])

        # GOGO plasticity only -- matches the experiment being investigated
        GO.gogoW = GO.update_weight(trial, exc_or_inh=2, weight_array=GO.get_gogoW())

        gogoW_history[trial] = GO.get_gogoW()
        newly_sat = (GO.get_gogoW() >= 0.99) & (saturated_trial == -1)
        saturated_trial[newly_sat] = trial

        spike_counts = GO.get_act().sum(axis=0)  # spikes per GO cell this trial
        hz = spike_counts / trial_seconds
        mean_hz_history[trial] = hz.mean()
        max_hz_history[trial] = hz.max()

        GR.updateFinalState()
        GR.reset_GPU_summed_act()
        MF.generate_MFisiDistribution()

        dt = time.time() - t0
        print(f"trial {trial:4d}  mean_hz={hz.mean():7.2f}  max_hz={hz.max():7.2f}  "
              f"gogoW[mean/max]={GO.gogoW.mean():.4f}/{GO.gogoW.max():.4f}  "
              f"saturated_cells={(saturated_trial >= 0).sum():4d}  ({dt:.2f}s)")

        if hz.mean() >= args.explode_hz:
            print(f"\n*** Mean firing rate crossed {args.explode_hz} Hz at trial "
                  f"{trial}. Stopping early -- divergence confirmed, no need "
                  f"to run the rest. ***\n")
            stop_trial = trial + 1
            break

    # ---- save + quick correlation check ----
    np.save(os.path.join(args.outdir, f"{tag}_gogoW_history.npy"), gogoW_history[:stop_trial])
    np.save(os.path.join(args.outdir, f"{tag}_mean_hz.npy"), mean_hz_history[:stop_trial])
    np.save(os.path.join(args.outdir, f"{tag}_max_hz.npy"), max_hz_history[:stop_trial])
    np.save(os.path.join(args.outdir, f"{tag}_saturated_trial.npy"), saturated_trial)

    ever_saturated = saturated_trial >= 0
    print(f"\n{ever_saturated.sum()} / {numGO} cells saturated gogoW (>=0.99) by trial {stop_trial}")

    if ever_saturated.sum() > 5:
        corr = np.corrcoef(go_in_degree, GO.gogoW)[0, 1]
        print(f"Correlation(in-degree, final gogoW) = {corr:.3f}  "
              f"(expect strongly NEGATIVE if the hypothesis is right: "
              f"low in-degree -> high/saturated gogoW)")

        sat_only = saturated_trial[ever_saturated]
        deg_only = go_in_degree[ever_saturated]
        corr2 = np.corrcoef(deg_only, sat_only)[0, 1]
        print(f"Correlation(in-degree, trial-of-saturation) = {corr2:.3f}  "
              f"(expect POSITIVE: lower in-degree cells saturate earlier)")
    else:
        print("Not enough saturated cells yet for a meaningful correlation -- "
              "consider a longer --numTrial or lowering --explode_hz.")

    print(f"\nAll arrays saved to {args.outdir}/{tag}_*.npy")


if __name__ == "__main__":
    main()