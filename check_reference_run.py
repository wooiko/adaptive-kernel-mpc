"""TZ 10, D16: reference run on a second machine, compared with the result JSON of this repository.

The rule is fixed before the run (TZ 14, D16):
  exact   - the same code fingerprint and seed, no cached runs, the same scenarios, configurations and segments,
            and the same constraint violations and ignition verdict in every run;
  bounded - in the scenarios inside the admissible set (S1-S4, S5a, S6), for iae_total, tc_total_variation, max_T
            and the steady-state error of every segment: |candidate - reference| <= FACTOR x the envelope of that
            scenario and metric. The envelope is the largest |difference| over all configurations of two runs of the
            same code on the same cloud machine that differ only in the OpenBLAS kernels (OPENBLAS_CORETYPE): the
            start-up data then differ at the 1e-12 mol/L level, and the closed loop amplifies it scenario by scenario.
            FACTOR = 10 is an a priori choice, not derived;
  printed - S5b (outside the set, no threshold, as D11 and D15), fallback steps, solver statuses, filter counts,
            shares and every other field: differences are printed without a threshold.
Time fields (step_ms, ident_ms, full_ms, wall_s, wall_s_total, machine) are not compared.

Standard library only, so it runs on any machine:
    python check_reference_run.py envelope REF PROBE [PROBE ...]   # writes outputs/reference_envelope.json
    python check_reference_run.py check REF CANDIDATE [--note ..]  # writes outputs/reference_run_check.json
"""
import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
IN_SET = ("S1", "S2", "S3", "S4", "S5a", "S6")      # as report_mpc.IN_SET (TZ 9)
BOUNDED = ("iae_total", "tc_total_variation", "max_T")
SEGMENT_BOUNDED = ("ss_error",)
EXACT_RUN = ("violations", "ignition")
TIME_KEYS = {"step_ms", "ident_ms", "full_ms", "wall_s", "wall_s_total", "machine"}
FACTOR = 10.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bounded_values(m):
    """{metric: [values]} of one run for the bounded metrics (segment metrics as lists, None kept)."""
    out = {k: [m[k]] for k in BOUNDED}
    for k in SEGMENT_BOUNDED:
        out[k] = [s[k] for s in m["segments"]]
    return out


def max_abs_diff(a, b):
    """Largest |a - b| over paired values; None must pair with None (otherwise inf)."""
    worst = 0.0
    for x, y in zip(a, b):
        if x is None or y is None:
            if (x is None) != (y is None):
                return float("inf")
            continue
        worst = max(worst, abs(float(x) - float(y)))
    return worst


def envelope(ref, probes):
    """Per in-set scenario and bounded metric: the largest |probe - reference| over all configurations and probes."""
    env = {}
    for sc in IN_SET:
        env[sc] = {}
        for k in BOUNDED + SEGMENT_BOUNDED:
            env[sc][k] = max(max_abs_diff(bounded_values(ref["results"][sc][c])[k],
                                          bounded_values(p["results"][sc][c])[k])
                             for p in probes for c in ref["results"][sc])
    return env


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in TIME_KEYS:
                yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def printed_diffs(ref, cand):
    """Every other differing field, grouped by scenario and field name (path without configuration and index)."""
    a, b = dict(leaves(ref["results"])), dict(leaves(cand["results"]))
    out = {}
    for p in sorted(set(a) | set(b), key=str):
        x, y = a.get(p), b.get(p)
        if x == y:
            continue
        key = p[0] + " / " + "/".join(str(s) for s in p[2:] if not isinstance(s, int))
        rec = out.setdefault(key, {"n": 0, "max_abs": 0.0, "example": None})
        rec["n"] += 1
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        d = abs(x - y) if num else float("inf")
        if rec["example"] is None or d > rec["max_abs"]:
            rec["max_abs"], rec["example"] = d, {"config": p[1], "reference": x, "candidate": y}
    return out


def check(ref, cand, env, factor=FACTOR):
    exact = []

    def need(name, ok, detail):
        exact.append({"check": name, "ok": bool(ok), "detail": detail})

    need("code fingerprint", ref["provenance"]["code_sha256_16"] == cand["provenance"]["code_sha256_16"],
         [ref["provenance"]["code_sha256_16"], cand["provenance"]["code_sha256_16"]])
    need("seed", ref["seed"] == cand["seed"], [ref["seed"], cand["seed"]])
    need("no cached runs", cand["provenance"]["runs_from_cache"] == 0, cand["provenance"]["runs_from_cache"])
    runs_r = {(sc, c) for sc, v in ref["results"].items() for c in v}
    runs_c = {(sc, c) for sc, v in cand["results"].items() for c in v}
    need("same scenarios and configurations", runs_r == runs_c,
         {"only_reference": sorted(map(list, runs_r - runs_c)), "only_candidate": sorted(map(list, runs_c - runs_r))})
    common = sorted(runs_r & runs_c)
    seg = [list(r) for r in common if len(ref["results"][r[0]][r[1]]["segments"])
           != len(cand["results"][r[0]][r[1]]["segments"])]
    need("same segments", not seg, seg)
    for k in EXACT_RUN:
        bad = [[sc, c, ref["results"][sc][c][k], cand["results"][sc][c][k]] for sc, c in common
               if ref["results"][sc][c][k] != cand["results"][sc][c][k]]
        need(f"same {k} in every run", not bad, bad)

    bounded = []
    for sc in IN_SET:
        for k in BOUNDED + SEGMENT_BOUNDED:
            cfgs = [c for s, c in common if s == sc]
            d, worst = 0.0, None
            for c in cfgs:
                dc = max_abs_diff(bounded_values(ref["results"][sc][c])[k], bounded_values(cand["results"][sc][c])[k])
                if worst is None or dc > d:
                    d, worst = dc, c
            tol = factor * env[sc][k]
            bounded.append({"scenario": sc, "metric": k, "max_abs_diff": d, "config": worst,
                            "envelope": env[sc][k], "tolerance": tol, "ok": d <= tol})

    ok = all(e["ok"] for e in exact) and all(b["ok"] for b in bounded)
    return {"rule": "TZ 14, D16", "factor": factor, "passed": ok, "exact": exact, "bounded": bounded,
            "printed": printed_diffs(ref, cand),
            "reference": {"machine": ref.get("machine"), "startup_block_sha256_16": ref["startup_block_sha256_16"]},
            "candidate": {"machine": cand.get("machine"), "provenance": cand["provenance"],
                          "startup_block_sha256_16": cand["startup_block_sha256_16"],
                          "tc_max_diff": abs(ref["tc_max"]["tc_max"] - cand["tc_max"]["tc_max"])}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("envelope")
    e.add_argument("ref")
    e.add_argument("probes", nargs="+")
    e.add_argument("--note", default="")
    c = sub.add_parser("check")
    c.add_argument("ref")
    c.add_argument("candidate")
    c.add_argument("--envelope", default=str(OUT / "reference_envelope.json"))
    c.add_argument("--out", default=str(OUT / "reference_run_check.json"))
    c.add_argument("--note", default="", help="e.g. library versions of the candidate run as reported")
    a = ap.parse_args(argv)
    if a.cmd == "envelope":
        ref, probes = load(a.ref), [load(p) for p in a.probes]
        res = {"rule": "TZ 14, D16", "note": a.note, "reference_startup_block": ref["startup_block_sha256_16"],
               "probes": [{"file": Path(p).name, "startup_block_sha256_16": q["startup_block_sha256_16"],
                           "code_sha256_16": q["provenance"]["code_sha256_16"]} for p, q in zip(a.probes, probes)],
               "envelope": envelope(ref, probes)}
        (OUT / "reference_envelope.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res["envelope"], indent=2))
        return 0
    res = check(load(a.ref), load(a.candidate), load(a.envelope)["envelope"])
    res["note"] = a.note
    Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    for x in res["exact"]:
        print(("ok  " if x["ok"] else "FAIL"), x["check"])
    for b in res["bounded"]:
        print(("ok  " if b["ok"] else "FAIL"),
              f"{b['scenario']:4s} {b['metric']:20s} diff {b['max_abs_diff']:.3e}  tol {b['tolerance']:.3e}  ({b['config']})")
    print("printed only:", len(res["printed"]), "fields differ")
    print("PASSED" if res["passed"] else "NOT PASSED")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
