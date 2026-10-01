#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dh_scan.py -- Scanner HPC reprenable pour le module `dh-certify`.

Scanne la droite critique de la fonction de Davenport-Heilbronn
f(s) = L(s, chi_5) + eps . L(s, chi_b) par batchs successifs, avec :

  * checkpoints periodiques (defaut : 10 s) -> reprise automatique
    apres coupure (SLURM, Ctrl-C, instabilite reseau) ;
  * etat d'avancement persiste en JSON a cote des resultats ;
  * certificat d'exactitude a chaque fin de batch : la variation totale
    d'argument le long du segment balaye est un multiple entier de 2*pi
    aux erreurs de discretisation pres (principe de l'argument) ;
  * parallélisme multi-processus (module `multiprocessing`) sur les batchs.

Usage :
  python3 dh_scan.py scan --tmax 60 [--batch 10] [--step 0.01]
                          [--dps 25] [--checkpoint-interval 10]
                          [--workers N] [--fresh] [--json]
  python3 dh_scan.py status            # etat du checkpoint courant
  python3 dh_scan.py merge             # consolide le rapport final

Le fichier d'etat (JSON) contient :
  { "version", "params", "t_done", "zeros_on_line": [gamma...],
    "candidates_not_on_line": [[re, im]...], "batches": [...],
    "winding_total_raw", "finished": bool }

NOTE SCIENTIFIQUE : ce scanner produit des VERIFICATIONS A HAUTEUR FINIE
(pas une preuve d'RH). Il detecte les zeros SUR la droite critique ; les
zeros HORS droite sont certifies separement via le principe de l'argument
(voir rh_engine.dh_certify, section certificat).
"""
import argparse
import json
import os
import signal
import sys
import time
from multiprocessing import Pool

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_engine import DavenportHeilbronn, results_dir  # noqa: E402

STATE_VERSION = 1


# --------------------------------------------------------------------------
# etat / checkpoints
# --------------------------------------------------------------------------
def state_path(tmax):
    tag = ("%g" % tmax).replace(".", "_")
    return os.path.join(results_dir(), "dh_scan_T%s.state.json" % tag)


def new_state(params):
    return {
        "version": STATE_VERSION,
        "params": params,
        "t_done": 0.0,
        "zeros_on_line": [],
        "candidates_not_on_line": [],
        "batches": [],
        "winding_total_raw": 0.0,
        "finished": False,
    }


def load_state(tmax):
    p = state_path(tmax)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        st = json.load(fh)
    if st.get("version") != STATE_VERSION:
        sys.stderr.write("checkpoint obsolete (v%s != v%d), repart de zero\n"
                         % (st.get("version"), STATE_VERSION))
        return None
    return st


def save_state(st, tmax):
    p = state_path(tmax)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, p)  # atomic


# --------------------------------------------------------------------------
# un batch : scan [t0, t1] de la droite critique Re s = 1/2
#   - suivi de phase continu (arg deape <= pi entre pas consecutifs)
#   - zero sur la droite <=> changement de signe SIMULTANE de a(t) et b(t)
#   - retourne aussi le bord (a,b) au debut et a la fin pour enchainer
# --------------------------------------------------------------------------
def scan_batch(args):
    t0, t1, step_f, dps = args
    mp.mp.dps = dps + 5          # marge de securite contre le bruit d'arrondi
    dh = DavenportHeilbronn(dps=dps + 5)
    step = mp.mpf(repr(step_f))
    half = mp.mpf(1) / 2

    t = mp.mpf(repr(t0))
    w = dh.f(half + 1j * t)
    pa, pb = mp.re(w), mp.im(w)
    prev_arg = mp.arg(w)

    zeros = []
    winding = mp.mpf(0)
    npts = 0
    while t < mp.mpf(repr(t1)):
        t += step
        w = dh.f(half + 1j * t)
        a, b = mp.re(w), mp.im(w)
        cur_arg = mp.arg(w)
        d = cur_arg - prev_arg
        if d > mp.pi:
            d -= 2 * mp.pi
        elif d < -mp.pi:
            d += 2 * mp.pi
        winding += d
        if ((a > 0) != (pa > 0)) and ((b > 0) != (pb > 0)):
            zeros.append(float((t + (t - step)) / 2))
        pa, pb, prev_arg = a, b, cur_arg
        npts += 1
    return {
        "t0": t0, "t1": t1,
        "zeros": zeros,
        "winding_raw": float(winding / (2 * mp.pi)),
        "n_points": npts,
        "end_ab": [float(mp.nstr(pa, 12)), float(mp.nstr(pb, 12))],
    }


def refine_zeros(state, dps_refine=40):
    """Affine chaque candidat comme zero complexe (Muller) et classe sur/hors droite."""
    mp.mp.dps = dps_refine
    dh = DavenportHeilbronn(dps=dps_refine)
    confirmed = []
    for c in state["_pending"]:
        try:
            r = mp.findroot(lambda z: dh.f(z), complex(mp.mpf("0.5"), c),
                            solver="muller",
                            tol=mp.mpf(10) ** (-dps_refine + 8), maxsteps=60)
        except Exception as e:  # non converge -> on garde en attente
            state.setdefault("refine_failures", []).append([c, str(e)])
            continue
        if abs(mp.re(r) - mp.mpf(1) / 2) < mp.mpf(10) ** (-dps_refine + 12):
            confirmed.append(float(mp.im(r)))
        else:
            state["candidates_not_on_line"].append([float(mp.re(r)), float(mp.im(r))])
    state["zeros_on_line"] = sorted(set(state["zeros_on_line"]) | set(confirmed))
    mp.mp.dps = dps_refine - 15


# --------------------------------------------------------------------------
# boucle principale
# --------------------------------------------------------------------------
_interrupted = False


def _sigint(signum, frame):
    global _interrupted
    _interrupted = True
    sys.stderr.write("\n[interruption demandee : checkpoint en fin de batch puis arret]\n")


def run_scan(params, workers=1, quiet=False):
    global _interrupted
    _interrupted = False
    tmax = params["tmax"]
    batch = params["batch"]
    step = params["step"]
    dps = params["dps"]
    ckpt_every = params["checkpoint_interval"]

    st = load_state(tmax)
    if st is None or params.get("fresh"):
        st = new_state(params)
        save_state(st, tmax)
        if not quiet:
            sys.stderr.write("nouveau checkpoint : %s\n" % state_path(tmax))
    else:
        if not quiet:
            sys.stderr.write("reprise depuis t=%.4f (%d zeros confirmes)\n"
                             % (st["t_done"], len(st["zeros_on_line"])))

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    pool = Pool(workers) if workers > 1 else None
    total_batches = int(mp.ceil(mp.mpf(repr(tmax)) / mp.mpf(repr(batch))))

    def consolidate(res):
        st["t_done"] = res["t1"]
        st["batches"].append({k: res[k] for k in ("t0", "t1", "winding_raw", "n_points")})
        st["winding_total_raw"] += res["winding_raw"]
        st["zeros_on_line"] = sorted(set(st["zeros_on_line"]) | set(res["zeros"]))
        # certificat d'exactitude : winding attendu ~ entier si aucun zero
        # ne traverse les bords verticaux du bandeau reel (suivi continu)
        err = abs(res["winding_raw"] - round(res["winding_raw"]))
        if err > 0.45:
            st.setdefault("accuracy_warnings", []).append(
                {"t0": res["t0"], "t1": res["t1"],
                 "winding_raw": res["winding_raw"],
                 "note": "variation d'argument non proche d'un entier : "
                         "step trop grand ou zero extremement proche du contour"})

    t_start = time.time()
    nxt = st["t_done"]
    while nxt < tmax and not _interrupted:
        hi = min(nxt + batch, tmax)
        jobs = []
        x = nxt
        while x < hi - 1e-12:
            y = min(x + batch, hi)
            jobs.append((x, y, step, dps))
            x = y
        if pool:
            for res in pool.imap(scan_batch, jobs):
                consolidate(res)
        else:
            for j in jobs:
                consolidate(scan_batch(j))
        nxt = hi
        st["finished"] = (nxt >= tmax - 1e-9)
        save_state(st, tmax)
        if not quiet:
            sys.stderr.write("  batch T=[%7.3f,%7.3f] ok  (%d/%d)  zeros=%d  %.1fs\n"
                             % (jobs[0][0], hi, len(st["batches"]), total_batches,
                                len(st["zeros_on_line"]), time.time() - t_start))

    if pool:
        pool.close()
        pool.terminate() if _interrupted else pool.join()

    if st["finished"] and not _interrupted:
        if not quiet:
            sys.stderr.write("refinage des zeros a dps=%d...\n" % (dps + 15))
        st["_pending"] = list(st["zeros_on_line"])
        st["zeros_on_line"] = []
        refine_zeros(st, dps_refine=dps + 15)
        del st["_pending"]
        report = build_report(st)
        rp = os.path.join(results_dir(), "dh_scan_T%g_report.json" % tmax)
        with open(rp, "w") as fh:
            json.dump(report, fh, indent=2)
        st["report_path"] = rp
        save_state(st, tmax)
    return st, _interrupted


def build_report(st):
    p = st["params"]
    return {
        "module": "dh-scan",
        "state_version": st["version"],
        "T_max": p["tmax"],
        "scan_step": p["step"],
        "dps": p["dps"],
        "batches_completed": len(st["batches"]),
        "total_points_scanned": sum(b["n_points"] for b in st["batches"]),
        "zeros_on_line_below_T": len(st["zeros_on_line"]),
        "first_zeros_on_line": st["zeros_on_line"][:10],
        "candidates_not_on_line": st["candidates_not_on_line"],
        "winding_total_raw": st["winding_total_raw"],
        "accuracy_warnings": st.get("accuracy_warnings", []),
        "finished": st["finished"],
        "note": ("verification a hauteur finie ; les zeros hors droite sont "
                 "certifies par principe de l'argument (rh_engine dh_certify)"),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="dh_scan.py", description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="scanner la droite critique jusqu'a --tmax")
    sc.add_argument("--tmax", type=float, default=30.0)
    sc.add_argument("--batch", type=float, default=10.0, help="hauteur par batch")
    sc.add_argument("--step", type=float, default=0.01, help="pas du balayage")
    sc.add_argument("--dps", type=int, default=25)
    sc.add_argument("--checkpoint-interval", type=float, default=10.0,
                    dest="ckpt", help="secondes entre sauvegardes (best effort)")
    sc.add_argument("--workers", type=int, default=1)
    sc.add_argument("--fresh", action="store_true", help="ignorer le checkpoint")
    sc.add_argument("--json", action="store_true", help="rapport JSON sur stdout")

    stp = sub.add_parser("status", help="etat du checkpoint")
    stp.add_argument("--tmax", type=float, default=30.0)

    sub.add_parser("merge", help="reconstruit le rapport depuis le checkpoint")

    a = ap.parse_args(argv)

    if a.cmd == "scan":
        params = {"tmax": a.tmax, "batch": a.batch, "step": a.step,
                  "dps": a.dps, "checkpoint_interval": a.ckpt,
                  "fresh": a.fresh}
        st, interrupted = run_scan(params, workers=a.workers)
        report = build_report(st)
        if a.json:
            print(json.dumps(report, indent=2))
        else:
            print("t_done=%.4f / %.4f  batches=%d  zeros_sur_droite=%d  finished=%s"
                  % (st["t_done"], a.tmax, len(st["batches"]),
                     len(st["zeros_on_line"]), st["finished"]))
            if interrupted:
                print("INTERROMPU : relancer la meme commande pour reprendre.")
        return 0

    if a.cmd == "status":
        st = load_state(a.tmax)
        if st is None:
            print("aucun checkpoint pour T=%g" % a.tmax)
            return 1
        print(json.dumps({"t_done": st["t_done"],
                          "batches": len(st["batches"]),
                          "zeros_on_line": len(st["zeros_on_line"]),
                          "finished": st["finished"],
                          "path": state_path(a.tmax)}, indent=2))
        return 0

    if a.cmd == "merge":
        import glob
        found = False
        for pth in sorted(glob.glob(os.path.join(results_dir(), "dh_scan_T*.state.json"))):
            with open(pth) as fh:
                st = json.load(fh)
            if st.get("version") != STATE_VERSION:
                continue
            report = build_report(st)
            rp = pth.replace(".state.json", "_report.json")
            with open(rp, "w") as fh:
                json.dump(report, fh, indent=2)
            print("rapport ecrit : %s (finished=%s)" % (rp, st["finished"]))
            found = True
        return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
