# -*- coding: utf-8 -*-
"""
test_rh_engine.py -- tests unitaires/integration pour rh_engine.py et dh_scan.py.

Marquage :
  * les tests marques `slow` sont exclus par defaut (lancer avec -m slow).
Execution :
  python3 -m pytest tests/            # rapide (~1-2 min)
  python3 -m pytest tests/ -m slow    # inclut le scan complet T=30
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mpmath as mp                      # noqa: E402
import rh_engine                         # noqa: E402
import dh_scan                           # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_results(tmp_path, monkeypatch):
    """Chaque test ecrit ses checkpoints/CSV dans un repertoire temporaire."""
    monkeypatch.setenv("RH_RESULTS_DIR", str(tmp_path))
    _current_tmp.d = str(tmp_path)


# --------------------------------------------------------------------------
# harnais rapides
# --------------------------------------------------------------------------
def run_py(args, timeout=300):
    env = dict(os.environ, RH_RESULTS_DIR=_current_tmp.d)
    return subprocess.run([sys.executable] + args, capture_output=True,
                          text=True, timeout=timeout, cwd=ROOT, env=env)


_current_tmp = type("_CT", (), {"d": "/tmp"})


def run_c(args, timeout=120):
    exe = os.path.join(ROOT, "millennium_engine")
    if not os.path.exists(exe):
        pytest.skip("millennium_engine non compile (lancer `make`)")
    return subprocess.run([exe] + args, capture_output=True,
                          text=True, timeout=timeout, cwd=ROOT)


# --------------------------------------------------------------------------
# imports / structure
# --------------------------------------------------------------------------
class TestImports:
    def test_rh_engine_imports(self):
        assert hasattr(rh_engine, "dh_certify")
        assert hasattr(rh_engine, "zeta_verify")
        assert hasattr(rh_engine, "frontier")

    def test_dh_scan_exports(self):
        for fn in ("scan_batch", "run_scan", "build_report", "main"):
            assert callable(getattr(dh_scan, fn))

    def test_results_dir_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RH_RESULTS_DIR", str(tmp_path / "sub" / "dir"))
        d = rh_engine.results_dir()
        assert os.path.isdir(d)


# --------------------------------------------------------------------------
# Davenport-Heilbronn : equation fonctionnelle & constantes
# --------------------------------------------------------------------------
class TestDHMath:
    def test_kappa_closed_form_matches_derived(self):
        dh = rh_engine.DavenportHeilbronn(dps=30)
        assert abs(dh.kappa - dh.kappa_closed) < mp.mpf("1e-25")

    def test_functional_equation_residue_small(self):
        dh = rh_engine.DavenportHeilbronn(dps=30)
        pts = [mp.mpf("0.7") + 3j, mp.mpf("0.31") + 41.7j,
               mp.mpf("-0.2") + 2.5j, mp.mpf("0.9") + 85j]
        assert max(dh.fe_residue(pts)) < mp.mpf("1e-20")

    def test_eps_is_unit_modulus(self):
        dh = rh_engine.DavenportHeilbronn(dps=30)
        assert abs(abs(dh.eps) - 1) < mp.mpf("1e-25")


# --------------------------------------------------------------------------
# dh-certify (rapide : hauteur faible)
# --------------------------------------------------------------------------
_CERTIFY_CACHE = {}


def _certify_result():
    if "r" not in _CERTIFY_CACHE:
        _CERTIFY_CACHE["r"] = rh_engine.dh_certify(T=12.0, dps=30)
    return _CERTIFY_CACHE["r"]


class TestDhCertifyFast:
    @pytest.fixture()
    def result(self):
        return _certify_result()

    def test_first_on_line_zero_near_5_094(self, result):
        assert result["zeros_on_line_below_T"] >= 1
        assert abs(result["first_zeros_on_line"][0] - 5.094159844571095) < 1e-6

    def test_offline_zero_certificate_winding_one(self, result):
        cert = result["offline_certificate"]
        assert abs(cert["winding_number"] - 1.0) < 0.05
        assert cert["verdict"].startswith("CERTIFIE")

    def test_offline_zero_really_off_line(self, result):
        re_, im = result["offline_zero"]
        assert abs(re_ - 0.8085165) < 1e-4   # valeur publiee (Spira)
        assert result["offline_zero_re_minus_half"] > 0.3

    def test_trap_ladder_decreasing(self, result):
        # |f| doit diminuer regulierement quand on s'approche de rho :
        # c'est exactement ce qui rend leurre "|f| != 0 donc pas un zero"
        lad = result["offline_zero_ladder"]
        vals = [e["log10_abs_f"] for e in lad]
        assert vals == sorted(vals, reverse=True)   # strictement decroissant
        assert vals[0] - vals[-1] > 1.5             # au moins ~3 ordres de grandeur


# --------------------------------------------------------------------------
# scanner HPC dh_scan.py
# --------------------------------------------------------------------------
class TestDhScan:
    def test_state_roundtrip(self):
        params = {"tmax": 20.0, "batch": 10.0, "step": 0.05, "dps": 15,
                  "checkpoint_interval": 10.0}
        st = dh_scan.new_state(params)
        st["t_done"] = 10.0
        st["batches"].append({"t0": 0, "t1": 10, "winding_raw": 0.3, "n_points": 200})
        st["zeros_on_line"] = [5.094]
        dh_scan.save_state(st, 20.0)
        st2 = dh_scan.load_state(20.0)
        assert st2["t_done"] == 10.0 and st2["zeros_on_line"] == [5.094]

    def test_checkpoint_version_guard(self):
        p = dh_scan.state_path(7.0)
        with open(p, "w") as fh:
            json.dump({"version": 999, "params": {}, "t_done": 0}, fh)
        assert dh_scan.load_state(7.0) is None

    def test_scan_batch_finds_known_zero(self):
        res = dh_scan.scan_batch((4.0, 6.0, 0.01, 20))
        assert len(res["zeros"]) == 1
        assert abs(res["zeros"][0] - 5.094159844571095) < 0.02
        assert res["n_points"] == 201   # pas inclusif en bornes
        # winding fractionnaire normal : un zero interieur tourne ~pi sur
        # le segment de droite critique (demi-contribution)
        assert abs(res["winding_raw"] - 0.28) < 0.05

    def test_resume_after_interruption(self, monkeypatch):
        calls = {"n": 0}
        orig = dh_scan.scan_batch

        def wrapped(job):
            calls["n"] += 1
            if calls["n"] == 2:
                dh_scan._interrupted = True
            return orig(job)

        monkeypatch.setattr(dh_scan, "scan_batch", wrapped)
        params = {"tmax": 6.0, "batch": 2.0, "step": 0.05, "dps": 15,
                  "checkpoint_interval": 10.0, "fresh": True}
        st, intr = dh_scan.run_scan(params, workers=1, quiet=True)
        assert intr and not st["finished"] and st["t_done"] < 6.0
        saved = dh_scan.load_state(6.0)
        assert saved["t_done"] == st["t_done"]

        # reprise : doit finir sans recalculer les batchs deja sauves
        monkeypatch.undo()
        dh_scan._interrupted = False
        params["fresh"] = False
        st2, intr2 = dh_scan.run_scan(params, workers=1, quiet=True)
        assert st2["finished"] and not intr2

    def test_merge_cli(self, tmp_path):
        r = run_py(["dh_scan.py", "merge"])
        assert r.returncode == 1  # aucun checkpoint -> code 1 propre
        params = {"tmax": 5.0, "batch": 5.0, "step": 0.1, "dps": 12,
                  "checkpoint_interval": 10.0, "fresh": True}
        dh_scan.run_scan(params, workers=1, quiet=True)
        r = run_py(["dh_scan.py", "merge"])
        assert r.returncode == 0
        reports = list(tmp_path.glob("dh_scan_T*_report.json"))
        assert reports, "le merge doit produire un rapport"

    def test_dh_certify_delegates_to_scanner(self, tmp_path):
        # dh_scan.py etant present, `dh-certify` delegue et cree son checkpoint
        r = run_py(["rh_engine.py", "dh-certify", "4"], timeout=180)
        assert r.returncode == 0
        assert (tmp_path / "dh_scan_T4.state.json").exists(), \
            "la delegation vers dh_scan.py doit creer son checkpoint"


# --------------------------------------------------------------------------
# CLI rh_engine.py
# --------------------------------------------------------------------------
class TestRhEngineCli:
    def test_unknown_module_exits_1(self):
        r = run_py(["rh_engine.py", "module-inexistant"])
        assert r.returncode == 1
        assert "sous-module inconnu" in r.stderr

    def test_help_exit_0(self):
        r = run_py(["rh_engine.py", "help"])
        assert r.returncode == 0
        assert "dh-certify" in r.stdout

    def test_frontier_json_valid(self):
        r = run_py(["rh_engine.py", "frontier"], timeout=120)
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert isinstance(out, dict) and "module" in out

    def test_robin_small_X(self):
        r = run_py(["rh_engine.py", "robin", "1e6"], timeout=120)
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert out["max_robin_ratio_over_X"] < 1.0  # Robin verifie jusque-la


# --------------------------------------------------------------------------
# moteur C
# --------------------------------------------------------------------------
class TestCEngine:
    def test_collatz_point_known_values(self):
        r = run_c(["collatz-point", "27", "9780657630"])
        assert r.returncode == 0
        assert "111" in r.stdout      # tst(27)
        assert "1132" in r.stdout     # tst(9780657630)

    def test_collatz_verify_small(self):
        r = run_c(["collatz-verify", "10000"])
        assert r.returncode == 0
        assert "collatz-verify" in r.stdout


# --------------------------------------------------------------------------
# tests lents (exclus par defaut)
# --------------------------------------------------------------------------
@pytest.mark.slow
class TestSlow:
    def test_full_scan_T30_consistent_with_legacy(self):
        params = {"tmax": 30.0, "batch": 10.0, "step": 0.01, "dps": 25,
                  "checkpoint_interval": 10.0, "fresh": True}
        st, intr = dh_scan.run_scan(params, workers=1, quiet=True)
        assert st["finished"] and not intr
        legacy = rh_engine.dh_certify(T=30.0, dps=40)
        assert len(st["zeros_on_line"]) == legacy["zeros_on_line_below_T"]

    def test_zeta_verify_small(self):
        r = run_py(["rh_engine.py", "zeta-verify", "50"], timeout=600)
        assert r.returncode == 0
        out = json.loads(r.stdout)
        assert "turing_gap" in out
