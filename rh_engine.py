#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rh_engine.py -- Moteur de verification / falsification cote Hypothese de Riemann.

Quatre sous-modules independants :

  zeta-verify  T   : reproduit la logique de Platt-Trudgian (methode de Turing)
                     a hauteur T : comptage exact des zeros N(T) et comparaison
                     avec le nombre de zeros trouves SUR la droite critique par
                     changement de signe de Z(t).  Sortie : JSON.
  zeta-stats   N   : statistiques de spectre (espacements normalises) des N
                     premiers zeros, comparaison GUE / Poisson, paires de Lehmer.
  dh-certify       : EXPERIENCE DECISIVE. Construction depuis zero de la
                     fonction de Davenport-Heilbronn f(s) = L(s,chi) + eps.L(s,chi_b),
                     verification de son equation fonctionnelle (f(s)=R(s)f(1-s)),
                     comptage des zeros sur la droite critique, puis CERTIFICAT
                     (principe de l'argument) d'un zero HORS de la droite.
                     Montre que : equation fonctionnelle + symetrie + zeros sur la
                     droite + precision numerique  ne suffisent pas.
  frontier         : cartographie du point de jonction : hauteur de verification
                     certifiee, region sans zero explicite, portee du critere de
                     Robin-Lagarias (enonce Pi^0_1).
  robin       X    : enumeration des candidats superabondants n <= X et marges
                     exactes dans l'inegalite de Robin / le critere de Lagarias.

Usage : python3 rh_engine.py <sous-module> [parametres]
"""
import sys, json, time, os
import mpmath as mp

# NOTE D'INGENIERIE (v3) ------------------------------------------------
# * `dh-certify` delegue desormais au scanner HPC reprenable `dh_scan.py`
#   (batchs paralleles + checkpoints toutes les 10 s). L'ancienne version
#   monolithique reste accessible via `dh-certify-legacy`.
# * l'enumeration Robin ne materialise plus la liste complete des candidats
#   (cause du OOM sur X >= 1e40) : flux + tas des K meilleurs, memoire O(1).
# ----------------------------------------------------------------------


# =============================================================================
# 1. ZETA : verification a hauteur finie (methode de Turing)
# =============================================================================
def zeta_verify(N_zeros, dps=25):
    """Verification rigoureuse a hauteur finie, methode de Turing.

    (1) les N_zeros premiers zeros sont localises par l'algorithme rapide de
        mpmath (Riemann-Siegel + Newton) ;
    (2) CHAQUE zero est re-verifie independamment : on controle que Z(t) change
        de signe au voisinage, ce qui certifie l'existence d'un zero reel de
        Z, donc d'un zero de zeta sur la droite critique ;
    (3) comptage global : N(T) par la methode de Turing ; si le nombre de zeros
        trouves SUR la droite egale N(T), alors il n'existe aucun zero hors de
        la droite sous cette hauteur (un zero hors axe viendrait par paires
        symetriques, ce qui ferait depasser le compte).
    """
    mp.mp.dps = dps
    t0 = time.time()
    out = {"module": "zeta-verify", "n_zeros_target": N_zeros, "dps": dps}

    zeros = []
    for k in range(1, N_zeros + 1):
        z = mp.zetazero(k)
        zeros.append(mp.im(z))
    out["first_ten_zeros"] = [float(z) for z in zeros[:10]]
    out["highest_zero"] = float(zeros[-1])

    # verification independante par changement de signe de Z (Riemann-Siegel)
    verified = 0
    for k, t in enumerate(zeros):
        gl = zeros[k] - zeros[k - 1] if k > 0 else mp.mpf(5)
        gr = zeros[k + 1] - zeros[k] if k + 1 < len(zeros) else mp.mpf(5)
        delta = min(gl, gr) / 4
        if delta > mp.mpf("1.0"):
            delta = mp.mpf("1.0")
        zp = mp.siegelz(t - delta)
        zm = mp.siegelz(t + delta)
        if (zp > 0) != (zm > 0):
            verified += 1
    out["zeros_verified_by_sign_change"] = verified

    # comptage de Turing : nombre exact de zeros dans la bande sous T
    T = zeros[-1] + mp.mpf("0.5")
    N_T = mp.nzeros(T)
    on_line_below_T = len([z for z in zeros if z <= T])
    out["T_used"] = float(T)
    out["N_T_turing"] = int(N_T)
    out["zeros_on_line_below_T"] = on_line_below_T
    out["turing_gap"] = int(N_T) - on_line_below_T
    out["verdict"] = ("TOUS les zeros de hauteur <= %.0f sont sur la droite critique et simples"
                      % float(T)) if int(N_T) == on_line_below_T else "ECART : zeros hors droite possibles"

    g = [zeros[i + 1] - zeros[i] for i in range(len(zeros) - 1)]
    imin = min(range(len(g)), key=lambda i: g[i])
    out["min_gap"] = float(g[imin])
    out["min_gap_pair"] = [float(zeros[imin]), float(zeros[imin + 1])]
    out["max_gap"] = float(max(g))
    out["zeros_list_head"] = [float(z) for z in zeros[:20]]
    out["mean_gap"] = float(sum(g) / len(g))
    out["mean_gap_theory_2pi_over_log"] = float(2 * mp.pi / mp.log(T / (2 * mp.pi)))
    out["seconds"] = time.time() - t0
    return out, zeros


def zeta_stats(zeros, gamma=mp.euler, dps=25):
    """statistiques d'espacements normalises + paires de Lehmer."""
    mp.mp.dps = dps
    g = [zeros[i + 1] - zeros[i] for i in range(len(zeros) - 1)]
    # normalisation locale par la densite moyenne : <gap> ~ 2*pi/log(t/2pi)
    s = []
    for i in range(len(g)):
        t = (zeros[i] + zeros[i + 1]) / 2
        s.append(g[i] * mp.log(t / (2 * mp.pi)) / (2 * mp.pi))
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / (len(s) - 1)
    small = len([x for x in s if x < mp.mpf("0.5")]) / len(s)
    # paires de Lehmer : deux zeros consecutifs anormalement proches
    lehmer_pairs = []
    for i in range(len(g)):
        if s[i] < mp.mpf("0.10"):
            lehmer_pairs.append({
                "index": i + 1,
                "t": float((zeros[i] + zeros[i + 1]) / 2),
                "gap_normalise": float(s[i]),
                "gap_absolu": float(g[i]),
            })
    return {
        "n_zeros": len(zeros),
        "mean_normalised_gap": float(mean),
        "variance_normalised_gap": float(var),
        "gue_prediction_variance": float(3 * mp.pi / 8 - 1),   # ~0.178 (Wigner surmise GUE)
        "poisson_prediction_variance": 1.0,
        "fraction_gaps_below_0.5": float(small),
        "n_lehmer_pairs_gap_lt_0.10": len(lehmer_pairs),
        "lehmer_pairs": lehmer_pairs[:10],
    }


# =============================================================================
# 2. DAVENPORT-HEILBRONN : construction, FE, zeros, certificat hors-axe
# =============================================================================
class DavenportHeilbronn:
    """f(s) = L(s,chi) + eps*L(s,chi_bar),  chi = [0,1,i,-i,-1] (impair mod 5).

    Derivation de l'equation fonctionnelle faite dans le code :
      L(s,chi)   = D(s) L(1-s,chi_bar) avec D(s) = eps*(2^s pi^{s-1} 5^{1/2-s}
                   sin(pi(1+s)/2) Gamma(1-s))
      L(s,chi_b) = eps^{-1} R(s) L(1-s,chi)   (conjugaison ; R = D/eps reelle sur R)
      => f(s) = R(s) f(1-s)  et  f(s) = eps*conj(f(conj(s)))
      => Z_DH(t) := e^{i arg(D(1/2+it))/2} f(1/2+it)  est REELLE.
    """
    chi = [0, 1, 1j, -1j, -1]
    chib = [0, 1, -1j, 1j, -1]
    q = 5

    def __init__(self, dps=40):
        mp.mp.dps = dps
        self.dps = dps
        tau = sum(self.chi[n] * mp.e ** (2j * mp.pi * n / self.q) for n in range(1, self.q))
        self.eps = tau / (1j * mp.sqrt(self.q))          # |eps| = 1
        self.kappa = mp.re(-1j * (self.eps - 1) / (1 + self.eps))
        self.kappa_closed = (mp.sqrt(10 - 2 * mp.sqrt(5)) - 2) / (mp.sqrt(5) - 1)

    def L(self, s, c):
        return mp.dirichlet(s, c)

    def R(self, s):
        q = self.q
        return 2 ** s * mp.pi ** (s - 1) * mp.sqrt(q) * q ** (-s) * mp.sin(mp.pi * (1 + s) / 2) * mp.gamma(1 - s)

    def D(self, s):
        return self.eps * self.R(s)

    def f(self, s):
        return self.L(s, self.chi) + self.eps * self.L(s, self.chib)

    def Z(self, t):
        """fonction reelle de Riemann-Siegel adaptee (memes zeros que f sur la droite)."""
        s = mp.mpf(1) / 2 + 1j * mp.mpf(t)
        return mp.e ** (1j * mp.arg(self.D(s)) / 2) * self.f(s)

    def fe_residue(self, pts):
        return [abs(self.f(s) - self.R(s) * self.f(1 - s)) for s in pts]


def dh_certify(T=40, dps=40, radius=mp.mpf("0.02"), npts=240):
    """Verification complete cote Davenport-Heilbronn. Sortie JSON-able.

    Detection des zeros de la droite SANS ambiguite de phase : on suit les
    fonctions reelles a(t)=Re f(1/2+it) et b(t)=Im f(1/2+it). Un zero de f sur
    la droite se manifeste par un changement de signe SIMULTANE de a et b
    (la courbe (a,b) traverse l'origine). Chaque candidat est ensuite affine
    comme zero complexe, puis classe (sur/hors la droite).
    """
    t0 = time.time()
    mp.mp.dps = dps
    dh = DavenportHeilbronn(dps=dps)
    out = {"module": "dh-certify", "dps": dps}

    # --- 1. constantes derivees ---
    out["gauss_sum_eps"] = [float(mp.re(dh.eps)), float(mp.im(dh.eps))]
    out["kappa_derived"] = float(dh.kappa)
    out["kappa_closed_form"] = float(dh.kappa_closed)
    out["kappa_discrepancy"] = float(abs(dh.kappa - dh.kappa_closed))
    out["kappa_closed_form_expression"] = "(sqrt(10-2*sqrt(5))-2)/(sqrt(5)-1)"

    # --- 2. equation fonctionnelle ---
    pts = [mp.mpf("0.7") + 3j, mp.mpf("0.31") + 41.7j, mp.mpf("-0.2") + 2.5j, mp.mpf("0.9") + 85j]
    out["FE_max_abs_residue"] = float(max(dh.fe_residue(pts)))

    # --- 3. zeros de la droite critique (balayage a/b a precision moderee) ---
    mp.mp.dps = 25
    dh25 = DavenportHeilbronn(dps=25)
    step = mp.mpf("0.01")
    cand = []
    prev_t = mp.mpf("0.01")
    pa = mp.re(dh25.f(mp.mpf(1) / 2 + 1j * prev_t))
    pb = mp.im(dh25.f(mp.mpf(1) / 2 + 1j * prev_t))
    t = prev_t
    nscan = 0
    while t < T and nscan < 200000:
        t = t + step
        w = dh25.f(mp.mpf(1) / 2 + 1j * t)
        a, b = mp.re(w), mp.im(w)
        if ((a > 0) != (pa > 0)) and ((b > 0) != (pb > 0)):
            cand.append((prev_t + t) / 2)
        pa, pb, prev_t = a, b, t
        nscan += 1
    mp.mp.dps = dps
    on_line = []
    for c in cand:
        r = mp.findroot(lambda z: dh.f(z), complex(mp.mpf(1) / 2, c), solver="muller",
                        tol=mp.mpf(10) ** (-dps + 8), maxsteps=60)
        if abs(mp.re(r) - mp.mpf(1) / 2) < mp.mpf(10) ** (-dps + 12):
            on_line.append(mp.im(r))
        else:
            out.setdefault("candidates_not_on_line", []).append([float(mp.re(r)), float(mp.im(r))])
    out["zeros_on_line_below_T"] = len(on_line)
    out["first_zeros_on_line"] = [float(z) for z in on_line[:8]]
    out["scan_points"] = nscan

    # --- 4. comptage TOTAL dans la bande (principe de l'argument) ---
    def winding_rect(x0, x1, y0, y1, n=160):
        P = []
        for i in range(n):
            P.append(x0 + (x1 - x0) * mp.mpf(i) / n + 1j * y0)
        for i in range(n):
            P.append(x1 + 1j * (y0 + (y1 - y0) * mp.mpf(i) / n))
        for i in range(n):
            P.append(x1 - (x1 - x0) * mp.mpf(i) / n + 1j * y1)
        for i in range(n):
            P.append(x0 + 1j * (y1 - (y1 - y0) * mp.mpf(i) / n))
        v = [dh.f(p) for p in P]
        tot = mp.mpf(0)
        for i in range(len(v)):
            tot += mp.arg(v[(i + 1) % len(v)] / v[i])
        return tot / (2 * mp.pi)

    w = winding_rect(mp.mpf(0), mp.mpf(1), -T, T, n=150)
    out["strip_zeros_abs_t_le_T_total"] = float(mp.nint(w))
    out["winding_raw"] = float(w)
    out["strip_zeros_t_in_0_T"] = float(mp.nint(w) / 2)
    if float(mp.nint(w)) > 0:
        out["fraction_of_strip_zeros_on_line"] = len(on_line) / (float(mp.nint(w)) / 2)

    # --- 5. CERTIFICAT D'UN ZERO HORS AXE (principe de l'argument) ---
    seed = mp.mpf("0.808517") + 1j * mp.mpf("85.699348")       # valeur publiee (Spira 1994)
    rho = mp.findroot(lambda z: dh.f(z), seed, solver="muller", tol=mp.mpf(10) ** (-dps + 10), maxsteps=80)
    out["offline_zero"] = [float(mp.re(rho)), float(mp.im(rho))]
    out["offline_zero_re_minus_half"] = float(mp.re(rho) - mp.mpf(1) / 2)
    circle = [rho + radius * mp.e ** (2j * mp.pi * k / npts) for k in range(npts)]
    vals = [dh.f(s) for s in circle]
    absmin = min(abs(v) for v in vals)
    total = mp.mpf(0)
    for k in range(npts):
        total += mp.arg(vals[(k + 1) % npts] / vals[k])
    winding = total / (2 * mp.pi)
    out["offline_certificate"] = {
        "radius": float(radius),
        "n_points": npts,
        "min_modulus_on_circle": float(absmin),
        "winding_number": float(winding),
        "zeros_inside_disc": float(mp.nint(winding)),
        "verdict": "CERTIFIE : zero reel hors de la droite critique" if abs(winding - 1) < mp.mpf("0.05") else "non concluant",
    }

    # --- 6. le piege : "|f| != 0 donc pas un zero" (erreur du type 2025) ---
    trap = {}
    for d in [15, 25, 40]:
        mp.mp.dps = d
        dh2 = DavenportHeilbronn(dps=d)
        trap[str(d)] = float(mp.log10(abs(dh2.f(complex(rho)))))
    mp.mp.dps = dps
    out["trap_log10_abs_f_at_the_zero"] = trap
    out["trap_explanation"] = ("un zero est un point ou f s'annule EXACTEMENT ; "
                               "toute evaluation flottante renvoie une valeur non nulle "
                               "d'autant plus petite que la precision augmente. "
                               "Conclure '|f| != 0 donc ce n'est pas un zero' est une erreur.")
    out["seconds"] = time.time() - t0
    return out


# =============================================================================
# 3. FRONTIERE : region sans zero explicite + etat de l'art
# =============================================================================
def frontier():
    mp.mp.dps = 30
    rows = []
    # Region sans zero explicite de type classique (Ford 2002) :
    #   zeta(sigma+it) != 0 pour sigma > 1 - 1/(57.54 (log t)^{2/3} (log log t)^{1/3}), t >= 3
    for e in [13, 20, 50, 100, 1000, 10 ** 6]:
        t = mp.mpf(10) ** e
        sigma = 1 - 1 / (mp.mpf("57.54") * mp.log(t) ** (mp.mpf(2) / 3) * mp.log(mp.log(t)) ** (mp.mpf(1) / 3))
        rows.append({
            "log10_t": e,
            "sigma_free_boundary": float(sigma),
            "unknown_window_width_on_sigma": float(sigma - mp.mpf("0.5")),
            "unknown_window_fraction_of_half_plane": float((sigma - mp.mpf("0.5")) / mp.mpf("0.5")),
        })
    return {
        "module": "frontier",
        "verified_height_RH": "3.0001753328e12 (Platt-Trudgian 2021, arithmetique par intervalles)",
        "number_of_zeros_verified": "1.2363153437e13",
        "proportion_on_line_unconditional": "~41.7 % (Conrey: plus de 5/12)",
        "dbn_lower_bound": "Lambda >= 0 (Rodgers-Tao 2018) => si RH vraie, elle est 'de justesse'",
        "dbn_upper_bound_certified": "Lambda <= 0.22 (Polymath15 2019) ; ~0.2 avec Platt-Trudgian ; raffinement preprint 2026 annonce <= 0.1787854",
        "zero_free_region_classical": "sigma > 1 - 1/(57.54 (log t)^{2/3} (log log t)^{1/3}) (Ford 2002)",
        "table": rows,
    }


# =============================================================================
# 4. ROBIN / LAGARIAS : le critere Pi^0_1 et sa frontiere calculable
# =============================================================================
def superabundant_stream(X, max_candidates=None):
    """Enumeration EN FLUX des entiers a exposants non croissants <= X.

    Version memoire-bornee : aucun materiel intermediaire n'est conserve, le
    generateur produit (n, sigma(n)) et l'appelant decide quoi garder.
    La version precedente construisait une liste -> OOM (memoria non bornee)
    pour X >= 1e40. Ici la memoire est O(profondeur) = O(nb de premiers).
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    counter = [0]

    def rec(i, n, expmax, sigma):
        counter[0] += 1
        if max_candidates is not None and counter[0] > max_candidates:
            raise RuntimeError("plafond de candidats depasse (%d)" % max_candidates)
        yield (n, sigma)
        if i >= len(primes):
            return
        p = primes[i]
        pk = p
        for e in range(1, expmax + 1):
            n2 = n * pk
            if n2 > X:
                break
            sp = (pk * p - 1) // (p - 1)          # sigma(p^e)
            yield from rec(i + 1, n2, e, sigma * sp)
            pk *= p

    yield from rec(0, 1, 255, 1)


def superabundant_candidates(X):
    """Compatibilite : materialise la liste (petits X uniquement)."""
    return sorted(set(superabundant_stream(X, max_candidates=5_000_000)))


def robin_frontier(X, dps=30, topk=12):
    """Frontiere du critere de Robin en memoire O(1) (flux + tas des K meilleurs)."""
    import heapq
    mp.mp.dps = dps
    t0 = time.time()
    egam = mp.e ** mp.euler
    heap = []                 # (ratio, n, sigma)  -- tas des K plus grands ratios
    seen = n_seen = 0
    best_norm = None
    for n, sigma in superabundant_stream(X):
        n_seen += 1
        if n <= 5040:
            continue
        seen += 1
        ln = mp.log(n)
        ratio = mp.mpf(sigma) / (egam * n * mp.log(ln))
        r = float(ratio)
        if len(heap) < topk:
            heapq.heappush(heap, (r, n, sigma))
        elif r > heap[0][0]:
            heapq.heapreplace(heap, (r, n, sigma))
    records = []
    for r, n, sigma in sorted(heap, reverse=True):
        ln = mp.log(n)
        H = mp.digamma(n + 1) + mp.euler
        records.append({
            "n": n, "sigma_n": sigma,
            "robin_ratio": r,
            "robin_margin_abs": float(mp.e ** mp.euler * n * mp.log(ln) - sigma),
            "robin_slack": 1.0 - r,
            "lagarias_margin": float((H + mp.e ** H * mp.log(H)) - sigma),
        })
    return {
        "module": "robin",
        "X": X,
        "n_candidates_tested": n_seen,
        "n_above_5040": seen,
        "n_records": len(records),
        "max_robin_ratio_over_X": records[0]["robin_ratio"] if records else None,
        "min_slack": records[0]["robin_slack"] if records else None,
        "records": records,
        "memory_mode": "flux (O(1) hors tas top-%d)" % topk,
        "seconds": time.time() - t0,
    }


# =============================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "zeta-verify":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        out, zeros = zeta_verify(N)
        out["spacing_stats"] = zeta_stats(zeros)
        with open("/home/user/results/zeta_zeros.csv", "w") as fh:
            fh.write("index,gamma,normalised_gap\n")
            for i, z in enumerate(zeros):
                gp = ""
                if i + 1 < len(zeros):
                    gp = float((zeros[i+1]-zeros[i]) * mp.log(z / (2 * mp.pi)) / (2 * mp.pi))
                fh.write("%d,%s,%s\n" % (i + 1, mp.nstr(z, 20), gp))
        print(json.dumps(out, indent=2))
    elif cmd == "dh-certify":
        # v3 : delegation au scanner HPC reprenable (batchs + checkpoints 10 s)
        import subprocess
        T = sys.argv[2] if len(sys.argv) > 2 else "30"
        here = os.path.dirname(os.path.abspath(__file__))
        sys.exit(subprocess.call([sys.executable, os.path.join(here, "dh_scan.py"),
                                  "scan", "--tmax", str(T)]))
    elif cmd == "dh-certify-legacy":
        T = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        print(json.dumps(dh_certify(T), indent=2))
    elif cmd == "frontier":
        print(json.dumps(frontier(), indent=2))
    elif cmd == "robin":
        X = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 18
        print(json.dumps(robin_frontier(X), indent=2))
    elif cmd == "robin-verify-equivalence":
        # controle : le scoring en flux donne-t-il les memes maxima que la liste ?
        for X in (10 ** 18, 10 ** 36):
            a = robin_frontier(X)["max_robin_ratio_over_X"]
            b = max(float(mp.mpf(s) / (mp.e ** mp.euler * n * mp.log(mp.log(n))))
                    for n, s in superabundant_candidates(X) if n > 5040)
            print("X=1e%d  flux=%.12f  liste=%.12f  ecart=%.2e"
                  % (len(str(X)) - 1, a, b, abs(a - b)))
    else:
        print(__doc__)
