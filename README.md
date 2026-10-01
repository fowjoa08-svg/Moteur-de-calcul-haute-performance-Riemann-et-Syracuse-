# High-Performance Computing (HPC) Simulation Engine

Moteur de simulation numérique asynchrone conçu pour auditer les frontières de la vérification computationnelle concernant l'**Hypothèse de Riemann** (RH) et la **conjecture de Collatz** (Syracuse).

> ⚠️ **Note d'honnêteté scientifique** : ni RH ni Collatz ne sont démontrées. Ce dépôt ne prétend produire aucune preuve ; il fournit des outils de vérification à hauteur finie, des certificats numériques et une cartographie de la « frontière » entre ce qui est calculable et ce qui reste ouvert. Le module `dh-certify` construit notamment un **contre-exemple de travail** (fonction de Davenport–Heilbronn) qui satisfait l'équation fonctionnelle avec des zeros sur la droite critique tout en ayant des zeros hors de la droite — illustrant pourquoi la seule vérification numérique ne prouve rien.

## 🚀 Démarrage rapide

```bash
pip install -r requirements.txt   # dependance python : mpmath
make                              # compile millennium_engine (-O3 -march=native -fopenmp)
make test                         # auto-tests de non-regression
```

## 🛠️ Architecture technique & optimisations

- **Empreinte mémoire :** pipeline de flux continu à empreinte O(1) ; la vérification Collatz par segments (`collatz-range`, `collatz-tst-stream`) plafonne l'espace d'adressage via `setrlimit` et rend un RSS borné (rapporté dans chaque sortie JSON) pour éviter les OOM sur les grands jeux de données.
- **Safepoints asynchrones :** checkpoints de reprise sérialisés en JSON toutes les 10 s (paramétrables), écriture atomique (`tmp` + `rename()`), thread écrivain dédié avec double tampon — latence d'E/S nulle pour le calcul, reprise propre après SIGINT/SIGTERM ou timeout.
- **Accélération matérielle :** modules C parallélisés par OpenMP (`-O3 -march=native -fopenmp`), arithmétique native `__uint128_t` pour suivre les pics de trajectoire au-delà de 2⁶⁴ sans perte de cas.

## 🔬 Protocoles implémentés

### `rh_engine.py` — Hypothèse de Riemann
| Sous-module | Description |
|---|---|
| `zeta-verify N` | Méthode de Turing (esprit Platt–Trudgian) : localisation des N premiers zeros, re-vérification indépendante par changement de signe de Z(t), comptage exact N(T) et verdict « tous les zeros ≤ T sont sur la droite critique ». Export CSV (`results/zeta_zeros.csv`). |
| `zeta-stats` (inclus) | Statistiques d'espacements normalisés : variance GUE (~0,178) vs Poisson, détection de paires de Lehmer. |
| `dh-certify [T]` | Construction *from scratch* de f(s) = L(s,χ) + ε·L(s,χ̄) (caractère de Davenport–Heilbronn, module 5), contrôle de l'équation fonctionnelle, comptage des zeros sur la droite, puis **certificat par principe de l'argument** d'un zero HORS de la droite critique. |
| `frontier` | Cartographie de la jonction : hauteur certifiée (Platt–Trudgian 2021), régions sans zero (Ford 2002), bornes De Bruijn–Newman, proportion connue de zeros sur la droite. |
| `robin X` | Critère de Robin/Lagarias (équivalent à RH) : énumération **en flux** des candidats superabondants n ≤ X, marges exactes, mémoire O(1) (tas des K meilleurs). |

### `millennium_engine.c` — Collatz / Syracuse
| Mode | Description |
|---|---|
| `collatz-verify N [hist.csv]` | Vérification exhaustive de la convergence pour tout n ≤ N **sans mémoire** (induction forte par descente sous le point de départ) ; records de σ(n), de pic, histogramme, compte des trajectoires sortant de la zone vérifiée. |
| `collatz-full N [budget_mb]` | Temps d'arrêt total tst(n) pour tout n ≤ N (tableau uint16) ; refuse explicitement si le budget mémoire est dépassé. |
| `collatz-tst-stream N [state.json]` | Variante à mémoire O(1) avec checkpoints et reprise. |
| `collatz-range lo hi state.json [batch] [ckpt_s] [budget_mb]` | Segments parallèles micro-batchés, thread écrivain, reprise sur arrêt. |
| `collatz-point n1 n2 ...` | Validation ponctuelle contre les tables publiques (ex. 27 → 111, 9780657630 → 1132). |
| `robin-score X_log10 [k]` | Scoring Robin en récurrence sur les exposants de premiers (long double), top-K en ligne. |

Sorties systématiquement en **JSON** (exploitables par `jq`) ; chemins de sortie Python configurables via `RH_RESULTS_DIR` (défaut : `./results/`, créé automatiquement).

## 📦 Structure du dépôt

```
├── millennium_engine.c   # moteur C (Collatz, robin-score) — OpenMP + __uint128_t
├── rh_engine.py          # moteur Python (RH : zeta, DH, frontier, robin) — mpmath
├── Makefile              # all / test / clean
├── requirements.txt      # mpmath
└── README.md
```

## 🧪 Exemples

```bash
./millennium_engine collatz-point 27 9780657630
./millennium_engine collatz-verify 1000000 | jq .record_sigma
python3 rh_engine.py zeta-verify 100 | jq .verdict
python3 rh_engine.py dh-certify 40 | jq .offline_certificate
python3 rh_engine.py robin 1e24 | jq .max_robin_ratio_over_X
```

---
*Projet de recherche indépendant explorant les invariants structurels et la conception de systèmes haute performance.*
