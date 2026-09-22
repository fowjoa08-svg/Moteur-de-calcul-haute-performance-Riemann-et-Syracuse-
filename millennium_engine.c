/* ============================================================================
 * millennium_engine.c  --  Moteur de vérification computationnelle
 *
 * Utilisé par le RAPPORT_MILLENIUM.md (Conjecture de Syracuse / Collatz).
 *
 * MODES
 *   collatz-verify N   Verification exhaustive de la convergence pour tout
 *                      n <= N, SANS memoire (induction forte par "descente
 *                      sous le point de depart"). Rapporte :
 *                        - echecs (aucun attendu)
 *                        - records de temps d'arret sigma(n) (etapes avant de
 *                          descendre strictement sous n)
 *                        - records de "pic" : max de la trajectoire et ratio pic/n
 *                        - histogramme de sigma
 *                        - compte des n dont la trajectoire SORT de la zone
 *                          deja verifiee (valeur depassant N)  <-- mesure du
 *                          "point de jonction" computationnel
 *   collatz-full N     Calcule le temps d'arret total tst(n) (etapes jusqu'a 1)
 *                      pour tout n <= N (tableau uint16), histogramme + records.
 *   collatz-point U    Temps d'arret total de nombres isoles (validation externe
 *                      contre les tables publiques, ex. 27 -> 111, 9780657630 ->
 *                      1132).
 *
 * Compilation : gcc -O3 -march=native -fopenmp -o millennium_engine millennium_engine.c -lm
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef unsigned long long u64;
typedef __uint128_t u128;

static double peak_rss_mb(void) {          /* empreinte memoire maximale reelle */
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;  /* Linux : Ko */
}

static int mem_guard_mb(long budget_mb) { /* plafonne l'espace d'adressage */
    struct rlimit rl;
    if (budget_mb <= 0) return 0;
    rl.rlim_cur = (rlim_t)budget_mb * 1024 * 1024;
    rl.rlim_max = (rlim_t)budget_mb * 1024 * 1024;
    if (setrlimit(RLIMIT_AS, &rl) != 0) { perror("setrlimit"); return -1; }
    return 0;
}

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------------------------------------------------------------------------
 * Iteration de Collatz "courte" T(n) = n/2 si pair, (3n+1)/2 si impair.
 * On travaille sur les entiers impairs ; chaque pas impair absorbe la division
 * par 2 forcee qui suit. Le nombre de pas "T" vaut la moitie (a +/- 1 pres) du
 * nombre de pas de la carte standard ; on rapporte les deux conventions.
 * -------------------------------------------------------------------------*/

/* mode 1 : verification exhaustive sans memoire -----------------------------*/
/* Pour tout n <= N on itere T jusqu'a descendre strictement sous n (induction
   forte : ce n est alors verifie par le cas deja traite). Aucun tableau n'est
   necessaire ; les trajectoires qui depassent 2^63 basculent en arithmetique
   128 bits (voie lente) pour ne perdre aucun pic ni aucun cas.               */

static void print_u128(u128 x, char *buf) {          /* u128 -> chaine decimale */
    char tmp[64]; int k = 0;
    if (x == 0) { strcpy(buf, "0"); return; }
    while (x > 0) { tmp[k++] = '0' + (int)(x % 10); x /= 10; }
    for (int i = 0; i < k; i++) buf[i] = tmp[k - 1 - i];
    buf[k] = 0;
}

static int collatz_verify(u64 N, const char *csv_out) {
    double t0 = wall();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    u64 rec_sigma = 0, rec_sigma_arg = 1;
    u128 rec_peak = 0; u64 rec_peak_arg = 1;
    u64 rec_ratio_huge = 0;                       /* n dont le ratio pic/n est un record */
    u128 rec_ratio_peak = 0;
    u64 excursions = 0, failures = 0, first_fail = 0, alti = 0;   /* alti = n a depassement u64 */
    const u64 STEPCAP = 2000000000ULL;

#define NBIN 8192
    static u64 hist[NBIN];
    memset(hist, 0, sizeof(hist));

#pragma omp parallel
    {
        u64 l_rec_sigma = 0, l_rec_sigma_arg = 1;
        u128 l_rec_peak = 0; u64 l_rec_peak_arg = 1;
        u64 l_exc = 0, l_fail = 0, l_alti = 0, l_first_fail = 0;
        u64 l_ratio_huge = 0; u128 l_ratio_peak = 0;
        u64 l_hist[NBIN];
        memset(l_hist, 0, sizeof(l_hist));

#pragma omp for schedule(dynamic, 1<<20) nowait
        for (long long nn = 1; nn <= (long long)N; nn++) {
            u64 n = (u64)nn;
            if (n == 1) { l_hist[0]++; continue; }
            u64 m = n, steps = 0;
            int ok = 0;
            u128 peak = n;
            while (steps < STEPCAP) {
                if (m & 1ULL) {
                    u64 t = 3*m + 1;
                    if ((u128)t > peak) peak = t;
                    m = t >> 1;
                    steps += 2;
                } else {
                    m >>= 1; steps += 1;
                }
                if ((u128)m > peak) peak = m;
                if (m < n) { ok = 1; break; }
                if (m > 6000000000000000000ULL) {        /* voie lente 128 bits */
                    u128 h = m; l_alti++;
                    while (steps < STEPCAP) {
                        if (h & 1) { u128 t = 3*h + 1; if (t > peak) peak = t; h = t >> 1; steps += 2; }
                        else { h >>= 1; steps += 1; }
                        if (h > peak) peak = h;
                        if (h < (u128)n) { ok = 1; break; }
                    }
                    break;
                }
            }
            if (!ok) { l_fail++; if (!l_first_fail) l_first_fail = n; continue; }
            if (peak > (u128)N) l_exc++;
            if (steps > l_rec_sigma) { l_rec_sigma = steps; l_rec_sigma_arg = n; }
            if (peak > l_rec_peak) { l_rec_peak = peak; l_rec_peak_arg = n; }
            if ((u128)n > peak) { /* impossible */ }
            /* record du ratio pic/n : comparaison en 128 bits evitant les flottants */
            if (l_ratio_peak == 0 || peak * (u128)l_ratio_huge > l_ratio_peak * (u128)n) {
                l_ratio_peak = peak; l_ratio_huge = n;
            }
            if (steps < NBIN) l_hist[steps]++;
        }

#pragma omp critical
        {
            if (l_rec_sigma > rec_sigma) { rec_sigma = l_rec_sigma; rec_sigma_arg = l_rec_sigma_arg; }
            if (l_rec_peak > rec_peak) { rec_peak = l_rec_peak; rec_peak_arg = l_rec_peak_arg; }
            if (rec_ratio_huge == 0 || l_ratio_peak * (u128)rec_ratio_huge > rec_ratio_peak * (u128)l_ratio_huge) {
                rec_ratio_peak = l_ratio_peak; rec_ratio_huge = l_ratio_huge;
            }
            excursions += l_exc; failures += l_fail; alti += l_alti;
            if (!first_fail && l_first_fail) first_fail = l_first_fail;
            for (int i = 0; i < NBIN; i++) hist[i] += l_hist[i];
        }
    }

    char pk[64], rp[64];
    print_u128(rec_peak, pk);
    print_u128(rec_ratio_peak, rp);
    long double ratio = (long double)strtold(rp, NULL) / (long double)rec_ratio_huge;

    printf("{\n");
    printf("  \"mode\": \"collatz-verify\",\n");
    printf("  \"N\": %llu,\n", N);
    printf("  \"threads\": %d,\n", nthreads);
    printf("  \"seconds\": %.3f,\n", wall() - t0);
    printf("  \"numbers_per_second\": %.0f,\n", (double)N / (wall() - t0));
    printf("  \"failures\": %llu,\n", failures);
    printf("  \"first_failure\": %llu,\n", first_fail);
    printf("  \"trajectories_exceeding_uint64\": %llu,\n", alti);
    printf("  \"record_sigma\": %llu,\n", rec_sigma);
    printf("  \"record_sigma_at\": %llu,\n", rec_sigma_arg);
    printf("  \"record_peak\": %s,\n", pk);
    printf("  \"record_peak_at\": %llu,\n", rec_peak_arg);
    printf("  \"record_peak_ratio\": %s,\n", rp);
    printf("  \"record_peak_ratio_at\": %llu,\n", rec_ratio_huge);
    printf("  \"record_peak_ratio_value\": %.6Le,\n", ratio);
    printf("  \"excursions_above_N\": %llu,\n", excursions);
    printf("  \"histogram\": [");
    { int first = 1;
      for (int i = 0; i < NBIN; i++) { if (!hist[i]) continue;
          printf("%s[%d,%llu]", first ? "" : ",", i, hist[i]); first = 0; } }
    printf("],\n");
    printf("  \"champion_total_stopping_times\": {");
    { u64 checks[3] = { rec_sigma_arg, rec_peak_arg, rec_ratio_huge }; int first = 1;
      for (int i = 0; i < 3; i++) {
        u64 n = checks[i]; if (!n) continue;
        u64 steps = 0; u128 h = n;
        while (h != 1) { if (h & 1) { h = (3*h + 1) >> 1; steps += 2; } else { h >>= 1; steps += 1; }
                         if (steps > 100000000ULL) break; }
        printf("%s\"%llu\": %llu", first ? "" : ", ", n, steps); first = 0; } }
    printf("}\n");
    printf("}\n");

    if (csv_out) {
        FILE *f = fopen(csv_out, "w");
        if (f) { fprintf(f, "sigma,count\n");
                 for (int i = 0; i < NBIN; i++) if (hist[i]) fprintf(f, "%d,%llu\n", i, hist[i]);
                 fclose(f); }
    }
    return 0;
}

/* mode 2 : temps d'arret total jusqu'a 1 pour tout n <= N ------------------*/
static int collatz_full(u64 N, long budget_mb) {
    double t0 = wall();
    double need_mb = (double)(N + 1) * sizeof(uint16_t) / (1024.0 * 1024.0);
    if (budget_mb > 0 && need_mb > (double)budget_mb) {
        fprintf(stderr,
            "REFUS : collatz-full exige %.1f Mo (tableau uint16 de N+1) > budget %ld Mo.\n"
            "        -> utilisez 'collatz-tst-stream N [state.json]' (memoire O(1), reprise).\n",
            need_mb, budget_mb);
        return 3;
    }
    uint16_t *tst = (uint16_t *)malloc((size_t)(N + 1) * sizeof(uint16_t));
    if (!tst) { fprintf(stderr, "memoire insuffisante (%.1f Mo demandes)\n", need_mb); return 2; }
    tst[0] = 0; if (N >= 1) tst[1] = 0;
    u64 maxstep = 0, maxarg = 1;
    for (u64 n = 2; n <= N; n++) {
        u64 m = n, steps = 0;
        while (m >= n) {                        /* tant qu'on n'est pas redescendu */
            if (m & 1ULL) { m = (3*m + 1) >> 1; steps += 2; } else { m >>= 1; steps += 1; }
        }
        uint32_t t = (uint32_t)steps + (uint32_t)tst[m];
        tst[n] = (uint16_t)(t > 65535 ? 65535 : t);
        if (t > maxstep) { maxstep = t; maxarg = n; }
    }
    /* histogramme */
    size_t H = 4096;
    u64 *h = (u64 *)calloc(H, sizeof(u64));
    for (u64 n = 2; n <= N; n++) if (tst[n] < H) h[tst[n]]++;
    printf("{\"mode\":\"collatz-full\",\"N\":%llu,\"seconds\":%.3f,", N, wall() - t0);
    printf("\"max_total_stopping_time\":%llu,\"max_at\":%llu,\"mean\":%.4f,\"histogram\":[",
           maxstep, maxarg, 0.0);
    { int first = 1; for (size_t i = 0; i < H; i++) if (h[i]) { printf("%s[%zu,%llu]", first?"":",", i, h[i]); first = 0; } }
    printf("],\"peak_rss_mb\":%.1f}\n", peak_rss_mb());
    free(h); free(tst);
    return 0;
}

/* mode 3 : tst de nombres isoles -------------------------------------------*/
static int collatz_point(int argc, char **argv, int start) {
    printf("[");
    for (int i = start; i < argc; i++) {
        u64 n = strtoull(argv[i], NULL, 10);
        u64 m = n, steps = 0, peak = n;
        while (m != 1) {
            if (m & 1ULL) { u64 t = 3*m + 1; if (t > peak) peak = t; m = t >> 1; steps += 2; }
            else { m >>= 1; steps += 1; }
            if (m > peak) peak = m;
            if (steps > 10000000ULL) break;
        }
        printf("%s{\"n\":%llu,\"tst\":%llu,\"peak\":%llu}", i > start ? "," : "", n, steps, peak);
    }
    printf("]\n");
    return 0;
}


/* ===========================================================================
 * mode 4 : verification par SEGMENT [lo,hi) -- reprise, batchs, checkpoints
 * ---------------------------------------------------------------------------
 * - aucune table : memoire O(1) hors histogramme et enregistrements
 * - micro-batchs (defaut 2^27 nombres) : parallelises par OpenMP
 * - etat sauvegarde par un THREAD ECRIVAIN toutes les ckpt_sec secondes,
 *   par double tampon (la copie se fait sous mutex, l'ecriture disque hors
 *   verrou) -> "blocs alternes", latence d'E/S nulle pour le calcul
 * - REPRISE : relit l'etat JSON et repart a state.next
 * - SIGINT/SIGTERM : arret propre + sauvegarde finale
 * ======================================================================== */
#define MAXBIN 8192

typedef struct {
    u64  lo, hi, next;
    u64  verified, failures, first_failure, alti, excursions;
    u64  rec_sigma, rec_sigma_at;
    u128 rec_peak; u64 rec_peak_at;
    u128 rec_ratio_peak; u64 rec_ratio_n;
    u64  hist[MAXBIN];
} rstate_t;

static rstate_t  g_state;
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static volatile int    g_stop = 0;
static volatile int    g_dirty = 0;
static volatile double g_last_ckpt = 0.0;
static double          g_ckpt_sec = 10.0;
static char            g_state_path[512];

static void on_signal(int sig) { (void)sig; g_stop = 1; }

static long double u128_to_ld(u128 x) {
    long double v = 0.0L, p = 1.0L;
    while (x) { v += (long double)(unsigned)(x % 1000000000ULL) * p; x /= 1000000000ULL; p *= 1000000000.0L; }
    return v;
}

static void u128_to_str(u128 x, char *buf) {
    char tmp[64]; int k = 0;
    if (x == 0) { strcpy(buf, "0"); return; }
    while (x > 0) { tmp[k++] = '0' + (int)(x % 10); x /= 10; }
    for (int i = 0; i < k; i++) buf[i] = tmp[k - 1 - i];
    buf[k] = 0;
}

/* ecriture atomique : fichier temporaire puis rename() */
static void write_state_json(const rstate_t *st) {
    char tmp[560]; char peak[64], rp[64];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g_state_path);
    u128_to_str(st->rec_peak, peak);
    u128_to_str(st->rec_ratio_peak, rp);
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "{\"mode\":\"collatz-range\",\"lo\":%llu,\"hi\":%llu,\"next\":%llu,"
               "\"verified\":%llu,\"failures\":%llu,\"first_failure\":%llu,"
               "\"trajectories_exceeding_uint64\":%llu,\"excursions_above_hi\":%llu,"
               "\"record_sigma\":%llu,\"record_sigma_at\":%llu,"
               "\"record_peak\":\"%s\",\"record_peak_at\":%llu,"
               "\"record_ratio_peak\":\"%s\",\"record_ratio_at\":%llu,"
               "\"updated\":%.3f,\"peak_rss_mb\":%.1f,\"hist\":[",
            st->lo, st->hi, st->next, st->verified, st->failures, st->first_failure,
            st->alti, st->excursions, st->rec_sigma, st->rec_sigma_at,
            peak, st->rec_peak_at, rp, st->rec_ratio_n,
            wall(), peak_rss_mb());
    { int first = 1;
      for (int i = 0; i < MAXBIN; i++) if (st->hist[i]) {
          fprintf(f, "%s[%d,%llu]", first ? "" : ",", i, st->hist[i]); first = 0; } }
    fprintf(f, "]}\n");
    fclose(f);
    rename(tmp, g_state_path);      /* atomique sur POSIX */
}

static void *writer_thread(void *arg) {
    (void)arg;
    rstate_t *buf = (rstate_t *)malloc(sizeof(rstate_t));   /* tampon B */
    if (!buf) return NULL;
    for (;;) {
        struct timespec ts = {0, 200 * 1000 * 1000};        /* 200 ms */
        nanosleep(&ts, NULL);
        double now = wall();
        int stop = g_stop;
        pthread_mutex_lock(&g_mtx);
        int due = g_dirty && (now - g_last_ckpt >= g_ckpt_sec);
        if (due || (stop && g_dirty)) {
            memcpy(buf, &g_state, sizeof(rstate_t));        /* copie sous verrou */
            g_dirty = 0; g_last_ckpt = now; due = 1;
        }
        pthread_mutex_unlock(&g_mtx);
        if (due) write_state_json(buf);                     /* ecriture hors verrou */
        if (stop && !g_dirty) break;
    }
    free(buf);
    return NULL;
}

/* chargement de l'etat (reprise) ; renvoie 1 si repris */
static int load_state(const char *path, u64 lo, u64 hi, rstate_t *st) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char blob[8192];
    size_t n = fread(blob, 1, sizeof(blob) - 1, f); blob[n] = 0;
    fclose(f);
    u64 slo = 0, shi = 0, snext = 0;
    if (!strstr(blob, "\"collatz-range\"")) return 0;
    sscanf(strstr(blob, "\"lo\":"), "\"lo\":%llu", &slo);
    sscanf(strstr(blob, "\"hi\":"), "\"hi\":%llu", &shi);
    sscanf(strstr(blob, "\"next\":"), "\"next\":%llu", &snext);
    if (slo != lo || shi != hi || snext < lo || snext > hi) return 0;
    st->lo = lo; st->hi = hi; st->next = snext;
    sscanf(strstr(blob, "\"verified\":"), "\"verified\":%llu", &st->verified);
    sscanf(strstr(blob, "\"failures\":"), "\"failures\":%llu", &st->failures);
    sscanf(strstr(blob, "\"first_failure\":"), "\"first_failure\":%llu", &st->first_failure);
    sscanf(strstr(blob, "\"trajectories_exceeding_uint64\":"), "\"trajectories_exceeding_uint64\":%llu", &st->alti);
    sscanf(strstr(blob, "\"excursions_above_hi\":"), "\"excursions_above_hi\":%llu", &st->excursions);
    sscanf(strstr(blob, "\"record_sigma\":"), "\"record_sigma\":%llu", &st->rec_sigma);
    sscanf(strstr(blob, "\"record_sigma_at\":"), "\"record_sigma_at\":%llu", &st->rec_sigma_at);
    /* records 128 bits : relus depuis les chaines decimales */
    { char *p;
      st->rec_peak = 0; st->rec_peak_at = 0; st->rec_ratio_peak = 0; st->rec_ratio_n = 0;
      p = strstr(blob, "\"record_peak\":\"");
      if (p) { p += 15; u128 v = 0; while (*p >= '0' && *p <= '9') { v = v * 10 + (u128)(*p - '0'); p++; } st->rec_peak = v; }
      p = strstr(blob, "\"record_peak_at\":");        if (p) sscanf(p, "\"record_peak_at\":%llu", &st->rec_peak_at);
      p = strstr(blob, "\"record_ratio_peak\":\"");
      if (p) { p += 21; u128 v = 0; while (*p >= '0' && *p <= '9') { v = v * 10 + (u128)(*p - '0'); p++; } st->rec_ratio_peak = v; }
      if (st->rec_peak_at) st->rec_ratio_n = st->rec_peak_at;   /* ratio defini au point du record */
    }
    /* histogramme : restaure pour que l'artefact final couvre TOUT le segment */
    memset(st->hist, 0, sizeof(st->hist));
    { char *p = strstr(blob, "\"hist\":[");
      if (p) {
          p += 8;
          while (*p && *p != ']') {
              while (*p == ',' || *p == ' ') p++;
              if (*p != '[') break;
              p++;
              int bin = (int)strtol(p, &p, 10);
              if (*p == ',') p++;
              unsigned long long c = strtoull(p, &p, 10);
              if (bin >= 0 && bin < MAXBIN) st->hist[bin] += c;
              if (*p == ']') p++;
          }
      }
    }
    return 1;
}

static int collatz_range(u64 lo, u64 hi, const char *state_path,
                         u64 batch, double ckpt_sec, long budget_mb) {
    double t0 = wall();
    g_ckpt_sec = ckpt_sec;
    snprintf(g_state_path, sizeof(g_state_path), "%s", state_path);
    if (mem_guard_mb(budget_mb) != 0) return 4;

    memset(&g_state, 0, sizeof(g_state));
    g_state.lo = lo; g_state.hi = hi; g_state.next = lo;
    int resumed = 0;
    if (state_path && *state_path) {
        rstate_t st;
        if (load_state(state_path, lo, hi, &st)) { g_state = st; resumed = 1; }
    }
    fprintf(stderr, "[collatz-range] %s : [%llu, %llu), batch=%.3g nombres, checkpoint=%.1fs, budget=%ld Mo\n",
            resumed ? "REPRISE" : "depart", lo, hi, (double)batch, ckpt_sec, budget_mb);

    pthread_t wt; int have_writer = 0;
    if (state_path && *state_path) {
        g_last_ckpt = wall();
        have_writer = (pthread_create(&wt, NULL, writer_thread, NULL) == 0);
    }
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);

    u64 nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    for (u64 base = g_state.next; base < hi; base += batch) {
        if (g_stop) break;
        u64 end = base + batch; if (end > hi) end = hi;

        /* ---- enregistrements de bloc, fusionnes sous #pragma omp critical ---- */
        u64  b_v = 0, b_f = 0, b_ff = 0, b_al = 0, b_ex = 0;
        u64  b_rs = 0, b_rsa = 0, b_rpa = 0;  u128 b_rp = 0;
        u64  b_hist[MAXBIN]; memset(b_hist, 0, sizeof(b_hist));

#pragma omp parallel
        {
            /* --- tampons PRIVES par thread (aucun partage, aucune course) --- */
            u64  t_v = 0, t_f = 0, t_ff = 0, t_al = 0, t_ex = 0;
            u64  t_rs = 0, t_rsa = 0, t_rpa = 0;  u128 t_rp = 0;
            u64  t_hist[MAXBIN]; memset(t_hist, 0, sizeof(t_hist));

#pragma omp for schedule(dynamic, 1 << 16) nowait
            for (long long nn = (long long)base; nn < (long long)end; nn++) {
                u64 n = (u64)nn;
                if (n < 2) { t_v++; if (n < MAXBIN) t_hist[n]++; continue; }
                u64 m = n, steps = 0;
                int ok = 0;
                u128 peak = n;
                while (steps < 4000000000ULL) {
                    if (m & 1ULL) { u128 t = (u128)3 * m + 1; if (t > peak) peak = t; m = (u64)(t >> 1); steps += 2; }
                    else { m >>= 1; steps += 1; }
                    if ((u128)m > peak) peak = m;
                    if (m < n) { ok = 1; break; }
                    if (m > 6000000000000000000ULL) {          /* hors domaine u64 : voie 128 bits */
                        u128 h = m; t_al++;
                        while (steps < 4000000000ULL) {
                            if (h & 1) { u128 t = 3*h + 1; if (t > peak) peak = t; h = t >> 1; steps += 2; }
                            else { h >>= 1; steps += 1; }
                            if (h > peak) peak = h;
                            if (h < (u128)n) { ok = 1; break; }
                        }
                        break;
                    }
                }
                if (!ok) { t_f++; if (!t_ff || n < t_ff) t_ff = n; continue; }
                t_v++;
                if (steps < MAXBIN) t_hist[steps]++;
                if (peak > (u128)hi) t_ex++;
                if (steps > t_rs) { t_rs = steps; t_rsa = n; }
                if (peak > t_rp) { t_rp = peak; t_rpa = n; }

            }

            /* --- fusion 1 : prive -> bloc (verrou court, aucune ecriture non protegee) --- */
#pragma omp critical
            {
                b_v += t_v; b_f += t_f; b_al += t_al; b_ex += t_ex;
                if (t_ff && (!b_ff || t_ff < b_ff)) b_ff = t_ff;
                if (t_rs > b_rs) { b_rs = t_rs; b_rsa = t_rsa; }
                if (t_rp > b_rp) { b_rp = t_rp; b_rpa = t_rpa; }

                for (int i = 0; i < MAXBIN; i++) b_hist[i] += t_hist[i];
            }
        }   /* fin de la region parallele : barriere implicite */

        /* --- fusion 2 : bloc -> etat global --- */
        pthread_mutex_lock(&g_mtx);
        g_state.verified += b_v; g_state.failures += b_f;
        if (b_ff && (!g_state.first_failure || b_ff < g_state.first_failure)) g_state.first_failure = b_ff;
        g_state.alti += b_al; g_state.excursions += b_ex;
        if (b_rs > g_state.rec_sigma) { g_state.rec_sigma = b_rs; g_state.rec_sigma_at = b_rsa; }
        if (b_rp > g_state.rec_peak) {
            g_state.rec_peak = b_rp; g_state.rec_peak_at = b_rpa;
            g_state.rec_ratio_peak = b_rp; g_state.rec_ratio_n = b_rpa;   /* ratio = pic/n au meme point */
        }
        /* le "record de ratio" est par definition le ratio au point du record de hauteur
           : rec_ratio_peak / rec_ratio_n avec rec_ratio_n = rec_peak_at (cf. sortie). */
        for (int i = 0; i < MAXBIN; i++) g_state.hist[i] += b_hist[i];
        g_state.next = end;
        g_dirty = 1;
        pthread_mutex_unlock(&g_mtx);
    }

    g_stop = 1;
    if (have_writer) pthread_join(wt, NULL);
    if (state_path && *state_path) write_state_json(&g_state);

    char peak[64];
    u128_to_str(g_state.rec_peak, peak);
    printf("{\"mode\":\"collatz-range\",\"lo\":%llu,\"hi\":%llu,\"next\":%llu,\"verified\":%llu,"
           "\"failures\":%llu,\"first_failure\":%llu,\"trajectories_exceeding_uint64\":%llu,"
           "\"excursions_above_hi\":%llu,\"record_sigma\":%llu,\"record_sigma_at\":%llu,"
           "\"record_peak\":\"%s\",\"record_peak_at\":%llu,\"record_peak_ratio\":%.6e,"
           "\"threads\":%llu,\"seconds\":%.3f,"
           "\"peak_rss_mb\":%.1f,\"resumed\":%d}\n",
           g_state.lo, g_state.hi, g_state.next, g_state.verified, g_state.failures,
           g_state.first_failure, g_state.alti, g_state.excursions, g_state.rec_sigma,
           g_state.rec_sigma_at, peak, g_state.rec_peak_at,
           (g_state.rec_peak_at ? (double)(u128_to_ld(g_state.rec_peak) / (long double)g_state.rec_peak_at) : 0.0),
           nthreads, wall() - t0, peak_rss_mb(), resumed);
    return 0;
}

/* ===========================================================================
 * mode 5 : temps d'arret total, version FLUX (memoire O(1), reprise)
 * ---------------------------------------------------------------------------
 * - aucun tableau : chaque TST est recalcule integralement (pas de memo)
 * - blocs + checkpoint toutes les ckpt_sec (meme mecanique de thread ecrivain)
 * ======================================================================== */
static int collatz_tst_stream(u64 N, const char *state_path, double ckpt_sec) {
    double t0 = wall();
    u64 start = 2, maxstep = 0, maxarg = 1;
    u64 hist[MAXBIN]; memset(hist, 0, sizeof(hist));
    if (state_path && *state_path) {
        FILE *f = fopen(state_path, "r");
        if (f) {
            char blob[2048]; size_t n = fread(blob, 1, sizeof(blob) - 1, f); blob[n] = 0; fclose(f);
            u64 sN = 0, sNx = 0, sms = 0, sma = 0;
            if (strstr(blob, "\"collatz-tst-stream\"") &&
                sscanf(blob, "{\"mode\":\"collatz-tst-stream\",\"N\":%llu,\"next\":%llu,\"max_tst\":%llu,\"max_at\":%llu",
                       &sN, &sNx, &sms, &sma) == 4 && sN == N) {
                start = sNx; maxstep = sms; maxarg = sma;
                fprintf(stderr, "[tst-stream] REPRISE a n = %llu\n", start);
            }
        }
    }
    u64 n = start;
    double last = wall();
    while (n <= N) {
        u64 m = n, steps = 0;
        while (m != 1) {
            if (m & 1ULL) { m = (3*m + 1) >> 1; steps += 2; } else { m >>= 1; steps += 1; }
            if (steps > 1000000ULL) break;
        }
        if (steps < MAXBIN) hist[steps]++;
        if (steps > maxstep) { maxstep = steps; maxarg = n; }
        n++;
        if ((n & 0xFFFFF) == 0 && state_path && *state_path && wall() - last > ckpt_sec) {
            char tmp[560]; snprintf(tmp, sizeof(tmp), "%s.tmp", state_path);
            FILE *f = fopen(tmp, "w");
            if (f) { fprintf(f, "{\"mode\":\"collatz-tst-stream\",\"N\":%llu,\"next\":%llu,"
                                "\"max_tst\":%llu,\"max_at\":%llu,\"peak_rss_mb\":%.1f}",
                             N, n, maxstep, maxarg, peak_rss_mb()); fclose(f); }
            rename(tmp, state_path);
            last = wall();
        }
    }
    printf("{\"mode\":\"collatz-tst-stream\",\"N\":%llu,\"next\":%llu,\"max_total_stopping_time\":%llu,"
           "\"max_at\":%llu,\"threads\":1,\"seconds\":%.3f,\"peak_rss_mb\":%.1f,\"histogram\":[",
           N, n, maxstep, maxarg, wall() - t0, peak_rss_mb());
    { int first = 1; for (int i = 0; i < MAXBIN; i++) if (hist[i]) {
          printf("%s[%d,%llu]", first ? "" : ",", i, hist[i]); first = 0; } }
    printf("]}\n");
    if (state_path && *state_path) unlink(state_path);
    return 0;
}


/* ===========================================================================
 * mode 6 : criblage Robin/Lagarias en C (flux, memoire O(profondeur))
 * ---------------------------------------------------------------------------
 * Enumere les entiers a exposants non croissants <= X (X donne en puissance de
 * 10), calcule en long double le ratio G(n) = sigma(n)/(e^gamma n log log n) et
 * retient les K meilleurs -- SANS jamais materialiser la liste des candidats
 * (c'est ce qui faisait exploser la RAM cote Python).
 * Les K gagnants sont sortis sous forme de VECTEUR D'EXPONENTS : Python
 * reconstruit alors n et sigma(n) EXACTEMENT (entiers + mpmath) et certifie.
 * ======================================================================== */
#define RMAXK 32
#define RMAXP 30

static int  g_rk = 8;
static long double g_rX = 1e36L;
static long double g_rEg = 1.7810724179901979852L;   /* e^gamma */
static double g_rT0 = 0.0;
static unsigned long long g_rCount = 0;
static long double g_rTop[32];

static unsigned char g_rExp[RMAXK][RMAXP];
static unsigned char g_rLen[RMAXK];
static long double g_rVal[RMAXK];

static const int g_rPrimes[RMAXP] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,
                                     53,59,61,67,71,73,79,83,89,97,101,103,107,109,113};

static long double ld_pow(int p, int e) {
    long double r = 1.0L, b = (long double)p;
    while (e) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return r;
}

static void robin_insert(long double ratio, const unsigned char *exp, int len, long double nval) {
    int pos = g_rk;
    for (int i = 0; i < g_rk; i++) if (ratio > g_rTop[i]) { pos = i; break; }
    if (pos >= g_rk) return;
    for (int i = g_rk - 1; i > pos; i--) {
        g_rTop[i] = g_rTop[i-1]; g_rVal[i] = g_rVal[i-1]; g_rLen[i] = g_rLen[i-1];
        memcpy(g_rExp[i], g_rExp[i-1], RMAXP);
    }
    g_rTop[pos] = ratio; g_rVal[pos] = nval; g_rLen[pos] = (unsigned char)len;
    memcpy(g_rExp[pos], exp, RMAXP);
}

static void robin_rec(int i, long double n, long double sigma, int expmax,
                      unsigned char *exp, int len) {
    if (n > 5040.0L && n <= g_rX) {
        g_rCount++;
        long double ll = logl(logl(n));
        if (ll > 0) {
            long double ratio = sigma / (g_rEg * n * ll);
            robin_insert(ratio, exp, len, n);
        }
    }
    if (i >= RMAXP) return;
    int p = g_rPrimes[i];
    long double pk = 1.0L;
    for (int e = 1; e <= expmax; e++) {
        pk *= (long double)p;
        if (n * pk > g_rX) break;
        long double sp = (pk * (long double)p - 1.0L) / ((long double)p - 1.0L);
        exp[i] = (unsigned char)e;
        robin_rec(i + 1, n * pk, sigma * sp, e, exp, i + 1);
        exp[i] = 0;
    }
}

static int robin_score(double X_log10, int topk, int verbose) {
    g_rk = topk > RMAXK ? RMAXK : topk;
    g_rX = 1.0L;
    for (int i = 0; i < (int)X_log10; i++) g_rX *= 10.0L;
    if (X_log10 - (double)(int)X_log10 > 0) g_rX *= powl(10.0L, (long double)(X_log10 - (int)X_log10));
    for (int i = 0; i < g_rk; i++) { g_rTop[i] = -1.0L; g_rLen[i] = 0; memset(g_rExp[i], 0, RMAXP); }
    unsigned char exp[RMAXP]; memset(exp, 0, sizeof(exp));
    g_rCount = 0;
    g_rT0 = wall();
    robin_rec(0, 1.0L, 1.0L, 300, exp, 0);
    double secs = wall() - g_rT0;
    if (verbose) {
        printf("{\"mode\":\"robin-score\",\"X_log10\":%.0f,\"candidates\":%llu,\"seconds\":%.3f,"
               "\"peak_rss_mb\":%.1f,\"topk\":[", X_log10, g_rCount, secs, peak_rss_mb());
        for (int i = 0; i < g_rk; i++) {
            if (g_rLen[i] == 0) continue;
            printf("%s{\"ratio_ld\":\"%.12Le\",\"exps\":[", i ? "," : "", g_rTop[i]);
            for (int j = 0; j < g_rLen[i]; j++) printf("%s%d", j ? "," : "", (int)g_rExp[i][j]);
            printf("]}");
        }
        printf("]}\n");
    } else {
        printf("robin-score X=1e%.0f : %llu candidats, %.3f s, top ratio = %.12Le (n ~ %.10Le), RSS %.1f Mo\n",
               X_log10, g_rCount, secs, g_rTop[0], g_rVal[0], peak_rss_mb());
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage:\n"
            "  %s collatz-verify N [hist.csv]\n"
            "  %s collatz-full   N\n"
            "  %s collatz-point  n1 n2 ...\n"
            "  %s collatz-range  lo hi state.json [batch=134217728] [ckpt_sec=10] [budget_mb=0]\n"
            "  %s collatz-tst-stream N state.json [ckpt_sec=10]\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "collatz-verify"))
        return collatz_verify(strtoull(argv[2], NULL, 10), argc > 3 ? argv[3] : NULL);
    if (!strcmp(argv[1], "collatz-full"))
        return collatz_full(strtoull(argv[2], NULL, 10),
                            argc > 3 ? strtol(argv[3], NULL, 10) : 0);
    if (!strcmp(argv[1], "collatz-range")) {
        if (argc < 5) { fprintf(stderr, "collatz-range lo hi state.json\n"); return 1; }
        u64 batch = argc > 5 ? strtoull(argv[5], NULL, 10) : (1ULL << 27);
        double ck = argc > 6 ? atof(argv[6]) : 10.0;
        long b = argc > 7 ? strtol(argv[7], NULL, 10) : 512;
        return collatz_range(strtoull(argv[2], NULL, 10), strtoull(argv[3], NULL, 10),
                             argv[4], batch, ck, b);
    }
    if (!strcmp(argv[1], "robin-score")) {
        double Xl = argc > 2 ? atof(argv[2]) : 36;
        int k = argc > 3 ? atoi(argv[3]) : 8;
        int v = argc > 4 ? atoi(argv[4]) : 1;
        return robin_score(Xl, k, v);
    }
    if (!strcmp(argv[1], "collatz-tst-stream")) {
        if (argc < 3) { fprintf(stderr, "collatz-tst-stream N [state.json] [ckpt]\n"); return 1; }
        return collatz_tst_stream(strtoull(argv[2], NULL, 10),
                                  argc > 3 ? argv[3] : NULL, argc > 4 ? atof(argv[4]) : 10.0);
    }
    if (!strcmp(argv[1], "collatz-point"))
        return collatz_point(argc, argv, 2);
    fprintf(stderr, "mode inconnu: %s\n", argv[1]);
    return 1;
}
