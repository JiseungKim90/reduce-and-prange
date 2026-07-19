// Full end-to-end RP pass simulation over a prime field.
//
// This checks that the recursive pass-success probability G_1 matches an
// actual bounded RP implementation: generate A, s, sparse error e, run the
// nested fixed-budget search, solve the final information set, and verify the
// full residual weight.
//
// Build:
//   g++ -O3 -std=c++17 -pthread rp_full_e2e.cpp -o rp_full_e2e

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::int64_t;
using std::string;
using std::vector;

static constexpr int64_t PRIME = 2147483647LL;

struct Case {
    string name;
    int m;
    int n;
    int t;
    vector<int> ns;
    int trials;
};

struct GlobalStats {
    std::atomic<int64_t> trials{0};
    std::atomic<int64_t> clean_leaf_success{0};
    std::atomic<int64_t> e2e_success{0};
    std::atomic<int64_t> false_accept{0};
    std::atomic<int64_t> clean_rank_fail{0};
    std::atomic<int64_t> leaves{0};
    std::atomic<int64_t> solves{0};
};

struct Instance {
    int m;
    int n;
    int t;
    vector<int64_t> a;
    vector<int64_t> s;
    vector<int64_t> b;
    vector<char> is_error;
};

struct PassState {
    bool saw_clean_leaf = false;
    bool e2e_success = false;
    bool false_accept = false;
    int64_t clean_rank_fail = 0;
    int64_t leaves = 0;
    int64_t solves = 0;
};

static int64_t add_mod(int64_t x, int64_t y) {
    int64_t z = x + y;
    if (z >= PRIME) z -= PRIME;
    return z;
}

static int64_t sub_mod(int64_t x, int64_t y) {
    int64_t z = x - y;
    if (z < 0) z += PRIME;
    return z;
}

static int64_t mul_mod(int64_t x, int64_t y) {
    return (x * y) % PRIME;
}

static int64_t pow_mod(int64_t a, int64_t e) {
    int64_t r = 1;
    while (e > 0) {
        if (e & 1) r = mul_mod(r, a);
        a = mul_mod(a, a);
        e >>= 1;
    }
    return r;
}

static long double lbinom(int n, int k) {
    if (k < 0 || k > n) return -INFINITY;
    return lgammal((long double)n + 1.0L) - lgammal((long double)k + 1.0L) -
           lgammal((long double)(n - k) + 1.0L);
}

static long double stage_prob(int m, int t, int mu, int ni) {
    return expl(lbinom(m - mu - t, ni) - lbinom(m - mu, ni));
}

static vector<long double> stage_probs(const Case& c) {
    vector<long double> ps;
    int mu = 0;
    for (int ni : c.ns) {
        ps.push_back(stage_prob(c.m, c.t, mu, ni));
        mu += ni;
    }
    return ps;
}

static vector<int> reps_from_probs(const vector<long double>& ps) {
    vector<int> reps;
    for (long double p : ps) reps.push_back((int)ceill(1.0L / p));
    return reps;
}

static long double recursive_success(const vector<long double>& ps, const vector<int>& reps) {
    long double g = 1.0L - powl(1.0L - ps.back(), reps.back());
    for (int i = (int)ps.size() - 2; i >= 0; --i) {
        g = 1.0L - powl(1.0L - ps[i] * g, reps[i]);
    }
    return g;
}

static vector<int> sample_subset(const vector<int>& available, int k, std::mt19937_64& rng) {
    vector<int> tmp = available;
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)tmp.size() - 1);
        int j = dist(rng);
        std::swap(tmp[i], tmp[j]);
    }
    tmp.resize(k);
    return tmp;
}

static vector<int> remove_chosen(const vector<int>& available, const vector<int>& chosen) {
    vector<char> mark;
    int maxv = *std::max_element(available.begin(), available.end());
    mark.assign(maxv + 1, 0);
    for (int x : chosen) mark[x] = 1;
    vector<int> out;
    out.reserve(available.size() - chosen.size());
    for (int x : available) {
        if (!mark[x]) out.push_back(x);
    }
    return out;
}

static Instance make_instance(const Case& c, std::mt19937_64& rng) {
    std::uniform_int_distribution<int64_t> field(0, PRIME - 1);
    std::uniform_int_distribution<int64_t> nonzero(1, PRIME - 1);

    Instance inst;
    inst.m = c.m;
    inst.n = c.n;
    inst.t = c.t;
    inst.a.assign((size_t)c.m * c.n, 0);
    inst.s.assign(c.n, 0);
    inst.b.assign(c.m, 0);
    inst.is_error.assign(c.m, 0);

    for (auto& x : inst.a) x = field(rng);
    for (auto& x : inst.s) x = field(rng);

    vector<int> coords(c.m);
    std::iota(coords.begin(), coords.end(), 0);
    vector<int> err = sample_subset(coords, c.t, rng);
    for (int idx : err) inst.is_error[idx] = 1;

    for (int row = 0; row < c.m; ++row) {
        int64_t acc = 0;
        for (int col = 0; col < c.n; ++col) {
            acc = add_mod(acc, mul_mod(inst.a[(size_t)row * c.n + col], inst.s[col]));
        }
        if (inst.is_error[row]) acc = add_mod(acc, nonzero(rng));
        inst.b[row] = acc;
    }
    return inst;
}

static bool solve_on_rows(const Instance& inst, const vector<int>& rows, vector<int64_t>& sol) {
    int n = inst.n;
    vector<int64_t> mat((size_t)n * (n + 1), 0);
    for (int i = 0; i < n; ++i) {
        int row = rows[i];
        for (int j = 0; j < n; ++j) mat[(size_t)i * (n + 1) + j] = inst.a[(size_t)row * n + j];
        mat[(size_t)i * (n + 1) + n] = inst.b[row];
    }

    for (int col = 0; col < n; ++col) {
        int pivot = -1;
        for (int r = col; r < n; ++r) {
            if (mat[(size_t)r * (n + 1) + col] != 0) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) return false;
        if (pivot != col) {
            for (int j = col; j <= n; ++j) {
                std::swap(mat[(size_t)pivot * (n + 1) + j], mat[(size_t)col * (n + 1) + j]);
            }
        }
        int64_t inv = pow_mod(mat[(size_t)col * (n + 1) + col], PRIME - 2);
        for (int j = col; j <= n; ++j) mat[(size_t)col * (n + 1) + j] = mul_mod(mat[(size_t)col * (n + 1) + j], inv);
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            int64_t factor = mat[(size_t)r * (n + 1) + col];
            if (factor == 0) continue;
            for (int j = col; j <= n; ++j) {
                mat[(size_t)r * (n + 1) + j] = sub_mod(mat[(size_t)r * (n + 1) + j],
                                                       mul_mod(factor, mat[(size_t)col * (n + 1) + j]));
            }
        }
    }

    sol.assign(n, 0);
    for (int i = 0; i < n; ++i) sol[i] = mat[(size_t)i * (n + 1) + n];
    return true;
}

static int residual_weight(const Instance& inst, const vector<int64_t>& sol) {
    int wt = 0;
    for (int row = 0; row < inst.m; ++row) {
        int64_t acc = 0;
        for (int col = 0; col < inst.n; ++col) {
            acc = add_mod(acc, mul_mod(inst.a[(size_t)row * inst.n + col], sol[col]));
        }
        if (sub_mod(inst.b[row], acc) != 0) ++wt;
    }
    return wt;
}

static bool recurse_pass(
    const Case& c,
    const vector<int>& reps,
    const Instance& inst,
    int level,
    const vector<int>& available,
    vector<int>& prefix,
    bool prefix_clean,
    std::mt19937_64& rng,
    PassState& st) {
    int ni = c.ns[level];
    for (int rep = 0; rep < reps[level]; ++rep) {
        vector<int> chosen = sample_subset(available, ni, rng);
        bool clean = prefix_clean;
        for (int row : chosen) clean = clean && !inst.is_error[row];
        prefix.insert(prefix.end(), chosen.begin(), chosen.end());

        if (level + 1 == (int)c.ns.size()) {
            st.leaves += 1;
            st.solves += 1;
            if (clean) st.saw_clean_leaf = true;
            vector<int64_t> sol;
            bool ok_rank = solve_on_rows(inst, prefix, sol);
            if (!ok_rank) {
                if (clean) st.clean_rank_fail += 1;
            } else {
                int wt = residual_weight(inst, sol);
                if (wt == c.t) {
                    st.e2e_success = true;
                    if (!clean) st.false_accept = true;
                    prefix.resize(prefix.size() - chosen.size());
                    return true;
                }
            }
        } else {
            vector<int> next_available = remove_chosen(available, chosen);
            if (recurse_pass(c, reps, inst, level + 1, next_available, prefix, clean, rng, st)) {
                prefix.resize(prefix.size() - chosen.size());
                return true;
            }
        }

        prefix.resize(prefix.size() - chosen.size());
    }
    return false;
}

static PassState run_one_pass(const Case& c, const vector<int>& reps, std::mt19937_64& rng) {
    Instance inst = make_instance(c, rng);
    vector<int> available(c.m);
    std::iota(available.begin(), available.end(), 0);
    vector<int> prefix;
    prefix.reserve(c.n);
    PassState st;
    recurse_pass(c, reps, inst, 0, available, prefix, true, rng, st);
    return st;
}

static void worker(const Case& c, const vector<int>& reps, int thread_id, int threads, uint64_t seed, GlobalStats& gs) {
    std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(thread_id + 1));
    for (int trial = thread_id; trial < c.trials; trial += threads) {
        PassState st = run_one_pass(c, reps, rng);
        gs.trials.fetch_add(1, std::memory_order_relaxed);
        if (st.saw_clean_leaf) gs.clean_leaf_success.fetch_add(1, std::memory_order_relaxed);
        if (st.e2e_success) gs.e2e_success.fetch_add(1, std::memory_order_relaxed);
        if (st.false_accept) gs.false_accept.fetch_add(1, std::memory_order_relaxed);
        gs.clean_rank_fail.fetch_add(st.clean_rank_fail, std::memory_order_relaxed);
        gs.leaves.fetch_add(st.leaves, std::memory_order_relaxed);
        gs.solves.fetch_add(st.solves, std::memory_order_relaxed);
    }
}

static string join_ns(const vector<int>& ns) {
    std::ostringstream ss;
    for (size_t i = 0; i < ns.size(); ++i) {
        if (i) ss << ",";
        ss << ns[i];
    }
    return ss.str();
}

static void run_case(const Case& c, int threads, uint64_t seed) {
    vector<long double> ps = stage_probs(c);
    vector<int> reps = reps_from_probs(ps);
    long double g1 = recursive_success(ps, reps);

    GlobalStats gs;
    vector<std::thread> pool;
    for (int tid = 0; tid < threads; ++tid) {
        pool.emplace_back(worker, std::cref(c), std::cref(reps), tid, threads, seed + (uint64_t)c.m * 1009ULL, std::ref(gs));
    }
    for (auto& th : pool) th.join();

    long double trials = (long double)gs.trials.load();
    long double clean = (long double)gs.clean_leaf_success.load() / trials;
    long double e2e = (long double)gs.e2e_success.load() / trials;
    long double ci = 1.96L * sqrtl(std::max((long double)0.0, e2e * (1.0L - e2e)) / trials);
    long double clean_ci = 1.96L * sqrtl(std::max((long double)0.0, clean * (1.0L - clean)) / trials);

    std::cout << std::left << std::setw(16) << c.name
              << std::right << std::setw(4) << c.m
              << std::setw(4) << c.n
              << std::setw(4) << c.t
              << std::setw(3) << c.ns.size()
              << "  " << std::left << std::setw(17) << join_ns(c.ns)
              << std::right << std::setw(8) << c.trials
              << std::setw(9) << (double)g1
              << std::setw(9) << (double)clean
              << std::setw(9) << (double)clean_ci
              << std::setw(9) << (double)e2e
              << std::setw(9) << (double)ci
              << std::setw(9) << (double)(e2e - g1)
              << std::setw(8) << gs.false_accept.load()
              << std::setw(9) << gs.clean_rank_fail.load()
              << std::setw(10) << (double)((long double)gs.solves.load() / trials)
              << "\n";
}

int main(int argc, char** argv) {
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    uint64_t seed = 20260627;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = (uint64_t)std::stoull(argv[++i]);
    }

    vector<Case> cases = {
        {"L2-balanced", 24, 12, 4, {6, 6}, 50000},
        {"L3-balanced", 24, 12, 4, {4, 4, 4}, 50000},
        {"L4-balanced", 24, 12, 4, {3, 3, 3, 3}, 35000},
        {"medium-32", 32, 16, 5, {6, 5, 5}, 30000},
        {"medium-40", 40, 20, 6, {8, 6, 6}, 20000},
        {"front-heavy", 40, 20, 6, {12, 4, 4}, 20000},
        {"back-heavy", 40, 20, 6, {4, 4, 12}, 20000},
        {"low-noise", 60, 30, 3, {10, 10, 10}, 12000},
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "case               m   n   t  L  Ns                 trials       G1    clean cleanCI      e2e    e2eCI   e2e-G1 false rankFail avgSolves\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------------------------\n";
    for (const auto& c : cases) run_case(c, threads, seed);
    std::cout << "Field prime q=" << PRIME << "; false=verified dirty accepts; rankFail=singular clean final systems.\n";
    return 0;
}
