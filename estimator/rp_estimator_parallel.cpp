// Parallel RP/regular-RP estimator for the revised success-probability accounting.
//
// Build examples:
//   g++ -O3 -std=c++17 -pthread rp_estimator_parallel.cpp -o rp_estimator_parallel
//   clang++ -O3 -std=c++17 rp_estimator_parallel.cpp -o rp_estimator_parallel
//   cl /O2 /std:c++17 /EHsc rp_estimator_parallel.cpp
//
// The implementation mirrors rp_estimator_parallel.py.  It evaluates the paper's
// threshold partition family and reports log2(B/G1), where G1 is the recursive
// continuous fixed-budget pass-success probability.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using std::string;
using std::vector;

static constexpr long double ONE_MINUS_INV_E = 1.0L - 1.0L / 2.7182818284590452353602874713526625L;
static constexpr long double OMEGA = 2.8L;
static constexpr long double NEG_INF = -std::numeric_limits<long double>::infinity();

struct Row {
    string table;
    string kind;
    long long m;
    long long n;
    long long t;
    int baseline;
    int paper_rp;
    bool quick;
};

struct Result {
    Row row;
    int levels = 0;
    long double best_delta = 0;
    long double log_cost = 0;
    long double paper_model_log_cost = 0;
    long double gamma = 0;
    long double delta_bits = 0;
    string partition_prefix;
};

static long double logaddexp(long double a, long double b) {
    if (std::isinf(a) && a < 0) return b;
    if (std::isinf(b) && b < 0) return a;
    if (a < b) std::swap(a, b);
    return a + log1pl(expl(b - a));
}

static long double lbinom(long long n, long long k) {
    if (k < 0 || k > n || n < 0) return NEG_INF;
    return lgammal((long double)n + 1.0L) - lgammal((long double)k + 1.0L) -
           lgammal((long double)(n - k) + 1.0L);
}

static long double log_body(long double m_eff, long double n_stage, bool leaf) {
    if (n_stage <= 0) return NEG_INF;
    long double val;
    if (leaf) {
        val = powl(n_stage, OMEGA) + m_eff * n_stage;
    } else {
        val = powl(n_stage, OMEGA) + n_stage * n_stage * (m_eff - n_stage) +
              n_stage * (m_eff - n_stage) * (m_eff - n_stage);
    }
    return logl(val);
}

static long double sd_logq(long long m, long long t, long long n) {
    return lbinom(m - t, n) - lbinom(m, n);
}

static long double sd_stage_logp(long long m, long long t, long long mu_prev, long long n_stage) {
    return lbinom(m - mu_prev - t, n_stage) - lbinom(m - mu_prev, n_stage);
}

static long double reg_logq(long long b, long long k_blocks, long long n) {
    if (n < 0) return NEG_INF;
    long long full = n / b;
    long long rem = n - full * b;
    if (full > k_blocks) return NEG_INF;
    if (full == k_blocks && rem) return NEG_INF;
    long double total = full ? (long double)b * (logl((long double)(k_blocks - full)) - logl((long double)k_blocks)) : 0.0L;
    if (rem) {
        long double x = 1.0L - 1.0L / (long double)(k_blocks - full);
        if (x <= 0) return NEG_INF;
        total += (long double)rem * logl(x);
    }
    return total;
}

static long double pass_success(const vector<long double>& stage_logps) {
    long double g = 1.0L;
    for (auto it = stage_logps.rbegin(); it != stage_logps.rend(); ++it) {
        long double p = expl(*it);
        long double pg = p * g;
        if (pg <= 0) {
            g = 0;
        } else if (p < 1e-8L) {
            g = -expm1l(-g);
        } else {
            long double exponent = log1pl(-pg) / p;
            g = -expm1l(exponent);
        }
        if (g < 0) g = 0;
        if (g > 1) g = 1;
    }
    return g;
}

static vector<long double> geometric_deltas(long long n) {
    long double lo = logl(2.0L);
    long double hi = OMEGA * logl((long double)std::max<long long>(n, 2));
    vector<long double> vals;
    const int grid = 260;
    vals.reserve(grid + 9);
    for (int i = 0; i < grid; ++i) {
        vals.push_back(expl(lo + (hi - lo) * (long double)i / (long double)(grid - 1)));
    }
    for (long double x : {1.0L, 1.25L, 1.5L, 2.0L, 3.0L, 4.0L, 8.0L, 16.0L, 32.0L}) vals.push_back(x);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return vals;
}

static Result eval_row(const Row& row) {
    long long m_eff = row.m;
    long long b = row.t;
    long long k_blocks = 0;
    if (row.kind == "rsd") {
        k_blocks = (row.m + b - 1) / b;
        m_eff = b * k_blocks;
    }

    auto cumulative_logq = [&](long long x) -> long double {
        if (row.kind == "sd") return sd_logq(row.m, row.t, x);
        return reg_logq(b, k_blocks, x);
    };
    auto stage_logp = [&](long long mu, long long x) -> long double {
        if (row.kind == "sd") return sd_stage_logp(row.m, row.t, mu, x);
        return reg_logq(b, k_blocks, mu + x) - reg_logq(b, k_blocks, mu);
    };

    long double log_t = -cumulative_logq(row.n);
    bool have_best = false;
    Result best;
    best.row = row;

    for (long double delta : geometric_deltas(row.n)) {
        long double threshold_log = logl(delta) + log_t;
        vector<long long> parts;
        long long mu = 0;
        while (mu < row.n && (long long)parts.size() < row.n + 1) {
            long long remain = row.n - mu;
            auto term_log = [&](long long x) -> long double {
                return log_body((long double)m_eff - (long double)mu, (long double)x, false) -
                       cumulative_logq(mu + x);
            };
            if (term_log(1) >= threshold_log) break;
            long long lo = 1, hi = remain, keep = 1;
            while (lo <= hi) {
                long long mid = (lo + hi) / 2;
                if (term_log(mid) < threshold_log) {
                    keep = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            parts.push_back(keep);
            mu += keep;
        }
        if (mu < row.n) parts.push_back(row.n - mu);

        long double log_b = NEG_INF;
        vector<long double> stage_logps;
        mu = 0;
        for (size_t i = 0; i < parts.size(); ++i) {
            bool leaf = i + 1 == parts.size();
            long long part = parts[i];
            long long mu_next = mu + part;
            log_b = logaddexp(log_b, log_body((long double)m_eff - (long double)mu, (long double)part, leaf) -
                                         cumulative_logq(mu_next));
            stage_logps.push_back(stage_logp(mu, part));
            mu = mu_next;
        }
        long double gamma = pass_success(stage_logps);
        long double corrected = (log_b - logl(gamma)) / logl(2.0L);
        if (!have_best || corrected < best.log_cost) {
            have_best = true;
            best.levels = (int)parts.size();
            best.best_delta = delta;
            best.log_cost = corrected;
            best.paper_model_log_cost = (log_b - logl(ONE_MINUS_INV_E)) / logl(2.0L);
            best.gamma = gamma;
            best.delta_bits = log2l(ONE_MINUS_INV_E / gamma);
            std::ostringstream ss;
            for (size_t j = 0; j < parts.size() && j < 10; ++j) {
                if (j) ss << ",";
                ss << parts[j];
            }
            if (parts.size() > 10) ss << ",...";
            best.partition_prefix = ss.str();
        }
    }
    return best;
}

static vector<Row> rows() {
    return {
        {"SD-low", "sd", 1LL << 10, 652, 57, 111, 105, false},
        {"SD-low", "sd", 1LL << 12, 1589, 98, 100, 94, false},
        {"SD-low", "sd", 1LL << 14, 3482, 198, 101, 97, false},
        {"SD-low", "sd", 1LL << 16, 7391, 389, 103, 99, false},
        {"SD-low", "sd", 1LL << 18, 15536, 760, 105, 108, false},
        {"SD-low", "sd", 1LL << 20, 32771, 1419, 102, 104, false},
        {"SD-low", "sd", 1LL << 22, 67440, 2735, 104, 107, false},
        {"SD-low", "sd", 1LL << 12, 3072, 44, 117, 111, false},
        {"SD-low", "sd", 1LL << 14, 12288, 39, 111, 107, false},
        {"SD-low", "sd", 1LL << 16, 49152, 34, 107, 104, false},
        {"SD-low", "sd", 1LL << 18, 196608, 32, 108, 106, false},
        {"SD-low", "sd", 1LL << 20, 786432, 31, 112, 110, true},
        {"SD-low", "sd", 1LL << 22, 3145728, 30, 116, 114, true},
        {"SD-low", "sd", 1LL << 24, 12582912, 29, 119, 118, true},
        {"SD-rec", "sd", 1LL << 12, 1321, 172, 128, 121, false},
        {"SD-rec", "sd", 1LL << 14, 2895, 338, 128, 122, false},
        {"SD-rec", "sd", 1LL << 16, 6005, 667, 128, 123, false},
        {"SD-rec", "sd", 1LL << 18, 12160, 1312, 128, 124, false},
        {"SD-rec", "sd", 1LL << 20, 25346, 2467, 128, 124, false},
        {"SD-rec", "sd", 1LL << 22, 50854, 4788, 128, 125, false},
        {"RSD-hi", "rsd", 1LL << 10, 652, 106, 178, 161, false},
        {"RSD-hi", "rsd", 1LL << 12, 1589, 172, 150, 141, false},
        {"RSD-hi", "rsd", 1LL << 14, 3482, 338, 149, 144, false},
        {"RSD-hi", "rsd", 1LL << 16, 7391, 667, 150, 142, false},
        {"RSD-hi", "rsd", 1LL << 18, 15336, 1312, 133, 145, false},
        {"RSD-hi", "rsd", 1LL << 20, 32771, 2467, 131, 148, false},
        {"RSD-hi", "rsd", 1LL << 22, 67440, 4788, 110, 150, false},
        {"RSD-mid", "rsd", 1LL << 10, 652, 57, 107, 105, false},
        {"RSD-mid", "rsd", 1LL << 12, 1589, 98, 99, 91, false},
        {"RSD-mid", "rsd", 1LL << 14, 3482, 198, 101, 94, false},
        {"RSD-mid", "rsd", 1LL << 16, 7391, 389, 103, 97, false},
        {"RSD-mid", "rsd", 1LL << 18, 15336, 760, 105, 100, false},
        {"RSD-mid", "rsd", 1LL << 20, 32771, 1419, 102, 102, false},
        {"RSD-mid", "rsd", 1LL << 22, 67440, 2735, 104, 105, false},
        {"RSD-34", "rsd", 1LL << 12, 3072, 44, 116, 107, false},
        {"RSD-34", "rsd", 1LL << 14, 12288, 39, 111, 105, false},
        {"RSD-34", "rsd", 1LL << 16, 49152, 34, 107, 101, false},
        {"RSD-34", "rsd", 1LL << 18, 196608, 32, 108, 104, false},
        {"RSD-34", "rsd", 1LL << 20, 786432, 31, 112, 110, false},
        {"RSD-34", "rsd", 1LL << 22, 3145728, 30, 116, 120, true},
        {"RSD-34", "rsd", 1LL << 24, 12582912, 29, 119, 124, true},
        {"RSD-rec", "rsd", 1LL << 12, 1377, 172, 128, 120, false},
        {"RSD-rec", "rsd", 1LL << 14, 2909, 338, 128, 118, false},
        {"RSD-rec", "rsd", 1LL << 16, 6091, 667, 128, 118, false},
        {"RSD-rec", "rsd", 1LL << 18, 14796, 1312, 128, 128, false},
        {"RSD-rec", "rsd", 1LL << 20, 30978, 2467, 128, 142, false},
        {"RSD-rec", "rsd", 1LL << 22, 75396, 4788, 128, 165, false},
    };
}

static void print_header() {
    std::cout << std::setw(8) << "table" << std::setw(6) << "log2m" << std::setw(10) << "n"
              << std::setw(7) << "t" << std::setw(6) << "old" << std::setw(9) << "new"
              << std::setw(6) << "ceil" << std::setw(9) << "G1" << std::setw(7) << "+bit"
              << std::setw(8) << "newImp" << "\n";
}

static void print_result(const Result& r) {
    int log2m = (int)std::llround(std::log2((long double)r.row.m));
    int ceil_cost = (int)std::ceil(r.log_cost - 1e-12L);
    long double new_imp = (long double)r.row.baseline - r.log_cost;
    std::cout << std::setw(8) << r.row.table << std::setw(6) << log2m << std::setw(10) << r.row.n
              << std::setw(7) << r.row.t << std::setw(6) << r.row.paper_rp << std::setw(9)
              << (double)r.log_cost << std::setw(6) << ceil_cost << std::setw(9)
              << (double)r.gamma << std::setw(7) << (double)r.delta_bits << std::setw(8)
              << (double)new_imp << "\n";
}

int main(int argc, char** argv) {
    long long max_n = std::numeric_limits<long long>::max();
    bool skip_quick = false;
    string only_table;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--max-n" && i + 1 < argc) max_n = std::stoll(argv[++i]);
        else if (arg == "--skip-quick") skip_quick = true;
        else if (arg == "--table" && i + 1 < argc) only_table = argv[++i];
    }

    vector<Row> tasks;
    for (const auto& r : rows()) {
        if (r.n > max_n) continue;
        if (skip_quick && r.quick) continue;
        if (!only_table.empty() && r.table != only_table) continue;
        tasks.push_back(r);
    }

    vector<std::future<Result>> futs;
    for (const auto& r : tasks) {
        futs.emplace_back(std::async(std::launch::async, [r]() { return eval_row(r); }));
    }
    std::cout << std::fixed << std::setprecision(2);
    print_header();
    std::cout.flush();
    size_t remaining = futs.size();
    vector<char> done(futs.size(), 0);
    while (remaining > 0) {
        bool progressed = false;
        for (size_t i = 0; i < futs.size(); ++i) {
            if (done[i]) continue;
            if (futs[i].wait_for(std::chrono::milliseconds(50)) == std::future_status::ready) {
                Result r = futs[i].get();
                print_result(r);
                std::cout.flush();
                done[i] = 1;
                --remaining;
                progressed = true;
            }
        }
        if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
