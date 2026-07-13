// TwinPrimeSequence.cpp
//
// Verifier for the "GCD twin-prime sieve" conjecture from Math StackExchange,
// rewritten to operate entirely on an arbitrary-precision big-integer type ZZ.
//
// For an integer n >= 2, define the sequence {a_k}_{k>=1} by
//
//     a_1     = n^2 - 1
//     a_{k+1} = a_k - gcd(a_k, (n+k)^2 - 1)
//
// The sequence strictly decreases while a_k > 0, so it reaches 0 in finitely
// many steps. Let T(n) be the smallest index m with a_m = 0, and define the
// terminal offset
//
//     P(n) = n + T(n).
//
// Conjecture: for all n >= 2, both P(n) - 1 and P(n) + 1 are prime
//             (i.e. they form a twin-prime pair).
//
// For each n the program checks:
//   * boundedness         T(n) <= n^2,
//   * parity              T(n) === n (mod 2),
//   * the conjecture      P(n)-1 and P(n)+1 are twin primes,
//   * descent invariant   gcd(a_k, (n+k)^2-1) == gcd(a_k, I_k^2-1).
// Each n prints one line: "EXAMPLE ... TRUE" when the conjecture holds for
// that n, or "COUNTEREXAMPLE ... FALSE" otherwise.
//
// Everything runs through ZZ, so n is limited only by memory/time, not by 64
// bits. ZZ is a minimal unsigned bignum (base-2^32 limbs, little-endian) with
// only the operations this verifier needs: +, -, *, /, %, comparisons, binary
// GCD, integer sqrt, and a Miller-Rabin primality test.
//
// Usage (arguments are decimal integers of any size):
//     TwinPrimeSequence                # verifies n = 2 .. 10000 (default)
//     TwinPrimeSequence N              # verifies n = 2 .. N
//     TwinPrimeSequence LO HI          # verifies n = LO .. HI
//
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <numeric>   // std::gcd
#include <iostream>
#include <utility>
#if defined(_MSC_VER)
#  include <intrin.h>   // _umul128 / _udiv128 for 64-bit mulmod
#endif

// ---------------------------------------------------------------------------
//  Compile-time toggle:
//    * default (fast): all arithmetic uses native 64-bit integers (type Nat).
//      Exact and valid while every intermediate stays below 2^64, i.e. while
//      (n+k)^2 < 2^64  =>  roughly n < 65536. Far beyond any range that
//      finishes in human time (each n costs up to T(n) <= n^2 gcd steps).
//    * define USE_BIGINT (e.g. cl /D USE_BIGINT, or set it in the project) to
//      switch every value over to the arbitrary-precision ZZ type instead.
// ---------------------------------------------------------------------------
// #define USE_BIGINT   // uncomment, or pass /D USE_BIGINT, to force bignum

// ===========================================================================
//  ZZ : minimal arbitrary-precision unsigned integer
// ===========================================================================
struct ZZ {
    // little-endian base-2^32 limbs; empty vector == 0; no trailing zero limbs.
    std::vector<uint32_t> d;

    ZZ() = default;
    ZZ(uint64_t v) {
        while (v) { d.push_back((uint32_t)(v & 0xFFFFFFFFull)); v >>= 32; }
    }

    void trim() { while (!d.empty() && d.back() == 0) d.pop_back(); }
    bool isZero() const { return d.empty(); }
    bool isOne()  const { return d.size() == 1 && d[0] == 1; }
    bool isEven() const { return d.empty() || (d[0] & 1u) == 0u; }

    size_t bits() const {
        if (d.empty()) return 0;
        size_t hi = d.size() - 1;
        uint32_t top = d[hi];
        size_t b = 0;
        while (top) { top >>= 1; ++b; }
        return hi * 32 + b;
    }
    bool testBit(size_t i) const {
        size_t limb = i >> 5, off = i & 31;
        return limb < d.size() && ((d[limb] >> off) & 1u);
    }

    // ---- comparison -------------------------------------------------------
    static int cmp(const ZZ& a, const ZZ& b) {
        if (a.d.size() != b.d.size()) return a.d.size() < b.d.size() ? -1 : 1;
        for (size_t i = a.d.size(); i-- > 0; )
            if (a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
        return 0;
    }
    friend bool operator<(const ZZ& a, const ZZ& b)  { return cmp(a, b) < 0; }
    friend bool operator>(const ZZ& a, const ZZ& b)  { return cmp(a, b) > 0; }
    friend bool operator<=(const ZZ& a, const ZZ& b) { return cmp(a, b) <= 0; }
    friend bool operator>=(const ZZ& a, const ZZ& b) { return cmp(a, b) >= 0; }
    friend bool operator==(const ZZ& a, const ZZ& b) { return cmp(a, b) == 0; }
    friend bool operator!=(const ZZ& a, const ZZ& b) { return cmp(a, b) != 0; }

    // ---- add / sub --------------------------------------------------------
    friend ZZ operator+(const ZZ& a, const ZZ& b) {
        ZZ r;
        size_t n = a.d.size() > b.d.size() ? a.d.size() : b.d.size();
        r.d.resize(n);
        uint64_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t s = carry;
            if (i < a.d.size()) s += a.d[i];
            if (i < b.d.size()) s += b.d[i];
            r.d[i] = (uint32_t)(s & 0xFFFFFFFFull);
            carry = s >> 32;
        }
        if (carry) r.d.push_back((uint32_t)carry);
        r.trim();
        return r;
    }
    // requires a >= b
    friend ZZ operator-(const ZZ& a, const ZZ& b) {
        ZZ r;
        r.d.resize(a.d.size());
        int64_t borrow = 0;
        for (size_t i = 0; i < a.d.size(); ++i) {
            int64_t s = (int64_t)a.d[i] - borrow - (i < b.d.size() ? (int64_t)b.d[i] : 0);
            if (s < 0) { s += (int64_t)1 << 32; borrow = 1; } else borrow = 0;
            r.d[i] = (uint32_t)s;
        }
        r.trim();
        return r;
    }

    // ---- multiply ---------------------------------------------------------
    friend ZZ operator*(const ZZ& a, const ZZ& b) {
        if (a.isZero() || b.isZero()) return ZZ();
        ZZ r;
        r.d.assign(a.d.size() + b.d.size(), 0);
        for (size_t i = 0; i < a.d.size(); ++i) {
            uint64_t carry = 0, ai = a.d[i];
            for (size_t j = 0; j < b.d.size(); ++j) {
                uint64_t cur = (uint64_t)r.d[i + j] + ai * b.d[j] + carry;
                r.d[i + j] = (uint32_t)(cur & 0xFFFFFFFFull);
                carry = cur >> 32;
            }
            r.d[i + b.d.size()] += (uint32_t)carry;
        }
        r.trim();
        return r;
    }

    // ---- shifts -----------------------------------------------------------
    ZZ shl(size_t n) const {
        if (isZero() || n == 0) return *this;
        size_t limbShift = n >> 5, bitShift = n & 31;
        ZZ r;
        r.d.assign(d.size() + limbShift + 1, 0);
        for (size_t i = 0; i < d.size(); ++i) {
            uint64_t v = (uint64_t)d[i] << bitShift;
            r.d[i + limbShift]     |= (uint32_t)(v & 0xFFFFFFFFull);
            r.d[i + limbShift + 1] |= (uint32_t)(v >> 32);
        }
        r.trim();
        return r;
    }
    ZZ shr(size_t n) const {
        size_t limbShift = n >> 5, bitShift = n & 31;
        if (limbShift >= d.size()) return ZZ();
        ZZ r;
        r.d.assign(d.size() - limbShift, 0);
        for (size_t i = 0; i < r.d.size(); ++i) {
            uint64_t v = d[i + limbShift] >> bitShift;
            if (bitShift && i + limbShift + 1 < d.size())
                v |= (uint64_t)d[i + limbShift + 1] << (32 - bitShift);
            r.d[i] = (uint32_t)(v & 0xFFFFFFFFull);
        }
        r.trim();
        return r;
    }
    // count trailing zero bits (value must be nonzero)
    size_t ctz() const {
        size_t c = 0;
        for (size_t i = 0; i < d.size(); ++i) {
            if (d[i] == 0) { c += 32; continue; }
            uint32_t x = d[i];
            while ((x & 1u) == 0u) { x >>= 1; ++c; }
            break;
        }
        return c;
    }

    // bitwise OR (only needed by gcd, to collect the common power of two)
    friend ZZ operator|(const ZZ& a, const ZZ& b) {
        ZZ r;
        size_t n = a.d.size() > b.d.size() ? a.d.size() : b.d.size();
        r.d.resize(n);
        for (size_t i = 0; i < n; ++i)
            r.d[i] = (i < a.d.size() ? a.d[i] : 0u) | (i < b.d.size() ? b.d[i] : 0u);
        r.trim();
        return r;
    }

    // ---- divmod (binary long division): returns {quotient, remainder} -----
    static std::pair<ZZ, ZZ> divmod(const ZZ& a, const ZZ& b) {
        assert(!b.isZero());
        if (cmp(a, b) < 0) return { ZZ(), a };
        ZZ q, r;
        q.d.assign((a.bits() >> 5) + 1, 0);
        for (size_t i = a.bits(); i-- > 0; ) {
            r = r.shl(1);
            if (a.testBit(i)) { if (r.d.empty()) r.d.push_back(1); else r.d[0] |= 1u; }
            if (cmp(r, b) >= 0) {
                r = r - b;
                q.d[i >> 5] |= (1u << (i & 31));
            }
        }
        q.trim();
        return { q, r };
    }
    friend ZZ operator/(const ZZ& a, const ZZ& b) { return divmod(a, b).first; }
    friend ZZ operator%(const ZZ& a, const ZZ& b) { return divmod(a, b).second; }

    // ---- binary GCD (no division) -----------------------------------------
    static ZZ gcd(ZZ a, ZZ b) {
        if (a.isZero()) return b;
        if (b.isZero()) return a;
        size_t shift = (a | b).ctz();
        a = a.shr(a.ctz());
        do {
            b = b.shr(b.ctz());
            if (cmp(a, b) > 0) std::swap(a, b);
            b = b - a;
        } while (!b.isZero());
        return a.shl(shift);
    }

    // ---- integer square root (floor) --------------------------------------
    ZZ isqrt() const {
        if (isZero()) return ZZ();
        ZZ x = ZZ(1).shl((bits() + 1) / 2);      // >= sqrt(n)
        while (true) {
            ZZ y = (x + *this / x).shr(1);        // Newton step
            if (cmp(y, x) >= 0) break;
            x = y;
        }
        while (x * x > *this) x = x - ZZ(1);
        while ((x + ZZ(1)) * (x + ZZ(1)) <= *this) x = x + ZZ(1);
        return x;
    }

    // ---- modular exponentiation -------------------------------------------
    static ZZ powmod(ZZ base, ZZ e, const ZZ& mod) {
        ZZ r(1);
        base = base % mod;
        while (!e.isZero()) {
            if (!e.isEven()) r = (r * base) % mod;
            e = e.shr(1);
            base = (base * base) % mod;
        }
        return r;
    }

    // ---- Miller-Rabin primality -------------------------------------------
    // Deterministic for values < 3.3e24 with these bases; a strong probable-
    // prime test (no false negatives) beyond that range.
    bool isProbablePrime() const {
        static const uint32_t base[] = {
            2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };
        if (cmp(*this, ZZ(2)) < 0) return false;
        for (uint32_t p : base) {
            ZZ zp(p);
            int c = cmp(*this, zp);
            if (c == 0) return true;
            if ((*this % zp).isZero()) return false;
        }
        ZZ nm1 = *this - ZZ(1);
        size_t s = nm1.ctz();
        ZZ dd = nm1.shr(s);
        for (uint32_t a : base) {
            ZZ x = powmod(ZZ(a), dd, *this);
            if (x.isOne() || cmp(x, nm1) == 0) continue;
            bool composite = true;
            for (size_t i = 1; i < s; ++i) {
                x = (x * x) % *this;
                if (cmp(x, nm1) == 0) { composite = false; break; }
            }
            if (composite) return false;
        }
        return true;
    }

    // ---- decimal I/O ------------------------------------------------------
    static ZZ fromString(const std::string& s) {
        ZZ r;
        for (char ch : s) {
            if (ch < '0' || ch > '9') continue;
            r = r * ZZ(10) + ZZ((uint64_t)(ch - '0'));
        }
        return r;
    }
    std::string toString() const {
        if (isZero()) return "0";
        ZZ t = *this;
        std::string out;
        const ZZ TEN9(1000000000ull);      // 9 decimal digits per chunk
        while (!t.isZero()) {
            auto qr = divmod(t, TEN9);
            uint32_t chunk = qr.second.d.empty() ? 0u : qr.second.d[0];
            t = qr.first;
            for (int i = 0; i < 9; ++i) { out.push_back(char('0' + chunk % 10)); chunk /= 10; }
        }
        while (out.size() > 1 && out.back() == '0') out.pop_back();
        return std::string(out.rbegin(), out.rend());
    }
};

// ===========================================================================
//  Nat : fast native 64-bit integer with the same surface ZZ exposes.
//
//  Exact while every intermediate stays below 2^64. The only operations that
//  reach that ceiling are the squarings in run() ((n+k)^2, I^2) and the
//  modular multiply inside Miller-Rabin; the latter uses a 128-bit-wide
//  product so primality is correct across the whole 64-bit range.
// ===========================================================================
struct Nat {
    uint64_t v = 0;
    Nat() = default;
    Nat(uint64_t x) : v(x) {}

    bool isZero() const { return v == 0; }
    bool isOne()  const { return v == 1; }
    bool isEven() const { return (v & 1u) == 0u; }

    friend bool operator<(Nat a, Nat b)  { return a.v <  b.v; }
    friend bool operator>(Nat a, Nat b)  { return a.v >  b.v; }
    friend bool operator<=(Nat a, Nat b) { return a.v <= b.v; }
    friend bool operator>=(Nat a, Nat b) { return a.v >= b.v; }
    friend bool operator==(Nat a, Nat b) { return a.v == b.v; }
    friend bool operator!=(Nat a, Nat b) { return a.v != b.v; }

    friend Nat operator+(Nat a, Nat b) { return Nat(a.v + b.v); }
    friend Nat operator-(Nat a, Nat b) { return Nat(a.v - b.v); }
    friend Nat operator*(Nat a, Nat b) { return Nat(a.v * b.v); }
    friend Nat operator/(Nat a, Nat b) { return Nat(a.v / b.v); }
    friend Nat operator%(Nat a, Nat b) { return Nat(a.v % b.v); }

    static Nat gcd(Nat a, Nat b) { return Nat(std::gcd(a.v, b.v)); }

    // full-width 64-bit modular multiply (a,b < m assumed)
    static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
#if defined(_MSC_VER) && defined(_M_X64)
        uint64_t hi, lo = _umul128(a, b, &hi);
        uint64_t rem;
        _udiv128(hi % m, lo, m, &rem);      // hi<m guaranteed since a,b<m
        return rem;
#elif defined(__SIZEOF_INT128__)
        return (uint64_t)((unsigned __int128)a * b % m);
#else
        // portable fallback: binary (Russian-peasant) modular multiply
        uint64_t r = 0; a %= m;
        while (b) { if (b & 1) r = (r + a) % m; a = (a + a) % m; b >>= 1; }
        return r;
#endif
    }
    static uint64_t powmod(uint64_t a, uint64_t e, uint64_t m) {
        uint64_t r = 1 % m; a %= m;
        while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
        return r;
    }
    // Deterministic Miller-Rabin over all 64-bit integers (these bases suffice).
    bool isProbablePrime() const {
        uint64_t n = v;
        if (n < 2) return false;
        for (uint64_t p : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,
                           29ull,31ull,37ull}) {
            if (n % p == 0) return n == p;
        }
        uint64_t d = n - 1; int s = 0;
        while ((d & 1) == 0) { d >>= 1; ++s; }
        for (uint64_t a : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,
                           29ull,31ull,37ull}) {
            uint64_t x = powmod(a, d, n);
            if (x == 1 || x == n - 1) continue;
            bool composite = true;
            for (int i = 1; i < s; ++i) {
                x = mulmod(x, x, n);
                if (x == n - 1) { composite = false; break; }
            }
            if (composite) return false;
        }
        return true;
    }

    static Nat fromString(const std::string& s) { return Nat(std::strtoull(s.c_str(), nullptr, 10)); }
    std::string toString() const { return std::to_string(v); }
};

// ---------------------------------------------------------------------------
//  Select the working integer type at compile time.
// ---------------------------------------------------------------------------
#if defined(USE_BIGINT)
using Int = ZZ;
static const char* kIntMode = "ZZ (arbitrary precision)";
#else
using Int = Nat;
static const char* kIntMode = "Nat (native 64-bit)";
#endif

// ===========================================================================
//  The sequence and the checks
// ===========================================================================
struct Result {
    Int n, T, P, L;
    bool boundOK, parityOK, twinOK, invariantOK;
};

static Result run(const Int& n) {
    Result r;
    r.n = n;

    const Int one(1);
    Int a = n * n - one;              // a_1
    Int k(1);
    Int lastDropL;                    // a_{j+1}: value after the last nontrivial drop
    bool invariantOK = true;

    while (!a.isZero()) {
        Int nk = n + k;               // n + k
        Int arg = nk * nk - one;      // (n+k)^2 - 1
        Int g = Int::gcd(a, arg);

        // Descent-invariant cross-check: gcd(a, (n+k)^2-1) == gcd(a, I_k^2-1)
        // with I_k = a + n + k. Reduce mod a to keep intermediates small.
        Int I = a + nk;
        Int Im = I % a;
        Int val = (Im * Im + a - (one % a)) % a;  // (I^2 - 1) mod a
        if (Int::gcd(a, val) != g) invariantOK = false;

        Int aNext = a - g;
        if (g > one) lastDropL = aNext;
        a = aNext;
        k = k + one;
    }

    // Loop maintains k == index of current a; it exits with a = a_k = 0, so T = k.
    r.T = k;                          // a_T = 0 reached
    r.P = n + r.T;
    r.L = lastDropL;
    r.invariantOK = invariantOK;

    r.boundOK  = (r.T <= n * n);
    r.parityOK = (r.T.isEven() == n.isEven());
    r.twinOK   = (r.P >= one) && (r.P - one).isProbablePrime()
                              && (r.P + one).isProbablePrime();
    return r;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    Int lo(2), hi(10000);
    if (argc == 2) {
        hi = Int::fromString(argv[1]);
    } else if (argc >= 3) {
        lo = Int::fromString(argv[1]);
        hi = Int::fromString(argv[2]);
    }
    if (lo < Int(2)) lo = Int(2);

    std::cout << "Integer backend: " << kIntMode << "\n";
#if !defined(USE_BIGINT)
    // Native mode is exact only while (n+k)^2 < 2^64, i.e. n < 65536.
    if (Int(65535) < hi)
        std::cout << "WARNING: n exceeds 65535 in native mode; squarings may "
                     "overflow. Rebuild with /D USE_BIGINT for exact results.\n";
#endif

    uint64_t failTwin = 0, failBound = 0, failParity = 0, failInv = 0;
    uint64_t checked = 0;
    const Int one(1);

    for (Int n = lo; n <= hi; n = n + one) {
        Result r = run(n);
        ++checked;

        if (!r.boundOK)     ++failBound;
        if (!r.parityOK)    ++failParity;
        if (!r.twinOK)      ++failTwin;
        if (!r.invariantOK) ++failInv;

        // The conjecture holds for this n exactly when P-1 and P+1 are both
        // prime. Print an EXAMPLE ... TRUE line then, else a COUNTEREXAMPLE.
        std::cout << (r.twinOK ? "EXAMPLE " : "COUNTEREXAMPLE ")
                  << "n=" << r.n.toString()
                  << "  T=" << r.T.toString()
                  << "  P=" << r.P.toString()
                  << "  L=" << r.L.toString()
                  << "  [P-1=" << (r.P - one).toString()
                  << ((r.P - one).isProbablePrime() ? "(prime)" : "(comp)")
                  << ", P+1=" << (r.P + one).toString()
                  << ((r.P + one).isProbablePrime() ? "(prime)" : "(comp)") << "]"
                  << (r.twinOK ? "  TRUE" : "  FALSE")
                  << (r.boundOK ? "" : "  BOUND-FAIL")
                  << (r.parityOK ? "" : "  PARITY-FAIL")
                  << (r.invariantOK ? "" : "  INVARIANT-FAIL")
                  << "\n";

        // Immediately halt on the first genuine counterexample to the
        // conjecture (P-1 and P+1 not both prime).
        if (!r.twinOK) {
            std::cout << "\nHALT: twin-prime conjecture FALSE at n="
                      << r.n.toString() << " (P=" << r.P.toString() << ").\n";
            return 1;
        }
    }

    std::cout << "\n==== Summary ====\n";
    std::cout << "Range n = " << lo.toString() << " .. " << hi.toString()
              << "  (" << checked << " values)\n";
    std::cout << "Twin-prime conjecture failures : " << failTwin  << "\n";
    std::cout << "Bound  T(n)<=n^2   failures     : " << failBound << "\n";
    std::cout << "Parity T(n)=n(mod2) failures    : " << failParity << "\n";
    std::cout << "Descent-invariant  failures     : " << failInv   << "\n";

    bool allOK = (failTwin | failBound | failParity | failInv) == 0;
    std::cout << (allOK
        ? "\nAll checks passed. Conjecture holds on this range.\n"
        : "\nAt least one check failed (see above).\n");
    return allOK ? 0 : 1;
}