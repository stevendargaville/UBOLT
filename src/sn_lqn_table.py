#!/usr/bin/env python3
"""Generate src/sn_lqn_table.hpp - the level-symmetric (LQn) quadrature table.

    python3 src/sn_lqn_table.py            # writes src/sn_lqn_table.hpp
    python3 src/sn_lqn_table.py --check    # regenerate and diff, no write

Run from anywhere; the output path is resolved relative to this file. The
output is deterministic: a re-run reproduces it byte for byte.

Why this exists
---------------
The 2D and 3D quadratures need the LQn constants to full double precision, and
the published tables (Lewis & Miller, *Computational Methods of Neutron
Transport*, Table 4-1) carry seven digits. Everything below is derived from the
defining conditions at 60 decimal digits instead, and the published values are
used only as root-finder seeds and as a cross-check.

The construction
----------------
A level-symmetric set of order N has n = N/2 levels of direction cosine, with
the squares in arithmetic progression:

    mu_i^2 = mu_1^2 + (i - 1) delta,   delta = 2 (1 - 3 mu_1^2) / (N - 2)

so mu_1 is the only free cosine. One octant holds the points (mu_i, mu_j, mu_k)
for every ORDERED index triple with i + j + k = n + 2 - which is exactly the
condition that makes the direction a unit vector - so there are n(n+1)/2 =
N(N+2)/8 of them. A point's weight depends only on the SORTED triple (its
permutation class), because the set has to look the same from every axis.

The defining conditions are the even axis moments through order N. Writing the
weights as fractions of the whole sphere (they sum to 1 over all 8 octants),

    sum_{p in octant} c_p mu_{i(p)}^{2m} = 1 / (8 (2m + 1)),   m = 0 .. n

with i(p) the level index of p's FIRST component. These reduce a long way. Put
x_l = mu_l^2 and let y_l be the total octant weight of the points whose first
index is l; then the conditions are a generalised Vandermonde system

    sum_l x_l^m y_l = 1 / (8 (2m + 1))

whose m = 1 row is a linear combination of the m = 0 row (summing the three
axes of a point gives mu_i^2 + mu_j^2 + mu_k^2 = 1, and the three axes carry
equal weight by symmetry). Dropping it leaves n equations for the n unknowns
y_l, so the LEVEL SUMS are unique for any mu_1. The same i + j + k = n + 2
identity that made the m = 1 row redundant then leaves exactly one condition
that y must satisfy to come from real per-class weights,

    sum_l (3 l - (n + 2)) y_l = 0

and THAT is what fixes mu_1 - one scalar root-find, at every order. The
resulting mu_1 agrees with the published table to its last tabulated digit for
every order it reaches (asserted below).

What is left is recovering the per-class weights c from the level sums y. That
map is an integer matrix and it is not injective at every order: it has a null
space of dimension 1 at S14 and S16, 2 at S18 and 3 at S20 (0 below S14). So
those orders have a FAMILY of level-symmetric sets, every member of which
satisfies every axis moment through order N exactly, and one has to be picked.

The rule is ONE least-squares problem: take the member minimising the error in
the MIXED even moments <mu^2a eta^2b xi^2c>, a + b + c <= n - the conditions the
axis moments do not imply. It is unique, so the choice is deterministic; at S14
it is satisfiable exactly and elsewhere it is a genuine fit. Where the set is
already unique (S2 to S12) there is nothing to pick and the rule does nothing.

This is NOT how the published tables resolve the freedom - their point weights
are tabulated, not derived - so the weights here agree with Lewis & Miller's
Table 4-1 only over S2 to S12, where the set is unique and they match to all
seven published digits.

WHERE THE TABLE STOPS is then not a choice either: the generator walks the
orders upwards and stops at the first one whose least-squares member has a
weight that is not strictly positive. That is S20, so the table runs to S18. A
negative weight is not something to work around - it is the level-symmetric
family running out, which is the well-known reason SN codes cap LQn somewhere
around here. If a higher order is wanted, it needs a different family (a product
or Gauss-Chebyshev set, say), not a different way of choosing within this one.
"""

import os
import sys
from fractions import Fraction

from mpmath import findroot, lu_solve, matrix, mp, mpf, sqrt

# Everything is derived at this precision and rounded to double on emission
mp.dps = 60

# The lowest order there is. The HIGHEST is not set here - the generator walks
# upwards and stops at the first order whose weights are not all positive (see
# the module docstring), which is what CANDIDATE_ORDERS bounds the search of.
# That bound is the published table's own extent: past it there is no seed for
# the root-find and nothing to cross-check the answer against, and the walk
# stops well before it anyway
MIN_ORDER = 2
CANDIDATE_ORDERS = range(2, 22, 2)

# Lewis & Miller, Computational Methods of Neutron Transport, Table 4-1. Used
# ONLY to seed the root-find and to cross-check what it converges to - none of
# these digits reach the generated header
PUBLISHED_MU1 = {
    2: "0.5773503", 4: "0.3500212", 6: "0.2666355", 8: "0.2182179",
    10: "0.1893213", 12: "0.1672126", 14: "0.1519859", 16: "0.1389568",
    18: "0.1293445", 20: "0.1206033",
}
# How far the derived mu_1 may sit from the published one: half a unit in its
# last tabulated digit, plus room for the table's own rounding
PUBLISHED_TOL = mpf("5e-7")

HEADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sn_lqn_table.hpp")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The combinatorics of one octant
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def octant_triples(n):
    """The ordered level-index triples of one octant, lexicographically.

    Lexicographic order is what makes the S4 set come out as (1,1,2), (1,2,1),
    (2,1,1) - the base-point order the quadrature classes have always used.
    """
    out = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            k = n + 2 - i - j
            if 1 <= k <= n:
                out.append((i, j, k))
    return out


def permutation_classes(n):
    """The sorted triples, in a fixed order - one weight per class."""
    return sorted({tuple(sorted(t)) for t in octant_triples(n)})


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exact rational linear algebra on the integer class map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rref(rows, n_cols):
    """Exact reduced row echelon form, tracking the row transform.

    Returns (R, T, pivots) with T * rows = R. Everything is a Fraction, so the
    pivoting is decided by exact zero tests rather than a tolerance.
    """
    n_rows = len(rows)
    A = [[Fraction(v) for v in r] for r in rows]
    T = [[Fraction(1 if i == j else 0) for j in range(n_rows)] for i in range(n_rows)]
    pivots = []
    r = 0
    for c in range(n_cols):
        pr = next((k for k in range(r, n_rows) if A[k][c] != 0), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        T[r], T[pr] = T[pr], T[r]
        inv = A[r][c]
        A[r] = [v / inv for v in A[r]]
        T[r] = [v / inv for v in T[r]]
        for k in range(n_rows):
            if k != r and A[k][c] != 0:
                f = A[k][c]
                A[k] = [a - f * b for a, b in zip(A[k], A[r])]
                T[k] = [a - f * b for a, b in zip(T[k], T[r])]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return A, T, pivots


def to_mpf(fraction):
    return mpf(fraction.numerator) / mpf(fraction.denominator)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The level cosines and the level sums
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def level_squares(n, mu1):
    """x_l = mu_l^2 for l = 1..n, from the arithmetic progression in mu^2."""
    if n == 1:
        # One level, and the unit-vector condition alone fixes it: 3 mu^2 = 1
        return [mpf(1) / 3]
    x1 = mu1 ** 2
    delta = (1 - 3 * x1) / (n - 1)
    return [x1 + i * delta for i in range(n)]


def level_sums(n, mu1):
    """y_l: the total octant weight carried by points whose first index is l.

    Unique for any mu_1 - the m = 1 axis moment is dropped because it is a
    multiple of the m = 0 one, leaving n independent conditions for n unknowns.
    """
    x = level_squares(n, mu1)
    moments = [0] + list(range(2, n + 1))
    V = matrix(n, n)
    rhs = matrix(n, 1)
    for r, m in enumerate(moments):
        for l in range(n):
            V[r, l] = x[l] ** m
        rhs[r] = mpf(1) / (8 * (2 * m + 1))
    return lu_solve(V, rhs)


def mu1_residual(n, mu1):
    """The one condition mu_1 has to satisfy: sum_l (3 l - (n + 2)) y_l = 0."""
    y = level_sums(n, mu1)
    return sum((3 * (l + 1) - (n + 2)) * y[l] for l in range(n))


def solve_mu1(n):
    if n == 1:
        return sqrt(mpf(1) / 3)
    seed = mpf(PUBLISHED_MU1[2 * n])
    return findroot(lambda t: mu1_residual(n, t),
                    (seed * mpf("0.999"), seed * mpf("1.001")),
                    solver="secant", tol=mpf("1e-90"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The mixed even moments of the sphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def double_factorial(k):
    out = 1
    while k > 1:
        out *= k
        k -= 2
    return out


def sphere_moment(a, b, c):
    """<mu^2a eta^2b xi^2c> averaged over the unit sphere, exactly."""
    return Fraction(double_factorial(2 * a - 1) * double_factorial(2 * b - 1)
                    * double_factorial(2 * c - 1),
                    double_factorial(2 * (a + b + c) + 1))


def mixed_moment_keys(n):
    """The mixed even moments of total degree <= N, one per permutation class.

    b > 0 skips the pure axis moments: those are already imposed exactly, and
    every member of the solution family has the same level sums, so they are
    identically satisfied across it.
    """
    return [(a, b, c)
            for a in range(n + 1)
            for b in range(min(a, n - a) + 1)
            for c in range(min(b, n - a - b) + 1)
            if b > 0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One order
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def least_squares(A, rhs):
    """Least-squares solution of A t = rhs through the normal equations.

    A has a handful of columns and is well conditioned at every order here; the
    residuals are asserted by the caller either way.
    """
    return lu_solve(A.T * A, A.T * rhs)


def build_order(sn_order):
    """Everything the header needs for one order, plus what to assert on it."""
    n = sn_order // 2
    mu1 = solve_mu1(n)
    x = level_squares(n, mu1)
    mu = [sqrt(v) for v in x]
    triples = octant_triples(n)
    classes = permutation_classes(n)
    cls_of = {c: a for a, c in enumerate(classes)}
    y = level_sums(n, mu1)

    def moment(weights, a, b, c):
        """sum_p c_p mu_i^2a mu_j^2b mu_k^2c over the octant."""
        return sum(weights[cls_of[tuple(sorted(t))]]
                   * x[t[0] - 1] ** a * x[t[1] - 1] ** b * x[t[2] - 1] ** c
                   for t in triples)

    # The integer map from class weights to level sums, and its null space
    class_to_level = [[0] * len(classes) for _ in range(n)]
    for t in triples:
        class_to_level[t[0] - 1][cls_of[tuple(sorted(t))]] += 1
    R, T, pivots = rref(class_to_level, len(classes))
    free = [c for c in range(len(classes)) if c not in pivots]

    # A particular solution (free classes at zero) and a basis of the freedom
    Ty = [sum(to_mpf(T[r][k]) * y[k] for k in range(n)) for r in range(n)]
    base = [mpf(0)] * len(classes)
    for row, p in enumerate(pivots):
        base[p] = Ty[row]
    null_basis = []
    for fc in free:
        v = [mpf(0)] * len(classes)
        v[fc] = mpf(1)
        for row, p in enumerate(pivots):
            v[p] = -to_mpf(R[row][fc])
        null_basis.append(v)

    keys = mixed_moment_keys(n)

    def mixed_error(weights):
        if not keys:
            return mpf(0)
        return max(abs(moment(weights, *k) / (to_mpf(sphere_moment(*k)) / 8) - 1)
                   for k in keys)

    def combine(t):
        return [base[k] + sum(t[j] * null_basis[j][k] for j in range(len(null_basis)))
                for k in range(len(classes))]

    if not null_basis:
        # Nothing to choose: the axis moments determine the weights outright
        weights = base
    else:
        # Least squares on the relative mixed-moment residuals, and that is the
        # whole rule - if it lands on a non-positive weight the caller stops
        # generating rather than reaching for a different member
        G = matrix(len(keys), len(null_basis))
        r0 = matrix(len(keys), 1)
        for row, k in enumerate(keys):
            want = to_mpf(sphere_moment(*k)) / 8
            r0[row] = (moment(base, *k) - want) / want
            for col, v in enumerate(null_basis):
                G[row, col] = moment(v, *k) / want
        weights = combine(least_squares(G, -r0))

    return {
        "sn_order": sn_order,
        "n_levels": n,
        "mu": mu,
        "mu1": mu1,
        "triples": triples,
        "classes": classes,
        "cls_of": cls_of,
        "weights": weights,
        "freedom": len(null_basis),
        "smallest_weight": min(weights),
        "mixed_error": mixed_error(weights),
        "moment": moment,
        "x": x,
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Checks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_order(order):
    """Assert everything the C++ side is entitled to assume. Returns the
    worst axis-moment residual AFTER rounding the table to double, which is
    what the quadrature will actually see."""
    N, n = order["sn_order"], order["n_levels"]
    mu, x = order["mu"], order["x"]
    triples, weights, cls_of = order["triples"], order["weights"], order["cls_of"]

    assert len(triples) == N * (N + 2) // 8, f"S{N}: wrong octant point count"
    assert all(0 < v < 1 for v in mu), f"S{N}: a level cosine left (0, 1)"
    assert all(mu[l] < mu[l + 1] for l in range(n - 1)), f"S{N}: levels not ascending"
    for t in triples:
        unit = x[t[0] - 1] + x[t[1] - 1] + x[t[2] - 1]
        assert abs(unit - 1) < mpf("1e-40"), f"S{N}: {t} is not a unit vector"
    assert all(w > 0 for w in weights), f"S{N}: a class weight is not positive"

    published = mpf(PUBLISHED_MU1[N])
    assert abs(order["mu1"] - published) < PUBLISHED_TOL, \
        f"S{N}: mu_1 {order['mu1']} is not the published {published}"

    # Axis moments at working precision, on all three axes (the y and z ones
    # hold by permutation symmetry, so they check the class bookkeeping)
    for m in range(n + 1):
        for axis in range(3):
            got = sum(weights[cls_of[tuple(sorted(t))]] * x[t[axis] - 1] ** m
                      for t in triples)
            assert abs(got * 8 * (2 * m + 1) - 1) < mpf("1e-30"), \
                f"S{N}: axis {axis} moment {m} is off by {got * 8 * (2 * m + 1) - 1}"

    # And again through the doubles that actually get emitted
    xd = [mpf(float(v)) ** 2 for v in mu]
    wd = [mpf(float(w)) for w in weights]
    worst = mpf(0)
    for m in range(n + 1):
        got = sum(wd[cls_of[tuple(sorted(t))]] * xd[t[0] - 1] ** m for t in triples)
        worst = max(worst, abs(got * 8 * (2 * m + 1) - 1))
    assert worst < mpf("1e-13"), f"S{N}: double-rounded axis moments off by {worst}"
    return worst


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Emission
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def as_double(value):
    """The shortest literal that reads back as this double, exactly."""
    return repr(float(value))


def emit(orders, worst_double, stopped_at):
    max_order = orders[-1]["sn_order"]
    out = []
    w = out.append
    w("#ifndef UBOLT_SN_LQN_TABLE_HPP")
    w("#define UBOLT_SN_LQN_TABLE_HPP")
    w("")
    w("// GENERATED BY src/sn_lqn_table.py - DO NOT EDIT, REGENERATE INSTEAD")
    w("//")
    w("// The level-symmetric (LQn) quadrature constants for even SN orders")
    w(f"// {MIN_ORDER} to {max_order}, derived from the even axis moments through order N at")
    w("// 60 decimal digits and rounded to double here. The script's docstring")
    w("// has the derivation and the assertions.")
    w("//")
    w("// A level-symmetric set of order N has N/2 levels of direction cosine,")
    w("// with mu_i^2 in arithmetic progression. One octant holds the points")
    w("// (mu_i, mu_j, mu_k) for every ORDERED index triple with i + j + k =")
    w("// N/2 + 2 - the condition that makes the direction a unit vector - and a")
    w("// point's weight depends only on the SORTED triple, so the set looks the")
    w("// same from every axis. The other seven octants are the sign flips, and")
    w("// 2D folds the xi > 0 half onto the xi < 0 one by doubling the weights.")
    w("//")
    w("// The points are listed in LEXICOGRAPHIC triple order, which is what")
    w("// makes S4 come out as (1,1,2), (1,2,1), (2,1,1) - the base-point order")
    w("// the quadrature classes have used since they were S2/S4 literals.")
    w("//")
    w("// Never include this from include/ubolt/ - it is a src/ implementation")
    w("// detail, the same rule the vendored JSON parser follows.")
    w("//")
    w("// order | levels | points/octant | weight freedom | smallest weight | mixed-moment error | axis moments at double")
    for o in orders:
        w(f"// {('S' + str(o['sn_order'])):>5} | {o['n_levels']:>6} | {len(o['triples']):>13}"
          f" | {o['freedom']:>14} | {float(o['smallest_weight']):>15.3e}"
          f" | {float(o['mixed_error']):>18.2e} | {float(o['double_residual']):>22.2e}")
    w("//")
    w("// 'weight freedom' is the dimension of the family of sets satisfying")
    w("// every axis moment through N. Where it is non-zero the member chosen is")
    w("// the one minimising the error in the MIXED even moments")
    w("// <mu^2a eta^2b xi^2c> of total degree <= N, which the axis moments do")
    w("// not imply; that error is reported here, not required. 'smallest weight'")
    w("// is a fraction of the whole sphere and has to be positive - which is")
    w("// exactly why the table stops where it does:")
    w("//")
    w(f"//   S{stopped_at['sn_order']} is NOT here because its smallest weight is"
      f" {float(stopped_at['smallest_weight']):.3e}.")
    w("//")
    w("// That is the level-symmetric family running out rather than anything to")
    w("// work around, and it is the well-known reason SN codes cap LQn around")
    w("// here. A higher order needs a DIFFERENT family - a product or")
    w("// Gauss-Chebyshev set - not a different way of choosing within this one.")
    w(f"// Worst axis-moment residual over every order, at double: {float(worst_double):.2e}")
    w("")
    w("#include <petscsys.h>")
    w("")
    w("namespace ubolt_lqn {")
    w("")
    w("// One point of the principal octant: the 1-based level indices of its")
    w("// three cosines, and its weight as a fraction of the WHOLE sphere (so")
    w("// the weights of all 8 * n_points ordinates sum to 1). The consumers")
    w("// scale that by 4 pi, and 2D by another factor of 2 for the fold")
    w("struct LqnPoint {")
    w("   PetscInt  i, j, k;")
    w("   PetscReal weight;")
    w("};")
    w("")
    w("// One order's table: the level cosines, then the octant points")
    w("struct LqnOrder {")
    w("   PetscInt         sn_order;")
    w("   PetscInt         n_levels;")
    w("   PetscInt         n_points;")
    w("   const PetscReal *mu;")
    w("   const LqnPoint  *points;")
    w("};")
    w("")
    for o in orders:
        N = o["sn_order"]
        w(f"// S{N}: {o['n_levels']} level(s), {len(o['triples'])} points per octant,"
          f" {len(o['classes'])} distinct weight(s)")
        w(f"static const PetscReal lqn_mu_s{N}[{o['n_levels']}] = {{")
        for value in o["mu"]:
            w(f"   {as_double(value)},")
        w("};")
        w(f"static const LqnPoint lqn_points_s{N}[{len(o['triples'])}] = {{")
        for t in o["triples"]:
            weight = o["weights"][o["cls_of"][tuple(sorted(t))]]
            w(f"   {{{t[0]}, {t[1]}, {t[2]}, {as_double(weight)}}},")
        w("};")
        w("")
    w(f"static const PetscInt lqn_n_orders = {len(orders)};")
    w(f"static const LqnOrder lqn_orders[{len(orders)}] = {{")
    for o in orders:
        N = o["sn_order"]
        w(f"   {{{N}, {o['n_levels']}, {len(o['triples'])}, lqn_mu_s{N}, lqn_points_s{N}}},")
    w("};")
    w("")
    w("// The orders the table carries, for the caller's error message")
    w(f"static const PetscInt lqn_min_order = {MIN_ORDER};")
    w(f"static const PetscInt lqn_max_order = {max_order};")
    w("")
    w("// The table for an order, or NULL if there is none - the caller owns")
    w("// the error message, so this says nothing about why")
    w("static inline const LqnOrder *lqn_find(PetscInt sn_order)")
    w("{")
    w("   for (PetscInt k = 0; k < lqn_n_orders; k++) {")
    w("      if (lqn_orders[k].sn_order == sn_order) return &lqn_orders[k];")
    w("   }")
    w("   return NULL;")
    w("}")
    w("")
    w("}")
    w("")
    w("#endif")
    w("")
    return "\n".join(out)


def main():
    orders = []
    worst_double = mpf(0)
    stopped_at = None

    # Walk upwards and stop at the first order whose weights are not all
    # positive: that is the level-symmetric family running out, and it is what
    # decides how far the table goes
    for sn_order in CANDIDATE_ORDERS:
        order = build_order(sn_order)
        print(f"S{order['sn_order']:<3} mu_1 = {order['mu1']} "
              f"(published {PUBLISHED_MU1[order['sn_order']]}), "
              f"{len(order['triples'])} points/octant, "
              f"{len(order['classes'])} weight(s), freedom {order['freedom']}, "
              f"smallest weight {float(order['smallest_weight']):.3e}, "
              f"mixed-moment error {float(order['mixed_error']):.2e}")
        if order["smallest_weight"] <= 0:
            stopped_at = order
            print(f"     ^ not positive, so the table stops below S{sn_order}")
            break
        order["double_residual"] = check_order(order)
        worst_double = max(worst_double, order["double_residual"])
        orders.append(order)

    assert orders, "no order produced an all-positive set"
    assert stopped_at is not None, \
        (f"every candidate order up to S{orders[-1]['sn_order']} has positive weights - "
         "widen CANDIDATE_ORDERS (and PUBLISHED_MU1 with it) so the table stops for a "
         "reason rather than at the end of the search")

    text = emit(orders, worst_double, stopped_at)
    if "--check" in sys.argv:
        with open(HEADER_PATH) as fh:
            existing = fh.read()
        if existing != text:
            print(f"\n{HEADER_PATH} is NOT what this script generates", file=sys.stderr)
            return 1
        print(f"\n{HEADER_PATH} is up to date")
        return 0
    with open(HEADER_PATH, "w") as fh:
        fh.write(text)
    print(f"\nwrote {HEADER_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
