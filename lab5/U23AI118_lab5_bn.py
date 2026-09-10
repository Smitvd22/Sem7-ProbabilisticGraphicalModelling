"""
AI453 - Practical #5: A Bayesian Network from Real Data
Department of Artificial Intelligence, SVNIT Surat

Plain Python 3, standard library only. Nothing to install.
Run:  python3 lab5_bn.py
It should print one line first: P(fail=1 | medu=1) = 0.099, then the output
for Problem Statements 1-8.

Data: student-por.csv, 649 secondary-school students from two Portuguese
schools (Cortez & Silva, 2008; UCI Machine Learning Repository).
Keep it in the same folder as this file.
"""

import csv
import itertools
import random

# --------------------------------------------------------------------------
# The eight variables, all binary, all cut from real columns of the CSV.
# --------------------------------------------------------------------------
#   medu    mother's education is secondary or above   (Medu >= 3)
#   goout   goes out with friends often                (goout >= 4)
#   fail    has failed a class before                  (failures >= 1)
#   higher  wants to go on to higher education         (higher == yes)
#   study   studies more than 5 hours a week            (studytime >= 3)
#   absent  six or more absences this year              (absences >= 6)
#   g1      first-period grade of 12 or more            (G1 >= 12)
#   grade   final grade of 12 or more                   (G3 >= 12)

VARS = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]

# --------------------------------------------------------------------------
# The network. Each entry is  child : [its parents].
# The ordering is temporal: family background, then things fixed early in the
# year, then the first-period grade, then the final grade.
# --------------------------------------------------------------------------
PARENTS = {
    "medu":   [],
    "goout":  [],
    "fail":   ["medu"],
    "higher": ["medu", "fail"],
    "study":  ["medu", "higher", "goout"],
    "absent": ["goout", "fail"],
    "g1":     ["fail", "study", "goout", "higher"],
    "grade":  ["g1", "absent"],
}

# VARS is already a topological order: every variable appears after all of its
# parents. medu, goout have no parents (both come first); fail depends only on
# medu; higher depends on medu, fail; study depends on medu, higher, goout;
# absent depends on goout, fail; g1 depends on fail, study, goout, higher;
# grade depends on g1, absent. Every parent named for a variable already
# appears earlier in VARS, so the order is valid.


# --------------------------------------------------------------------------
# Helpers, written for you
# --------------------------------------------------------------------------

def load(path="lab5/student-por.csv"):
    """Read the CSV and cut every column down to 0/1. Returns a list of dicts."""
    rows = []
    for r in csv.DictReader(open(path)):
        g = lambda k: r[k].strip('"')
        rows.append({
            "medu":   1 if int(g("Medu")) >= 3 else 0,
            "goout":  1 if int(g("goout")) >= 4 else 0,
            "fail":   1 if int(g("failures")) >= 1 else 0,
            "higher": 1 if g("higher") == "yes" else 0,
            "study":  1 if int(g("studytime")) >= 3 else 0,
            "absent": 1 if int(g("absences")) >= 6 else 0,
            "g1":     1 if int(g("G1")) >= 12 else 0,
            "grade":  1 if int(g("G3")) >= 12 else 0,
        })
    return rows


def count(data, cond):
    """How many rows satisfy every key=value pair in cond. count(data, {}) is len(data)."""
    return sum(1 for d in data if all(d[k] == v for k, v in cond.items()))


def configs(parents):
    """All parent assignments, as tuples, in a fixed order. configs([]) gives [()]."""
    return list(itertools.product([0, 1], repeat=len(parents)))


def sample_bernoulli(p):
    """From Practical 3. Still the only random primitive you need."""
    return 1 if random.random() < p else 0


# --------------------------------------------------------------------------
# The worked example: one CPT row, straight from counting.
#
#   P(fail = 1 | medu = 1) = (rows with fail=1 and medu=1) / (rows with medu=1)
#
# Every CPT in this lab is this line with a different variable and a different
# parent assignment. That is all maximum likelihood estimation is here --
# exactly the counting you did for the die in Practical 3, T8.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    data = load()
    num = count(data, {"fail": 1, "medu": 1})
    den = count(data, {"medu": 1})
    print("P(fail=1 | medu=1) =", round(num / den, 3))


# ==========================================================================
# CORE MACHINERY  (used by every Problem Statement below)
# ==========================================================================

def fit(data, alpha=0.0, parents=PARENTS):
    """cpt[v][parent_tuple] = P(v = 1 | parents = parent_tuple).

    P(v=1 | pa) = (n1 + alpha) / (n0 + n1 + 2*alpha)
    n1, n0 are counted directly from the data for each parent configuration.
    A configuration with n0 = n1 = 0 (never observed) gives 0/0 -> nan when
    alpha = 0; that is deliberate, see Problem Statement 6.
    """
    cpt = {}
    for v in VARS:
        pa_list = parents[v]
        cpt[v] = {}
        for pa in configs(pa_list):
            cond = dict(zip(pa_list, pa))
            n1 = count(data, {**cond, v: 1})
            n0 = count(data, {**cond, v: 0})
            denom = n0 + n1 + 2 * alpha
            cpt[v][pa] = (n1 + alpha) / denom if denom > 0 else float("nan")
    return cpt


def joint(assignment, cpt, parents=PARENTS):
    """P(assignment) = product over v of P(v = assignment[v] | its parents)."""
    p = 1.0
    for v in VARS:
        pa = tuple(assignment[x] for x in parents[v])
        p1 = cpt[v][pa]
        p *= p1 if assignment[v] == 1 else (1 - p1)
    return p


def infer(query, evidence, cpt, parents=PARENTS):
    """P(query | evidence) by brute-force enumeration over all 2**8 assignments."""
    num = 0.0
    den = 0.0
    for bits in itertools.product([0, 1], repeat=len(VARS)):
        a = dict(zip(VARS, bits))
        if all(a[k] == v for k, v in evidence.items()):
            j = joint(a, cpt, parents)
            den += j
            if all(a[k] == v for k, v in query.items()):
                num += j
    return num / den if den > 0 else float("nan")


def data_fraction(data, query, evidence=None):
    """The plain fraction observed in the data, plus the count it rests on."""
    evidence = evidence or {}
    n_evidence = count(data, evidence) if evidence else len(data)
    cond = dict(evidence)
    cond.update(query)
    n_both = count(data, cond)
    frac = n_both / n_evidence if n_evidence > 0 else float("nan")
    return frac, n_evidence


def forward(cpt, parents=PARENTS):
    """One simulated student: walk VARS in topological order, sampling each
    variable from its own CPT row using the values already drawn for its
    parents. This only works because VARS lists every parent before its
    child -- see T/Q y) below."""
    a = {}
    for v in VARS:
        pa = tuple(a[x] for x in parents[v])
        a[v] = sample_bernoulli(cpt[v][pa])
    return a


def rejection_sample(cpt, query, evidence, M, parents=PARENTS):
    """Draw M students, keep the ones consistent with evidence, and return the
    fraction of survivors that also match query, plus the retention rate."""
    kept = 0
    matched = 0
    for _ in range(M):
        s = forward(cpt, parents)
        if all(s[k] == v for k, v in evidence.items()):
            kept += 1
            if all(s[k] == v for k, v in query.items()):
                matched += 1
    est = matched / kept if kept > 0 else float("nan")
    retention = kept / M
    return est, retention, kept


def fmt(x):
    return "nan" if x != x else round(x, 3)


# ==========================================================================
# PROBLEM STATEMENT 1 - Data Preparation and Marginals
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 1: Data Preparation and Marginals =====")

    # a) row count
    print("a) number of records:", len(data), "(must be 649)")

    # b) marginals P(v=1)
    print("b) marginals P(v = 1):")
    marginals = {}
    for v in VARS:
        marginals[v] = count(data, {v: 1}) / len(data)
        print(f"   P({v}=1) = {marginals[v]:.3f}")
    # Sanity check requested: P(grade=1) ~= 0.536 -- matches above.

    # c) written answer
    print("c) [see comment in source for the written answer]")
    # c) The three continuous columns that got thresholded are absences
    #    (0-32 in this file -> absent = absences>=6), G1 (0-19 -> g1 = G1>=12)
    #    and G3 (0-19 -> grade = G3>=12). Binarising throws away the distance
    #    from the cut: a student with 6 absences and one with 30 are both
    #    "absent=1", and a G1 of 19 and a G1 of 12 are both "g1=1", while a
    #    G1 of 11 and a G1 of 0 are both "g1=0" even though 11 and 12 are
    #    practically the same student. Any dose-response relationship
    #    (more absences -> steadily worse outcome) is collapsed into a single
    #    step, and students near the cut are the ones most likely to be
    #    mis-bucketed.
    #    The variable whose marginal is extreme enough to cause trouble later
    #    is "higher": P(higher=1) = 0.894, so only ~10.6% of the 649 students
    #    have higher=0. Since higher is a parent of study and of g1, every
    #    parent configuration with higher=0 is starved of data -- this is
    #    exactly why most of the degenerate CPT rows found in Problem
    #    Statement 6 involve higher=0.


# ==========================================================================
# PROBLEM STATEMENT 2 - Parameter Estimation by Counting
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 2: Parameter Estimation by Counting =====")

    # d) fit() is defined above, in the CORE MACHINERY section.
    cpt0 = fit(data, alpha=0.0)

    # e) CPT of grade: parents are (g1, absent), so 4 rows.
    print("e) CPT of grade, parents (g1, absent):")
    for pa in configs(PARENTS["grade"]):
        cond = dict(zip(PARENTS["grade"], pa))
        n1 = count(data, {**cond, "grade": 1})
        n0 = count(data, {**cond, "grade": 0})
        p = cpt0["grade"][pa]
        print(f"   g1={pa[0]} absent={pa[1]}: P(grade=1|..)={p:.3f}  "
              f"(n1={n1}, n0={n0}, total={n1+n0})")

    # f) written answer, quantified from the four rows just printed:
    #    g1=0,absent=0 -> 0.200   g1=1,absent=0 -> 0.954   (swing = 0.754)
    #    g1=0,absent=1 -> 0.175   g1=1,absent=1 -> 0.809   (swing = 0.634)
    #    Holding absent fixed and flipping g1 moves P(grade=1) by roughly
    #    0.63-0.75. Holding g1 fixed and flipping absent instead:
    #    absent=0->1 at g1=0: 0.200 -> 0.175 (swing = -0.025)
    #    absent=0->1 at g1=1: 0.954 -> 0.809 (swing = -0.145)
    #    g1's swing is about 5-20x larger than absent's. g1 is by far the more
    #    informative parent of grade; absent nudges the answer but g1 all but
    #    decides it.
    print("f) [see comment: g1 swings P(grade=1) by ~0.63-0.75, "
          "absent swings it by only ~0.03-0.15 -> g1 carries far more information]")

    # g) total free parameters vs. an unrestricted joint over 8 binary variables
    n_params = sum(2 ** len(PARENTS[v]) for v in VARS)
    print(f"g) free parameters in the network: {n_params} "
          f"(one CPT row per parent configuration, per variable)")
    print(f"   an unrestricted joint over 8 binary variables needs 2**8 - 1 = "
          f"{2**8 - 1} free parameters")
    # The graph needs 40 numbers instead of 255 -- a >6x reduction -- and that
    # reduction is entirely a consequence of which conditional independences
    # the graph asserts.


# ==========================================================================
# PROBLEM STATEMENT 3 - Factorization of the Joint Distribution
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 3: Factorization of the Joint Distribution =====")

    # h) joint() is defined above, in the CORE MACHINERY section.

    # i) sum over all 256 assignments
    cpt1 = fit(data, alpha=1.0)  # alpha=1 avoids the nan rows from Problem Statement 6
    total = 0.0
    for bits in itertools.product([0, 1], repeat=len(VARS)):
        a = dict(zip(VARS, bits))
        total += joint(a, cpt1)
    print(f"i) sum of joint() over all 256 assignments = {total:.10f} (expect 1.0)")

    # j) written answer:
    print("j) [see comment in source for the written answer]")
    # j) If the total were not 1, the two possible faults look different from
    #    the printed output alone. A CPT row that does not sum to 1 (e.g. a
    #    bug that returns P(v=1|pa) and P(v=0|pa) computed from different
    #    denominators) will generally push the *total* away from 1 by an
    #    amount roughly proportional to how often that row's parent
    #    configuration occurs -- rare misconfigurations barely move the sum,
    #    common ones move it a lot, and the error direction is consistent
    #    across every assignment that uses that row. A parent looked up in the
    #    wrong order (e.g. reading cpt["grade"][(absent, g1)] when the table
    #    was built keyed by (g1, absent)) instead scrambles which row is used
    #    for which assignment; every row still individually sums to 1 (since
    #    each is a valid p and 1-p pair), so the grand total still comes out
    #    at 1.0, but per-query answers computed with infer() would be visibly
    #    wrong even though this global check passes. So: total far from 1 ->
    #    look for a bad denominator; total exactly 1 but infer() answers look
    #    wrong -> look for a parent-order mismatch.


# ==========================================================================
# PROBLEM STATEMENT 4 - Exact Inference by Enumeration
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 4: Exact Inference by Enumeration =====")

    # k) infer() is defined above, in the CORE MACHINERY section.

    # l) six queries, model vs. data
    queries = [
        ({"grade": 1}, {}),
        ({"grade": 1}, {"g1": 1}),
        ({"grade": 1}, {"fail": 0}),
        ({"grade": 1}, {"g1": 1, "absent": 1}),
        ({"grade": 1}, {"study": 1, "goout": 0}),
        ({"grade": 1}, {"fail": 1}),
    ]
    print("l) query                                model    data     n")
    for q, e in queries:
        model = infer(q, e, cpt1)
        frac, n = data_fraction(data, q, e)
        label = "P(grade=1" + ("" if not e else " | " + ", ".join(f"{k}={v}" for k, v in e.items())) + ")"
        print(f"   {label:38s} {model:.3f}    {frac:.3f}    {n}")

    # m) written answer:
    print("m) [see comment in source for the written answer]")
    # m) Five queries land within about 0.03-0.06 of the data fraction, well
    #    inside what 649 (or fewer, once conditioned) students would produce
    #    just from sampling noise. The one that does not agree is
    #    P(grade=1 | fail=1): the model says 0.281 but only 7.0% (7/100) of
    #    students who had failed before actually pass. That gap (0.28 vs
    #    0.07) is far too large to blame on the n=100 sample size -- it is
    #    set aside here and investigated in Problem Statement 8.

    # n) written answer:
    print("n) [see comment in source for the written answer]")
    # n) infer() visits every one of the 2**k assignments to the k variables
    #    (k=8 here, so 256), regardless of how small the evidence or query is,
    #    because joint() has to be evaluated once per assignment before you
    #    know whether it is consistent with the evidence. That cost is
    #    exponential in the number of variables. 256 is nothing for a
    #    computer, but the same routine on, say, 40 binary variables would
    #    need 2**40 (about 10^12) evaluations -- this is the point at which
    #    naive enumeration stops being usable and you need either a smarter
    #    exact algorithm (variable elimination / junction trees, later in this
    #    course) or an approximate one (Problem Statement 7).


# ==========================================================================
# PROBLEM STATEMENT 5 - Conditional Independence and Explaining Away
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 5: Conditional Independence and Explaining Away =====")

    # o) three pairs the graph claims are marginally independent
    print("o) pairs with no edge and no common ancestor:")
    pairs = [("medu", "goout"), ("goout", "fail"), ("goout", "higher")]
    for a, b in pairs:
        p_ab = count(data, {a: 1, b: 1}) / len(data)
        p_a = marginals[a]
        p_b = marginals[b]
        print(f"   P({a}=1,{b}=1)={p_ab:.4f}   P({a}=1)P({b}=1)={p_a*p_b:.4f}")
    # All three are close (within about 0.006-0.01 of each other), which is
    # small next to n=649 -- the data is broadly consistent with the graph's
    # claim that goout carries no information about medu, fail, or higher.

    # p) conditioning on the collider g1
    print("p) conditioning on the collider g1:")
    p_medu, _ = data_fraction(data, {"medu": 1}, {})
    p_medu_g0, n1 = data_fraction(data, {"medu": 1}, {"g1": 0})
    p_medu_g0_go0, n2 = data_fraction(data, {"medu": 1}, {"g1": 0, "goout": 0})
    p_medu_g0_go1, n3 = data_fraction(data, {"medu": 1}, {"g1": 0, "goout": 1})
    print(f"   P(medu=1)                     = {p_medu:.3f}")
    print(f"   P(medu=1 | g1=0)              = {p_medu_g0:.3f}  (n={n1})")
    print(f"   P(medu=1 | g1=0, goout=0)     = {p_medu_g0_go0:.3f}  (n={n2})")
    print(f"   P(medu=1 | g1=0, goout=1)     = {p_medu_g0_go1:.3f}  (n={n3})")

    # q) written answer:
    print("q) [see comment in source for the written answer]")
    # q) medu is not a direct parent of g1, but its downstream effects
    #    (fail, higher, study) are, so g1 is a common effect of paths that
    #    start at medu and paths that start at goout -- conditioning on it
    #    is exactly the sprinkler/rain situation from Practical 2, T7: g1=0
    #    is the "wet grass", and low medu (via fail/higher/study) and high
    #    goout are the two competing explanations (the "sprinkler" and the
    #    "rain"). Among students who scored low (g1=0), those who also went
    #    out a lot (goout=1) already have an explanation for the low score,
    #    so the model needs the other explanation less -- P(medu=1) rises
    #    back toward its unconditional value (0.451, close to the base rate
    #    0.484). Students who scored low but did NOT go out a lot (goout=0)
    #    are left needing some other explanation, and low maternal education
    #    (acting through fail/higher/study) is more often that explanation --
    #    so P(medu=1) drops further, to 0.367. That is explaining away: one
    #    confirmed cause reduces the inferred likelihood of the other.

    # r) repeat conditioning on grade (a descendant of the collider) instead of g1
    print("r) conditioning on grade instead of g1 (a descendant of the collider):")
    p_medu_gr0, m1 = data_fraction(data, {"medu": 1}, {"grade": 0})
    p_medu_gr0_go0, m2 = data_fraction(data, {"medu": 1}, {"grade": 0, "goout": 0})
    p_medu_gr0_go1, m3 = data_fraction(data, {"medu": 1}, {"grade": 0, "goout": 1})
    print(f"   P(medu=1 | grade=0)               = {p_medu_gr0:.3f}  (n={m1})")
    print(f"   P(medu=1 | grade=0, goout=0)      = {p_medu_gr0_go0:.3f}  (n={m2})")
    print(f"   P(medu=1 | grade=0, goout=1)      = {p_medu_gr0_go1:.3f}  (n={m3})")
    # The same direction shows up (goout=1 still nudges P(medu=1) up: 0.391
    # vs. 0.369) but the gap is smaller than it was through g1 (0.451 vs.
    # 0.367). Conditioning on a descendant of a collider (grade, which is
    # downstream of g1) opens the same path between medu and goout, because
    # learning grade is informative about g1 which is informative about the
    # collider structure -- but the signal is diluted by the extra edge
    # (g1 -> grade also passes through absent, and grade is a noisier,
    # once-more-removed proxy for the original collider), so the induced
    # dependence is weaker, not absent.

    # s) an independence the graph gets wrong
    print("s) checking the graph's stronger claim: grade _||_ fail | (g1, absent)")
    for g1v in (0, 1):
        for av in (0, 1):
            p_f1, n_f1 = data_fraction(data, {"grade": 1}, {"fail": 1, "g1": g1v, "absent": av})
            p_f0, n_f0 = data_fraction(data, {"grade": 1}, {"fail": 0, "g1": g1v, "absent": av})
            print(f"   g1={g1v}, absent={av}: P(grade=1|fail=1,..)={p_f1:.3f} (n={n_f1})   "
                  f"P(grade=1|fail=0,..)={p_f0:.3f} (n={n_f0})")
    # By the local Markov property, the graph asserts grade is independent of
    # its non-descendant fail once g1 and absent (grade's parents) are known
    # -- i.e. it claims fail adds nothing about grade beyond what g1 and
    # absent already say. The data flatly rejects this: in every one of the
    # four (g1, absent) cells, having failed before roughly halves-to-tenths
    # the chance of a passing final grade compared to otherwise identical
    # students who had not failed (e.g. g1=0, absent=0: 0.017 vs. 0.260).
    # The repair is a direct edge fail -> grade, which is exactly the edge
    # proposed and tested in Problem Statement 8.


# ==========================================================================
# PROBLEM STATEMENT 6 - Sparse Counts and Laplace Smoothing
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 6: Sparse Counts and Laplace Smoothing =====")

    # t) list every degenerate row of g1 at alpha=0
    print("t) degenerate rows of g1 at alpha=0  (parents: fail, study, goout, higher):")
    degenerate = []
    for pa in configs(PARENTS["g1"]):
        cond = dict(zip(PARENTS["g1"], pa))
        n1 = count(data, {**cond, "g1": 1})
        n0 = count(data, {**cond, "g1": 0})
        if n0 + n1 == 0:
            print(f"   fail={pa[0]} study={pa[1]} goout={pa[2]} higher={pa[3]}: "
                  f"undefined (0/0), n=0")
            degenerate.append(pa)
        elif n1 == 0 or n0 == 0:
            p = n1 / (n0 + n1)
            print(f"   fail={pa[0]} study={pa[1]} goout={pa[2]} higher={pa[3]}: "
                  f"P(g1=1|..)={p:.1f}  (n1={n1}, n0={n0}, n={n0+n1})")
            degenerate.append(pa)
    print(f"   total degenerate rows: {len(degenerate)}")

    # u) written answer:
    print("u) [see comment in source for the written answer]")
    # u) A CPT row that is exactly 0 (or exactly 1) forces P(assignment) to 0
    #    for every one of the 2**8 full assignments that pass through that
    #    row -- infer() will then confidently rule out that entire branch of
    #    the joint distribution, even though "never observed among 649
    #    students" is not the same claim as "impossible". A 0/0 row is worse:
    #    cpt[v][pa] is nan, so any joint() call that needs it returns nan, and
    #    any infer() query whose evidence forces that parent configuration
    #    divides nan by nan -- the routine cannot answer the query at all
    #    rather than answering it wrongly.

    # v) refit with alpha=1
    cpt_smoothed = fit(data, alpha=1.0)
    print("v) same rows, refit with alpha=1:")
    for pa in degenerate:
        cond = dict(zip(PARENTS["g1"], pa))
        n1 = count(data, {**cond, "g1": 1})
        n0 = count(data, {**cond, "g1": 0})
        p = cpt_smoothed["g1"][pa]
        print(f"   fail={pa[0]} study={pa[1]} goout={pa[2]} higher={pa[3]}: "
              f"P(g1=1|..)={p:.3f}  (was n1={n1}, n0={n0})")
    # Every row is now a genuine probability strictly between 0 and 1. This is
    # the same fix as Practical 3, T8: the die that never showed a 6 in 30
    # rolls got assigned P(6)=0 under plain maximum likelihood; Laplace
    # smoothing (alpha=1 is "add one success, one failure" per outcome) pulls
    # every estimate away from the edges of [0,1] in proportion to how little
    # data supports it, without changing well-supported rows very much.

    # w) alpha sweep for P(grade=1 | g1=1)
    print("w) P(grade=1 | g1=1) as alpha varies:")
    for a in (0.01, 0.1, 0.5, 1.0, 2.0):
        cpt_a = fit(data, alpha=a)
        val = infer({"grade": 1}, {"g1": 1}, cpt_a)
        print(f"   alpha={a:<5} -> P(grade=1|g1=1) = {val:.4f}")
    # The answer barely moves (0.918 down to 0.907) because grade's own CPT
    # rows are all well-supported (each of the 4 rows in Problem Statement 2e
    # rests on 68-240 students) -- smoothing only matters where counts are
    # thin, as in g1's table above. Bigger alpha means more smoothing: it
    # protects against the 0/1/undefined rows seen in t), at the cost of
    # dragging every estimate toward 0.5 even where the data was already
    # confident. Small alpha stays faithful to well-supported rows but
    # reintroduces instability wherever a parent configuration is rare.


# ==========================================================================
# PROBLEM STATEMENT 7 - Ancestral and Rejection Sampling
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 7: Ancestral and Rejection Sampling =====")
    random.seed(42)  # reproducible sample counts for this run

    # x) forward() is defined above, in the CORE MACHINERY section.
    cpt_s = fit(data, alpha=1.0)
    print("x) one simulated student:", forward(cpt_s))

    # y) written answer:
    print("y) [see comment in source for the written answer]")
    # y) forward() draws v with p = cpt[v][pa], where pa is read off the
    #    values *already stored* in the assignment dict for v's parents. That
    #    only produces a correct sample from the joint distribution if every
    #    parent has already been sampled by the time its child's turn comes --
    #    which is exactly what a topological order guarantees. Draw a
    #    variable before one of its parents and pa would either be missing
    #    (KeyError) or would have to fall back to some placeholder value,
    #    silently sampling from the wrong conditional distribution and
    #    breaking the factorisation the whole network relies on.

    # z) rejection sampling: P(grade=1 | fail=1)
    exact_fail = infer({"grade": 1}, {"fail": 1}, cpt_s)
    print(f"z) exact P(grade=1 | fail=1) = {exact_fail:.4f}")
    for M in (2000, 50000):
        est, ret, kept = rejection_sample(cpt_s, {"grade": 1}, {"fail": 1}, M)
        print(f"   M={M:<6} rejection estimate = {est:.4f}   "
              f"retained {ret*100:.2f}% ({kept}/{M})")

    # aa) rejection sampling with more specific evidence
    ev2 = {"g1": 0, "goout": 1, "study": 1, "higher": 0}
    exact_ev2 = infer({"grade": 1}, ev2, cpt_s)
    print(f"aa) exact P(grade=1 | {ev2}) = {exact_ev2:.4f}")
    for M in (2000, 50000):
        est, ret, kept = rejection_sample(cpt_s, {"grade": 1}, ev2, M)
        print(f"    M={M:<6} rejection estimate = {est:.4f}   "
              f"retained {ret*100:.2f}% ({kept}/{M})")

    # ab) written answer:
    print("ab) [see comment in source for the written answer]")
    # ab) Retention collapses from roughly 15-16% for the single-variable
    #    evidence {fail=1} down to well under 1% for the four-variable
    #    evidence in aa), because each extra evidence variable multiplies the
    #    survival probability by another factor less than 1 -- retention
    #    falls off exponentially in the number of (and rarity of) evidence
    #    variables. At M=2000 with the four-variable evidence, only a
    #    handful of samples survive at all, so the estimate is extremely
    #    noisy (it can be far from the exact answer or even undefined by
    #    chance); M=50000 helps, but only by brute force, and real networks
    #    have far more variables and far rarer evidence combinations than
    #    this one does. That is why rejection sampling is a teaching tool
    #    rather than a practical algorithm, and why the course moves on to
    #    likelihood weighting and Markov Chain Monte Carlo (Gibbs sampling),
    #    which fix every sample to the evidence instead of throwing samples
    #    away.


# ==========================================================================
# PROBLEM STATEMENT 8 - Model Criticism
# ==========================================================================
if __name__ == "__main__":
    print("\n===== PROBLEM STATEMENT 8: Model Criticism =====")

    # ac) model vs. observed
    cpt_c = fit(data, alpha=1.0)
    model_val = infer({"grade": 1}, {"fail": 1}, cpt_c)
    obs_val, obs_n = data_fraction(data, {"grade": 1}, {"fail": 1})
    print(f"ac) model P(grade=1|fail=1) = {model_val:.3f}   "
          f"observed = {obs_val:.3f} (n={obs_n})   "
          f"discrepancy = {model_val - obs_val:.3f}")

    # ad) alpha sweep to rule out smoothing as the cause
    print("ad) P(grade=1|fail=1) as alpha varies:")
    for a in (0.01, 0.1, 1.0, 2.0):
        cpt_a = fit(data, alpha=a)
        val = infer({"grade": 1}, {"fail": 1}, cpt_a)
        print(f"    alpha={a:<5} -> {val:.4f}")
    # The estimate moves from about 0.235 to 0.312 as alpha runs from 0.01 to
    # 2.0 -- real, but a fraction of the roughly 0.21 gap between the model
    # (~0.28) and the data (0.07). No value of alpha gets anywhere close to
    # 0.07, so smoothing is not the explanation; the graph's structure is.

    # ae) trace paths from fail to grade, add the missing edge, refit
    print("ae) [see comment in source: paths fail->grade and the fix]")
    # ae) Every directed path from fail to grade goes through at least one of
    #    g1 or absent: fail->g1->grade; fail->higher->g1->grade;
    #    fail->higher->study->g1->grade; fail->absent->grade. Each of these
    #    paths transmits fail's effect only by nudging a binary intermediate
    #    variable, and a binary cut can only carry so much: a fail=1 student
    #    who happens to still clear the g1>=12 and absences<6 thresholds
    #    looks, to this graph, exactly like a fail=0 student in the same
    #    (g1, absent) cell -- but Problem Statement 5s) showed that is false:
    #    conditioning on g1 and absent does not screen off fail's effect on
    #    grade at all. The missing piece is a direct edge fail -> grade,
    #    representing whatever about having failed before (weaker
    #    fundamentals, a harder recovery) keeps depressing the final grade
    #    even among students who matched their g1/absent peers.
    PARENTS_FIXED = dict(PARENTS)
    PARENTS_FIXED["grade"] = ["g1", "absent", "fail"]
    cpt_fixed = fit(data, alpha=1.0, parents=PARENTS_FIXED)
    corrected = infer({"grade": 1}, {"fail": 1}, cpt_fixed, parents=PARENTS_FIXED)
    print(f"    corrected model (with fail -> grade added): "
          f"P(grade=1|fail=1) = {corrected:.3f}   observed = {obs_val:.3f}")
    # Adding the single edge moves the estimate from 0.281 to about 0.120 --
    # much closer to the observed 0.070, and the remaining gap is consistent
    # with the small sample (only 100 students have fail=1).

    # af) written answer:
    print("af) [see comment in source for the written answer]")
    # af) absences accumulates over the entire academic year, while G1 (and
    #    so g1) is fixed at roughly the midpoint of that same year. That means
    #    a large share of the absences behind "absent=1" for many students
    #    happened *after* g1 was already recorded -- so g1 cannot possibly
    #    have screened off the part of absent's influence on grade that comes
    #    from second-half absences, because that information did not exist
    #    yet when g1 was measured. The graph's edges absent->grade and
    #    g1->grade quietly assume all the variables are measured at
    #    comparable times; here they are not. A correct treatment would split
    #    absences into a pre-midpoint count (which g1 could plausibly
    #    mediate) and a post-midpoint count (which cannot be screened off by
    #    g1 at all and should point at grade directly, much like fail).

    # ag) written conclusion:
    print("ag) [see comment in source for the written conclusion]")
    # ag) This DAG was written down from a story about the order events
    #    happen in a school year, not learned from the 649 students it was
    #    then fit to -- and the data contradicted that story in specific,
    #    checkable places: fail's effect on grade survives conditioning on
    #    g1 and absent (Problem Statements 5s and 8ac-ae), and absent mixes
    #    timescales that g1 cannot mediate between (8af). Both are structural
    #    complaints, not estimation-noise complaints, and no amount of
    #    re-fitting or smoothing (8ad) fixes a wrong edge set. The appropriate
    #    remedy is to let the data help choose the graph rather than
    #    asserting it up front -- adding edges such as fail->grade where
    #    independence tests fail, and more generally running a structure-
    #    learning algorithm (constraint-based methods built on conditional
    #    independence tests, or score-based search over DAGs) -- which is the
    #    topic this course turns to next: learning the structure of a
    #    Bayesian network, not just its parameters.
