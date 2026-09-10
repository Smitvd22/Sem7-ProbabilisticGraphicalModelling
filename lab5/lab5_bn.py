"""
AI453 - Practical #5: A Bayesian Network from Real Data
Department of Artificial Intelligence, SVNIT Surat

Plain Python 3, standard library only. Nothing to install.
Run:  python3 lab5_bn.py
It should print one line: P(fail=1 | medu=1) = 0.099

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
#   study   studies more than 5 hours a week           (studytime >= 3)
#   absent  six or more absences this year             (absences >= 6)
#   g1      first-period grade of 12 or more           (G1 >= 12)
#   grade   final grade of 12 or more                  (G3 >= 12)

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
# parents. Check that for yourself before you rely on it in Part D.


# --------------------------------------------------------------------------
# Helpers, written for you
# --------------------------------------------------------------------------

def load(path="student-por.csv"):
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
# PART A - fitting the network
# ==========================================================================

# T1. Print the number of rows, and P(v=1) for each of the eight variables.
#     Sanity check: you should have 649 rows and P(grade=1) near 0.536.

# T2. fit(data, alpha=0.0) -> a dict cpt[v][parent_tuple] = P(v = 1 | parents).
#     For each variable v and each assignment to its parents, count as in the
#     worked example. alpha is a smoothing constant; leave it at 0 for now and
#     come back to it in T7.
#         P(v=1 | pa) = (n1 + alpha) / (n0 + n1 + 2*alpha)
#     Print the CPT of 'grade'. Four rows, one per (g1, absent) pair.


# ==========================================================================
# PART B - the graph is a claim about the distribution
# ==========================================================================

# T3. joint(assignment, cpt) -> the probability of one full assignment to all
#     eight variables, as the product over variables of P(v | its parents).
#     That product is the whole content of the graph.
#     Then loop over all 2^8 = 256 assignments and add up the results.
#     If your factorisation is right the total is 1. If it is not, you have a
#     CPT row that does not sum to 1, or a parent lookup in the wrong order.

# T4. infer(query, evidence, cpt) -> P(query | evidence), by enumeration:
#     sum joint() over every assignment consistent with the evidence, keep a
#     second running total for those that also match the query, divide.
#     This is the slow, honest, exponential way. It is the ground truth every
#     faster algorithm later in the course gets checked against.
#
#     Print, side by side, the model's answer and the raw fraction in the data:
#         P(grade=1)                 P(grade=1 | g1=1)
#         P(grade=1 | fail=0)        P(grade=1 | g1=1, absent=1)
#     Also print how many students the data fraction is based on. Where the
#     model and the data disagree, note whether the count is small.


# ==========================================================================
# PART C - checking what the graph predicts
# ==========================================================================

# T5. Read the graph, not the data. Three pairs have no edge and no common
#     ancestor, so the graph claims they are independent outright:
#         medu and goout,   goout and fail,   goout and higher
#     For each, print P(a=1, b=1) against P(a=1) P(b=1), the way you did in
#     Practical 2, T2. One line: does the data back the graph up?

# T6. Now the same thing with a collider in the way. The graph says medu and
#     goout are independent -- but they are both parents of g1, so learning g1
#     should make them dependent. Print
#         P(medu=1)      P(medu=1 | g1=0)
#         P(medu=1 | g1=0, goout=0)      P(medu=1 | g1=0, goout=1)
#     Two lines in a comment: among students who scored poorly, why does
#     learning that they go out a lot make an educated mother MORE likely?
#     This is the sprinkler from Practical 2, T7, in a real school.


# ==========================================================================
# PART D - sampling, smoothing, and where the model breaks
# ==========================================================================

# T7. Run fit() with alpha=0 and look for CPT rows that are 0, 1, or 0/0.
#     Print them. There are eight, all in g1 -- which has four parents, so 16
#     rows to fill from 649 students, and one parent combination that no
#     student in the file matches at all.
#     This is Practical 3, T8 again: the die fitted on 30 rolls gave face 6 a
#     probability of exactly 0. Refit with alpha=1 and confirm they are gone.
#     One line: what does a CPT row of exactly 0 do to any query that needs it?

# T8. forward(cpt) -> one simulated student. Walk VARS in order; each variable
#     is a sample_bernoulli() (Practical 3) with p read off its CPT row, using
#     the values you already drew for its parents. That is the only reason the
#     topological order matters.
#     Then rejection sampling: draw M students, throw away the ones that do not
#     match the evidence, and take the fraction of the survivors that match the
#     query. Estimate P(grade=1 | fail=1) with M = 2000 and M = 50000 and
#     compare with your exact answer from T4.
#     Print the percentage of samples you kept. One line: what happens to that
#     percentage as the evidence gets more specific, and why is that a problem?

# T9. The model against reality. Print the model's P(grade=1 | fail=1) beside
#     the fraction in the data. They are far apart, and re-running T7 with a
#     smaller alpha does not close the gap -- so this is not a smoothing
#     problem. Look at the graph and find every path from fail to grade.
#     Two or three lines: what edge would you add, and what does that tell you
#     about where the DAG came from in the first place?
