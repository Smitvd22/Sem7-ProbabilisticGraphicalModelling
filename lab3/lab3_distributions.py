"""
AI453  --  Practical #3: Distributions, Sampling and Maximum Likelihood

Deck 3 and the first half of Deck 4, in code. You write eight small
functions; each one is a few lines.

Plain Python 3, standard library only. Nothing to install.

Run:  python3 lab3_distributions.py
It prints the worked example, then stops at T1.
"""

import math
import random

# A fixed seed, so your own runs repeat. It does NOT make your numbers match
# your neighbour's -- as soon as two people draw a different number of samples,
# the streams diverge. Anything you estimate from samples will differ a little
# from everyone else's, and that is the subject of T3. Compare your sampled
# numbers against the formulas, never against each other.
random.seed(453)


# ======================================================================
# THE DATA  --  both of these are printed on your handout.
# ======================================================================

# Twenty flips of a coin, in order. 1 = heads.
COIN = "11000100100000101000"

# A loaded six-sided die: P(1), P(2), ..., P(6). Note the last one.
DIE = [0.25, 0.20, 0.20, 0.15, 0.18, 0.02]


# ======================================================================
# GIVEN TO YOU  --  use these, do not edit them.
# ======================================================================

def mean_of(xs):
    """Average of a list."""
    return sum(xs) / len(xs)


def var_of(xs):
    """Variance of a list, as E[X^2] - (E[X])^2."""
    m = mean_of(xs)
    return sum(x * x for x in xs) / len(xs) - m * m


def ascii_hist(values):
    """Draw a histogram of a list of whole numbers."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    for v in range(min(values), max(values) + 1):
        c = counts.get(v, 0)
        print(f"    {v:>4} | {'#' * round(54 * c / top):<54} {c}")


def check(label, got, want, tol=1e-9):
    """Compare a number with what it should be, and say so."""
    ok = abs(got - want) <= tol
    print(f"[{'  ok  ' if ok else ' FAIL '}] {label:<34} "
          f"got {got:>10.4f}   want {want:>10.4f}")


# ----------------------------------------------------------------------
# WORKED EXAMPLE -- nothing to write.
#
# A Bernoulli draw: 1 with probability p, else 0. This is the one-line
# pattern that every sampler below is built out of.
# ----------------------------------------------------------------------

def sample_bernoulli(p):
    return 1 if random.random() < p else 0


print("=" * 74)
print("WORKED EXAMPLE  --  a Bernoulli draw")
print("=" * 74)

draws = [sample_bernoulli(0.3) for _ in range(10000)]
check("fraction of 1s in 10000 draws", mean_of(draws), 0.3, tol=0.02)
print()


# ======================================================================
# PART A  --  SAMPLING
# ======================================================================

# T1. Write sample_die(). It should return one roll of the loaded die
#     above, as an index 0..5 (so face 1 is 0, face 6 is 5).
#
#     The method is the INVERSE CDF, and it is four lines:
#
#         draw u = random.random(), a number in [0, 1)
#         keep a running total, adding DIE[0], then DIE[1], and so on
#         the moment the running total goes past u, return that index
#
#     Picture the line from 0 to 1 cut into six pieces, of lengths 0.25,
#     0.20, 0.20, 0.15, 0.18, 0.02. The question is: which piece did u
#     land in? This is how every discrete distribution in this course is
#     sampled.
#
#     Then draw 10000 rolls and print the fraction of each face. They
#     should match DIE.
# ----------------------------------------------------------------------

def sample_die():
    u = random.random()
    total = 0.0
    for i, p in enumerate(DIE):
        total += p
        if total > u:
            return i
    return len(DIE) - 1


print("=" * 74)
print("T1  sampling the die")
print("=" * 74)

draws = [sample_die() for _ in range(10000)]
fractions = [draws.count(i) / 10000 for i in range(6)]
for i in range(6):
    check(f"face {i+1} fraction", fractions[i], DIE[i], tol=0.02)


# T2. Write sample_binomial(n, p): the number of 1s in n Bernoulli draws.
#     Do NOT use a formula with factorials. Call sample_bernoulli(p) n
#     times and add up what you get -- that is what "the Binomial is a
#     sum of Bernoullis" means, and T5 depends on your having seen it
#     that way.
#
#     Draw 10000 values of sample_binomial(10, 0.3) and check that their
#     mean is about n*p = 3 and their variance about n*p*(1-p) = 2.1.
# ----------------------------------------------------------------------

def sample_binomial(n, p):
    return sum(sample_bernoulli(p) for _ in range(n))


print("=" * 74)
print("T2  the Binomial is a sum of Bernoullis")
print("=" * 74)

draws = [sample_binomial(10, 0.3) for _ in range(10000)]
check("binomial mean", mean_of(draws), 10 * 0.3, tol=0.1)
check("binomial var", var_of(draws), 10 * 0.3 * 0.7, tol=0.1)


# ======================================================================
# PART B  --  EXPECTATION AND VARIANCE
# ======================================================================

# T3. Deck 3 gives the mean and variance of the die by formula:
#
#         E[X]   = sum over k of  k * DIE[k]
#         Var(X) = sum over k of  k*k * DIE[k]  -  E[X]^2
#
#     (k is the 0-based index, matching what sample_die returns.)
#
#     Compute both from the formula and print them. Then draw n rolls
#     and estimate both with mean_of() and var_of(), for
#     n = 100, 1000, 10000, 100000. Print a small table of the errors.
#
#     Do the two formula values on paper first -- it is twelve
#     multiplications -- then check the computer agrees.
#
#     One line in a comment: each time n goes up by a factor of 100,
#     what happens to the error? Deck 3 predicted this.
# ----------------------------------------------------------------------

print("=" * 74)
print("T3  mean and variance: formula against data")
print("=" * 74)

mean_formula = sum(k * DIE[k] for k in range(6))
var_formula = sum(k * k * DIE[k] for k in range(6)) - mean_formula ** 2

print(f"Formula mean: {mean_formula:.4f}, variance: {var_formula:.4f}")

for n in [100, 1000, 10000, 100000]:
    draws = [sample_die() for _ in range(n)]
    m = mean_of(draws)
    v = var_of(draws)
    print(f"n = {n:<6}: mean error = {abs(m - mean_formula):.4f}, var error = {abs(v - var_formula):.4f}")

# ANSWER:
# Each time n goes up by a factor of 100, the error goes down by a factor of 10.
# This happens because the standard error is proportional to 1/sqrt(n).


# T4. Variance of a sum. Draw 20000 pairs (x, y), twice over:
#
#         independent:  x and y each a separate sample_bernoulli(0.3)
#         coupled:      x a sample_bernoulli(0.3), and then y = x
#
#     Both times, estimate Var(X + Y) with var_of(). Compare each against
#     Var(X) + Var(Y) = 0.21 + 0.21 = 0.42.
#
#     One agrees and one does not. In a comment: what is the extra term
#     in the general formula, what is its value in the coupled case, and
#     when does it vanish?
# ----------------------------------------------------------------------

print("=" * 74)
print("T4  variance of a sum")
print("=" * 74)

draws_indep_x = [sample_bernoulli(0.3) for _ in range(20000)]
draws_indep_y = [sample_bernoulli(0.3) for _ in range(20000)]
sum_indep = [x + y for x, y in zip(draws_indep_x, draws_indep_y)]

draws_coupled_x = [sample_bernoulli(0.3) for _ in range(20000)]
sum_coupled = [x + x for x in draws_coupled_x]

var_indep = var_of(sum_indep)
var_coupled = var_of(sum_coupled)

check("Var(X + Y) independent", var_indep, 0.42, tol=0.02)
check("Var(X + Y) coupled", var_coupled, 0.42, tol=0.02)

# ANSWER:
# The extra term in the general formula Var(X + Y) is 2 * Cov(X, Y).
# In the coupled case (Y = X), this term is 2 * Var(X) = 0.42.
# It vanishes when X and Y are uncorrelated (which includes when they are independent).


# ======================================================================
# PART C  --  WHY GAUSSIANS ARE EVERYWHERE
# ======================================================================

# T5. Deck 3 claimed that adding up many independent things gives you a
#     bell curve, whatever you started from. Check it.
#
#     For k = 1, 5, 30, 100: build a list of 5000 values, each one the
#     sum of k draws of sample_bernoulli(0.1), and ascii_hist() it.
#
#     Bernoulli(0.1) is about as far from a bell as a distribution gets
#     -- at k = 1 your histogram is two bars. Look at k = 100.
#
#     Hint: sample_binomial(k, 0.1) already IS the sum of k draws of
#     sample_bernoulli(0.1). You wrote it in T2.
#
#     For each k, also print the mean and variance of your 5000 sums
#     beside k*0.1 and k*0.09, which is what Deck 3 predicts for a sum of
#     k independent copies.
#
#     Two lines in a comment. The shape changed completely between k = 1
#     and k = 100, but those two predictions held at every k, including
#     k = 1 where nothing is remotely bell-shaped. So what did the
#     central limit theorem actually give you -- the mean, the variance,
#     or the shape?
# ----------------------------------------------------------------------

print("=" * 74)
print("T5  sums of many things")
print("=" * 74)

for k in [1, 5, 30, 100]:
    sums = [sample_binomial(k, 0.1) for _ in range(5000)]
    print(f"\n--- k = {k} ---")
    ascii_hist(sums)
    print(f"Mean: got {mean_of(sums):.4f} want {k * 0.1:.4f}")
    print(f"Var:  got {var_of(sums):.4f} want {k * 0.09:.4f}")

# ANSWER:
# The central limit theorem actually gave us the shape (the bell shape).
# The mean and variance predictions hold regardless of the distribution's shape, as long as the variables are independent.


# ======================================================================
# PART D  --  MAXIMUM LIKELIHOOD
# ======================================================================

# T6. The twenty coin flips at the top of the file. Read them with
#         flips = [int(c) for c in COIN]
#
#     Write likelihood(theta, flips): the probability of getting exactly
#     these flips if the coin comes up heads with probability theta. It
#     is theta for each head and (1 - theta) for each tail, all
#     multiplied together.
#
#     Then try every theta from 0.00 to 1.00 in steps of 0.01, and print
#     the one that gives the largest value. Compare it with
#     (number of heads) / 20.
#
#     You are not being asked to trust the lecture's formula. You are
#     checking every possible theta by brute force, and finding that the
#     best one is where the lecture said it would be.
# ----------------------------------------------------------------------

def likelihood(theta, flips):
    p = 1.0
    for f in flips:
        if f == 1:
            p *= theta
        else:
            p *= (1 - theta)
    return p


print("=" * 74)
print("T6  maximum likelihood, by brute force")
print("=" * 74)

flips = [int(c) for c in COIN]
best_theta = 0.0
best_l = -1.0
for i in range(101):
    theta = i / 100.0
    l = likelihood(theta, flips)
    if l > best_l:
        best_l = l
        best_theta = theta

print(f"Best theta: {best_theta:.2f}")
print(f"Heads fraction: {sum(flips) / 20:.2f}")


# T7. Print likelihood(0.3, flips * 65). That is the same twenty flips
#     repeated 65 times: 1300 flips, which is a small dataset.
#
#     Look at what comes out. Then write log_likelihood(theta, flips),
#     which adds up math.log(...) instead of multiplying, and print that
#     for the same 1300 flips.
#
#     One line in a comment: what happened to the first number, and why
#     does every library in this course work in logs?
# ----------------------------------------------------------------------

def log_likelihood(theta, flips):
    ll = 0.0
    for f in flips:
        if f == 1:
            ll += math.log(theta)
        else:
            ll += math.log(1 - theta)
    return ll


print("=" * 74)
print("T7  why everything is done in logs")
print("=" * 74)

flips = [int(c) for c in COIN]
l_1300 = likelihood(0.3, flips * 65)
print(f"Likelihood of 1300 flips: {l_1300}")

print(f"Log-likelihood of 1300 flips: {log_likelihood(0.3, flips * 65)}")

# ANSWER:
# The first number underflows to 0.0 because multiplying many small probabilities results in a value too small for floating-point representation.
# Every library works in logs to prevent this underflow, by turning multiplication into addition.


# T8. Maximum likelihood for the die is just counting: the estimate of
#     P(face k) is (number of times face k came up) / (total rolls).
#
#     Write fit_die(rolls) returning a list of six such estimates.
#     Check it on 100000 rolls -- you should get DIE back.
#
#     NOW BREAK IT. Face 6 has probability 0.02, so in a short run it
#     often does not appear at all. Fit the die on just 30 rolls, and do
#     that 100 times over. Count how many of the 100 estimates give face
#     6 a probability of exactly 0. Compare with 0.98^30, which you can
#     work out with a pen.
#
#     Then take one of those estimates and use it to score a NEW
#     sequence of ten rolls in which face 6 appears once -- multiply the
#     ten probabilities together. Print the result.
#
#     Two or three lines in a comment. Your estimate says face 6 is
#     impossible. It is not: you were told at the top of this file that
#     its probability is 0.02, and the die does not care what you saw in
#     thirty rolls. Is "I have never seen it, so it cannot happen" a
#     reasonable thing for an estimate to say? If not, what would you
#     rather it did -- and where could that information come from, given
#     that it is definitely not in your thirty rolls?
#
#     You are not expected to know the fix. The next lecture is the
#     answer.
# ----------------------------------------------------------------------

def fit_die(rolls):
    counts = [0] * 6
    for r in rolls:
        counts[r] += 1
    return [c / len(rolls) for c in counts]


print("=" * 74)
print("T8  where maximum likelihood breaks")
print("=" * 74)

rolls_100k = [sample_die() for _ in range(100000)]
fit_100k = fit_die(rolls_100k)
print("Fit on 100000 rolls:", [round(x, 4) for x in fit_100k])
print("True DIE:          ", DIE)

count_zero = 0
one_estimate_with_zero = None
for _ in range(100):
    rolls_30 = [sample_die() for _ in range(30)]
    fit_30 = fit_die(rolls_30)
    if fit_30[5] == 0:
        count_zero += 1
        one_estimate_with_zero = fit_30

print(f"Times face 6 had probability 0: {count_zero} (out of 100)")
print(f"0.98^30 is approx: {0.98**30:.4f}")

new_rolls = [0]*9 + [5]
prob_new = 1.0
for r in new_rolls:
    prob_new *= one_estimate_with_zero[r]
print(f"Probability of new 10 rolls: {prob_new}")

# ANSWER:
# "I have never seen it, so it cannot happen" is not a reasonable thing for an estimate to say, because it makes the model completely intolerant to new evidence.
# We would rather it assigned a small non-zero probability, which could come from a prior belief (like Laplace smoothing/pseudocounts) rather than just the observed data.


# ======================================================================
# IF YOU FINISH EARLY
# ======================================================================
#
# 1. Uncorrelated is not independent. Let X be equally likely to be -1,
#    0 or 1, and let Y = X * X. Estimate their covariance from samples:
#        cov = mean_of([x*y ...]) - mean_of(xs) * mean_of(ys)
#    You will get 0. But Y is computed from X, so they are about as
#    dependent as two things can be. Deck 3 asserted this; build the
#    counterexample yourself.
#
# 2. Sample a Gaussian. Box-Muller turns two uniforms into one standard
#    normal draw:
#        z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
#    Then mu + sigma * z has mean mu and variance sigma^2. Check it with
#    mean_of() and var_of(). (Why 1 - u1 and not u1?)
#
# 3. In T5, how good is the bell at k = 100 in the middle, and how good
#    is it at the far right edge? The answer to the second half is why
#    Unit 4 needs sampling methods at all.
