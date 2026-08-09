"""
AI453 Probabilistic Graphical Models  --  Practical #1: Introduction to Probability Theory
SVNIT Surat, Department of Artificial Intelligence

You are given ONE joint distribution over three binary variables A, B, C.
Everything you compute today comes out of that one table. Nothing is loaded,
downloaded, or estimated from data.

Run:  python3 lab1_simple.py
Dependencies: NONE. Plain Python 3.
"""

# ----------------------------------------------------------------------
# THE JOINT DISTRIBUTION P(A, B, C)
#
# Three binary variables A, B, C, each 0 or 1.  Eight combinations, eight
# numbers.  The key (a, b, c) means "A=a and B=b and C=c".
#
#       P[(1, 0, 1)]  is  P(A=1, B=0, C=1)  =  0.06
# ----------------------------------------------------------------------
P = {
    #  A  B  C        probability
    (0, 0, 0): 0.06,
    (0, 0, 1): 0.24,
    (0, 1, 0): 0.04,
    (0, 1, 1): 0.16,
    (1, 0, 0): 0.09,
    (1, 0, 1): 0.06,
    (1, 1, 0): 0.21,
    (1, 1, 1): 0.14,
}


# ----------------------------------------------------------------------
# WORKED EXAMPLE  --  read this carefully, every task below is this loop again
#
#   P(A=1)  =  sum of P(A=1, B=b, C=c)  over every b and every c
#
# In words: walk through all eight rows, and add up the ones where A is 1.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        total += p
print("P(A=1) =", total)

# That is the whole idea. A marginal is a sum over the rows that match.
# A conditional is one such sum divided by another.


# ----------------------------------------------------------------------
# T1.  Check that the table is a valid distribution: all eight numbers
#      must add up to 1.  Print the total.
# ----------------------------------------------------------------------
total_T1 = sum(P.values())
print("T1: Total probability =", total_T1)


# ----------------------------------------------------------------------
# T2.  Compute and print P(B=1).
#      Same loop as the worked example, different condition.
# ----------------------------------------------------------------------
total_T2 = 0.0
for (a, b, c), p in P.items():
    if b == 1:
        total_T2 += p
print("T2: P(B=1) =", total_T2)


# ----------------------------------------------------------------------
# T3.  Compute and print P(C=1).
# ----------------------------------------------------------------------
total_T3 = 0.0
for (a, b, c), p in P.items():
    if c == 1:
        total_T3 += p
print("T3: P(C=1) =", total_T3)


# ----------------------------------------------------------------------
# T4.  Compute and print the joint P(A=1, B=1).
#      Now the condition has two parts.
# ----------------------------------------------------------------------
total_T4 = 0.0
for (a, b, c), p in P.items():
    if a == 1 and b == 1:
        total_T4 += p
print("T4: P(A=1, B=1) =", total_T4)


# ----------------------------------------------------------------------
# T5.  Compute and print the conditional P(C=1 | A=1).
#
#                        P(A=1, C=1)
#      P(C=1 | A=1)  =  --------------
#                          P(A=1)
#
#      Two sums, one divided by the other. Compute the top and the bottom
#      in the same loop if you like.
# ----------------------------------------------------------------------
p_a1_c1 = 0.0
p_a1 = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        p_a1 += p
        if c == 1:
            p_a1_c1 += p
print("T5: P(C=1 | A=1) =", p_a1_c1 / p_a1)


# ----------------------------------------------------------------------
# T6.  Compute and print P(B=1 | A=0, C=1).
#      Two things known, one thing asked. Same pattern.
# ----------------------------------------------------------------------
p_b1_a0_c1 = 0.0
p_a0_c1 = 0.0
for (a, b, c), p in P.items():
    if a == 0 and c == 1:
        p_a0_c1 += p
        if b == 1:
            p_b1_a0_c1 += p
print("T6: P(B=1 | A=0, C=1) =", p_b1_a0_c1 / p_a0_c1)


# ----------------------------------------------------------------------
# T7.  THE CHAIN RULE.  In class we showed that for any three variables
#
#          P(A,B,C)  =  P(A) * P(B|A) * P(C|A,B)
#
#      Check it numerically. For every one of the eight rows (a,b,c):
#        - look up P(A=a, B=b, C=c) straight from the table
#        - separately compute P(A=a), then P(B=b|A=a), then P(C=c|A=a,B=b)
#          and multiply the three together
#        - print both numbers side by side and say whether they match
#          (allow a tiny difference, e.g. 1e-9, for floating point)
#
#      Then answer in a comment: does the chain rule hold only for THIS
#      table, or for every joint distribution? Why?
# ----------------------------------------------------------------------
print("T7: Checking the chain rule P(A,B,C) = P(A) * P(B|A) * P(C|A,B)")
for (a_val, b_val, c_val), p_abc in P.items():
    p_a = 0.0
    p_a_b = 0.0
    for (a, b, c), p in P.items():
        if a == a_val:
            p_a += p
            if b == b_val:
                p_a_b += p
                
    p_b_given_a = p_a_b / p_a
    p_c_given_ab = p_abc / p_a_b
    chain_rule_result = p_a * p_b_given_a * p_c_given_ab
    
    match = abs(p_abc - chain_rule_result) < 1e-9
    print(f"Row ({a_val},{b_val},{c_val}): Table={p_abc:.4f}, ChainRule={chain_rule_result:.4f}, Match={match}")

# Answer: The chain rule holds for EVERY joint distribution. It is derived from the 
# basic definition of conditional probability: P(X, Y) = P(X) * P(Y | X). By applying 
# this iteratively, any joint distribution over any number of variables can be 
# decomposed into a product of conditional distributions.


# ----------------------------------------------------------------------
# T8.  BAYES' RULE.  You know P(A=1) already -- that was the worked
#      example. Now suppose you are told that C = 1. Compute
#
#          P(A=1 | C=1)
#
#      and compare it with P(A=1). Did learning C=1 make A=1 more likely
#      or less likely? Write ONE line saying by how much, and in which
#      direction.
# ----------------------------------------------------------------------
p_a1_c1_t8 = 0.0
p_c1_t8 = 0.0
p_a1_t8 = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        p_a1_t8 += p
    if c == 1:
        p_c1_t8 += p
        if a == 1:
            p_a1_c1_t8 += p

p_a1_given_c1 = p_a1_c1_t8 / p_c1_t8
print("T8: P(A=1 | C=1) =", p_a1_given_c1)
print("    P(A=1)       =", p_a1_t8)

# Learning C=1 made A=1 LESS likely, dropping its probability from 0.50 to approximately 0.33.
