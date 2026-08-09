/*
 * AI453 Probabilistic Graphical Models -- Practical #1 (C version)
 * SVNIT Surat, Department of Artificial Intelligence
 *
 * Same joint distribution, same tasks as lab1_simple.py. Use whichever
 * language you are more comfortable in -- the probability is identical.
 *
 * Build and run:   gcc -o lab1 lab1_simple.c && ./lab1
 */
#include <stdio.h>

/* ------------------------------------------------------------------
 * THE JOINT DISTRIBUTION P(A, B, C)
 *
 * Three binary variables A, B, C, each 0 or 1. Eight combinations,
 * eight numbers. P[a][b][c] is P(A=a, B=b, C=c), so
 *
 *      P[1][0][1]  is  P(A=1, B=0, C=1)  =  0.06
 * ------------------------------------------------------------------ */
double P[2][2][2] = {
    /*        C=0     C=1  */
    { /* A=0 */
        { 0.06, 0.24 },   /* B=0 */
        { 0.04, 0.16 }    /* B=1 */
    },
    { /* A=1 */
        { 0.09, 0.06 },   /* B=0 */
        { 0.21, 0.14 }    /* B=1 */
    }
};

int main(void)
{
    int a, b, c;
    double total;

    /* --------------------------------------------------------------
     * WORKED EXAMPLE -- read this carefully, every task is this loop.
     *
     *   P(A=1) = sum of P(A=1, B=b, C=c) over every b and every c
     *
     * In words: walk through all eight entries, and add up the ones
     * where A is 1.
     * -------------------------------------------------------------- */
    total = 0.0;
    for (a = 0; a < 2; a++)
        for (b = 0; b < 2; b++)
            for (c = 0; c < 2; c++)
                if (a == 1)
                    total += P[a][b][c];
    printf("P(A=1) = %g\n", total);

    /* That is the whole idea. A marginal is a sum over the entries
     * that match. A conditional is one such sum divided by another. */

    /* T1. Check the table is a valid distribution: all eight numbers
     *     must add up to 1. Print the total. */
    /* your code here */

    /* T2. Compute and print P(B=1). */
    /* your code here */

    /* T3. Compute and print P(C=1). */
    /* your code here */

    /* T4. Compute and print the joint P(A=1, B=1). */
    /* your code here */

    /* T5. Compute and print the conditional P(C=1 | A=1):
     *
     *                       P(A=1, C=1)
     *     P(C=1 | A=1)  =  -------------
     *                         P(A=1)
     *
     *     Two sums, one divided by the other. */
    /* your code here */

    /* T6. Compute and print P(B=1 | A=0, C=1). */
    /* your code here */

    /* T7. THE CHAIN RULE.  For any three variables,
     *
     *         P(A,B,C) = P(A) * P(B|A) * P(C|A,B)
     *
     *     Check it numerically. For each of the eight (a,b,c): read
     *     P[a][b][c] straight from the table; separately compute
     *     P(A=a), P(B=b|A=a), P(C=c|A=a,B=b) and multiply the three.
     *     Print both numbers side by side and say whether they match
     *     (allow a tiny difference, e.g. 1e-9, for floating point).
     *
     *     Then answer in a comment: does the chain rule hold only for
     *     THIS table, or for every joint distribution? Why? */
    /* your code here */

    /* T8. BAYES' RULE. You know P(A=1) already. Now suppose you are
     *     told that C = 1. Compute P(A=1 | C=1) and compare it with
     *     P(A=1). Did learning C=1 make A=1 more likely or less
     *     likely? One line, by how much and in which direction. */
    /* your code here */

    return 0;
}
