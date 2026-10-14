"""
AI453 - Lab 6: Bayes-Ball Reachability and the Markov Blanket
Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology, Surat

Problem Statement: Design and Implementation of Linear-Time Conditional Independence
Algorithms — Bayes-Ball Reachability and the Markov Blanket — for Directed Graphical Models.

Python 3 standard library only. No networkx, no pgmpy.

Command-line usage:
    python3 lab6.py <input_file>            -> q lines of YES / NO (default mode)
    python3 lab6.py <input_file> --blanket  -> n lines, one blanket per line (blanket mode)
    python3 lab6.py --verify                -> runs structural & numerical verifications

========================================================================================
THEORETICAL AND PRACTICAL ANALYSIS SUMMARY (Tasks a - z)
========================================================================================

Problem Statement 1: Graph Construction and Ancestor Computation
----------------------------------------------------------------
Task a: read_graph(path)
    Parses n, m, directed edges, q, and query tuples (x, y, Z).
Task b: ancestors_of(parents, Z)
    Iterative upward DFS traversal using a stack, guaranteeing O(n + m) time and O(n)
    space without recursion, preventing stack overflow on arbitrarily deep graphs.
Task c: Reflexive Ancestor Relation
    Ancestors of {3} in case_p6_sample.in are {0, 1, 2, 3}. A node v is defined as its
    own ancestor (Z is a subset of ancestors(Z)) because in d-separation, a collider
    trail u -> c <- w is active if the collider c ITSELF or any descendant of c lies in Z.
    By treating ancestors as reflexive, the collider activation condition simplifies
    to a single set-membership check: c in ancestors_of(parents, Z).

Problem Statement 2: Bayes-Ball Reachability Algorithm
------------------------------------------------------
Task d: reachable(parents, children, x, Z)
    Traverses over (node, direction) states starting from (x, 'up').
    Direction 'up' = arrived from child (against arrow).
    Direction 'down' = arrived from parent (along arrow).
Task e: Why Visited Must Track (Node, Direction) Pairs
    The behavior of the ball at node v depends fundamentally on its arrival direction.
    If 'visited' stored only node indices, a node v visited in one direction (which may
    dead-end or get blocked downstream) would mark v as visited. Any subsequent active
    trail entering v from the opposite direction would be dropped.
    Minimal 4-node counterexample:
        Nodes: {0, 1, 2, 3}. Edges: 0->2, 0->3, 1->2, 2->3. Query: x=3, y=1, Z={2}.
        Active trail: 3 <- 0 -> 2 <- 1 (collider at 2 is open because 2 in Z).
        Ground truth: NOT d-separated (NO).
        If visited tracks nodes only:
          At (3, 'up'), (2, 'up') is queued. When (2, 'up') is popped, 2 in Z blocks it,
          but node 2 is added to visited. Later, the trail from parent 0 pushes (2, 'down').
          Because node 2 is already in visited, (2, 'down') is skipped! The ball never
          passes through collider 2 to reach node 1, falsely returning YES (d-separated).
        Tracking pairs (v, dir) keeps (2, 'down') distinct from (2, 'up'), correctly
        reaching node 1.
Task f: Excluding Evidence Z from Result
    The reachability algorithm finds unobserved variables conditionally dependent on x.
    Observed evidence nodes Z are fixed/conditioned on. While the ball may visit (v, 'down')
    for v in Z to bounce upward through an open collider, v itself cannot be an unobserved
    reachable variable.
Task g: Nodes Visited in Both Directions
    In case_p6_student.in, Query 4 (0, 1, {6}): nodes 2 (fail), 3 (higher), 4 (study)
    are visited both 'down' (from parent 0) and 'up' (bouncing off collider 6 in Z).

Problem Statement 3 & 4: d-Separation Queries & Complexity Analysis
-------------------------------------------------------------------
Task h, i, j, k:
    Tested against Lab 4 trail-enumeration program:
    case_p6_sample.in (12 queries): 0 mismatches (diff produces empty output).
    case_p6_student.in (12 queries): 0 mismatches.
    case_p6_main.in (40 queries): 0 mismatches. Output written to lab6_output.txt.
Task l, m, n, o:
    Empirical complexity across all test networks:
      case_p6_sample.in  (n=5,  m=4,   2n=10):  max popped=7,  wall-clock=0.055 ms
      case_p6_student.in (n=8,  m=14,  2n=16):  max popped=13, wall-clock=0.102 ms
      case_p6_main.in    (n=15, m=22,  2n=30):  max popped=21, wall-clock=0.357 ms
      case_p6_wide.in    (n=91, m=120, 2n=182): max popped=92, wall-clock=0.274 ms
    case_p6_wide.in contains 2^30 (~1.07 * 10^9) directed paths between endpoints.
    Bayes-Ball completes in 0.27 ms. Lab 4 trail enumeration was run and abandoned after
    10 seconds because exploring 10^9 paths would take days.
    Bound: State space has at most 2n pairs. Each state is popped at most once, and each
    edge is examined at most twice (once forward, once backward). The algorithm is strictly
    O(n + m), completely independent of the number of trails.

Problem Statement 5 & 6: Markov Blanket & Structural Verification
-----------------------------------------------------------------
Task p, q: blanket(parents, children, v) returns sorted unique list excluding v.
Task r: Blanket sizes in case_p6_main.in:
    Smallest blanket: Node 14 (size 2, blanket: [2, 10]).
    Largest blanket:  Node 10 (size 9, blanket: [0, 2, 4, 5, 7, 8, 9, 13, 14]).
Task s: Node with no children:
    If children[v] is empty, v has no co-parents. Hence MB(v) = Parents(v), and
    |MB(v)| = in-degree of v.
Task t: Full blanket d-separation check:
    For all v, for all w not in MB(v) U {v}: 0 failures across all 4 networks!
Task u: Omission of co-parents introduces:
    case_p6_sample.in:  3 failures
    case_p6_student.in: 20 failures
    case_p6_main.in:    86 failures
    case_p6_wide.in:    60 failures
Task v: Failure in case_p6_sample.in:
    Query 2: (0, 2, {1}) fails if co-parent 2 is omitted from MB(0) = {1, 2}.
    Conditioning on child 1 opens collider 0 -> 1 <- 2. To keep 0 independent of 2,
    co-parent 2 MUST be conditioned on.

Problem Statement 7: Numerical Verification on Real Data
--------------------------------------------------------
Task w: Blankets for student-por network:
    medu:   ['goout', 'fail', 'higher', 'study'] (size 4)
    goout:  ['medu', 'fail', 'higher', 'study', 'absent', 'g1'] (size 6)
    fail:   ['medu', 'goout', 'higher', 'study', 'absent', 'g1'] (size 6)
    higher: ['medu', 'goout', 'fail', 'study', 'g1'] (size 5)
    study:  ['medu', 'goout', 'fail', 'higher', 'g1'] (size 5)
    absent: ['goout', 'fail', 'g1', 'grade'] (size 4)
    g1:     ['goout', 'fail', 'higher', 'study', 'absent', 'grade'] (size 6)
    grade:  ['absent', 'g1'] (size 2)
    grade has the smallest blanket because it is a sink/leaf node with no children.
Task x, y: Numerical agreement:
    P(g1=1 | MB(g1)=1)            = 0.595744680851064
    P(g1=1 | MB(g1)=1, medu=1)    = 0.595744680851064  (Abs diff = 1.11e-16)
    P(grade=1 | MB(grade)=1)         = 0.800000000000000
    P(grade=1 | MB(grade)=1, medu=1) = 0.800000000000000  (Abs diff = 1.11e-16)
    P(study=1 | MB(study)=1)         = 0.173228346456693
    P(study=1 | MB(study)=1, grade=1)= 0.173228346456693  (Abs diff = 0.00e+00)
Task z: Network of 10,000 nodes:
    Only the Markov blanket of the variable (typically 5 to 10 nodes) needs to be
    examined to resample it. Gibbs Sampling directly depends on this locality property.
========================================================================================
"""

import sys
import os


# ==========================================================================
# PROBLEM STATEMENT 1: Graph Construction and Ancestor Computation
# ==========================================================================

def read_graph(path):
    """
    Parses the graph input file.
    
    File Format:
      n m                     n nodes numbered 0..n-1, m directed edges
      u v                     m lines, one per directed edge u -> v
      q                       number of queries
      x y z1 z2 ... zk        q lines: target pair x, y and evidence set Z
      
    Returns:
      n: int (number of nodes)
      parents: dict mapping node -> list of parent nodes
      children: dict mapping node -> list of child nodes
      queries: list of tuples (x, y, Z_set)
    """
    with open(path, 'r') as f:
        lines = [ln.split() for ln in f if ln.strip() != ""]
    if not lines:
        return 0, {}, {}, []
        
    n, m = int(lines[0][0]), int(lines[0][1])
    parents = {v: [] for v in range(n)}
    children = {v: [] for v in range(n)}
    
    for u, v in ((int(a), int(b)) for a, b in lines[1:1 + m]):
        parents[v].append(u)
        children[u].append(v)
        
    q = int(lines[1 + m][0])
    queries = []
    for ln in lines[2 + m: 2 + m + q]:
        nums = [int(t) for t in ln]
        queries.append((nums[0], nums[1], set(nums[2:])))
        
    return n, parents, children, queries


def ancestors_of(parents, Z):
    """
    Computes the ancestor closure of the evidence set Z by iterative upward traversal.
    
    A node v in Z is included in the returned set (ancestor relation is reflexive).
    This ensures that when a collider c is itself in Z, it satisfies c in ancestors(Z),
    which correctly opens the collider.
    
    Uses an explicit stack rather than recursion to avoid stack overflow on deep graphs.
    """
    seen = set()
    stack = list(Z)
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(parents[v])
    return seen


# ==========================================================================
# PROBLEM STATEMENT 2: Bayes-Ball Reachability Algorithm
# ==========================================================================

def reachable(parents, children, x, Z, track_pairs=False):
    """
    Bayes-Ball reachability algorithm.
    Starts from (x, 'up') and returns the set of nodes reachable (not d-separated from x by Z).
    
    Search state: (v, direction)
      - 'up':   arrived at v from a child (moving against directed arrows)
      - 'down': arrived at v from a parent (moving along directed arrows)
      
    Traversal rules (Shachter, 1998):
      State (v, up):
        - if v not in Z: pass to parents as 'up', children as 'down'
        - if v in Z:     trail blocked, pass nothing
      State (v, down):
        - if v not in Z: pass to children as 'down'
        - if v in Z or v in ancestors(Z):
                         pass to parents as 'up' (collider opened by evidence)
                         
    Notes on requirements:
      1. 'visited' must store (node, direction) pairs, NEVER bare nodes.
         Storing bare nodes blocks trails entering from the other direction,
         causing false d-separations.
      2. Nodes collected into reachable set are only those NOT in Z.
      3. Popped state counter is tracked and can be reported to stderr.
    """
    anc = ancestors_of(parents, Z)
    visited = set()
    stack = [(x, 'up')]
    reachable_nodes = set()
    popped_count = 0
    
    while stack:
        v, direction = stack.pop()
        if (v, direction) in visited:
            continue
        visited.add((v, direction))
        popped_count += 1
        
        # Collect node only if not in evidence set Z
        if v not in Z:
            reachable_nodes.add(v)
            
        if direction == 'up':
            if v not in Z:
                for p in parents[v]:
                    stack.append((p, 'up'))
                for c in children[v]:
                    stack.append((c, 'down'))
        elif direction == 'down':
            if v not in Z:
                for c in children[v]:
                    stack.append((c, 'down'))
            if v in anc or v in Z:
                for p in parents[v]:
                    stack.append((p, 'up'))
                    
    if track_pairs:
        return reachable_nodes, popped_count
    return reachable_nodes


# ==========================================================================
# PROBLEM STATEMENT 5: Markov Blanket Computation
# ==========================================================================

def blanket(parents, children, v):
    """
    Computes the Markov blanket of node v:
      - parents of v
      - children of v
      - co-parents (other parents of the children of v)
      
    Returns a sorted list of unique node indices, guaranteed not to contain v itself.
    """
    mb = set(parents[v]) | set(children[v])
    for c in children[v]:
        for p in parents[c]:
            if p != v:
                mb.add(p)
    mb.discard(v)
    return sorted(mb)


# ==========================================================================
# VERIFICATION SUITES (Self-checks for Problems 1 - 7)
# ==========================================================================

def run_verifications():
    """Executes all structural and numerical verifications across test networks."""
    print("=" * 70)
    print("RUNNING LAB 6 STRUCTURAL & NUMERICAL VERIFICATIONS")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(base_dir, "case_p6_sample.in")
    student_file = os.path.join(base_dir, "case_p6_student.in")
    main_file = os.path.join(base_dir, "case_p6_main.in")
    wide_file = os.path.join(base_dir, "case_p6_wide.in")
    
    # 1. Ancestor verification on case_p6_sample.in
    if os.path.exists(sample_file):
        _, par, ch, _ = read_graph(sample_file)
        anc3 = sorted(ancestors_of(par, {3}))
        print(f"[Problem 1] Ancestors of {{3}} in sample: {anc3} (Expected: [0, 1, 2, 3])")
        assert anc3 == [0, 1, 2, 3], "Ancestor verification failed!"
        
    # 2. Structural verification of Markov blanket across all 4 networks
    print("\n[Problem 6] Verifying that Markov blanket d-separates across all test files:")
    for fn in [sample_file, student_file, main_file, wide_file]:
        if not os.path.exists(fn):
            continue
        n, par, ch, _ = read_graph(fn)
        full_failures = 0
        no_cp_failures = 0
        for v in range(n):
            b_full = set(blanket(par, ch, v))
            b_no_cp = set(par[v]) | set(ch[v])
            b_no_cp.discard(v)
            for w in range(n):
                if w != v and w not in b_full:
                    if w in reachable(par, ch, v, b_full):
                        full_failures += 1
                if w != v and w not in b_no_cp:
                    if w in reachable(par, ch, v, b_no_cp):
                        no_cp_failures += 1
        name = os.path.basename(fn)
        print(f"  {name:<18}: Full blanket failures = {full_failures} (Pass), "
              f"Without co-parents failures = {no_cp_failures}")
        assert full_failures == 0, f"Full blanket failure on {name}"
        
    # 3. Main network blanket sizes
    if os.path.exists(main_file):
        n, par, ch, _ = read_graph(main_file)
        sizes = [(len(blanket(par, ch, v)), v) for v in range(n)]
        sizes.sort()
        print(f"\n[Problem 5] case_p6_main.in:")
        print(f"  Smallest blanket: Node {sizes[0][1]} with size {sizes[0][0]}")
        print(f"  Largest blanket:  Node {sizes[-1][1]} with size {sizes[-1][0]}")
        
    # 4. Numerical verification on real student data
    csv_file = os.path.join(base_dir, "..", "lab5", "student-por.csv")
    bn_file = os.path.join(base_dir, "..", "lab5", "U23AI118_lab5_bn.py")
    if os.path.exists(csv_file) and os.path.exists(bn_file):
        print("\n[Problem 7] Numerical Verification on student-por.csv (alpha=1.0):")
        sys.path.insert(0, os.path.abspath(os.path.join(base_dir, "..", "lab5")))
        import lab5_bn
        import U23AI118_lab5_bn
        
        VARS = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]
        V_IDX = {v: i for i, v in enumerate(VARS)}
        _, par, ch, _ = read_graph(student_file)
        data = lab5_bn.load(csv_file)
        cpt = U23AI118_lab5_bn.fit(data, alpha=1.0)
        
        test_pairs = [
            ("g1", "medu"),
            ("grade", "medu"),
            ("study", "grade"),
            ("absent", "higher")
        ]
        print(f"  {'Node':<8} | {'Outside':<8} | {'P(node=1|MB=1)':<20} | "
              f"{'P(node=1|MB=1,out=1)':<22} | {'Abs Diff':<10}")
        print("  " + "-" * 75)
        for target, outside in test_pairs:
            mb_vars = [VARS[j] for j in blanket(par, ch, V_IDX[target])]
            ev1 = {v: 1 for v in mb_vars}
            ev2 = dict(ev1)
            ev2[outside] = 1
            p1 = U23AI118_lab5_bn.infer({target: 1}, ev1, cpt)
            p2 = U23AI118_lab5_bn.infer({target: 1}, ev2, cpt)
            print(f"  {target:<8} | {outside:<8} | {p1:<20.15f} | {p2:<22.15f} | {abs(p1 - p2):.2e}")
            assert abs(p1 - p2) < 1e-10, "Numerical verification failed to agree!"
            
    print("\nAll verifications completed successfully!")
    print("=" * 70)


# ==========================================================================
# MAIN EXECUTION ROUTINE
# ==========================================================================

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python3 lab6.py <input_file>            # q lines of YES / NO")
        print("  python3 lab6.py <input_file> --blanket  # n lines, one blanket each")
        print("  python3 lab6.py --verify                # run self-verifications")
        return

    if "--verify" in args:
        run_verifications()
        return

    blanket_mode = False
    input_file = None

    for arg in args:
        if arg == "--blanket":
            blanket_mode = True
        elif not arg.startswith("--") and input_file is None:
            input_file = arg

    if input_file is None:
        sys.stderr.write("Error: No input file specified.\n")
        sys.exit(1)

    n, parents, children, queries = read_graph(input_file)

    if blanket_mode:
        # Blanket mode: print n lines; line i holds the Markov blanket of node i
        # as node numbers in increasing order separated by single spaces.
        # An empty blanket prints an empty line.
        for i in range(n):
            mb = blanket(parents, children, i)
            print(" ".join(map(str, mb)))
    else:
        # Default mode: print YES if x and y are d-separated by Z, otherwise NO,
        # one line per query in input order. Capitals only, nothing else on the line.
        # Pairs popped counter is instrumented and printed to stderr to keep stdout clean.
        for q_idx, (x, y, Z) in enumerate(queries):
            reachable_nodes, popped = reachable(parents, children, x, Z, track_pairs=True)
            sys.stderr.write(f"[Query {q_idx + 1}] x={x}, y={y}, |Z|={len(Z)}: popped {popped} pairs\n")
            if y not in reachable_nodes:
                print("YES")
            else:
                print("NO")


if __name__ == "__main__":
    main()
