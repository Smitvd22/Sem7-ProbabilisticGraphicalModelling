import sys
import os

def parse_graph(path):
    with open(path, 'r') as fh:
        rows = [line.split() for line in fh if line.strip()]

    if not rows:
        return 0,{},{},[]

    n,m = int(rows[0][0]),int(rows[0][1])

    pa = {v: [] for v in range(n)}
    ch = {v: [] for v in range(n)}

    for row in rows[1:1 + m]:
        u, v = int(row[0]), int(row[1])
        pa[v].append(u)
        ch[u].append(v)

    q = int(rows[1 + m][0])
    queries = []
    for row in rows[2 + m: 2 + m + q]:
        vals = [int(tok) for tok in row]
        queries.append((vals[0], vals[1], set(vals[2:])))

    return n, pa, ch, queries

def ancestor_set(pa, Z):
    closed = set()
    pending = list(Z)

    while pending:
        v = pending.pop()
        if v in closed:
            continue
        closed.add(v)
        pending.extend(pa[v])

    return closed

def bayes_ball(pa, ch, x, Z, count_states=False):
    anc = ancestor_set(pa, Z)
    seen = set()
    work = [(x, 'U')]
    hit = set()
    expanded = 0

    while work:
        v, direction = work.pop()
        key = (v, direction)
        if key in seen:
            continue
        seen.add(key)
        expanded += 1

        if v not in Z:
            hit.add(v)

        if direction == 'U':
            if v not in Z:
                for p in pa[v]:
                    work.append((p, 'U'))
                for c in ch[v]:
                    work.append((c, 'D'))
        else:
            if v not in Z:
                for c in ch[v]:
                    work.append((c, 'D'))
            if v in anc or v in Z:
                for p in pa[v]:
                    work.append((p, 'U'))

    if count_states:
        return hit, expanded
    return hit

def markov_blanket(pa, ch, v):
    mb = set(pa[v]) | set(ch[v])

    for child in ch[v]:
        for co_parent in pa[child]:
            if co_parent != v:
                mb.add(co_parent)

    mb.discard(v)
    return sorted(mb)

def run_self_checks():
    print("LAB 6 — STRUCTURAL & NUMERICAL SELF CHECKS")

    here = os.path.dirname(os.path.abspath(__file__))
    f_sample = os.path.join(here, "case_p6_sample.in")
    f_student = os.path.join(here, "case_p6_student.in")
    f_main = os.path.join(here, "case_p6_main.in")
    f_wide = os.path.join(here, "case_p6_wide.in")

    if os.path.exists(f_sample):
        _, pa, _, _ = parse_graph(f_sample)
        got = sorted(ancestor_set(pa, {3}))
        print(f"[Part 1] ancestors of {{3}} in sample file: {got} (expected [0, 1, 2, 3])")
        assert got == [0, 1, 2, 3], "ancestor closure check failed"

    print("\n[Part 4] verifying Markov blanket separation on every network:")
    for fn in (f_sample, f_student, f_main, f_wide):
        if not os.path.exists(fn):
            continue

        n, pa, ch, _ = parse_graph(fn)
        fail_full = 0
        fail_nocp = 0

        for v in range(n):
            mb_full = set(markov_blanket(pa, ch, v))
            mb_partial = set(pa[v]) | set(ch[v])
            mb_partial.discard(v)

            for w in range(n):
                if w == v:
                    continue
                if w not in mb_full and w in bayes_ball(pa, ch, v, mb_full):
                    fail_full += 1
                if w not in mb_partial and w in bayes_ball(pa, ch, v, mb_partial):
                    fail_nocp += 1

        print(f"  {os.path.basename(fn):<18}: full-blanket failures = {fail_full} (expected 0), "
              f"no-co-parent failures = {fail_nocp}")
        assert fail_full == 0, f"full blanket failed to separate on {os.path.basename(fn)}"

    if os.path.exists(f_main):
        n, pa, ch, _ = parse_graph(f_main)
        ranked = sorted((len(markov_blanket(pa, ch, v)), v) for v in range(n))
        print(f"\n[Part 4] case_p6_main.in:")
        print(f"  smallest blanket: node {ranked[0][1]} (size {ranked[0][0]})")
        print(f"  largest blanket : node {ranked[-1][1]} (size {ranked[-1][0]})")

    csv_path = os.path.join(here, "..", "lab5", "student-por.csv")
    bn_path = os.path.join(here, "..", "lab5", "U23AI118_lab5_bn.py")

    if os.path.exists(csv_path) and os.path.exists(bn_path):
        print("\n[Part 5] numerical check on student-por.csv (alpha = 1.0)")

        lab5_dir = os.path.abspath(os.path.join(here, "..", "lab5"))
        sys.path.insert(0, lab5_dir)

        import lab5_bn
        import U23AI118_lab5_bn

        VARS = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]
        IDX = {name: i for i, name in enumerate(VARS)}

        _, pa, ch, _ = parse_graph(f_student)
        data = lab5_bn.load(csv_path)
        cpt = U23AI118_lab5_bn.fit(data, alpha=1.0)

        trials = [
            ("g1", "medu"),
            ("grade", "medu"),
            ("study", "grade"),
            ("absent", "higher"),
        ]

        header = f"  {'node':<8} | {'outside':<8} | {'P(node=1|MB=1)':<20} | " \
                 f"{'P(node=1|MB=1,out=1)':<22} | {'|delta|':<10}"
        print(header)
        print("  " + "-" * 75)

        for target, outside in trials:
            mb_names = [VARS[j] for j in markov_blanket(pa, ch, IDX[target])]
            ev_full = {name: 1 for name in mb_names}
            ev_extra = dict(ev_full)
            ev_extra[outside] = 1

            p_full = U23AI118_lab5_bn.infer({target: 1}, ev_full, cpt)
            p_extra = U23AI118_lab5_bn.infer({target: 1}, ev_extra, cpt)

            print(f"  {target:<8} | {outside:<8} | {p_full:<20.15f} | "
                  f"{p_extra:<22.15f} | {abs(p_full - p_extra):.2e}")
            assert abs(p_full - p_extra) < 1e-10, "blanket locality check failed"

    print("\nAll self-checks passed.")
    print("=" * 70)

def main():
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python3 lab6.py <input_file>             # q lines of YES / NO")
        print("  python3 lab6.py <input_file> --blanket   # n lines, one blanket each")
        print("  python3 lab6.py --verify                 # run self-checks")
        return

    if "--verify" in args:
        run_self_checks()
        return

    blanket_mode = False
    input_file = None

    for token in args:
        if token == "--blanket":
            blanket_mode = True
        elif not token.startswith("--") and input_file is None:
            input_file = token

    if input_file is None:
        sys.stderr.write("Error: no input file supplied.\n")
        sys.exit(1)

    n, pa, ch, queries = parse_graph(input_file)

    if blanket_mode:
        for v in range(n):
            print(" ".join(str(u) for u in markov_blanket(pa, ch, v)))
    else:
        for q_idx, (x, y, Z) in enumerate(queries):
            hit, pops = bayes_ball(pa, ch, x, Z, count_states=True)
            sys.stderr.write(
                f"[query {q_idx + 1}] x={x}, y={y}, |Z|={len(Z)}: expanded {pops} states\n"
            )
            print("YES" if y not in hit else "NO")

if __name__ == "__main__":
    main()