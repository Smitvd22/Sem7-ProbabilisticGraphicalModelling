"""
LAB 9 & 10: Variable Elimination and Belief Propagation on Factor Graphs
"""
import numpy as np
import itertools
import time
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination as PgmpyVE
from pgmpy.utils import get_example_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

#Factor
class Factor:
    """Discrete factor over a list of variables."""

    def __init__(self, variables, cardinality, values):
        self.variables = list(variables)
        self.cardinality = dict(cardinality)
        self.values = np.asarray(values, dtype=float)
        if self.variables:
            shape = tuple(self.cardinality[v] for v in self.variables)
            assert self.values.shape == shape

    def __repr__(self):
        return f"Factor({self.variables}, shape={self.values.shape})"

    def copy(self):
        return Factor(self.variables, self.cardinality, self.values.copy())

    def _align(self, target):
        order = [v for v in target if v in self.variables]
        vals = self.values
        if order != self.variables:
            vals = np.transpose(vals, [self.variables.index(v) for v in order])
        shape = [self.cardinality[v] if v in self.variables else 1 for v in target]
        if not shape:
            return np.array([float(self.values.ravel()[0])])
        return vals.reshape(shape)

    def product(self, other):
        if not self.variables:
            s = float(self.values.ravel()[0])
            return Factor(other.variables, other.cardinality, s * other.values)
        if not other.variables:
            s = float(other.values.ravel()[0])
            return Factor(self.variables, self.cardinality, s * self.values)
        new_vars = self.variables + [v for v in other.variables
                                     if v not in self.variables]
        card = {v: self.cardinality.get(v, other.cardinality.get(v))
                for v in new_vars}
        return Factor(new_vars, card,
                      self._align(new_vars) * other._align(new_vars))

    def marginalize(self, var):
        if var not in self.variables:
            return self.copy()
        if len(self.variables) == 1:
            return Factor([], {}, np.array(self.values.sum()))
        axis = self.variables.index(var)
        rest = [v for v in self.variables if v != var]
        return Factor(rest, {v: self.cardinality[v] for v in rest},
                      self.values.sum(axis=axis))

    def max_marginalize(self, var):
        if var not in self.variables:
            return self.copy()
        if len(self.variables) == 1:
            return Factor([], {}, np.array(self.values.max()))
        axis = self.variables.index(var)
        rest = [v for v in self.variables if v != var]
        return Factor(rest, {v: self.cardinality[v] for v in rest},
                      self.values.max(axis=axis))

    def reduce(self, evidence):
        fix = {v: e for v, e in evidence.items() if v in self.variables}
        if not fix:
            return self.copy()
        idx = tuple(fix.get(v, slice(None)) for v in self.variables)
        vals = self.values[idx]
        rest = [v for v in self.variables if v not in fix]
        if not rest:
            return Factor([], {}, np.array(np.asarray(vals).ravel()[0]))
        return Factor(rest, {v: self.cardinality[v] for v in rest},
                      np.asarray(vals).reshape([self.cardinality[v] for v in rest]))

    def normalize(self):
        s = self.values.sum()
        return self.copy() if s == 0 else Factor(self.variables, self.cardinality,
                                                 self.values / s)


# Model helpers
def cpd_to_factor(cpd, model):
    f = cpd.to_factor()
    vs = list(f.variables)
    card = {v: model.get_cardinality(v) for v in vs}
    return Factor(vs, card, np.asarray(f.values, dtype=float))

def model_to_factors(model):
    return [cpd_to_factor(c, model) for c in model.get_cpds()]

def student_model():
    m = BN([("D", "G"), ("I", "G"), ("I", "S"), ("G", "L")])
    m.add_cpds(
        TabularCPD("D", 2, [[0.6], [0.4]],
                   state_names={"D": ["easy", "hard"]}),
        TabularCPD("I", 2, [[0.7], [0.3]],
                   state_names={"I": ["dumb", "smart"]}),
        TabularCPD("G", 3,
                   [[0.3, 0.05, 0.9, 0.5],
                    [0.4, 0.25, 0.08, 0.3],
                    [0.3, 0.7, 0.02, 0.2]],
                   evidence=["I", "D"], evidence_card=[2, 2],
                   state_names={"G": ["A", "B", "C"],
                                "I": ["dumb", "smart"],
                                "D": ["easy", "hard"]}),
        TabularCPD("S", 2, [[0.95, 0.2], [0.05, 0.8]],
                   evidence=["I"], evidence_card=[2],
                   state_names={"S": ["low", "high"],
                                "I": ["dumb", "smart"]}),
        TabularCPD("L", 2, [[0.1, 0.4, 0.99], [0.9, 0.6, 0.01]],
                   evidence=["G"], evidence_card=[3],
                   state_names={"L": ["weak", "strong"],
                                "G": ["A", "B", "C"]}))
    return m

def joint_probability(model, assign):
    p = 1.0
    for cpd in model.get_cpds():
        f = cpd.to_factor()
        p *= float(np.asarray(f.values)[tuple(assign[v] for v in f.variables)])
    return p

def enumerate_marginal(model, query, evidence={}):
    free = [v for v in model.nodes() if v != query and v not in evidence]
    out = np.zeros(model.get_cardinality(query))
    for q in range(len(out)):
        for combo in itertools.product(*[range(model.get_cardinality(v))
                                         for v in free]):
            full = dict(evidence, **{query: q}, **dict(zip(free, combo)))
            out[q] += joint_probability(model, full)
    return out / out.sum()

def enumerate_map(model, evidence={}):
    free = [v for v in model.nodes() if v not in evidence]
    best_p, best = -1, None
    for combo in itertools.product(*[range(model.get_cardinality(v))
                                     for v in free]):
        full = dict(evidence, **dict(zip(free, combo)))
        p = joint_probability(model, full)
        if p > best_p:
            best_p, best = p, full
    return best, best_p

#Variable elimination
def eliminate(factors, query_vars, evidence={}, order=None, max_product=False):
    factors = [f.reduce(evidence) for f in factors]

    if order is None:
        order, seen = [], set()
        for f in factors:
            for v in f.variables:
                if v not in seen and v not in query_vars and v not in evidence:
                    order.append(v)
                    seen.add(v)
    else:
        order = [v for v in order if v not in query_vars and v not in evidence]

    largest, largest_vars = 0, None
    for var in order:
        involved = [f for f in factors if var in f.variables]
        if not involved:
            continue
        rest = [f for f in factors if var not in f.variables]
        combined = involved[0]
        for f in involved[1:]:
            combined = combined.product(f)
        if combined.values.size > largest:
            largest = combined.values.size
            largest_vars = list(combined.variables)
        combined = (combined.max_marginalize(var) if max_product
                    else combined.marginalize(var))
        factors = rest + [combined]

    result = factors[0]
    for f in factors[1:]:
        result = result.product(f)
    for v in list(result.variables):
        if v not in query_vars:
            result = (result.max_marginalize(v) if max_product
                      else result.marginalize(v))
    return result, largest, largest_vars


# Moral graph & orderings
def moral_graph(model):
    g = {v: set() for v in model.nodes()}
    for a, b in model.edges():
        g[a].add(b)
        g[b].add(a)
    for cpd in model.get_cpds():
        parents = list(cpd.variables[1:])
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                g[parents[i]].add(parents[j])
                g[parents[j]].add(parents[i])
    return g


def min_degree_order(graph):
    g = {v: set(nb) for v, nb in graph.items()}
    order = []
    while g:
        v = min(g, key=lambda x: (len(g[x]), x))
        order.append(v)
        for u in g[v]:
            g[u].discard(v)
        del g[v]
    return order


def min_fill_order(graph):
    g = {v: set(nb) for v, nb in graph.items()}
    order = []
    while g:
        def fill(v):
            nb = list(g[v])
            return sum(1 for i in range(len(nb)) for j in range(i + 1, len(nb))
                       if nb[j] not in g[nb[i]])
        v = min(g, key=lambda x: (fill(x), len(g[x]), x))
        order.append(v)
        nb = list(g[v])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                g[nb[i]].add(nb[j])
                g[nb[j]].add(nb[i])
        for u in nb:
            g[u].discard(v)
        del g[v]
    return order


# Factor graph & message passing
def build_factor_graph(model):
    factors = model_to_factors(model)
    v2f = defaultdict(list)
    for i, f in enumerate(factors):
        for v in f.variables:
            v2f[v].append(i)
    return factors, v2f


def _run_bp(factors, v2f, max_iter, combine):
    card = {}
    for f in factors:
        for v in f.variables:
            card[v] = f.cardinality[v]

    var_to_f = {(v, fi): np.ones(card[v])
                for v, fl in v2f.items() for fi in fl}
    f_to_var = {(fi, v): np.ones(card[v])
                for v, fl in v2f.items() for fi in fl}

    for _ in range(max_iter):
        # factor → variable
        for fi, f in enumerate(factors):
            for v in f.variables:
                others = [u for u in f.variables if u != v]
                prod = f.values.copy()
                for u in others:
                    shp = [1] * len(f.variables)
                    shp[f.variables.index(u)] = card[u]
                    prod = prod * var_to_f[(u, fi)].reshape(shp)
                axes = tuple(f.variables.index(u) for u in others)
                msg = combine(prod, axis=axes) if axes else prod
                s = msg.max() if combine is np.max else msg.sum()
                if s > 0:
                    msg = msg / s
                f_to_var[(fi, v)] = msg

        # variable → factor
        for v, fl in v2f.items():
            for fi in fl:
                prod = np.ones(card[v])
                for fj in fl:
                    if fj != fi:
                        prod = prod * f_to_var[(fj, v)]
                s = prod.max() if combine is np.max else prod.sum()
                if s > 0:
                    prod = prod / s
                var_to_f[(v, fi)] = prod

    # beliefs
    beliefs = {}
    for v, fl in v2f.items():
        b = np.ones(card[v])
        for fi in fl:
            b = b * f_to_var[(fi, v)]
        s = b.sum()
        beliefs[v] = b / s if s > 0 else b
    return beliefs


def sum_product(factors, v2f, max_iter=100):
    return _run_bp(factors, v2f, max_iter, combine=np.sum)


def max_product_map(factors, v2f, max_iter=100):
    beliefs = _run_bp(factors, v2f, max_iter, combine=np.max)
    return {v: int(np.argmax(b)) for v, b in beliefs.items()}


#Parts
def part1():
    print("\n[Part 1] Factor operations on Student network")
    model = student_model()
    factors = model_to_factors(model)

    joint = factors[0]
    for f in factors[1:]:
        joint = joint.product(f)
    print(f"  joint table sum = {joint.values.sum():.6f}")

    P_L = joint
    for v in [v for v in P_L.variables if v != "L"]:
        P_L = P_L.marginalize(v)
    P_L = P_L.normalize()

    enum = enumerate_marginal(model, "L")
    print(f"  P(L) via factors   : {P_L.values}")
    print(f"  P(L) via enumerate : {enum}")
    print(f"  max diff           : {np.abs(P_L.values - enum).max():.2e}")

def part2():
    print("\n[Part 2] Variable elimination on Student network")
    model = student_model()
    factors = model_to_factors(model)
    pgm = PgmpyVE(model)

    def to_states(ev):
        return {v: model.get_cpds(v).state_names[v][i] for v, i in ev.items()}

    queries = [
        (["D"], {"L": 1}, ["I", "G", "S", "L"], "P(D | L=strong)"),
        (["I"], {"G": 2, "S": 1}, ["D", "G", "L", "S"], "P(I | G=C, S=high)"),
    ]
    for q, ev, order, label in queries:
        res, size, lvars = eliminate(factors, q, ev, order=order)
        mine = res.normalize().values
        enum = enumerate_marginal(model, q[0], ev)
        pgm_r = pgm.query(q, evidence=to_states(ev), show_progress=False).values
        print(f"  {label}")
        print(f"    mine    : {mine}")
        print(f"    enum    : {enum}")
        print(f"    pgmpy   : {pgm_r}")
        print(f"    largest : {size} over {lvars}")

def part3():
    print("\n[Part 3] Elimination orderings on Alarm network")
    model = get_example_model("alarm")
    n = len(model.nodes())
    print(f"  {n} vars, {len(model.edges())} edges "
          f"(enumeration would need ~2^{n} = {2**n:,} assignments)")

    factors = model_to_factors(model)
    query = next(iter(model.nodes()))

    rng = np.random.default_rng(0)
    random_order = list(rng.permutation([v for v in model.nodes() if v != query]))
    g = moral_graph(model)

    for name, order in [("random", random_order),
                        ("min-degree", min_degree_order(g)),
                        ("min-fill", min_fill_order(g))]:
        t0 = time.time()
        _, size, _ = eliminate(factors, [query], {}, order=order)
        dt = (time.time() - t0) * 1000
        print(f"  {name:11s}: largest factor = {size:>10,}, time = {dt:.1f} ms")


def part4():
    print("\n[Part 4] Sum-product on Student factor graph")
    model = student_model()
    factors, v2f = build_factor_graph(model)
    nv, nf = len(model.nodes()), len(factors)
    ne = sum(len(f) for f in v2f.values())
    is_tree = (nv + nf - 1 == ne)
    print(f"  {nv} variables, {nf} factors, {ne} edges — tree? {is_tree}")

    beliefs = sum_product(factors, v2f, max_iter=50)
    for v in model.nodes():
        ve, _, _ = eliminate(factors, [v], {})
        ve = ve.normalize().values
        print(f"  P({v}) = {beliefs[v]}  diff={np.abs(beliefs[v] - ve).max():.2e}")


def part5a():
    print("\n[Part 5a] Max-product MAP on Student with L=weak")
    model = student_model()
    factors, _ = build_factor_graph(model)
    evidence = {"L": 0}

    # Reduce factors by evidence, then rebuild the factor graph
    reduced = [f.reduce(evidence) for f in factors]
    v2f = defaultdict(list)
    for i, f in enumerate(reduced):
        for v in f.variables:
            v2f[v].append(i)

    mp = max_product_map(reduced, v2f)
    mp_prob = joint_probability(model, dict(evidence, **mp))

    bf, bf_prob = enumerate_map(model, evidence)
    match = all(mp[v] == bf[v] for v in mp)

    print(f"  max-product : {mp}  P={mp_prob:.4e}")
    print(f"  brute-force : {bf}  P={bf_prob:.4e}")
    print(f"  match: {match}")

def part5b():
    print("\n[Part 5b] Loopy BP on 4x4 grid MRF")
    N = 4
    nodes = [(i, j) for i in range(N) for j in range(N)]

    agree, disagree = np.exp(0.5), np.exp(-0.5)
    factors, v2f = [], defaultdict(list)
    idx = 0

    for node in nodes:
        factors.append(Factor([node], {node: 2}, np.ones(2)))
        v2f[node].append(idx)
        idx += 1

    for i, j in nodes:
        for di, dj in [(0, 1), (1, 0)]:
            nb = (i + di, j + dj)
            if nb in nodes:
                f = Factor([(i, j), nb], {(i, j): 2, nb: 2},
                           np.array([[agree, disagree], [disagree, agree]]))
                factors.append(f)
                v2f[(i, j)].append(idx)
                v2f[nb].append(idx)
                idx += 1

    exact = {}
    for v in nodes:
        r, _, _ = eliminate(factors, [v], {})
        exact[v] = r.normalize().values

    errors = []
    for it in range(1, 21):
        b = sum_product(factors, v2f, max_iter=it)
        errors.append(max(np.abs(b[v] - exact[v]).max() for v in nodes))

    print(f"  max error after 20 iterations: {errors[-1]:.4f}")

    if HAS_PLT:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 21), errors, "o-")
        plt.xlabel("iteration")
        plt.ylabel("max marginal error")
        plt.title("Loopy BP convergence on 4x4 grid MRF")
        plt.grid(True)
        plt.savefig("lbp_error.png", dpi=100, bbox_inches="tight")
        print("  plot saved to lbp_error.png")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    part5a()
    part5b()