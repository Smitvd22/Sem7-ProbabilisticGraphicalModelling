import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 U23AI118_Lab4.py <input_file>")
        return
        
    input_file = sys.argv[1]
    
    with open(input_file, 'r') as f:
        lines = f.read().splitlines()
        
    if not lines:
        return
        
    n_str, m_str = lines[0].split()
    n = int(n_str)
    m = int(m_str)
    
    edges = set()
    adj = {i: set() for i in range(n)}
    
    line_idx = 1
    for _ in range(m):
        u, v = map(int, lines[line_idx].split())
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        line_idx += 1
        
    q = int(lines[line_idx])
    line_idx += 1
    
    # Precompute descendants for all nodes
    # desc[u] = set of all descendants of u, including u itself.
    desc = {i: set([i]) for i in range(n)}
    
    directed_adj = {i: set() for i in range(n)}
    for u, v in edges:
        directed_adj[u].add(v)
        
    def get_descendants(node):
        visited = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                for neighbor in directed_adj[curr]:
                    stack.append(neighbor)
        return visited
        
    for i in range(n):
        desc[i] = get_descendants(i)

    def is_blocked(trail, z_set):
        for i in range(1, len(trail) - 1):
            u = trail[i-1]
            v = trail[i]
            w = trail[i+1]
            
            is_collider = (u, v) in edges and (w, v) in edges
            
            if is_collider:
                if len(desc[v].intersection(z_set)) == 0:
                    return True
            else:
                if v in z_set:
                    return True
        return False
        
    for _ in range(q):
        parts = list(map(int, lines[line_idx].split()))
        line_idx += 1
        x = parts[0]
        y = parts[1]
        z_set = set(parts[2:])
        
        # Enumerate all trails and check if all are blocked
        visited = set([x])
        
        all_blocked = True
        
        def dfs(u, trail):
            nonlocal all_blocked
            if not all_blocked:
                return
                
            if u == y:
                if not is_blocked(trail, z_set):
                    all_blocked = False
                return
                
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    trail.append(v)
                    dfs(v, trail)
                    trail.pop()
                    visited.remove(v)
                    
        dfs(x, [x])
        
        if all_blocked:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()
