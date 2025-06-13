from heapq import heappush, heappop

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def manhattan(s):
    d = 0
    for i, t in enumerate(s):
        if t:
            g = GOAL.index(t)
            d += abs(i // 3 - g // 3) + abs(i % 3 - g % 3)
    return d


def moves(s):
    i = s.index(0)
    r, c = divmod(i, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            lst = list(s)
            lst[i], lst[j] = lst[j], lst[i]
            yield tuple(lst)


def solvable(s):
    seq = [x for x in s if x]
    inv = sum(a > b for i, a in enumerate(seq) for b in seq[i + 1 :])
    return inv % 2 == 0


def astar(start):
    if not solvable(start):
        return None
    open_set = []
    heappush(open_set, (manhattan(start), 0, start))
    g = {start: 0}
    parent = {}
    while open_set:
        f, cost, state = heappop(open_set)
        if state == GOAL:
            path = [state]
            while state in parent:
                state = parent[state]
                path.append(state)
            return path[::-1]
        for nxt in moves(state):
            nc = cost + 1
            if nxt not in g or nc < g[nxt]:
                g[nxt] = nc
                parent[nxt] = state
                heappush(open_set, (nc + manhattan(nxt), nc, nxt))
    return None


def show(s):
    print(f"{s[:3]}\n{s[3:6]}\n{s[6:]}\n")


def inversions(state):
    seq = [x for x in state if x]
    return sum(seq[i] > seq[j] for i in range(len(seq)) for j in range(i + 1, len(seq)))


def try_solve(state):
    print(f"State: {state}")
    print(f"Inversions: {inversions(state)}")
    print(f"Solvable: {solvable(state)}")
    print()

    path = astar(state)
    if path:
        print(f"\nSolution in {len(path)-1} moves:\n")
        for p in path:
            show(p)
    else:
        print("No solution")


if __name__ == "__main__":
    try_solve((1, 2, 3, 4, 5, 6, 8, 7, 0))  # unsolvable
    try_solve((2, 3, 8, 1, 6, 4, 7, 0, 5))  # solvable
