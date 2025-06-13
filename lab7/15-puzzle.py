from heapq import heappush, heappop

N = 4
GOAL = tuple(list(range(1, N * N)) + [0])


def manhattan(s):
    d = 0
    for i, t in enumerate(s):
        if t:
            g = GOAL.index(t)
            d += abs(i // N - g // N) + abs(i % N - g % N)
    return d


def moves(s):
    i = s.index(0)
    r, c = divmod(i, N)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            j = nr * N + nc
            lst = list(s)
            lst[i], lst[j] = lst[j], lst[i]
            yield tuple(lst)


def solvable(s):
    seq = [x for x in s if x]
    inv = sum(a > b for i, a in enumerate(seq) for b in seq[i + 1 :])
    blank_row_from_bottom = N - (s.index(0) // N)
    return (blank_row_from_bottom % 2) != (inv % 2)


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
    for r in range(0, N * N, N):
        print(s[r : r + N])
    print()


def inversions(state):
    seq = [x for x in state if x]
    return sum(seq[i] > seq[j] for i in range(len(seq)) for j in range(i + 1, len(seq)))


def try_solve(state):
    print(f"State: {state}")
    print(f"Inversions: {inversions(state)}")
    print(f"Solvable: {solvable(state)}\n")
    path = astar(state)
    if path:
        print(f"Solution in {len(path)-1} moves:\n")
        for p in path:
            show(p)
    else:
        print("No solution\n")


if __name__ == "__main__":
    try_solve(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0)
    )  # unsolvable example

    try_solve(
        (5, 1, 2, 3, 6, 0, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12)
    )  # solvable example
