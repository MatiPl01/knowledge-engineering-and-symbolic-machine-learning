s(0, 1).
s(N, X) :- N1 is N - 1, s(N1, X1), X is N * X1