import qecc as q
import numpy as np
import copy

n = 2

N = 2**(n**2 + 2*n)
for j in range(1, n+1):
    N *= 4**j - 1

Cliff_n = q.clifford_group(n, consider_phases=True)
Circuit = []

C_n = np.zeros(shape=(N, 2*n, 2*n), dtype=complex)

for i in range(N):
    a = copy.deepcopy(next(Cliff_n))
    Circuit.append(a)
    C_n[i] = a.as_unitary()
    print(a,"\n")


# print(Circuit[10000].circuit_decomposition(include_pauli=True), '\n')
# print(Circuit[0].as_unitary(),'\n',Circuit[3].as_unitary(), '\n')
# print(Circuit[10000].as_unitary(), '\n\n')
# print(C_n[3], '\n')
# print(C_n[4], '\n')
# print(Circuit[16], '\n\n')

