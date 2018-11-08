import qecc as q
import numpy as np

n = 1

N = 2**(n**2 + 2*n)
for j in range(1, n+1):
    N *= 4**j - 1

Cliff_n = q.clifford_group(n, consider_phases=True)
Circuit = []

C_n = np.zeros(shape=(N, 2*n, 2*n), dtype=complex)

for i in range(N):
    a = next(Cliff_n)
    Circuit.append(a)
    C_n[i] = a.as_unitary()

# print(Circuit[999].circuit_decomposition(include_pauli=True))
# print(Circuit[3], '\n\n')
# print(Circuit[4], '\n\n')
# print(Circuit[16], '\n\n')

