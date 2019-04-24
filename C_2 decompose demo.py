import numpy as np
# import qecc as q
# import copy
import itertools

# def Generate_symp_basis(n):
#     P = np.zeros(shape=(2*n, 2**n, 2**n), dtype=complex)
#     for i in range(n):  #P裡的第i及i+n個矩陣
#         for j in range(n):  #第i / i+n個矩陣的第j個kron為X / Z
#             if j == 0:
#                 if j == i:
#                     a = X
#                     b = Z
#                 else:
#                     a = I
#                     b = I
#             else:
#                 if j == i:
#                     a = np.kron(a, X)
#                     b = np.kron(b, Z)
#                 else:
#                     a = np.kron(a, I)
#                     b = np.kron(b, I)
#         P[i] = copy.deepcopy(a)
#         P[i+n] = copy.deepcopy(b)
#     return P
#
# def Is_pauli(M):
#     for i in range(len(P_n)):
#         if np.allclose(M, P_n[i], rtol=1e-05, atol=1e-08, equal_nan=False):
#             return True
#         else:
#             continue
#     return False
#
# def Is_clifford(M):
#     for i in range(len(Sym_set)):
#         a = Sym_set[i]
#         N = a.dot(M).dot(a.conj().T)
#         return Is_pauli(N)

#Generate a set [I, X, Y, Z]

#Generate n-qubit pauli group [I, X, Y, Z]
# def Generate_pauli(n):
#     a = q.pauli_group(n)
#     next(a)
#     P = []
#     for i in range(4**n-1):
#         b = copy.deepcopy(next(a)).as_unitary()
#         P.append(b)
#         P.append(-b)
#     # return P

#swap M(q1, q2) to M(q2, q1). For 4x4 matrix only.
# def Pauli_mat():
#     a = q.pauli_group(1)
#     P = []
#     for i in range(4):
#         b = copy.deepcopy(next(a)).as_unitary()
#         P.append(b)
#     return P

def swap(M):
    # SWAP = q.swap(2, 0, 1).as_unitary()
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)

'''
Functions "Gates_multi" and "Is_New" act on the following matrix object:
M = [ [decompose series], [matrix form] ]
"Decompose series" is in the form [generator1, qn], [generator2, qn]......
'''
#Matrix multiplication
def Gates_multi(M1, M2):
    a = M1[0] + M2[0]
    b = np.dot(M1[1], M2[1])
    return [a, b]

#Determine whether a matrix M is new to a set M_array.
def Is_New(M_array, M):
    for n in range(len(M_array)):
        if np.allclose(np.absolute(np.trace(np.dot(M_array[n][1].conj().T, M[1]))), 4):
            # print(M_array[n][0], "\n", M[0], "\n\n")
            return False
        else:
            continue
    return True

# global P_n
# P_n = Generate_pauli(2)
# global Pauli
# Pauli = Pauli_mat()
global I
global X
global Y
global Z
I = np.identity(2)
X = np.array([[0,   1], [1,  0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1,   0], [0, -1]])
# global Sym_set
# Sym_set = Generate_symp_basis(2)

Z_2 = (1/np.sqrt(2))*np.array([[1+1j, 0+0j], [0+0j, 1-1j]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
NCNOT = np.dot(np.dot(np.kron(X, I), CNOT), np.kron(X, I))
Zv_2 = (1/np.sqrt(2))*np.array([[1-1j, 0+0j], [0+0j, 1+1j]])

#primitive gates
#error adds here
X_2 = (1/np.sqrt(2))*np.array([[1+0j, 0-1j], [0-1j, 1+0j]])
X_CROT = np.dot(np.kron(Z_2, X_2), CNOT)
Z_CROT = np.dot(np.kron(Z_2, I), NCNOT)
CROT = np.dot(np.kron(Z_2, I), CNOT)

#generate possible Zv pulses set Zv_q1 and Zv_q2
Zv = [Zv_2]
for i in range(1, 4):
    Zv.append(np.dot(Zv[i-1], Zv_2))

Zv_q1 = [ [[['Zv(pi/2)', 1]]      ],
          [[['Zv(pi)', 1]]        ],
          [[['Zv(3pi/2)', 1]]     ],
          [[['Zv(2pi)', 1]]       ] ]

Zv_q2 = [ [[['Zv(pi/2)', 2]]      ],
          [[['Zv(pi)', 2]]        ],
          [[['Zv(3pi/2)', 2]]     ],
          [[['Zv(2pi)', 2]]       ] ]

for i in range(4):
    Zv_q1[i].append(np.kron(Zv[i], I))
    Zv_q2[i].append(np.kron(I, Zv[i]))

#generate all possible primitive gates set
Prim = [ [[['X(pi/2)', 2]], np.kron(I, X_2)], [[['X(pi/2)', 1]], swap(np.kron(I, X_2))],
         [[['X(pi/2)+CROT', 2]], X_CROT], [[['X(pi/2)+CROT', 1]], swap(X_CROT)],
         [[['Z(pi/2)+CROT', 2]], Z_CROT], [[['Z(pi/2)+CROT', 1]], swap(Z_CROT)],
         [[['CROT', 2]], CROT], [[['CROT', 1]], swap(CROT)] ]

#generate set contains all possible primitive gates with following Zv(4 possibilities each). 8*4=32 elements.
Prim_Zv = []
for i in range(8):
    if i%2==0:
        for j in range(4):
            Prim_Zv.append(Gates_multi(Prim[i], Zv_q2[j]))
    else:
        for j in range(4):
            Prim_Zv.append(Gates_multi(Prim[i], Zv_q1[j]))

'''
Computational searching all possible clifford elements.
'''

Cliff_2 = []
L = []
c = 0
#Iterate first Zv's
for i, j in itertools.product(range(4), range(4)):
    a = Gates_multi(Zv_q1[i], Zv_q2[j])
    if Is_New(Cliff_2, a):
        Cliff_2.append(a)
        c += 1
    else:
        continue
L.append(c)
c = 0

l = 2
t = 1
d = 0
while t<=l:
    if t>1:
        d += L[t-2]
    for i in range(d, len(Cliff_2)):
        for j in range(len(Prim_Zv)):
            a = Gates_multi(Cliff_2[i], Prim_Zv[j])
            if Is_New(Cliff_2, a):
                Cliff_2.append(a)
                c += 1
            else:
                continue
    L.append(c)
    c = 0
    t += 1

print(L, "\n")
# print(np.absolute(np.trace(np.dot(Cliff_2[16][1].conj().T, Cliff_2[528][1]))), "\n")
# print(np.allclose(np.absolute(np.trace(np.dot(Cliff_2[16][1].conj().T, Cliff_2[528][1]))), 4), "\n")
# print(Is_New(Cliff_2, Cliff_2[528]), "\n")
# print(np.absolute(np.trace(np.dot(Cliff_2[20][1].conj().T, 1j*Cliff_2[20][1]))), "\n")
print(Cliff_2[199][0], "\n\n", Cliff_2[199][1])
# print(Cliff_2[16][1], "\n\n", Cliff_2[528][1])




