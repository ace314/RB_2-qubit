import numpy as np
# import copy
import itertools
import pickle


def swap(M):
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

def Is_enough(L, num_array, c):
    if c == num_array[L]:
        return True
    else:
        return False


global I
global X
global Y
global Z
I = np.identity(2)
X = np.array([[0,   1], [1,  0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1,   0], [0, -1]])


Z_2 = (1/np.sqrt(2))*np.array([[1-1j, 0+0j], [0+0j, 1+1j]])
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

Zv_q1 = [ [[8]      ],                  #['Zv(pi/2)', 1]        = 8
          [[9]      ],                  #['Zv(pi)', 1]          = 9
          [[10]     ],                  #['Zv(3pi/2)', 1]       = 10
          [[]       ] ]                 #['Zv(2pi)', 1]         = identity = none

Zv_q2 = [ [[11]     ],                  #['Zv(pi/2)', 2]        = 11
          [[12]     ],                  #['Zv(pi)', 2]          = 12
          [[13]     ],                  #['Zv(3pi/2)', 2]       = 13
          [[]       ] ]                 #['Zv(2pi)', 2]         = identity = none

for i in range(4):
    Zv_q1[i].append(np.kron(Zv[i], I))
    Zv_q2[i].append(np.kron(I, Zv[i]))

#generate all possible primitive gates set
Prim = [ [[0], np.kron(I, X_2)],        #['X(pi/2)', 2]         = 0
         [[1], swap(np.kron(I, X_2))],  #['X(pi/2)', 1]         = 1
         [[2], X_CROT],                 #['X(pi/2)+CROT', 2]    = 2
         [[3], swap(X_CROT)],           #['X(pi/2)+CROT', 1]    = 3
         [[4], Z_CROT],                 #['Z(pi/2)+CROT', 2]    = 4
         [[5], swap(Z_CROT)],           #['Z(pi/2)+CROT', 1]    = 5
         [[6], CROT],                   #['CROT', 2]            = 6
         [[7], swap(CROT)] ]            #['CROT', 1]            = 7

#generate set contains all possible primitive gates with following Zv in the circuit(4 possibilities each). 8*4=32 elements.
Prim_Zv = []
for i in range(8):
    if i%2==0:
        for j in range(4):
            Prim_Zv.append(Gates_multi(Zv_q2[j], Prim[i]))
    else:
        for j in range(4):
            Prim_Zv.append(Gates_multi(Zv_q1[j], Prim[i]))

'''
Computational searching all possible clifford elements.
'''

Cliff_2 = []
Cliff_index = []
L = []
c = 0
count = 0
#Iterate first Zv's
for i, j in itertools.product(range(4), range(4)):
    a = Gates_multi(Zv_q1[i], Zv_q2[j])
    if Is_New(Cliff_2, a):
        Cliff_2.append(a)
        Cliff_index.append(a[0])
        c += 1
        count += 1
        print(count)
    else:
        continue
L.append(c)
c = 0

n_Cliff = [16, 384, 4176, 6912, 32]
l = 4
t = 1
d = 0
while t<=l:
    if t>1:
        d += L[t-2]
    for i in range(d, len(Cliff_2)):
        for j in range(len(Prim_Zv)):
            a = Gates_multi(Prim_Zv[j], Cliff_2[i])
            if Is_New(Cliff_2, a):
                Cliff_2.append(a)
                Cliff_index.append(a[0])
                c += 1
                count += 1
                print(count)
            if Is_enough(t, n_Cliff, c):
                print("breaked!", c)
                break
        else:
            continue
        break
    L.append(c)
    c = 0
    t += 1

with open('Cliff_indices_test.pkl', 'wb') as f:
    pickle.dump(Cliff_index, f)

print(L, "\n")
# print(Cliff_2[199][0], "\n\n", Cliff_2[199][1])
# print(Cliff_index)





