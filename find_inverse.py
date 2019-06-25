import pickle
import numpy as np

with open('Cliff_indices_4.pkl', 'rb') as f:
    Cliff_2 = pickle.load(f)

def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)

def Is_Inverse(A, B):
    if np.allclose(np.absolute(np.trace(np.dot(A, B))), 4):
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

X_2 = (1/np.sqrt(2))*np.array([[1+0j, 0-1j], [0-1j, 1+0j]])
X_CROT = np.dot(np.kron(Z_2, X_2), CNOT)
Z_CROT = np.dot(np.kron(Z_2, I), NCNOT)
CROT = np.dot(np.kron(Z_2, I), CNOT)

Prim = [ np.kron(I, X_2),        # ['X(pi/2)', 2]         = 0
         swap(np.kron(I, X_2)),  # ['X(pi/2)', 1]         = 1
         X_CROT,                 # ['X(pi/2)+CROT', 2]    = 2
         swap(X_CROT),           # ['X(pi/2)+CROT', 1]    = 3
         Z_CROT,                 # ['Z(pi/2)+CROT', 2]    = 4
         swap(Z_CROT),           # ['Z(pi/2)+CROT', 1]    = 5
         CROT,                   # ['CROT', 2]            = 6
         swap(CROT) ]            # ['CROT', 1]            = 7

# generate possible Zv pulses set Zv_q1 and Zv_q2
Zv = [Zv_2]
for i in range(1, 4):
    Zv.append(np.dot(Zv[i-1], Zv_2))

for i in range(3):
    Prim.append(np.kron(Zv[i], I))
for i in range(3):
    Prim.append(np.kron(I, Zv[i]))

# ['Zv(pi/2)', 1]        = 8
# ['Zv(pi)', 1]          = 9
# ['Zv(3pi/2)', 1]       = 10
# ['Zv(2pi)', 1]         = identity = none

# ['Zv(pi/2)', 2]        = 11
# ['Zv(pi)', 2]          = 12
# ['Zv(3pi/2)', 2]       = 13
# ['Zv(2pi)', 2]         = identity = none

# print(Prim)

C_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# Testing operator C_t initialized as 2-qubit identity (4x4 matrix).

for i in range(len(Cliff_2)):
    for j in range(len(Cliff_2[i])):
        C_t = np.dot(C_t, Prim[Cliff_2[i][j]])
    for k in range(14):
        if Is_Inverse(C_t, Prim[k]):
            print(str(Cliff_2[i]) + " is the inverse of " + str(k) + ".\n")
    C_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])    # Reset for each search.

Inverse = [[12, 0, 12],     # Inverse of ['X(pi/2)', 2]         = 0
           [9, 1, 9],       # Inverse of ['X(pi/2)', 1]         = 1
           [12, 2, 12],     # Inverse of ['X(pi/2)+CROT', 2]    = 2
           [9, 3, 9],       # Inverse of ['X(pi/2)+CROT', 1]    = 3
           [4, 9],          # Inverse of ['Z(pi/2)+CROT', 2]    = 4
           [9, 5, 9],       # Inverse of ['Z(pi/2)+CROT', 1]    = 5
           [6, 9],          # Inverse of ['CROT', 2]            = 6
           [9, 7, 9],       # Inverse of ['CROT', 1]            = 7
           [10],            # Inverse of ['Zv(pi/2)', 1]        = 8
           [9],             # Inverse of ['Zv(pi)', 1]          = 9
           [8],             # Inverse of ['Zv(3pi/2)', 1]       = 10
           [13],            # Inverse of ['Zv(pi/2)', 2]        = 11
           [12],            # Inverse of ['Zv(pi)', 2]          = 12
           [11] ]           # Inverse of ['Zv(3pi/2)', 2]       = 13

with open('Inverse_of_prim.pkl', 'wb') as f:
    pickle.dump(Inverse, f)
