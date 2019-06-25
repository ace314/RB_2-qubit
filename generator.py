import numpy as np
from scipy.linalg import expm, sinm, cosm
import math

def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)


Ez = 3.933e10
dEz = 1.326e7
Ω = 410000.0
J = 1.59e6

n = 1
T_pi_2 = math.sqrt(16*n**2-1)/4/J
# T_pi_2 = 1.0/(4.0*Ω)

I = np.identity(2)
X = np.array([[0,   1], [1,  0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1,   0], [0, -1]])

Z_2_1 = (1/np.sqrt(2))*np.array([[1-1j, 0+0j], [0+0j, 1+1j]])
CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
NCNOT = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
X_2_1 = (1/np.sqrt(2))*np.array([[1+0j, 0-1j], [0-1j, 1+0j]])

# Following generators operate on Q1.
X_2_q1 = np.kron(X_2_1, I)
X_CROT_q1 = np.dot(np.kron(X_2_1, Z_2_1), CNOT)
Z_CROT_q1 = np.dot(np.kron(I, Z_2_1), NCNOT)
CROT_q1 = np.dot(np.kron(I, Z_2_1), CNOT)

# Following generators operate on Q2.
X_2_q2 = swap(X_2_q1)
X_CROT_q2 = swap(X_CROT_q1)
Z_CROT_q2 = swap(Z_CROT_q1)
CROT_q2 = swap(CROT_q1)

def H_RWA1_1d(w):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0]])

def H_RWA1_1u(w):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0]])

def H_RWA2_1d(w, t):
    return 1/2*2*np.pi*w*np.array( [[                     0, 0, np.exp(-1j*J*2*np.pi*t), 0],
                                    [                     0, 0,                       0, 0],
                                    [np.exp(1j*J*2*np.pi*t), 0,                       0, 0],
                                    [                     0, 0,                       0, 0]] )

def H_RWA2_1u(w, t):
    return 1/2*2*np.pi*w*np.array( [[0,                       0, 0,                       0],
                                    [0,                       0, 0,  np.exp(1j*J*2*np.pi*t)],
                                    [0,                       0, 0,                       0],
                                    [0, np.exp(-1j*J*2*np.pi*t), 0,                       0]] )

def H_RWA1_2d(w):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]])

def H_RWA1_2u(w):
    return 1/2*2*np.pi*w*np.array( [[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]])

def H_RWA2_2d(w, t):
    return 1/2*2*np.pi*w*np.array( [[                     0, np.exp(-1j*J*2*np.pi*t), 0, 0],
                                    [np.exp(1j*J*2*np.pi*t),                       0, 0, 0],
                                    [                     0,                       0, 0, 0],
                                    [                     0,                       0, 0, 0]] )

def H_RWA2_2u(w, t):
    return 1/2*2*np.pi*w*np.array( [[0, 0,                       0,                      0],
                                    [0, 0,                       0,                      0],
                                    [0, 0,                       0, np.exp(1j*J*2*np.pi*t)],
                                    [0, 0, np.exp(-1j*J*2*np.pi*t),                      0]] )

def H_RWA_1d(w, t):
    return H_RWA1_1d(w) + H_RWA2_1d(w, t)

def H_RWA_1u(w, t):
    return H_RWA1_1u(w) + H_RWA2_1u(w, t)

def H_RWA_2d(w, t):
    return H_RWA1_2d(w) + H_RWA2_2d(w, t)

def H_RWA_2u(w, t):
    return H_RWA1_2u(w) + H_RWA2_2u(w, t)

def Fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M)))/4


''' 
some test for no-crosstalk-term Hamiltonians.
A = expm(-1j*H_RWA1_1d(Ω)*T_pi_2)   #U_1d_pi/2
B = expm(-1j*H_RWA1_1u(Ω)*T_pi_2)   #U_1u_pi/2

C = expm(1j*H_RWA1_1d(Ω)*T_pi_2)   #U_1d_-pi/2
D = expm(1j*H_RWA1_1u(Ω)*T_pi_2)   #U_1u_-pi/2

print(np.dot(B, A)) #X/2 ; 1
print(np.dot(B, C)) #X/2+CROT ; 1
print(np.dot(B, B)) #Z-CROT ; 1
print(np.dot(C, C)) #CROT ; 1
'''


delta = 50

L = np.linspace(0, T_pi_2, delta)

U_1d_plus = np.identity(4)   # Unitary transformation from f_1d pulse. "Plus" indicates that it is a pi/2 rotation.
U_1u_plus = np.identity(4)
U_1d_minus = np.identity(4)
U_1u_minus = np.identity(4)
U_2d_plus = np.identity(4)   # Unitary transformation from f_1d pulse. "Plus" indicates that it is a pi/2 rotation.
U_2u_plus = np.identity(4)
U_2d_minus = np.identity(4)
U_2u_minus = np.identity(4)

for t in L:
    U_1d_plus = np.dot(expm(-1j*H_RWA_1d(Ω, t)*L[1]), U_1d_plus)
    U_1u_plus = np.dot(expm(-1j*H_RWA_1u(Ω, t)*L[1]), U_1u_plus)
    U_1d_minus = np.dot(expm(1j*H_RWA_1d(Ω, t)*L[1]), U_1d_minus)   #TODO:Figure out how to construct -pi/2 pulse physically.
    U_1u_minus = np.dot(expm(1j*H_RWA_1u(Ω, t)*L[1]), U_1u_minus)

    U_2d_plus = np.dot(expm(-1j * H_RWA_2d(Ω, t) * L[1]), U_2d_plus)
    U_2u_plus = np.dot(expm(-1j * H_RWA_2u(Ω, t) * L[1]), U_2u_plus)
    U_2d_minus = np.dot(expm(1j * H_RWA_2d(Ω, t) * L[1]), U_2d_minus)
    U_2u_minus = np.dot(expm(1j * H_RWA_2u(Ω, t) * L[1]), U_2u_minus)

X_2_exp_q1 = np.dot(U_1u_plus, U_1d_plus)
X_CROT_exp_q1 = np.dot(U_1u_plus, U_1d_minus)
Z_CROT_exp_q1 = np.dot(U_1u_plus, U_1u_plus)
CROT_exp_q1 = np.dot(U_1d_minus, U_1d_minus)

X_2_exp_q2 = np.dot(U_2u_plus, U_2d_plus)
X_CROT_exp_q2 = np.dot(U_2u_plus, U_2d_minus)
Z_CROT_exp_q2 = np.dot(U_2u_plus, U_2u_plus)
CROT_exp_q2 = np.dot(U_2d_minus, U_2d_minus)

# print(U_1d_plus)

print("F(X_2_q1): ", Fidelity(X_2_exp_q1, X_2_q1))
print("F(X_CROT_q1): ", Fidelity(X_CROT_exp_q1, X_CROT_q1))
print("F(Z_CROT_q1): ", Fidelity(Z_CROT_exp_q1, Z_CROT_q1))
print("F(CROT_q1): ", Fidelity(CROT_exp_q1, CROT_q1))
print("\n")
print("F(X_2_q2): ", Fidelity(X_2_exp_q2, X_2_q2))
print("F(X_CROT_q2): ", Fidelity(X_CROT_exp_q2, X_CROT_q2))
print("F(Z_CROT_q2): ", Fidelity(Z_CROT_exp_q2, Z_CROT_q2))
print("F(CROT_q2): ", Fidelity(CROT_exp_q2, CROT_q2))








