import numpy as np
from scipy.linalg import expm
import math
from random import randrange

def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)


Ez = 3.933e10
dEz = 1.326e7
Ω = 410000.0
J = 1.59e6
phi = 0.031675*np.pi    # crosstalk phase error

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

def H_RWA1_1d(w, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0, np.exp(-1j*p)],
                                    [0, 0, 0, 0],
                                    [0, np.exp(1j*p), 0, 0]])

def H_RWA1_1u(w, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, np.exp(-1j*p), 0],
                                    [0, 0, 0, 0],
                                    [np.exp(1j*p), 0, 0, 0],
                                    [0, 0, 0, 0]])

def H_RWA2_1d(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[                     0, 0, np.exp(1j*(J*2*np.pi*t - p)), 0],
                                    [                     0, 0,                       0, 0],
                                    [np.exp(-1j*(J*2*np.pi*t - p)), 0,                       0, 0],
                                    [                     0, 0,                       0, 0]] )

def H_RWA2_1u(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0,                       0, 0,                       0],
                                    [0,                       0, 0,  np.exp(-1j*(J*2*np.pi*t + p))],
                                    [0,                       0, 0,                       0],
                                    [0, np.exp(1j*(J*2*np.pi*t + p)), 0,                       0]] )

def H_RWA1_2d(w, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, np.exp(-1j*p)],
                                    [0, 0, np.exp(1j*p), 0]])

def H_RWA1_2u(w, p):
    return 1/2*2*np.pi*w*np.array( [[0, np.exp(-1j*p), 0, 0],
                                    [np.exp(1j*p), 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]])

def H_RWA2_2d(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[                     0, np.exp(1j*(J*2*np.pi*t - p)), 0, 0],
                                    [np.exp(-1j*(J*2*np.pi*t - p)),                       0, 0, 0],
                                    [                     0,                       0, 0, 0],
                                    [                     0,                       0, 0, 0]] )

def H_RWA2_2u(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0,                       0,                      0],
                                    [0, 0,                       0,                      0],
                                    [0, 0,                       0, np.exp(-1j*(J*2*np.pi*t + p))],
                                    [0, 0, np.exp(1j*(J*2*np.pi*t + p)),                      0]] )

def H_RWA_1d(w, t, p):
    return H_RWA1_1d(w, p) + H_RWA2_1d(w, t, p)

def H_RWA_1u(w, t, p):
    return H_RWA1_1u(w, p) + H_RWA2_1u(w, t, p)

def H_RWA_2d(w, t, p):
    return H_RWA1_2d(w, p) + H_RWA2_2d(w, t, p)

def H_RWA_2u(w, t, p):
    return H_RWA1_2u(w, p) + H_RWA2_2u(w, t, p)

def Fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M)))/4

def pulse_generate(k, T, w, p, delta):      #TODO: check it out
    L = np.linspace(0, T, delta+1)
    M = np.identity(4)
    if k == 0:
        for t in L[1:]:
            M = np.dot(expm(-1j * H_RWA_1u(w, t - (L[1] / 2), p[2] - p[0]) * L[1]), M)
    elif k == 1:
        for t in L[1:]:
            M = np.dot(expm(-1j * H_RWA_1d(w, t - (L[1] / 2), p[3] - p[1]) * L[1]), M)
    elif k == 2:
        for t in L[1:]:
            M = np.dot(expm(-1j * H_RWA_2u(w, t - (L[1] / 2), p[1] - p[0]) * L[1]), M)
    elif k == 3:
        for t in L[1:]:
            M = np.dot(expm(-1j * H_RWA_2d(w, t - (L[1] / 2), p[3] - p[2]) * L[1]), M)
    return M

def ideal_pulse_generate(k, T, w):
    if k == 0:
        return expm(-1j * T * H_RWA1_1u(w, 0))
    elif k == 1:
        return expm(-1j * T * H_RWA1_1d(w, 0))
    elif k == 2:
        return expm(-1j * T * H_RWA1_2u(w, 0))
    elif k == 3:
        return expm(-1j * T * H_RWA1_2d(w, 0))

def phase_rec(k, p):
    p_err = 0.031675 * np.pi
    if k == 0:
        p[1] += p_err
        p[3] -= p_err
    elif k == 1:
        p[0] -= p_err
        p[2] += p_err
    elif k == 2:
        p[2] += p_err
        p[3] -= p_err
    elif k == 3:
        p[0] -= p_err
        p[1] += p_err


U = np.identity(4)          # with phase correction algorithm
U_nc = np.identity(4)       # no correction
U_ideal = np.identity(4)    # ideal gates (no crosstalk error

p_dummy = [0, 0, 0, 0]  # dummy phase input for non-compensated pulses
p_rec = [0, 0, 0, 0]    # phase record for compensation

N = 50

for i in range(N):
    index = randrange(4)
    print(i, " ", index)
    U = pulse_generate(index, T_pi_2, Ω, p_rec, 1000)@U
    U_nc = pulse_generate(index, T_pi_2, Ω, p_dummy, 1000)@U_nc
    U_ideal = ideal_pulse_generate(index, T_pi_2, Ω)@U_ideal
    phase_rec(index, p_rec)

proj_uu = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
proj_ud = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
proj_du = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
proj_dd = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

print("\nInitial state is uu")
print("\nU_exp projection (with correction):")
print("%.5f" % abs(np.trace(proj_uu @ U @ proj_uu @ U.conj().T)))
print("%.5f" % abs(np.trace(proj_ud @ U @ proj_uu @ U.conj().T)))
print("%.5f" % abs(np.trace(proj_du @ U @ proj_uu @ U.conj().T)))
print("%.5f" % abs(np.trace(proj_dd @ U @ proj_uu @ U.conj().T)))

print("\nU_exp projection (no correction):")
print("%.5f" % abs(np.trace(proj_uu @ U_nc @ proj_uu @ U_nc.conj().T)))
print("%.5f" % abs(np.trace(proj_ud @ U_nc @ proj_uu @ U_nc.conj().T)))
print("%.5f" % abs(np.trace(proj_du @ U_nc @ proj_uu @ U_nc.conj().T)))
print("%.5f" % abs(np.trace(proj_dd @ U_nc @ proj_uu @ U_nc.conj().T)))

print("\nU_ideal projection:")
print("%.5f" % abs(np.trace(proj_uu @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_ud @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_du @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_dd @ U_ideal @ proj_uu @ U_ideal.conj().T)))
