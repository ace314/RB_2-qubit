import numpy as np
from scipy.linalg import expm
import math
import pickle
from random import randrange

def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)

with open("Prim_perfect.pkl", "rb") as f:
    Prim_perfect = pickle.load(f)
f.close()


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

def dH(std_uu, std_ud, std_du, std_dd):
    return np.array([[np.random.normal(0.0, std_uu), 0, 0, 0],
                     [0, np.random.normal(0.0, std_ud), 0, 0],
                     [0, 0, np.random.normal(0.0, std_du), 0],
                     [0, 0, 0, np.random.normal(0.0, std_dd)]])
    # return np.array([[0, 0, 0, 0],
    #                  [0, 0, 0, 0],
    #                  [0, 0, 0, 0],
    #                  [0, 0, 0, 0]])

# combine perfect and cross error
def H_RWA_1d(w, t, p, noise_std):
    return H_RWA1_1d(w, p) + H_RWA2_1d(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_1u(w, t, p, noise_std):
    return H_RWA1_1u(w, p) + H_RWA2_1u(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_2d(w, t, p, noise_std):
    return H_RWA1_2d(w, p) + H_RWA2_2d(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_2u(w, t, p, noise_std):
    return H_RWA1_2u(w, p) + H_RWA2_2u(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])

def Fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M)))/4

def pulse_generate(k, T, w, p, delta, noise_std, QUASI_STATIC, sig=1):  # k: index for 1u, 1d, 2u, 2d ; sig: pi/2 or -pi/2 pulse (1 or -1)
    L = np.linspace(0, T, delta+1)
    M = np.identity(4)
    if QUASI_STATIC:    # quasi-static noise
        dH_static = dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])
        if k == 0:
            for t in L[1:]:
                H = H_RWA1_1u(w, p[2] - p[0]) + H_RWA2_1u(w, t - (L[1] / 2), p[2] - p[0]) + dH_static
                M = np.dot(expm(-1j * sig * H * L[1]), M)
        elif k == 1:
            for t in L[1:]:
                H = H_RWA1_1d(w, p[3] - p[1]) + H_RWA2_1d(w, t - (L[1] / 2), p[3] - p[1]) + dH_static
                M = np.dot(expm(-1j * sig * H * L[1]), M)
        elif k == 2:
            for t in L[1:]:
                H = H_RWA1_2u(w, p[1] - p[0]) + H_RWA2_2u(w, t - (L[1] / 2), p[1] - p[0]) + dH_static
                M = np.dot(expm(-1j * sig * H * L[1]), M)
        elif k == 3:
            for t in L[1:]:
                H = H_RWA1_2d(w, p[3] - p[2]) + H_RWA2_2d(w, t - (L[1] / 2), p[3] - p[2]) + dH_static
                M = np.dot(expm(-1j * sig * H * L[1]), M)
    else:   # stochastic noise
        if k == 0:
            for t in L[1:]:
                M = np.dot(expm(-1j * sig * H_RWA_1u(w, t - (L[1] / 2), p[2] - p[0], noise_std) * L[1]), M)
        elif k == 1:
            for t in L[1:]:
                M = np.dot(expm(-1j * sig * H_RWA_1d(w, t - (L[1] / 2), p[3] - p[1], noise_std) * L[1]), M)
        elif k == 2:
            for t in L[1:]:
                M = np.dot(expm(-1j * sig * H_RWA_2u(w, t - (L[1] / 2), p[1] - p[0], noise_std) * L[1]), M)
        elif k == 3:
            for t in L[1:]:
                M = np.dot(expm(-1j * sig * H_RWA_2d(w, t - (L[1] / 2), p[3] - p[2], noise_std) * L[1]), M)
    return M

def ideal_pulse_generate(k, T, w, sgn):
    if k == 0:
        return expm(-1j * sgn * T * H_RWA1_1u(w, 0))
    elif k == 1:
        return expm(-1j * sgn * T * H_RWA1_1d(w, 0))
    elif k == 2:
        return expm(-1j * sgn * T * H_RWA1_2u(w, 0))
    elif k == 3:
        return expm(-1j * sgn * T * H_RWA1_2d(w, 0))

def phase_rec(k, p):
    p_err = 0.03167654250993053 * np.pi
    if k == 0:      # 1u
        p[1] += p_err
        p[3] -= p_err
    elif k == 1:    # 1d
        p[0] -= p_err
        p[2] += p_err
    elif k == 2:    # 2u
        p[2] += p_err
        p[3] -= p_err
    elif k == 3:    # 2d
        p[0] -= p_err
        p[1] += p_err

def v_z(k, p):
    if math.floor(k / 11):  # key = 11, 12, 13
        phase = (k - 10) * np.pi / 4
        p[0] += phase
        p[1] -= phase
        p[2] += phase
        p[3] -= phase
    else:                   # key = 8, 9, 10
        phase = (k - 7) * np.pi / 4
        p[0] += phase
        p[1] += phase
        p[2] -= phase
        p[3] -= phase


def get_gates(p_l, prim_key, w=Ω, delta=1001, T=T_pi_2, noise_std=[0, 0, 0, 0], QUASI_STATIC=True):
    if prim_key < 8:
        if prim_key == 0:
            A = pulse_generate(2, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(2, p_l)
            B = pulse_generate(3, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(3, p_l)
        elif prim_key == 1:
            A = pulse_generate(0, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(0, p_l)
            B = pulse_generate(1, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(1, p_l)
        elif prim_key == 2:
            A = pulse_generate(2, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(2, p_l)
            B = pulse_generate(3, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(3, p_l)
        elif prim_key == 3:
            A = pulse_generate(0, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(0, p_l)
            B = pulse_generate(1, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(1, p_l)
        elif prim_key == 4:
            A = pulse_generate(2, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(2, p_l)
            B = pulse_generate(2, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(2, p_l)
        elif prim_key == 5:
            A = pulse_generate(0, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(0, p_l)
            B = pulse_generate(0, T, w, p_l, delta, noise_std, QUASI_STATIC, 1)
            phase_rec(0, p_l)
        elif prim_key == 6:
            A = pulse_generate(3, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(3, p_l)
            B = pulse_generate(3, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(3, p_l)
        elif prim_key == 7:
            A = pulse_generate(1, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(1, p_l)
            B = pulse_generate(1, T, w, p_l, delta, noise_std, QUASI_STATIC, -1)
            phase_rec(1, p_l)
        return np.dot(B, A)
    else:
        v_z(prim_key, p_l)
        return np.identity(4)

def ideal_get_gates(prim_key, T, w):
    if prim_key < 8:
        if prim_key == 0:           # ['X(pi/2)', 2]
            A = ideal_pulse_generate(2, T, w, 1)
            B = ideal_pulse_generate(3, T, w, 1)
        elif prim_key == 1:         # ['X(pi/2)', 1]
            A = ideal_pulse_generate(0, T, w, 1)
            B = ideal_pulse_generate(1, T, w, 1)
        elif prim_key == 2:         # ['X(pi/2)+CROT', 2]
            A = ideal_pulse_generate(2, T, w, 1)
            B = ideal_pulse_generate(3, T, w, -1)
        elif prim_key == 3:         # ['X(pi/2)+CROT', 1]
            A = ideal_pulse_generate(0, T, w, 1)
            B = ideal_pulse_generate(1, T, w, -1)
        elif prim_key == 4:         # ['Z(pi/2)+CROT', 2]
            A = ideal_pulse_generate(2, T, w, 1)
            B = ideal_pulse_generate(2, T, w, 1)
        elif prim_key == 5:         # ['Z(pi/2)+CROT', 1]
            A = ideal_pulse_generate(0, T, w, 1)
            B = ideal_pulse_generate(0, T, w, 1)
        elif prim_key == 6:         # ['CROT', 2]
            A = ideal_pulse_generate(3, T, w, -1)
            B = ideal_pulse_generate(3, T, w, -1)
        elif prim_key == 7:         # ['CROT', 1]
            A = ideal_pulse_generate(1, T, w, -1)
            B = ideal_pulse_generate(1, T, w, -1)
        return np.dot(B, A)
    else:                           # Zv-gates (for prim_key >= 8
        return Prim_perfect[prim_key]


U = np.identity(4)          # with phase correction algorithm
U_nc = np.identity(4)       # no correction
U_ideal = np.identity(4)    # ideal gates (no crosstalk error

p_dummy = [0, 0, 0, 0]  # dummy phase input for non-compensated pulses
p_rec = [0, 0, 0, 0]    # phase record for compensation

# Random pulses test (only +pi/2 pulses are involved without Zv-gates
'''
N = 50

for i in range(N):
    index = randrange(4)
    print(i, " ", index)
    U = pulse_generate(index, T_pi_2, Ω, p_rec, 1000)@U
    # U_nc = pulse_generate(index, T_pi_2, Ω, p_dummy, 1000)@U_nc
    U_ideal = ideal_pulse_generate(index, T_pi_2, Ω)@U_ideal
    phase_rec(index, p_rec)
'''

x = 3
std_uu = 16100 * x
std_ud = 10100 * x
std_du = 21000 * x
std_dd = 0 * x
# Random primitive gates test
Cliff_set = [[0, 9, 5, 8, 1, 8, 11], [10, 5, 11, 2, 9, 13], [11, 0, 8, 5, 12, 0, 8, 12]]
for i in range(len(Cliff_set)):
    cliff = Cliff_set[i]
    for j in reversed(range(len(cliff))):
        U = get_gates(p_rec, cliff[j], noise_std=[std_uu, std_ud, std_du, std_dd])@U
        U_ideal = ideal_get_gates(cliff[j], T_pi_2, Ω)@U_ideal

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

# print("\nU_exp projection (no correction):")
# print("%.5f" % abs(np.trace(proj_uu @ U_nc @ proj_uu @ U_nc.conj().T)))
# print("%.5f" % abs(np.trace(proj_ud @ U_nc @ proj_uu @ U_nc.conj().T)))
# print("%.5f" % abs(np.trace(proj_du @ U_nc @ proj_uu @ U_nc.conj().T)))
# print("%.5f" % abs(np.trace(proj_dd @ U_nc @ proj_uu @ U_nc.conj().T)))

print("\nU_ideal projection:")
print("%.5f" % abs(np.trace(proj_uu @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_ud @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_du @ U_ideal @ proj_uu @ U_ideal.conj().T)))
print("%.5f" % abs(np.trace(proj_dd @ U_ideal @ proj_uu @ U_ideal.conj().T)))
