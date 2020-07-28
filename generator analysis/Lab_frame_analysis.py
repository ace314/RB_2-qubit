import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math

global Ez
global dEz
global Ω
global J

Ez = 3.933e10
dEz = 1.326e7
Ω = 410000.0
J = 1.59e6
phi = np.pi/7     # phase offset of ac field

T_pi_2 = 1.0/(4.0*Ω)

f_1u = Ez + (dEz+J)/2
f_1d = Ez + (dEz-J)/2
f_2u = Ez + (-dEz+J)/2
f_2d = Ez + (-dEz-J)/2

def Fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M)))/4

def B_ac(w, f, t, p):  # w: Amplitude ; f: excitation frequency
    return w * np.exp(-1j*(2*np.pi*f*t + p))

def H(B):
    return 1/2 * 2*np.pi * np.array([[2*Ez, B, B, 0],
                                     [np.conj(B), dEz-J, 0, B],
                                     [np.conj(B), 0, -dEz-J, B],
                                     [0, np.conj(B), np.conj(B), -2*Ez]])

def R(t):
    return np.diag([np.exp(1j*Ez*2*np.pi*t), np.exp(1j*(dEz-J)*2*np.pi*t/2), np.exp(-1j*(dEz-J)*2*np.pi*t/2), np.exp(-1j*Ez*2*np.pi*t)])

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
                                    [np.exp(-1j*(J*2*np.pi*t - p)),                     0, 0, 0],
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


U_lab = np.identity(4)
U_rwa = np.identity(4)
U_crosstalk = np.identity(4)
U_crosstalk_p = np.identity(4)

L = np.linspace(0, T_pi_2, 1001)

for t in L[1:]:
    # U_lab = np.dot(expm(-1j * H(B_ac(Ω, f_1d, t-L[1]/2, phi)) * L[1]), U_lab)
    # U_rwa = np.dot(expm(-1j * H_RWA_1d(Ω, t-(L[1]/2), phi) * L[1]), U_rwa)
    U_crosstalk = np.dot(expm(-1j * H_RWA_2d(Ω, t - (L[1]/2), 0) * L[1]), U_crosstalk)
    # U_crosstalk_p = np.dot(expm(-1j * H_RWA2_1d(Ω, t - (L[1]/2), phi) * L[1]), U_crosstalk_p)


# M = R(T_pi_2).conj().T @ U_rwa
# print(Fidelity(M, U_lab))
# print(U_rwa)
print(U_crosstalk)
# print(U_crosstalk_p)