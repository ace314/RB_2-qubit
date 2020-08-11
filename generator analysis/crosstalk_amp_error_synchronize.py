import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math

def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)


Ez = 3.933e10
dEz = 1.326e7
Ω = 410000.0
J = 1.59e6

n = 1
# T_pi_2 = 6.089596456300971e-07
# T_pi_2 = math.sqrt(16*n**2-1)/4/J
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

def H_RWA_1d(w, t):
    return H_RWA1_1d(w) + H_RWA2_1d(w, t)

def H_RWA_1u(w, t):
    return H_RWA1_1u(w) + H_RWA2_1u(w, t)

def Fidelity(M_exp, M):
    return np.absolute((np.trace(np.dot(M_exp.conj().T, M)))**2)/16

n = 1
T_pi_2 = math.sqrt(16*n**2-1)/4/J
T = 1.0/(4.0*Ω)
delta = 100


#### Resonant and off-resonant transition probability comparison.
M2 = np.identity(4)
M1 = np.identity(4)
M3 = np.identity(4)
prob2 = [0]
prob1 = [0]
prob3 = [0]
L = np.linspace(0, T_pi_2, 12001)
proj_1u = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

M_test = np.identity(2)
for t in L[1:]:
    M2 = np.dot(expm(-1j * H_RWA_1d(Ω, t-(L[1]/2)) * L[1]), M2)
    ini2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    fin2 = M2 @ ini2 @ M2.conj().T
    prob2.append(abs(np.trace(proj_1u @ fin2)))

    # M1 = np.dot(expm(-1j * H_RWA1_1d(Ω) * L[1]), M1)
    # ini1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    # fin1 = M1 @ ini1 @ M1.conj().T
    # prob1.append(abs(np.trace(proj_1u @ fin1)))
    #
    # M3 = np.dot(expm(-1j * H_RWA_1d(Ω, t-(L[1]/2)) * L[1]), M3)
    # fin3 = M3 @ ini1 @ M3.conj().T
    # prob3.append(abs(np.trace(proj_1u @ fin3)))


print(Fidelity(M2, expm(-1j*H_RWA1_1d(Ω)*T_pi_2)))
# print(M2)
print(np.arccos(M2[0][0].real)/np.pi)
print(np.arcsin(M2[0][0].imag)/np.pi)

plt.figure(1)
figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
# prob2_plot = [x * 5 for x in prob2]
plt.plot(L*10**9, prob2, 'r-')
# plt.plot(L*10**9, prob1, 'b-')
# plt.axvline(x=T_pi_2*10**9, c='k', ls='--')
# plt.axvline(x=2*T_pi_2*10**9, c='k', ls='--')
# for i in range(1,12):
#     plt.axvline(x=math.sqrt(16*i**2-1)/4/J*10**9, c='k', ls='--')
for i in range(3):
     plt.axvline(x=1/(2*Ω)*(2*i+1/2)*10**9, c='k', ls='--')
plt.xlim(left=0, right=12*T_pi_2*10**9)
plt.xlabel("T(ns)")
plt.ylabel("prob(1↑)")
plt.title("synchronized resonant (blue) and off-resonant (red) transition probabilities")

# plt.figure(2)
# figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
# plt.plot(L*10**9, prob3, 'b-')
# for i in range(1,12):
#     plt.axvline(x=math.sqrt(16*i**2-1)/4/J*10**9, c='k', ls='--')
# plt.xlim(left=0, right=12 * T_pi_2 * 10 ** 9)
# plt.xlabel("T(ns)")
# plt.ylabel("prob(1↑)")
# plt.title("Combined transition probabilities")


plt.show()

'''
#### phase error compensate.
L = np.linspace(0, T_pi_2, 12001)
A = np.identity(4)

for t in L[1:]:
    A = np.dot(expm(-1j * H_RWA2_1d(Ω, t-(L[1]/2)) * L[1]), A)

# proj_1u = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
# ini2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
# fin2 = A @ ini2 @ A.conj().T
# print(abs(np.trace(proj_1u @ fin2)))
# print(A @ np.array(([0], [0], [1], [0])))

true_phase = 0
fidelity_comparator = 0
phase = np.linspace(0, 4*np.pi, 10001)
U_1d_plus = expm(-1j*H_RWA1_1d(Ω)*(1.0/(4.0*Ω)))
for p in phase:
    B = expm(-1j * p/2 * np.kron(Z, I)) @ A
    F = Fidelity(B, np.identity(4))
    if F > fidelity_comparator:
        fidelity_comparator = F
        true_phase = p
        C = B
print(fidelity_comparator)
print(C)
print(p)
'''