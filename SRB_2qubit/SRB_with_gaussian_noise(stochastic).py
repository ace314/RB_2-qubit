# Generate pulses with Gaussian noise
import numpy as np
from scipy.linalg import expm
import math
import pickle
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing as mp

# random.seed(30)

Ez = 3.933e10
dEz = 1.326e7
Ω = 410000.0
J = 1.59e6
# supp eqn.7
n = 1
T_pi_2 = math.sqrt(16 * n ** 2 - 1) / 4 / J

# stochastic noise deviation
# std_uu = 16100
# std_ud = 10100
# std_du = 21000
# std_dd = 0

std_uu = 0
std_ud = 0
std_du = 0
std_dd = 0




# open Cliff_indices1 put into data
with open("Cliff_indices_4.pkl", "rb") as f:
    data = pickle.load(f)
f.close()

with open("Prim_perfect.pkl", "rb") as f4:
    Prim_perfect = pickle.load(f4)
f4.close()


def fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M))) / 4


# perfect gate
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


# cross error
def H_RWA2_1d(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, np.exp(1j*(J*2*np.pi*t - p)), 0],
                                    [0, 0, 0, 0],
                                    [np.exp(-1j*(J*2*np.pi*t - p)), 0, 0, 0],
                                    [0, 0, 0, 0]] )

def H_RWA2_1u(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0,  np.exp(-1j*(J*2*np.pi*t + p))],
                                    [0, 0, 0, 0],
                                    [0, np.exp(1j*(J*2*np.pi*t + p)), 0, 0]] )


def H_RWA2_2d(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0, np.exp(1j*(J*2*np.pi*t - p)), 0, 0],
                                    [np.exp(-1j*(J*2*np.pi*t - p)), 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]] )

def H_RWA2_2u(w, t, p):
    return 1/2*2*np.pi*w*np.array( [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, np.exp(-1j*(J*2*np.pi*t + p))],
                                    [0, 0, np.exp(1j*(J*2*np.pi*t + p)), 0]] )


def dH(std_uu, std_ud, std_du, std_dd):
    # return np.array([[np.random.normal(0.0, std_uu), 0, 0, 0],
    #                  [0, np.random.normal(0.0, std_ud), 0, 0],
    #                  [0, 0, np.random.normal(0.0, std_du), 0],
    #                  [0, 0, 0, np.random.normal(0.0, std_dd)]])
    return np.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])


# combine perfect and cross error
def H_RWA_1d(w, t, p, noise_std):
    return H_RWA1_1d(w, p) + H_RWA2_1d(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_1u(w, t, p, noise_std):
    return H_RWA1_1u(w, p) + H_RWA2_1u(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_2d(w, t, p, noise_std):
    return H_RWA1_2d(w, p) + H_RWA2_2d(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


def H_RWA_2u(w, t, p, noise_std):
    return H_RWA1_2u(w, p) + H_RWA2_2u(w, t, p) + dH(noise_std[0], noise_std[1], noise_std[2], noise_std[3])


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

def phase_rec(k, p):
    p_err = 0.031675 * np.pi    # TODO: More precise crosstalk error can be modified here
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
        # print(prim_key, p_l)
        return np.dot(A, B)
    else:
        return Prim_perfect[prim_key]


# for i in range(14):
#     print(fidelity(Prim_perfect[i], get_gates(prim_key=i, T_pi_2=T_pi_2, noise_std=[std_uu, std_ud, std_du, std_dd])))

def error_initial_state(e1, e2, e3):
    return np.array([[(1-e1-e2-e3), 0, 0, 0],
                     [0, e1, 0, 0],
                     [0, 0, e2, 0],
                     [0, 0, 0, e3]])

def Is_Inverse(A, B):
    if np.allclose(np.absolute(np.trace(np.dot(A, B))), 4):
        return True
    else:
        return False

def fidelity(M_exp, M):
    return np.absolute(np.trace(np.dot(M_exp.conj().T, M))) / 4

def get_s(g1, phi):  # given a Cliff list then return experimental composition gate of all the Cliff elements
    p = np.identity(4)  # 計數器for p@p1@p2....
    for i in range(len(g1)):
        array = g1[i]   # decomposition of i-th Cliff
        for j in reversed(range(len(array))):
            a = get_gates(phi, array[j], noise_std=[std_uu, std_ud, std_du, std_dd])
            print_phi = [x/0.031675/np.pi for x in phi]
            print(print_phi)
            p = a @ p
    return p

def get_s_perfect(g1):
    p = np.identity(4)  # 計數器for p@p1@p2....
    for i in range(len(g1)):
        array = g1[i]
        for j in reversed(range(len(array))):
            p = Prim_perfect[array[j]] @ p
    return p

def get_perfect_unitary(seq):  # Get perfect gate from decomposition info. in "Cliff_indices_4.pkl"
    p = np.identity(4)
    for i in reversed(range(len(seq))):
        p = Prim_perfect[seq[i]] @ p
    return p

def get_nonperfect_unitary(seq, phi):  # Get perfect gate from decomposition info. in "Cliff_indices_4.pkl"
    p = np.identity(4)
    for i in reversed(range(len(seq))):
        a = get_gates(phi, seq[i], noise_std=[std_uu, std_ud, std_du, std_dd])
        print_phi = [x / 0.031675 / np.pi for x in phi]
        print(print_phi)
        p = a @ p
    return p

# given list , return c^-1
def c_m(c1, phi):
    a = get_s_perfect(c1)
    for k in range(len(data)):
        b = get_perfect_unitary(data[k])
        if Is_Inverse(b, a):
            print(data[k])
            return get_nonperfect_unitary(data[k], phi)

def RB_single_seq(L, repetition=125, ini_error=[0, 0, 0], rd_error=[0, 0, 0]):
    initial = error_initial_state(ini_error[0], ini_error[1], ini_error[2])
    num_uu = 0
    list = random.choices(data, k=L)  # data 隨便可重複取m個
    print(list)
    for j in range(repetition):
        phase = [0, 0, 0, 0]
        A = get_s(list, phase)
        B = c_m(list, phase)
        # print(phase)
        seq_k = B @ A  # k_th
        print(fidelity(np.identity(4), seq_k))

        final_state = seq_k @ initial @ seq_k.conj().T
        proj_measure = error_initial_state(rd_error[0], rd_error[1], rd_error[2])
        final_prob = np.trace(proj_measure @ final_state)
        print(final_prob)
        uu_count = [1, 0]
        a = random.choices(uu_count, weights=[final_prob, 1-final_prob], k=1)
        num_uu = num_uu + a[0]
    return num_uu / repetition



#define the gate length series chosen to run RB protocol
l1 = np.arange(1, 20, 1)
l2 = np.arange(20, 40, 2)
l3 = np.arange(40, 65, 5)
# x = np.hstack((l1, l2, l3))
x = [2]


y = []
yerr = []

# N = 2  # samples for each data point (算標準差即是使用每個data point對應的N個數據)
K = 1  # choices of S sequence 相同長度 重複取k次不同seq(等同K_L參數)
s_re = 1  # repeated times for each sequence
c = 0
initial_error = [0, 0, 0]   #[e_ud, e_du, e_dd]
readout_error = [0, 0, 0]

def RB_loop(L):
    print(L)
    return RB_single_seq(L, repetition=s_re, ini_error=initial_error, rd_error=readout_error)


if __name__ == '__main__':
    for m in x:
        a = [m]*K
        pool = mp.Pool()
        res = pool.map(RB_loop, a)  # RB rep)
        y.append(np.mean(res))
        yerr.append(np.std(res))
    pool.close()
    pool.join()

    # saving the results
    f5 = open('2q_RB_simu_L.pkl', 'wb')
    pickle.dump(x, f5)
    f5.close()

    f6 = open('2q_RB_simu_y.pkl', 'wb')
    pickle.dump(y, f6)
    f6.close()

    f7 = open('2q_RB_simu_yerr.pkl', 'wb')
    pickle.dump(yerr, f7)
    f7.close()

    # Fitting function
    def func(x, A, B, r):
        return A * (1 - 4 / 3 * r) ** x + B

    # def func(x, A):
    #     return  * (1 - 4/3*0.053)**x + 0.25

    popt, pcov = curve_fit(func, x, y, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
    # p0 is the guess of the parameters. Guess B ~ 0 (ideally be 0.25) and r ~ 0 (no noise model now so r should be ultra low)
    print("F_Ciff = 1 - r = ", 1 - popt[2])
    print("A = ", popt[0])
    print("B = ", popt[1])

    plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=4, capsize=3)
    plt.plot(x, func(x, *popt), 'r-')
    # plt.plot(x, func(x, 0.75, 0.25, 0.053), 'b-')
    plt.ylim(top=1.0)
    plt.xlabel("Number of Cliffords (L)")
    plt.ylabel("Proj. State Prob.")
    plt.title("Two-qubit RB Fitting")
    plt.show()