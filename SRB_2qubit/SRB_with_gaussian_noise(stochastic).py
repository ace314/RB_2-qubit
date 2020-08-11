import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import multiprocessing as mp
from lib.twoqrb import *

# stochastic noise deviation
std_uu = 16100
std_ud = 10100
std_du = 21000
std_dd = 0

# std_uu = 0
# std_ud = 0
# std_du = 0
# std_dd = 0

noise = [std_uu, std_ud, std_du, std_dd]

def RB_single_seq(L, repetition=125, ini_error=None, rd_error=None):
    initial = error_initial_state(ini_error[0], ini_error[1], ini_error[2])
    num_uu = 0
    cliff_seq = random.choices(Cliff_decompose, k=L)  # Cliff_decompose 隨便可重複取L個
    for j in range(repetition):
        phase = [0, 0, 0, 0]
        g1 = get_seq(cliff_seq, phase, noise_std=noise, noise_type=QUASI_STATIC)
        g2 = get_seq_inverse(cliff_seq, phase, noise_std=noise, noise_type=QUASI_STATIC)
        seq_k = g2 @ g1  # k_th

        final_state = seq_k @ initial @ seq_k.conj().T
        proj_measure = error_initial_state(rd_error[0], rd_error[1], rd_error[2])
        final_prob = abs(np.trace(proj_measure @ final_state))
        print(final_prob)
        # uu_count = [1, 0]
        # a = random.choices(uu_count, weights=[final_prob, 1-final_prob], k=1)
        # num_uu = num_uu + a[0]
        num_uu += final_prob
    return num_uu / repetition


# define the gate length series chosen to run RB protocol
l1 = np.arange(1, 20, 1)
l2 = np.arange(20, 40, 2)
l3 = np.arange(40, 65, 5)
# x = np.hstack((l1, l2, l3))
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

y = []
yerr = []

# N = 2  # samples for each Cliff_decompose point (算標準差即是使用每個data point對應的N個數據)
K = 1  # choices of S sequence 相同長度 重複取k次不同seq(等同K_L參數)
s_re = 1  # repeated times for each sequence
initial_error = [0, 0, 0]   # [e_ud, e_du, e_dd]
readout_error = [0, 0, 0]

def RB_loop(L):
    print(L)
    return RB_single_seq(L, repetition=s_re, ini_error=initial_error, rd_error=readout_error)


if __name__ == '__main__':
    for m in x:
        a = [m]*K
        pool = mp.Pool()
        res = pool.map(RB_loop, a)  # RB rep
        y.append(np.mean(res))
        yerr.append(np.std(res))
        # print(np.std(res))
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
    # p0 is the guess of the parameters.
    # Guess B ~ 0 (ideally be 0.25) and r ~ 0 (no noise model now so r should be ultra low)
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
