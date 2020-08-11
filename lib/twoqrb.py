"""
DblQuaDotRandomizedBenchmarking
=====

Provides
  1. Randomized benchmarking for 2 qubit system

"""

import pickle
from scipy.linalg import sqrtm
from lib.dblqdot import *

'''
Clifford decomposition
'''


'''
Basic 2 qubit gate operations
'''

# fidelity of two gates
def gate_fidelity(m_exp, m):
    return np.absolute(np.trace(np.dot(m_exp.conj().T, m)))/4

# fidelity of two probability distribution (two density matrices)
def prob_fidelity(r1, r2):
    a = sqrtm(r1)
    b = sqrtm(a @ r2 @ a)
    return np.trace(b)

# check whether two gates m1 and m2 are the same
def is_inverse(m1, m2):
    if np.allclose(np.absolute(np.trace(np.dot(m1, m2))), 4):
        return True
    else:
        return False


'''
initialization
'''

# initial "00" state mixed with diagonal error
def error_initial_state(e1, e2, e3):
    return np.array([[(1-e1-e2-e3), 0, 0, 0],
                     [0, e1, 0, 0],
                     [0, 0, e2, 0],
                     [0, 0, 0, e3]])


'''
RB sub-sequences
'''

# 2q Cliff elements decomposed by primitive gate indices saved in "Cliff_decompose"
with open("Cliff_indices_4.pkl", "rb") as f1:
    Cliff_decompose = pickle.load(f1)
f1.close()

# perfect primitive gates saved in "Prim_perfect"
with open("Prim_perfect.pkl", "rb") as f2:
    Prim_perfect = pickle.load(f2)
f2.close()

# following functions are included with phase correction algorithm. So there is a recording phase list p imported.
# keys: list of the prim_key's that combined to a Cliff element.
# prim_key is structured as follows:
# 'X(pi/2)' on Q2         = 0
# 'X(pi/2)', Q1           = 1
# 'X(pi/2)+CROT' on Q2    = 2
# 'X(pi/2)+CROT' on Q1    = 3
# 'Z(pi/2)+CROT' on Q2    = 4
# 'Z(pi/2)+CROT' on Q1    = 5
# 'CROT' on Q2            = 6
# 'CROT' on Q1            = 7
# 'Zv(pi/2)' on Q1        = 8
# 'Zv(pi)' on Q1          = 9
# 'Zv(3pi/2)' on Q1       = 10
# 'Zv(pi/2)' on Q2        = 11
# 'Zv(pi)' on Q2          = 12
# 'Zv(3pi/2)' on Q2       = 13
# seq: list of Cliff's expressed as prim_key lists (so it's a 2D array - list of prim_key lists.)
# p: current phase record (based on applied pulses) ;
# delta: total time slice ;
# noise_std: an 4-elements array with Gaussian energy fluctuation standard deviations for 4 states ;
# noise_type: QUASI_STATIC or STOCHASTIC ;

# given a single Cliff element decomposition keys then return experimental gate of the Cliff element
def get_cliff(keys, p, delta=1001, noise_std=None, noise_type=QUASI_STATIC):
    if noise_std is None:
        noise_std = [0, 0, 0, 0]
    g_exp = np.identity(4)
    for i in reversed(range(len(keys))):
        a = get_gates(p, keys[i], delta=delta, noise_std=noise_std, noise_type=noise_type)
        g_exp = a @ g_exp
    return g_exp

# given a single Cliff element decomposition keys then return perfect gate of the Cliff element
def get_perfect_cliff(keys):
    g_pf = np.identity(4)
    for i in reversed(range(len(keys))):
        g_pf = Prim_perfect[keys[i]] @ g_pf
    return g_pf

# given a Cliff decomposition sequence then return experimental combining gate of all the Cliff elements
def get_seq(seq, p, delta=1001, noise_std=None, noise_type=QUASI_STATIC):
    g = np.identity(4)
    for i in range(len(seq)):
        keys = seq[i]   # decomposition of i-th Cliff
        a = get_cliff(keys, p, delta=delta, noise_std=noise_std, noise_type=noise_type)
        g = a @ g
    return g

# given a Cliff decomposition sequence then return perfect combining gate of all the Cliff elements
def get_perfect_seq(seq):
    g = np.identity(4)
    for i in range(len(seq)):
        keys = seq[i]   # decomposition of i-th Cliff
        a = get_perfect_cliff(keys)
        g = a @ g
    return g

# given Cliff sequence , return its experimental inverse gate
def get_seq_inverse(seq, p, delta=1001, noise_std=None, noise_type=QUASI_STATIC):
    a = get_perfect_seq(seq)
    for i in range(len(Cliff_decompose)):
        b = get_perfect_cliff(Cliff_decompose[i])
        if is_inverse(b, a):
            return get_cliff(Cliff_decompose[i], p, delta=delta, noise_std=noise_std, noise_type=noise_type)
