from lib.twoqrb import *
import matplotlib.pyplot as plt

std_uu = 16100
std_ud = 10100
std_du = 21000
std_dd = 0

F = 0

# Single primitive sequence [4, 9, 4] testing with different noise strength
'''
p_rec = [0, 0, 0, 0]
c = 0
n = 20
x = []
y = []
initial_state = error_initial_state(0, 0, 0)

for m in range(11):
    std0 = 16100 * m
    std1 = 10100 * m
    std2 = 21000 * m
    std3 = 0 * m
    noise = [std0, std1, std2, std3]
    while c < n:
        state = error_initial_state(0, 0, 0)
        g1 = get_gates(p_rec, 4, noise_std=noise, noise_type=QUASI_STATIC)
        g2 = get_gates(p_rec, 9, noise_std=noise, noise_type=QUASI_STATIC)
        g3 = get_gates(p_rec, 4, noise_std=noise, noise_type=QUASI_STATIC)
        state = g3 @ g2 @ g1 @ state @ g1.conj().T @ g2.conj().T @ g3.conj().T
        F += abs(prob_fidelity(state, initial_state)/n)
        c += 1
    print(m, ";", F)
    x.append(m)
    y.append(F)
    F = 0
    c = 0

f1 = open('x.pkl', 'wb')
pickle.dump(x, f1)
f1.close()

f2 = open('y.pkl', 'wb')
pickle.dump(y, f2)
f2.close()

plt.plot(x, y, 'bo')
plt.ylim(top=1.0)
plt.xlabel("Noise multiplier")
plt.ylabel("Fidelity")
plt.title("Energy Fluctuation Noise Strength Analysis")
plt.show()
'''

# Clifford average fidelity

noise_std = [std_uu, std_ud, std_du, std_dd]
N = len(Cliff_decompose)

for i in range(N):
    p_rec = [0, 0, 0, 0]
    state = error_initial_state(0, 0, 0)
    g_exp = get_seq([Cliff_decompose[i]], [0, 0, 0, 0], noise_std=noise_std, noise_type=QUASI_STATIC)
    state = g_exp @ state @ g_exp.conj().T

    initial_state = error_initial_state(0, 0, 0)
    g_pft = get_perfect_seq([Cliff_decompose[i]])
    initial_state = g_pft @ initial_state @ g_pft.conj().T

    f = np.real(prob_fidelity(state, initial_state))
    print(i, f)
    F += f/N

print(F)

