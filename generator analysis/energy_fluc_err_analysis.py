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
'''
noise_std = [std_uu, std_ud, std_du, std_dd]
N = len(Cliff_decompose)
p_rec = [0, 0, 0, 0]

for i in range(N):
    p_rec = [0, 0, 0, 0]
    state = error_initial_state(0, 0, 0)
    g_exp = get_seq([Cliff_decompose[i]], p_rec, noise_std=noise_std, noise_type=QUASI_STATIC)
    # g_exp = get_perfect_seq([Cliff_decompose[i]])
    state = g_exp @ state @ g_exp.conj().T

    initial_state = np.array([1, 0, 0, 0])
    g_pf = get_perfect_seq([Cliff_decompose[i]])
    initial_state = g_pf @ initial_state.T

    # print(abs(state))
    # print(abs(initial_state))
    f = np.real(prob_fidelity(abs(state), abs(initial_state), r2_pure=True))
    print(i, f)
    F += f

print(F/N)
'''


# one-qubit model testing

# simple X rotation Hamiltonian with Gaussian Z noise.
def h_1q(a, std):
    h = 1/2*2*np.pi*a*np.array([[1, 0], [0, -1]])
    np.random.seed()
    h_err = 2*np.pi*np.random.normal(0.0, std[0])*np.array([[1, 0], [0, -1]])
    return h + h_err

# X-2pi rotation
def x_2pi_1q(a=Omega, delta=101, t_total=4*T_pi_2, noise_std=None):
    if noise_std is None:
        noise_std = [0, 0]
    gate = np.identity(2)
    t_slice = np.linspace(0, t_total, delta + 1)
    h = h_1q(a, noise_std)
    for t in t_slice[1:]:
        gate = np.dot(expm(-1j * h * t_slice[1]), gate)
    return gate


c = 0
n = 500
x = []
y = []
perfect_state = np.array([1, 0]).T

for m in range(11):
    std0 = 21000 * m
    std1 = 21000 * m
    noise = [std0, std1]
    while c < n:
        # state = np.array([[1, 0], [0, 0]])
        g1 = x_2pi_1q(noise_std=noise)
        # g2 = x_2pi_1q(noise_std=noise)
        # g3 = x_2pi_1q(noise_std=noise)
        # g4 = x_2pi_1q(noise_std=noise)
        # g5 = x_2pi_1q(noise_std=noise)
        # g1 = g1 @ g1 @ g1 @ g1 @ g1
        # state = g1 @ state @ g1.conj().T
        # f = np.real(prob_fidelity(abs(state), abs(perfect_state), r2_pure=True))
        f = (np.absolute(np.trace(np.dot(g1.conj().T, np.identity(2)))))**2/4
        F += f/n
        c += 1
    print(m, ";", F)
    x.append(m)
    y.append(F)
    F = 0
    c = 0

f1 = open('x_1q.pkl', 'wb')
pickle.dump(x, f1)
f1.close()

f2 = open('y_1q.pkl', 'wb')
pickle.dump(y, f2)
f2.close()

plt.plot(x, y, 'bo')
plt.ylim(top=1.0)
plt.xlabel("Noise multiplier")
plt.ylabel("Fidelity")
plt.title("Energy Fluctuation Noise Strength Analysis")
plt.show()
