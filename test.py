import qecc as q
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#### Generating noisy Clifford array
''' 
a = q.clifford_group(1, consider_phases=True)
a1 = list(q.pauli_group(1))

# print(len(a))
b = []
c = []

for i in range(24):
    b.append(next(a).as_unitary())
    c.append(b[i].conj().T)
    print(b[i], '\n')

for i in range(24):
    phi = np.random.normal(0.0, 0.01*np.pi)
    M = np.array([[np.exp(-1j*phi), 0], [0, np.exp(1j*phi)]])
    b[i] = M @ b[i]

for i in range(24):
    phi = np.random.normal(0.0, 0.01*np.pi)
    M = np.array([[np.exp(-1j*phi), 0], [0, np.exp(1j*phi)]])
    c[i] = M @ c[i]

file = open('Cliff_1.dat', 'w')

for i in range(24):
    for j in range(2):
        for k in range(2):
            s1 = str(b[i][j][k].real)
            s2 = str(b[i][j][k].imag)
            file.write(s1 + '\t' + s2 + '\n')
    file.write('\n')

file.close()

file2 = open('Cliff_1_H.dat', 'w')

for i in range(24):
    for j in range(2):
        for k in range(2):
            s1 = str(c[i][j][k].real)
            s2 = str(c[i][j][k].imag)
            file2.write(s1 + '\t' + s2 + '\n')
    file2.write('\n')

file2.close()
'''

#### Function fitting and plot

x=[]
y=[]
with open('RB.dat') as handler:
    for line in handler:
        parts = line.split()
        x.append(int(parts[0]))
        y.append(float(parts[1]))


def func(x, A, B, r):
    return A * (1 - 2 * r) ** x + B


# def func(x, A):
#     return  * (1 - 4/3*0.053)**x + 0.25

popt, pcov = curve_fit(func, x, y, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
# p0 is the guess of the parameters. Guess B ~ 0 (ideally be 0.25) and r ~ 0 (no noise model now so r should be ultra low)
print("F_Ciff = 1 - r = ", 1 - popt[2])
print("A = ", popt[0])
print("B = ", popt[1])

plt.errorbar(x, y, fmt='o', markersize=4, capsize=3)
# plt.plot(x, func(x, *popt), 'r-')
# plt.plot(x, func(x, 0.75, 0.25, 0.053), 'b-')
plt.ylim(top=1.5, bottom=0.5)
plt.xlabel("Number of Cliffords (L)")
plt.ylabel("Proj. State Prob.")
plt.title("One-qubit RB (No noise)")
plt.show()