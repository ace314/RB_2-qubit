import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math

n = 4
omega = 10
J = n*omega
total_time = 1/omega
time_div = 1000

def H(w, j, t):
    return 1/2*2*np.pi*w*np.array( [[0, np.exp(1j*j*2*np.pi*t)],
                                    [np.exp(-1j*j*2*np.pi*t), 0]] )


L = np.linspace(0, total_time, time_div+1)
proj_u = np.array([[1, 0], [0, 0]])
M = np.identity(2)
prob_data = [0]
temp1 = 0
temp2 = 0
c = 0

for t in L[1:]:
    M = np.dot(expm(1j * H(omega, J, t-(L[1]/2)) * L[1]), M)
    ini = np.array([[0, 0], [0, 1]])
    fin = M @ ini @ M.conj().T
    prob = abs(np.trace(proj_u @ fin))
    prob_data.append(prob)
    if (temp2>temp1 and temp2<prob and c>(time_div/10/n)):
        print(temp2, t)
        print(M)
        c = 0
    temp1 = temp2
    temp2 = prob
    c += 1


figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
# prob2_plot = [x * 5 for x in prob2]
plt.plot(L, prob_data, 'r-')
# plt.xlim(left=0, right=12*T_pi_2*10**9)
plt.xlabel("T(s)")
plt.ylabel("prob(1↑)")
plt.title("synchronized resonant (blue) and off-resonant (red) transition probabilities")
plt.show()