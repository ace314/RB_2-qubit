import qecc as q
import pickle

it = q.clifford_group(2, consider_phases=True)
Cliff_2 = []
for i in range(11520):
    Cliff_2.append(it.__next__().as_unitary())

print(Cliff_2)

with open('Cliff2_unitary.pkl', 'wb') as f:
    pickle.dump(Cliff_2, f)