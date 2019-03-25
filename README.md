# RB_2-qubit

[1] About "C_2 decompose.py"

"C_2 decompose.py" is a simple python file to decompose two qubit clifford gates by generators given in arXiv:1805.05027v3 (Hereafter refered to as "original paper".

Now the output in "C_2 decompose.py" is temporarily an array, which contains number of C_2 gates made up by 0,1,2....l primitive gates, corresponding to the "Extended Data Table I" in the original paper. One can change the variable "l" in line 169 (l=1 by default).

To extract the information about C_2 gates in terms of the generators, the output form should be modified as ones need.


[2] About "generator.py"

"generator.py" is a simulation file to construct the generators in Fig. 4 in the original paper by the Hamiltonian given in the supplemental information.

Now the output is the fedilities of 4 generators (X/2, X/2+CROT, Z-CROT, CROT), comparing to the ideal cases.
