# RB_2-qubit
## Installation

After cloning all files, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the modules needed into your virtual environment.

```bash
pip -r /path/to/requirements.txt
```

## About "C_2 decompose.py"

"C_2 decompose.py" is a simple python file to decompose two qubit clifford gates by generators given in arXiv:1805.05027v3 (Hereafter refered to as "original paper").

Now the output in "C_2 decompose.py" is temporarily an array, which contains number of C_2 gates made up by 0,1,2....l (lowercase "L") primitive gates, corresponding to the "Extended Data Table I" in the original paper. One can change the variable "l" (lowercase "L") in line 169 (l=1 by default).

```bash
line 169: l = 1
```

To extract the information about C_2 gates in terms of the generators, the output form should be modified as ones need.


## About "generator.py"

"generator.py" is a simulation file to construct the generators in Fig. 4 in the original paper by the Hamiltonian given in the supplemental information. Here there is only crosstalk error without any noise model introduced.

Now the output is the fedilities of 4 generators (X/2, X/2+CROT, Z-CROT, CROT), comparing to the ideal cases. For example, the output form is of the following:

```bash
F(X_2):  0.9950523449918148
F(X_CROT):  0.9950524587591166
F(Z_CROT):  0.9901292722503505
F(CROT):  0.9901292722503505
```
