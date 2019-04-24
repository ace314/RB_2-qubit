# RB_2-qubit
## Installation

After cloning all files, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the modules needed into your virtual environment.

```bash
pip -r /path/to/requirements.txt
```

## About "C_2 decompose demo.py"

"C_2 decompose demo.py" is a simple illustrative python file to decompose two qubit clifford gates by generators given in arXiv:1805.05027v3 (Hereafter refered to as "original paper").

Now the output in "C_2 decompose demo.py" is an array, which contains number of C_2 gates made up by 0,1,2....l (lowercase "L") primitive gates, corresponding to the "Extended Data Table I" in the original paper. One can change the variable "l" (lowercase "L") in line 169 (l=1 by default).

```bash
line 170: l = 1
```

To extract the information about C_2 gates in terms of the generators, the output form should be modified as ones need.

## About "C_2 decompose.py"

"C_2 decompose.py" is a python file that output the C_2 gates in terms of the indices of generators. The indices convertion are in "index_conv.txt", as follows:

```bash
['X(pi/2)', 2]         = 0
['X(pi/2)', 1]         = 1
['X(pi/2)+CROT', 2]    = 2
['X(pi/2)+CROT', 1]    = 3
['Z(pi/2)+CROT', 2]    = 4
['Z(pi/2)+CROT', 1]    = 5
['CROT', 2]            = 6
['CROT', 1]            = 7
['Zv(pi/2)', 1]        = 8
['Zv(pi)', 1]          = 9
['Zv(3pi/2)', 1]       = 10
['Zv(pi/2)', 2]        = 11
['Zv(pi)', 2]          = 12
['Zv(3pi/2)', 2]       = 13
```

The output of "C_2 decompose.py" is a pickle file, containing a list with digital-array-like elements. Each element corresponds to a Cliff_2 element and is decomposed into indices labelling the primitive and virtual-Z gates.

## About "generator.py"

"generator.py" is a simulation file to construct the generators in Fig. 4 in the original paper by the Hamiltonian given in the supplemental information. Here there is only crosstalk error without any noise model introduced.

Now the output is the fedilities of 4 generators (X/2, X/2+CROT, Z-CROT, CROT), comparing to the ideal cases. For example, the output form is of the following:

```bash
dt = T/100000
F(X_2):  0.9950523449918148
F(X_CROT):  0.9950524587591166
F(Z_CROT):  0.9901292722503505
F(CROT):  0.9901292722503505

dt = T/50
F(X_2):  0.9945902054775931
F(X_CROT):  0.9950645845697599
F(Z_CROT):  0.9896792799268357
F(CROT):  0.9896792799268357
```
