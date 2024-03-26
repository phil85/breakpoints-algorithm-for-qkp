# Breakpoints Algorithm
A scalable algorithm for quadratic knapsack problems

# Installation

1. Clone the repository.

2. Compile the C code of the simple parametric cut procedure `pseudopar.c` with a GNU C compiler.

    For Linux users:
    1. Open a terminal and navigate to the folder that contains the file `pseudopar.c`.
    2. Compile the C code with the command:
       ```bash
       gcc pseudopar.c -o pseudo_par
       ```

    For Windows users:
    1. Install [Cygwin](https://www.cygwin.com/).
    2. Open the Cygwin shell and navigate to the folder that contains the file `pseudopar.c`.
    3. Compile the C code with the command:
       ```bash
       gcc pseudopar.c pseudo_par.exe
       ```

3. All set, you can now run the algorithm in the `main.py` file.