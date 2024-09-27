![Cover Image](README/breakpoints.jpg)

# Description

This repository contains the source code of the breakpoints algorithm for the Quadratic Knapsack Problem (QKP). 

The quadratic knapsack problem (QKP) is a variation of the classical knapsack problem, where the objective is to maximize a quadratic objective function. In this case, instead of simply selecting items to maximize a linear profit under a weight constraint, the profit includes interactions between pairs of items, making the profit a quadratic function. The goal is to select a subset of items such that the total weight does not exceed the capacity of the knapsack, and the sum of individual profits and pairwise interaction profits is maximized. This problem is more complex than the linear version due to the quadratic nature of the objective.

# Installation

1. Clone the repository.

2. Compile the C code of the simple parametric cut procedure `QKPsimparamHPF.c` with a GNU C compiler.

    For Linux/Mac users:
    1. Open a terminal and navigate to the folder that contains the file `QKPsimparamHPF.c`.
    2. Compile the C code with the command:
       ```bash
       gcc QKPsimparamHPF.c -o QKPsimparamHPF.exe
       ```

    For Windows users:
    1. Install [Cygwin](https://www.cygwin.com/).
    2. Open the Cygwin shell and navigate to the folder that contains the file `QKPsimparamHPF.c`.
    3. Compile the C code with the command:
       ```bash
       gcc QKPsimparamHPF.c -o QKPsimparamHPF.exe
       ```

3. All set, you can now run the algorithm in the `main.py` file.

## Reference

Please cite the following paper if you use this code.

**Hochbaum, D. S., Baumann, P., Goldschmidt O., Zhang Y.** (2024): A Fast and Effective Breakpoints Algorithm for the Quadratic Knapsack Problem. under review.

Bibtex:
```
@article{hochbaum2024fast,
	author={Hochbaum, Dorit S., Baumann, Philipp, Goldschmidt Olivier and Zhang Yiqing},
	title = {A Fast and Effective Breakpoints Algorithm for the Quadratic Knapsack Problem},
	year={2024},
	url = {https://arxiv.org/abs/2408.12183},
	doi = {https://doi.org/10.48550/arXiv.2408.12183},
	journal = {under review},
}
```
[->Link to paper](https://arxiv.org/abs/2408.12183)

## Links to related repositories

- [Repository containing all benchmark instances](https://github.com/phil85/benchmark-instances-for-qkp)
- [Repository containing all results](https://github.com/phil85/results-for-qkp-benchmark-instances)
- [Repository containing the code for Gurobi-based approach](https://github.com/phil85/gurobi-based-approach-for-qkp)
- [Repository containing the code for Hexaly-based approach](https://github.com/phil85/hexaly-based-approach-for-qkp)
- [Repository containing the code for the relative greedy algorithm](https://github.com/phil85/greedy-algorithm-for-qkp)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details