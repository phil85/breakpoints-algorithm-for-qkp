![Cover Image](README/cover_image.png)

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

**Hochbaum, D. S., Baumann, P., Goldschmidt O., Zhang Y.** (2024): A Fast and Effective  Breakpoints Algorithm for the Quadratic Knapsack Problem. under review.

Bibtex:
```
@article{hochbaum2024fast,
	author={Hochbaum, Dorit S., Baumann, Philipp, Goldschmidt Olivier, Zhang Yiqing},
	title = {A Fast and Effective  Breakpoints Algorithm for the Quadratic Knapsack Problem},
	year={2024},
	journal = {under review},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details