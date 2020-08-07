# RPI 2020 REU [Notebook](notebook.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3937514.svg)](https://doi.org/10.5281/zenodo.3937514)

This repository contains my notebook, presentations, reports, and code from the
NSF REU at Rensselaer Polytechnic Institute in the summer of 2020. Due to
COVID-19, the REU was conducted remotely with my advisor, [Professor Vincent
Meunier](https://faculty.rpi.edu/vincent-meunier). To read the contents of the
project, start with the [final report](final-report/final-report-striffa.pdf).
The details, like the implementation of the Wang-Landau algorithm, are found in
the [extensive notebook](notebook.pdf).

## Building the notebook

The notebook is automatically compiled from individual entries in the `entries`
subdirectory and corresponding code in the `python-notebooks` directory.

I currently do not upload my Jupyter notebooks, but instead generate both code
source files and PDFs with output from them. This way, changes to the code of a
notebook are easily version-controlled, and the output is nicely readable. For
my uses, there is no loss of relevant information in the notebook to source
conversion, so it should be possible to (automatically) put the source back into
a notebook to be able to run it interactively. (This wasn't a priority for the
project.)

### Dependencies

- GNU Make
- Pandoc
- A Jupyter installation (for `jupyter-nbconvert`)
- A LaTeX installation with packages like `siunitx` and `minted`. I compile with XeTeX.
- Latexmk
- Python (to run a script to make notebook TeX use `minted` for code).

### Build Instructions

You should be able to just run `make` from the root of the repository. You can
also run `make clean` or `make distclean` at any level to (recursively) remove
generated files. In the `python-notebooks` subdirectory, `make edits` runs
`jupyter notebook` as a convenience.

