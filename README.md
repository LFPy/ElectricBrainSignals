# Electric Brain Signals

Notebooks and material for "Electric Brain Signals" written by
Geir Halnes, Torbjørn V. Ness, Solveig Næss, Espen Hagen, Klas H. Pettersen and Gaute T. Einevoll,
now published by Cambridge University Press.

These materials are made freely available under the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) [LICENSE](https://github.com/LFPy/ElectricBrainSignals/blob/main/LICENSE), unless otherwise noted.
Materials derived from other work adhere to the license of the original work (e.g., [GPL-3](https://www.gnu.org/licenses/gpl-3.0.txt)).

Copyright (c) 2022 Torbjørn V. Ness, Espen Hagen, Geir Halnes, Solveig Næss, Klas H. Pettersen, Gaute T. Einevoll.

## Citation info

If you use this material for your published works, please cite our book as follows

```
Halnes, G., Ness, T. V., Næss, S., Hagen, E., Pettersen, K. H., & Einevoll, G. T. (2024). Electric Brain Signals: Foundations and Applications of Biophysical Modeling. Cambridge: Cambridge University Press.
```

as well as this code repository as (replace version number and Zenodo-provided DOI accordingly):

```
Torbjørn V. Ness & Espen Hagen. (2023). LFPy/ElectricBrainSignals: ElectricBrainSignals-1.0.0r5 (v1.0.0rc5). Zenodo. https://doi.org/10.5281/zenodo.8414232
```

Bibtex format:
```
@book{Halnes_Ness_Næss_Hagen_Pettersen_Einevoll_2024,
  place={Cambridge},
  title={Electric Brain Signals: Foundations and Applications of Biophysical Modeling},
  publisher={Cambridge University Press}, author={Halnes, Geir and Ness, Torbjørn V. and Næss, Solveig and Hagen, Espen and Pettersen, Klas H. and Einevoll, Gaute T.},
  year={2024}
}

@software{espen_hagen_2023_8414232,
  author       = {Espen Hagen and
                  Torbjørn Vefferstad Ness},
  title        = {{LFPy/ElectricBrainSignals: 
                   ElectricBrainSignals-v1.0.0rc5}},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.0rc5},
  doi          = {10.5281/zenodo.8414232},
  url          = {https://doi.org/10.5281/zenodo.8414232}
}
```

## Getting the data

To use these codes, [clone](https://github.com/LFPy/ElectricBrainSignals) this repository by pressing that green button above and follow the instructions.

This repository also uses [git LFS](https://git-lfs.com) for large, non-code files.
After cloning the repository, initialize git LFS locally by issuing in the terminal:

```
cd </path/to/>ElectricBrainSignals
git lfs install
git pull
```

## File organization

Overview of files and folders:

```
- root level
    |_ README.md
    |_ CHANGELOG.md
    |_ CONTRIBUTING.md
    |_ setup.py
    |_ Dockerfile
    |_ requirements.txt
    |_ data/  # shared data directory
    |_ brainsignals/  # python extension folder
    |_ notebooks/  # jupyter notebooks
        |_ Ch-1/
            |_ Figure-01.ipynb
            |_ Figure-02.ipynb
            |_ ...
        |_ Ch-2/
        |_ Ch-.../
        |_ Appendix/
        |_ Misc/
```

## Python extension

In the current Python environment, issue

```
pip install -e .
```

in order to install the Python package `brainsignals` which may be required by the provided Jupyter notebooks.

## Issues and other problems

In case you encounter some problem running these codes, encounter a bug or similar, please do not hesitate to create a new GitHub [issue](https://github.com/LFPy/ElectricBrainSignals/issues/new/choose).

## Developer notes

- Don't assume that the user will have write access. If notebooks require modifying local files, do this in `/tmp/`
- Don't put files here that are not ours (neuron model files etc.), or not clearly free for use as we wish. It's better if these can be downloaded by the different notebooks as required.
- For large binary files (.zip, .tar.gz, .pdf, etc.), make sure that these (and similar files) are tracked using git LFS:

  ``` bash
  git lfs track "*.<file extension>"
  git add .gitattributes
  git add /path/to/binary/file
  ```

- Clear output from notebooks before committing.
