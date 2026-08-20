# Comet Centroiding Benchmark: Astrometric & Morphological Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22027201.svg)](https://doi.org/10.5281/zenodo.22027201)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

Official benchmarking and validation suite designed to rigorously evaluate and compare **astrometric centroiding algorithms** against complex cometary morphologies, including jets, diffuse comae, background gradients, and low-SNR regimes.

The results and methodology validated through this benchmark inform the core algorithms implemented in **[Kometra](https://github.com/Genofabio/kometra)** (developed within the research activities at [PoliTO Astronomy](https://github.com/PoliTO-Astronomy/Kometra)).

---

## 1. Overview

Traditional stellar centroiding algorithms, such as DAOPhot, standard center-of-mass methods, or basic Gaussian fits, can fail or introduce severe systematic biases when applied to cometary targets due to their non-Gaussian profiles, active jets, diffuse comae, and dense stellar fields.

This repository provides:

1. **The Generation Pipeline**
   Python scripts to synthesize realistic FITS images with known ground-truth coordinates (`true_x`, `true_y`).

2. **The Benchmark Engine**
   Automated testing frameworks to evaluate centroiding accuracy, robustness, window-size sensitivity, and offset drift.

3. **The Official Dataset**
   A pre-generated, standardized validation suite openly available through Zenodo.

---

## 2. Repository Structure

```text
comet-centroiding-benchmark/
├── src/                    # Core source code for simulation, benchmark, and analysis
├── data/                   # Local working directory for datasets and ground truth
├── results/                # Output directory for benchmark data and analysis reports
├── 01_generate_dataset.py  # Generate synthetic FITS images and CSV annotations
├── 02_run_benchmark.py     # Execute the centroiding benchmark tests
├── 03_analyze_results.py   # Generate statistical reports and validation plots
├── run.py                  # Master control script for the pipeline
├── pyproject.toml          # Package configuration and dependencies
└── LICENSE                 # GNU General Public License v3.0
```

---

## 3. Installation

Clone the repository and install the package in editable mode together with its dependencies:

```bash
git clone https://github.com/Genofabio/comet-centroiding-benchmark.git
cd comet-centroiding-benchmark
pip install -e .
```

---

## 4. Usage: Running with the Official Zenodo Dataset

For reproducible benchmarking, it is recommended to use the official standardized test suite published on Zenodo.

> **Dataset Reference:** Genovese, F. (2026). *Astrometric and Morphological Validation Dataset for Kometra (Synthetic Comet Benchmark Suite)*. Zenodo.
> DOI: [10.5281/zenodo.22014532](https://doi.org/10.5281/zenodo.22014532)

### Setup Steps

1. Download and extract the dataset archive from Zenodo.

2. Place the contents of `main_benchmark/` inside:

```text
data/synthetic_test/
```

3. Copy `synthetic_ground_truth.csv` into:

```text
data/
```

4. Execute the evaluation and analysis steps, skipping dataset generation:

```bash
python run.py benchmark
python run.py analyze
```

---

## 5. Usage: Generating the Dataset from Scratch

If you prefer to generate the synthetic images locally using the built-in physical models, run the complete pipeline through the master script.

The complete workflow consists of:

**Generation → Benchmark → Analysis**

```bash
python run.py all
```

The individual pipeline components are also available through the dedicated scripts:

```bash
python 01_generate_dataset.py
python 02_run_benchmark.py
python 03_analyze_results.py
```

---

## 6. Citation

If you use **Kometra**, its benchmark suite, or the associated dataset in your research, please cite both the software repository and the official dataset.

### Dataset Citation

```bibtex
@dataset{genovese_2026_dataset,
  author       = {Genovese, Fabio},
  title        = {{Synthetic Cometary Dataset for Centroiding Algorithm Selection (Kometra Benchmark Suite)}},
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.22014532},
  url          = {[https://doi.org/10.5281/zenodo.22014532](https://doi.org/10.5281/zenodo.22014532)}
}
```

---

## 7. License

This project is open-source software licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the [`LICENSE`](LICENSE) file for the complete license terms.

---

## 8. Reproducibility

The official Zenodo dataset provides a fixed and versioned benchmark reference for reproducible evaluation.

For published research, it is recommended to report:

* the Kometra Benchmark version;
* the Zenodo dataset version;
* the benchmark configuration;
* the centroiding methods evaluated;
* the ROI/window sizes used;
* the resulting astrometric error metrics.

This ensures that benchmark results can be independently reproduced and compared across implementations.

---

## 9. Project Links

* **Source repository (Benchmark):** https://github.com/Genofabio/comet-centroiding-benchmark
* **Kometra (Core Software):** https://github.com/Genofabio/kometra
* **PoliTO Astronomy Fork:** https://github.com/PoliTO-Astronomy/Kometra
* **Official dataset (Zenodo):** https://doi.org/10.5281/zenodo.22014532
* **Dataset DOI:** `10.5281/zenodo.22014532`
* **License:** https://www.gnu.org/licenses/gpl-3.0
