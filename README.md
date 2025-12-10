# DimCiM Research Project: HDC-on-CiM Noise Tolerance and Energy Savings

This repository contains the simulator and experiment scripts for:

> **Thibaud Clement**,  
> *How Much Precision Do We Need? Quantifying HDC-on-CiM Noise Tolerance and Energy Savings*,  
> CS349H – Software Techniques for Emerging Hardware Platforms, Stanford University, 2025.

The goal of this project is to quantify how analog noise and ADC bit-depth affect the accuracy and energy efficiency of Hyperdimensional Computing (HDC)-style classifiers when executed on Compute-in-Memory (CiM) hardware. The simulator supports both a classical HDC model with random projections and a gradient-trained HDC-style (LDC) model used for the MNIST experiments, and it models CiM-style analog non-idealities (additive and multiplicative noise plus limited ADC precision) along with a parametric energy model to explore accuracy–noise–precision trade-offs.

---

## Installation

This project targets Python 3.10+ (tested with 3.11/3.12).

1. Clone the repository:

```bash
git clone https://github.com/thibaudclement/dimcim-simulation.git
cd dimcim-simulation
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the simulator

The main entry point is `dim_cim.py`, which exposes a command-line interface to:

- Choose a dataset (`--dataset`),
- Set HDC dimensionalities (`--features`, `--d`),
- Select a noise model (`--noise`),
- Configure quantization (`--per-dim-quant`, bit-depth sweep),
- Configure seeds and model initializations (`--seeds`, `--proj-seeds`),
- Select an energy regime (`--regime` or `--e-mac-pj`, `--e-adc8-pj`),
- Set output directory and plotting limits.

---

## Reproducing the MNIST experiments from the paper

The paper evaluates a 10-class MNIST classifier with:

- PCA to 128 features (`--features 128`),
- HDC projection dimensionality `D = 1024` (`--d 1024`),
- Bit-depths `b ∈ {3, 4, 5, 6, 8}`,
- Noise levels `σ ∈ [0, 0.20]` sampled in 17 steps,
- Per-dimension quantization enabled (`--per-dim-quant`),
- 3 model initializations and 10 noise realizations per (σ, b).

### 1. Additive noise — Default energy regime (baseline)

```bash
python dim_cim.py --dataset mnist --features 128 --d 1024 --noise additive --sigma-max 0.2 --sigma-steps 17 --per-dim-quant --seeds 10 --proj-seeds 3 --regime default --ylim-bits-min 0.9070 --ylim-bits-max 0.9130 --ylim-sigma-min 0.9070 --ylim-sigma-max 0.9130 --outdir data/output/mnist_add_default_ldc_tight
```

### 2. Multiplicative noise — Default energy regime (sanity check)

```bash
python dim_cim.py --dataset mnist --features 128 --d 1024 --noise multiplicative --sigma-max 0.2 --sigma-steps 17 --per-dim-quant --seeds 10 --proj-seeds 3 --regime default --ylim-bits-min 0.9070 --ylim-bits-max 0.9130 --ylim-sigma-min 0.9070 --ylim-sigma-max 0.9130 --outdir data/output/mnist_mul_default_ldc_tight
```

_**Note:** As discussed in the paper, multiplicative noise has an even smaller impact on accuracy than additive noise. The corresponding plots are therefore omitted from the main text and used primarily as a sanity check confirming that additive perturbations—not gain variation—are the dominant accuracy-relevant non-idealities in this model._

### 3. Additive noise — ADC-dominated regime (steeper knee)

```bash
python dim_cim.py --dataset mnist --features 128 --d 1024 --noise additive --sigma-max 0.2 --sigma-steps 17 --per-dim-quant --seeds 10 --proj-seeds 3 --regime adc_dominated --ylim-bits-min 0.9070 --ylim-bits-max 0.9130 --ylim-sigma-min 0.9070 --ylim-sigma-max 0.9130 --outdir data/output/mnist_add_adcdom_ldc_tight
```

---

## Interpreting the results

- The gradient-trained HDC-style classifier is **remarkably stable** across both noise and ADC precision.
- Across the 3–8 bit range and additive noise up to $\sigma = 0.20$, accuracies differ from the 8-bit baseline by at most about 0.4 percentage points (and by only a few thousandths for 4–8 bits).
- In the **default energy** regime: ≈10–12% energy savings when going from 8 → 4 bits.
- In the **ADC-dominated** regime: ≈50–55% energy savings when going from 8 → 4 bits.

In practical terms: **4–5 ADC bits are “enough”** for HDC-style inference on CiM for MNIST, preserving baseline accuracy while substantially reducing modeled energy.

---

## Extending the simulator

Future extensions include device-aware noise models, spatially structured variation, ADC asymmetry, IR-drop correlation, and in-memory similarity search.