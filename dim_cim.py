"""
DimCiM HDC-on-CiM Simulator

This script implements a compact, NumPy-based simulator for
Hyperdimensional Computing (HDC) classifiers executed on
Compute-in-Memory (CiM) hardware.

It supports:
  - Synthetic, digits, and MNIST datasets (with optional PCA).
  - HDC encoding via random bipolar projections and class prototypes.
  - Train-learned 8-bit quantization ranges (global or per-dimension).
  - Additive and multiplicative Gaussian noise on MAC outputs.
  - A parametric energy model for MAC and ADC costs.
  - Sweeps over noise and ADC bit-depth, with averaging across
    multiple projections and noise draws.
  - Generation of CSV summaries and publication-ready plots.

Associated paper:
  Thibaud Clement,
  "How Much Precision Do We Need? Quantifying HDC-on-CiM Noise Tolerance
   and Energy Savings", CS349H, Stanford University, 2025.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- Data loaders --------------------

def load_synthetic(n_classes=10, n_features=64,
                   n_train_per_class=150, n_test_per_class=60,
                   sep=1.0, cov_sigma=0.25, seed=42):
    """
    Generate a synthetic classification dataset with Gaussian clusters.

    Each class c is centered at a point drawn from a base Gaussian, then
    linearly scaled and shifted to control inter-class separation.

    Args:
        n_classes: Number of distinct classes.
        n_features: Dimensionality of the feature space.
        n_train_per_class: Training samples per class.
        n_test_per_class: Test samples per class.
        sep: Scalar controlling separation between class centers.
        cov_sigma: Standard deviation of the isotropic covariance.
        seed: RNG seed for reproducibility.

    Returns:
        (X_train, y_train, X_test, y_test) with features in [0,1].
    """
    rng = np.random.default_rng(seed)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Sample class centers and rescale to control separation
    centers = rng.normal(0, 0.5, size=(n_classes, n_features))
    centers = (centers - centers.min()) / (centers.max() - centers.min())
    centers = (centers - 0.5) * sep + 0.5

    cov = np.eye(n_features) * (cov_sigma ** 2)
    for c in range(n_classes):
        Xc_train = rng.multivariate_normal(mean=centers[c], cov=cov, size=n_train_per_class)
        Xc_test  = rng.multivariate_normal(mean=centers[c], cov=cov, size=n_test_per_class)
        X_train_list.append(Xc_train); y_train_list.append(np.full(n_train_per_class, c))
        X_test_list.append(Xc_test);   y_test_list.append(np.full(n_test_per_class, c))

    X_train = np.clip(np.vstack(X_train_list), 0, 1)
    y_train = np.concatenate(y_train_list)
    X_test  = np.clip(np.vstack(X_test_list), 0, 1)
    y_test  = np.concatenate(y_test_list)
    return X_train.astype(np.float32), y_train.astype(int), \
           X_test.astype(np.float32),  y_test.astype(int)


def load_digits_split(test_size=0.25, seed=42):
    """
    Load the sklearn digits dataset and return a train/test split.
    If sklearn is not available (or fails), falls back to a synthetic dataset.
    """
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        X, y = load_digits(return_X_y=True)
        X = (X - X.min()) / (X.max() - X.min())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        return X_train.astype(np.float32), y_train.astype(int), \
               X_test.astype(np.float32),  y_test.astype(int)
    except Exception as e:
        print("[warn] sklearn not available or load_digits failed:", e)
        print("[warn] Falling back to synthetic data.")
        return load_synthetic(n_classes=10, n_features=64, sep=1.0, cov_sigma=0.25, seed=seed)


def load_mnist_split(test_size=0.25, seed=42, n_components=None):
    """
    Load MNIST from OpenML (via sklearn). If n_components (<784) is provided, apply PCA.

    Args:
        test_size: Fraction of data allocated to the test set.
        seed: Random seed for the train/test split and PCA.
        n_components: If set and <784, number of PCA components; otherwise
            use raw 784-dimensional pixels.

    Returns:
        (X_train, y_train, X_test, y_test) as float32 / int arrays.
    """
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = (X - X.min()) / (X.max() - X.min())
        y = y.astype(int)

        if n_components is not None and 0 < n_components < X.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=int(n_components), random_state=seed, svd_solver="auto")
            X = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        return X_train.astype(np.float32), y_train.astype(int), \
               X_test.astype(np.float32),  y_test.astype(int)
    except Exception as e:
        print("[warn] MNIST load failed:", e)
        print("[warn] Falling back to digits dataset.")
        return load_digits_split(test_size=test_size, seed=seed)

# -------------------- HDC model --------------------

def bipolar_sign(x):
    """
    Map real-valued inputs to bipolar representation.
    This is the core discretization step for HDC hypervectors.
    """
    return np.where(x >= 0, 1.0, -1.0)

class HDCClassifier:
    """
    Simple HDC classifier with random projection and class prototypes.

    - Random bipolar projection P scaled by sqrt(n_features)
    - Train-learned 8-bit quantization step/range (per-dim optional)
    - At inference, coarsen step for requested bit-depth: step_b = step_8 * 2^(8-b)

    The classifier:
      1. Projects input features into a D-dimensional space.
      2. Binarizes the projection to a bipolar hypervector.
      3. Builds class prototypes by bundling (summing) and re-binarizing
         hypervectors for each class.
      4. Classifies new samples via similarity (dot product) to prototypes.
    """
    def __init__(self, d=512, n_features=64, n_classes=10, seed=123):
        self.d = d
        self.n_features = n_features
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        # Random bipolar projection matrix with 1/sqrt(n_features) scaling
        P = rng.choice([-1.0, 1.0], size=(d, n_features)).astype(np.float32)
        self.P = P / np.sqrt(n_features)
        self.class_hv = None
        self.q_step_8 = None
        self.q_rng    = None

    def _quantize_with_bits(self, y, bits: int):
        """
        Quantize analog projection y using an effective ADC resolution 'bits'.

        The 8-bit step and range are learned at training time; lower-precision
        step sizes are derived by coarsening the 8-bit step.
        """
        if self.q_step_8 is None or self.q_rng is None:
            return y
        scale = 2 ** (8 - int(bits))
        step_b = self.q_step_8 * scale
        y = np.clip(y, -self.q_rng, self.q_rng)
        return np.round(y / step_b) * step_b

    def encode(self, x, noise=None, quant_bits=None):
        """
        Project a batch of inputs and optionally apply noise + quantization,
        then binarize to bipolar hypervectors.
        """
        y = x @ self.P.T
        if noise is not None and noise.get('sigma', 0) > 0:
            sigma = float(noise['sigma'])
            if noise['type'] == 'additive':
                y = y + np.random.normal(0, sigma, size=y.shape)
            elif noise['type'] == 'multiplicative':
                y = y * (1.0 + np.random.normal(0, sigma, size=y.shape))
        if quant_bits is not None:
            y = self._quantize_with_bits(y, quant_bits)
        return bipolar_sign(y)

    def fit(self, X, y, quant_bits=8, per_dim=False):
        """
        Fit class prototypes and learn 8-bit quantization ranges from training data.

        Args:
            X: Training features [n_samples, n_features].
            y: Integer labels [n_samples].
            quant_bits: Number of bits for the "reference" quantization (default 8).
            per_dim: If True, use per-dimension statistics for quantization;
                     otherwise use a single global standard deviation.
        """
        # Learn quantization ranges from training projections
        Ytr = X @ self.P.T
        if per_dim:
            std = np.std(Ytr, axis=0, keepdims=True) + 1e-9
        else:
            std = np.array([[np.std(Ytr) + 1e-9]])
        levels_8 = 2 ** 8
        self.q_rng   = 3.0 * std
        self.q_step_8 = (2.0 * self.q_rng) / levels_8

        # Encode training data at 8 bits and build class prototypes
        enc = self.encode(X, noise=None, quant_bits=8)
        class_sum = np.zeros((self.n_classes, self.d), dtype=np.float32)
        for c in range(self.n_classes):
            class_sum[c] = enc[y == c].sum(axis=0)
        self.class_hv = bipolar_sign(class_sum)

    def predict(self, X, noise=None, quant_bits=None):
        """
        Predict labels for a batch of inputs, with optional noise/quantization.
        """
        enc = self.encode(X, noise=noise, quant_bits=quant_bits)
        sims = enc @ self.class_hv.T
        return np.argmax(sims, axis=1)
    

# -------------------- LDC-style gradient-trained HDC model --------------------

class LDCClassifier:
    """
    Gradient-trained HDC-style classifier (LDC-like):
    - Learnable projection W (D x F) and class hypervectors H (C x D).
    - Trained with softmax cross-entropy via manual gradient descent.
    - After training, learn 8-bit quantization ranges from training projections.
      and reuse the same noise + quantization path as HDCClassifier.
    """
    def __init__(self, d=512, n_features=64, n_classes=10, seed=123,
                 lr=0.05, epochs=20, batch_size=256, l2_reg=1e-4):
        self.d = d
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2_reg = float(l2_reg)

        rng = np.random.default_rng(seed)
        # Learnable projection and class weights (real-valued)
        self.W = rng.normal(0, 1.0 / np.sqrt(n_features),
                            size=(d, n_features)).astype(np.float32)
        self.H = rng.normal(0, 1.0 / np.sqrt(d),
                            size=(n_classes, d)).astype(np.float32)

        self.q_step_8 = None
        self.q_rng = None

    @staticmethod
    def _softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y, n_classes):
        N = y.shape[0]
        Y = np.zeros((N, n_classes), dtype=np.float32)
        Y[np.arange(N), y] = 1.0
        return Y

    def _quantize_with_bits(self, y, bits: int):
        """
        Quantize analog projection y using an effective ADC resolution 'bits'.
        """
        if self.q_step_8 is None or self.q_rng is None:
            return y
        scale = 2 ** (8 - int(bits))
        step_b = self.q_step_8 * scale
        y = np.clip(y, -self.q_rng, self.q_rng)
        return np.round(y / step_b) * step_b

    def encode(self, X, noise=None, quant_bits=None, binarize=False):
        """
        Project a batch of inputs with learned W, optionally apply noise + quantization.
        For LDC/MAP-style inference we keep y real-valued by default.
        """
        y = X @ self.W.T

        if noise is not None and noise.get('sigma', 0) > 0:
            sigma = float(noise['sigma'])
            if noise['type'] == 'additive':
                y = y + np.random.normal(0, sigma, size=y.shape)
            elif noise['type'] == 'multiplicative':
                y = y * (1.0 + np.random.normal(0, sigma, size=y.shape))

        if quant_bits is not None:
            y = self._quantize_with_bits(y, quant_bits)

        if binarize:
            y = bipolar_sign(y)

        return y

    def fit(self, X, y, quant_bits=8, per_dim=False, verbose=False):
        """
        Train W and H with gradient descent, then learn 8-bit quantization ranges
        from training projections.

        Args:
            X: Training features [n_samples, n_features].
            y: Integer labels [n_samples].
            quant_bits: Unused (kept for API compatibility).
            per_dim: If True, use per-dimension statistics for quantization.
            verbose: If True, prints simple training loss/accuracy per epoch.
        """
        rng = np.random.default_rng(12345)
        N = X.shape[0]
        C = self.n_classes

        for epoch in range(self.epochs):
            # Shuffle indices
            idx = rng.permutation(N)
            X_shuf = X[idx]
            y_shuf = y[idx]

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                B = xb.shape[0]
                if B == 0:
                    continue

                # Forward pass
                z = xb @ self.W.T
                logits = z @ self.H.T
                P = self._softmax(logits)
                Y = self._one_hot(yb, C)

                # Cross-entropy gradient wrt logits
                grad_logits = (P - Y) / float(B)

                # Gradients
                grad_H = grad_logits.T @ z
                grad_z = grad_logits @ self.H
                grad_W = grad_z.T @ xb

                # L2 regularization
                if self.l2_reg > 0.0:
                    grad_W += self.l2_reg * self.W
                    grad_H += self.l2_reg * self.H

                # Parameter update
                self.W -= self.lr * grad_W
                self.H -= self.lr * grad_H

            if verbose:
                # Simple epoch-level accuracy estimate
                z_full = X @ self.W.T
                logits_full = z_full @ self.H.T
                preds = np.argmax(logits_full, axis=1)
                acc = (preds == y).mean()
                print(f"[LDC] epoch {epoch+1}/{self.epochs} train_acc={acc:.4f}")

        # After training, learn quantization ranges from training projections
        Ytr = X @ self.W.T
        if per_dim:
            std = np.std(Ytr, axis=0, keepdims=True) + 1e-9
        else:
            std = np.array([[np.std(Ytr) + 1e-9]])
        levels_8 = 2 ** 8
        self.q_rng   = 3.0 * std
        self.q_step_8 = (2.0 * self.q_rng) / levels_8

    def predict(self, X, noise=None, quant_bits=None):
        """
        Predict labels for a batch of inputs, with optional noise/quantization.

        We use real-valued encoded features and class weights (LDC/MAP-style),
        so inference is consistent with how the model was trained.
        """
        enc = self.encode(X, noise=noise, quant_bits=quant_bits, binarize=False)
        logits = enc @ self.H.T  # [N, C]
        return np.argmax(logits, axis=1)

# -------------------- Energy model --------------------

class EnergyModel:
    """
    Simple energy model for CiM-style HDC inference.

    E_total ≈ (#MACs) * E_MAC + (#ADC conversions) * E_ADC(bits),
    with E_ADC(bits) ~ E_ADC(8) * 2^(bits-8).

    This model abstracts away detailed circuit behavior in favor of a
    parametric scaling law that captures the exponential cost of ADC
    resolution and the linear cost of MAC operations.
    """
    def __init__(self, d, n_features, n_classes, e_mac_pj=0.5, e_adc_8bit_pj=10.0):
        self.d = d
        self.n_features = n_features
        self.n_classes = n_classes
        self.e_mac_pj = float(e_mac_pj)
        self.e_adc_8bit_pj = float(e_adc_8bit_pj)

    def adc_energy(self, bits: int):
        """
        Energy (pJ) of one ADC conversion at 'bits' effective resolution.
        """
        return self.e_adc_8bit_pj * (2 ** (int(bits) - 8))

    def per_sample(self, bits: int):
        """
        Estimate per-sample inference energy (pJ) for a given ADC bit-depth.
        """
        # Encoding MACs: D × F, similarity MACs: D × C
        macs_encode = self.d * self.n_features
        macs_sim    = self.d * self.n_classes
        e_mac = (macs_encode + macs_sim) * self.e_mac_pj
        
        # Conservatively assume one ADC conversion per projected dimension
        n_adc = self.d
        e_adc = n_adc * self.adc_energy(bits)
        return e_mac + e_adc

# -------------------- Sweep & plots --------------------

def run_sweep(X_train, y_train, X_test, y_test,
              noise_type='additive',
              sigmas=(0, 0.02, 0.05, 0.08, 0.10),
              bits_list=(3, 4, 5, 6, 8),
              d=512, n_features=64, n_classes=10,
              seeds=1, proj_seed=123, proj_seeds=1, quant_per_dim=False,
              e_mac_pj=0.5, e_adc_8bit_pj=10.0, verbose=True,
              training='vanilla'):
    """
    Sweep over noise amplitudes and bit-depths, averaging across projections and noise draws.

    Fit once per projection at 8-bit to learn an 8-bit step, then evaluate across
    requested bit-depths. Returns a DataFrame with mean accuracy, std, and energy.

    Args:
        training: 'vanilla' uses HDCClassifier (random projection + bundling),
                  'ldc' uses LDCClassifier (gradient-trained).
    """
    em  = EnergyModel(d=d, n_features=n_features, n_classes=n_classes,
                      e_mac_pj=e_mac_pj, e_adc_8bit_pj=e_adc_8bit_pj)

    records = []
    total = proj_seeds * len(bits_list) * len(sigmas) * seeds
    step  = max(1, total // 50)  # print ~50 progress ticks
    k = 0

    if verbose:
        print(f"[sweep] noise={noise_type} | proj_seeds={proj_seeds} | seeds={seeds} "
              f"| points={len(bits_list)}x{len(sigmas)} => {total} evals")
        print(f"[sweep] training={training}")

    for pidx in range(proj_seeds):
        # New random init per pidx
        if training == 'ldc':
            clf = LDCClassifier(d=d, n_features=n_features, n_classes=n_classes,
                                seed=proj_seed + pidx)
        else:
            clf = HDCClassifier(d=d, n_features=n_features, n_classes=n_classes,
                                seed=proj_seed + pidx)

        clf.fit(X_train, y_train, quant_bits=8, per_dim=quant_per_dim)

        for bits in bits_list:
            for sigma in sigmas:
                for s in range(seeds):
                    # Seed the noise generator for reproducibility of noise draws
                    np.random.seed(1000 + s)
                    preds = clf.predict(
                        X_test,
                        noise={'type': noise_type, 'sigma': float(sigma)},
                        quant_bits=bits
                    )
                    acc = (preds == y_test).mean()
                    records.append({
                        'noise_type': noise_type,
                        'sigma': float(sigma),
                        'bits': int(bits),
                        'acc': float(acc),
                        'energy_pj': float(em.per_sample(bits))
                    })
                    k += 1
                    if verbose and (k % step == 0 or k == total):
                        print(f"[sweep] progress {k}/{total} ({100.0*k/total:.1f}%)")

    df_runs = pd.DataFrame.from_records(records)
    grp = df_runs.groupby(['noise_type','sigma','bits'], as_index=False)
    df = grp.agg(acc_mean=('acc','mean'), acc_std=('acc','std'),
                 energy_pj=('energy_pj','first'))
    return df


def plot_acc_vs_sigma(df, bits_fixed, noise_type, out_dir: Path, fname,
                      ylim_min=None, ylim_max=None, echo_ylim_label=None):
    """
    Plot accuracy vs noise for a fixed bit-depth.
    """
    sub = df[(df['bits'] == bits_fixed) & (df['noise_type'] == noise_type)].sort_values('sigma')
    plt.figure(figsize=(6,4))
    yerr = sub['acc_std'].values if 'acc_std' in sub else None
    plt.errorbar(sub['sigma'], sub['acc_mean'], yerr=yerr, marker='o')
    plt.xlabel(r'Noise $\sigma$'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Noise (b={bits_fixed}, {noise_type})')
    if ylim_min is not None and ylim_max is not None:
        plt.ylim(ylim_min, ylim_max)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160); plt.close()
    if echo_ylim_label:
        ylo, yhi = sub['acc_mean'].min(), sub['acc_mean'].max()
        print(f"[ylims:{echo_ylim_label}] suggested (sigma-plot) "
              f"min={ylo:.6f}, max={yhi:.6f}")


def plot_acc_vs_bits(df, sigma_fixed, noise_type, out_dir: Path, fname,
                     ylim_min=None, ylim_max=None, echo_ylim_label=None):
    """
    Plot accuracy vs ADC bit-depth for a fixed noise level.
    """
    sub = df[(df['sigma'] == sigma_fixed) & (df['noise_type'] == noise_type)].sort_values('bits')
    plt.figure(figsize=(6,4))
    plt.plot(sub['bits'], sub['acc_mean'], marker='s')
    plt.xlabel('ADC bits'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Bits ($\\sigma={sigma_fixed}$, {noise_type})')
    if ylim_min is not None and ylim_max is not None:
        plt.ylim(ylim_min, ylim_max)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160); plt.close()
    if echo_ylim_label:
        ylo, yhi = sub['acc_mean'].min(), sub['acc_mean'].max()
        print(f"[ylims:{echo_ylim_label}] suggested (bits-plot) "
              f"min={ylo:.6f}, max={yhi:.6f}")


def plot_energy_vs_acc(df, sigma_fixed, noise_type, out_dir: Path, fname):
    """
    Plot energy per inference vs accuracy for a fixed noise level.
    """
    sub = df[(df['sigma'] == sigma_fixed) & (df['noise_type'] == noise_type)].sort_values('energy_pj')
    plt.figure(figsize=(6,4))
    plt.plot(sub['energy_pj'], sub['acc_mean'], marker='^')
    plt.xlabel('Energy per inference (pJ)'); plt.ylabel('Accuracy')
    plt.title(f'Energy vs Accuracy ($\\sigma={sigma_fixed}$, {noise_type})')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160); plt.close()

# -------------------- Main --------------------

def main():
    """
    Parse command-line arguments, load the requested dataset, configure the
    energy model, run the noise/bit-depth sweep, and write CSV + plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="synthetic",
                    choices=["synthetic", "digits", "mnist"])
    ap.add_argument("--outdir", type=str, default="",
                    help="Output dir (default: ./data/dim_cim_results)")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--features", type=int, default=64,
                    help="#features (synthetic); for MNIST, if <784 uses PCA to this size; if >=784 uses raw 784")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1,
                    help="Noise draws averaged per (sigma,bits,projection)")
    ap.add_argument("--proj-seed", type=int, default=123,
                    help="Base seed for random projection / init")
    ap.add_argument("--proj-seeds", type=int, default=1,
                    help="Number of different projections/inits to average over")
    ap.add_argument("--noise", type=str, default="additive",
                    choices=["additive","multiplicative"])
    ap.add_argument("--sep", type=float, default=1.0)
    ap.add_argument("--cov-sigma", type=float, default=0.25)
    ap.add_argument("--per-dim-quant", action="store_true")
    ap.add_argument("--sigma-max", type=float, default=0.10)
    ap.add_argument("--sigma-steps", type=int, default=6)

    ap.add_argument("--regime", type=str, default="default",
                    choices=["default","adc_dominated","mac_dominated"],
                    help="Energy presets: default (your prior), adc_dominated (converter heavy), mac_dominated (array heavy)")
    ap.add_argument("--e-mac-pj", type=float, default=None,
                    help="Override E_MAC pJ (takes precedence over --regime if set)")
    ap.add_argument("--e-adc8-pj", type=float, default=None,
                    help="Override 8-bit ADC energy pJ (takes precedence over --regime if set)")

    # Optional axis-freeze for comparable panels
    ap.add_argument("--ylim-bits-min", type=float, default=None)
    ap.add_argument("--ylim-bits-max", type=float, default=None)
    ap.add_argument("--ylim-sigma-min", type=float, default=None)
    ap.add_argument("--ylim-sigma-max", type=float, default=None)

    args = ap.parse_args()

    # Resolve output directory
    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir \
              else (Path.cwd() / "data" / "dim_cim_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset selection and loading
    if args.dataset == "digits":
        X_train, y_train, X_test, y_test = load_digits_split(seed=42)
        n_features = X_train.shape[1]; n_classes = len(np.unique(y_train))
        training_mode = 'vanilla'
    elif args.dataset == "mnist":
        ncomp = args.features if args.features > 0 and args.features < 784 else None
        X_train, y_train, X_test, y_test = load_mnist_split(seed=42, n_components=ncomp)
        n_features = X_train.shape[1]; n_classes = len(np.unique(y_train))
        # Use gradient-trained LDCClassifier for MNIST
        training_mode = 'ldc'
    else:
        X_train, y_train, X_test, y_test = load_synthetic(
            n_classes=args.classes, n_features=args.features,
            sep=args.sep, cov_sigma=args.cov_sigma, seed=42
        )
        n_features = args.features; n_classes = args.classes
        training_mode = 'vanilla'

    # Energy presets
    # Defaults from earlier runs: E_MAC=0.5 pJ, E_ADC8=10 pJ
    preset = {'e_mac_pj': 0.5, 'e_adc8_pj': 10.0}
    if args.regime == "adc_dominated":
        preset = {'e_mac_pj': 0.2, 'e_adc8_pj': 40.0}
    elif args.regime == "mac_dominated":
        preset = {'e_mac_pj': 0.8, 'e_adc8_pj': 6.0}
    # User overrides trump presets
    e_mac_pj = args.e_mac_pj if args.e_mac_pj is not None else preset['e_mac_pj']
    e_adc8_pj = args.e_adc8_pj if args.e_adc8_pj is not None else preset['e_adc8_pj']

    # Sweep grid for noise and bits
    sigmas = np.linspace(0.0, float(args.sigma_max), int(args.sigma_steps))
    bits_list = (3, 4, 5, 6, 8)

    # Run sweep with projection/init + noise averaging and progress logs
    df = run_sweep(
        X_train, y_train, X_test, y_test,
        noise_type=args.noise,
        sigmas=tuple(sigmas), bits_list=bits_list,
        d=args.d, n_features=n_features, n_classes=n_classes,
        seeds=args.seeds, proj_seed=args.proj_seed, proj_seeds=args.proj_seeds,
        quant_per_dim=args.per_dim_quant, e_mac_pj=e_mac_pj, e_adc_8bit_pj=e_adc8_pj,
        verbose=True, training=training_mode
    )

    # Save aggregated results
    csv_name = f"results_{args.noise}_{args.dataset}.csv"
    df.to_csv(out_dir / csv_name, index=False)

    # Pick plotting slice (midpoint of noise range)
    mid_sigma = round(float(args.sigma_max)/2, 6)

    # Plots (with optional axis freezing for consistent panels)
    plot_acc_vs_sigma(
        df, bits_fixed=4, noise_type=args.noise, out_dir=out_dir,
        fname=f"acc_vs_sigma_{args.noise}_{args.dataset}_b4.png",
        ylim_min=args.ylim_sigma_min, ylim_max=args.ylim_sigma_max,
        echo_ylim_label=f"{args.noise}-{args.dataset}-b4"
    )

    plot_acc_vs_bits(
        df, sigma_fixed=mid_sigma, noise_type=args.noise, out_dir=out_dir,
        fname=f"acc_vs_bits_{args.noise}_{args.dataset}_sMid.png",
        ylim_min=args.ylim_bits_min, ylim_max=args.ylim_bits_max,
        echo_ylim_label=f"{args.noise}-{args.dataset}-sMid"
    )

    plot_energy_vs_acc(
        df, sigma_fixed=mid_sigma, noise_type=args.noise, out_dir=out_dir,
        fname=f"energy_vs_acc_{args.noise}_{args.dataset}_sMid.png"
    )

    print("Saved files in:", out_dir)
    for p in sorted(out_dir.iterdir()):
        print("-", p.name)

if __name__ == "__main__":
    main()