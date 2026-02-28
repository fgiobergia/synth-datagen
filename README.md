# Differentiable Synthetic Dataset Generation

Reference implementation for:

> **Differentiable Synthetic Dataset Generation for Non-Trivial Regression Tasks**  
> DARLI-AP 2026 (EDBT/ICDT 2026 Workshop)

This repository implements a method to generate **synthetic regression datasets** that are:

- Learnable by a nonlinear model,
- Intentionally difficult for linear regression,
- Fully reproducible and controllable.

Instead of generating targets from inputs, we fix the target vector and **optimize the input features directly** using gradient descent.

---

## Method Summary

Given:

- A frozen nonlinear neural network $ f_\theta $
- A fixed target vector $ y $

We optimize the feature matrix $ X $ to minimize:

$$
\mathcal{L} = \text{MSE}(f_\theta(X), y) - \lambda \cdot \text{MSE}(X\beta, y)
$$

where:

- The first term enforces learnability by the nonlinear model.
- The second term penalizes linear regression.
- $ \lambda $ controls task difficulty.

---

## Requirements

- Python ≥ 3.8  
- PyTorch  
- NumPy  
- pandas  
- scikit-learn  

Install with:

```bash
pip install torch numpy pandas scikit-learn
```

---

## Usage

Run dataset generation from the command line:

```bash
python generate.py \
    --n_pts 10000 \
    --n_attr 100 \
    --n_epochs 5000 \
    --y_distrib bimodal \
    --lmbda 0.15 \
    --lr 0.01 \
    --random_state 42 \
    --output dataset.csv
```

---

## Arguments

| Argument | Default | Description |
|----------|----------|-------------|
| `--n_pts` | 10000 | Number of data points |
| `--n_attr` | 100 | Number of attributes |
| `--n_epochs` | 5000 | Optimization epochs |
| `--y_distrib` | bimodal | Target distribution (`bimodal`, `normal`, `lognormal`, `uniform`) |
| `--lmbda` | 0.15 | Baseline penalization strength |
| `--lr` | 0.01 | Learning rate for feature optimization |
| `--random_state` | 42 | Random seed |
| `--output` | dataset.csv | Output CSV file name |

---

## Expected Behavior

After generation:

- Linear Regression → low $ R^2 $
- KNN / nonlinear models → high $ R^2 $

This confirms that the dataset contains nonlinear structure and that the baseline penalization is effective.

---

## Architecture

The underlying nonlinear process is a randomly initialized neural network:

- 2-layer encoder (ReLU)
- 3-layer decoder
- Model parameters frozen (learning rate = 0)
- Only the feature matrix $ X $ is optimized

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{giobergia2026synthetic,
  title={Differentiable Synthetic Dataset Generation for Non-Trivial Regression Tasks},
  author={Giobergia, Flavio and Savelli, Claudio},
  booktitle={DARLI-AP 2026},
  year={2026}
}
```
