"""Mean squared error and softmax cross-entropy, with reverse-mode gradients.

What problem is being solved?
    Turn a network output into a scalar training objective, and return
    dL/d(output) so that affine layers can continue backpropagation.

What assumptions are required?
    MSE compares arrays of equal shape. Softmax cross-entropy takes integer
    class labels in ``{0, ..., C-1}``. Reduction is a mean over the batch
    (and, for MSE, over all remaining elements), matching PyTorch defaults.

Why was this method chosen?
    These two losses cover regression and multiclass classification on the
    synthetic DGPs in this laboratory. Softmax is fused with cross-entropy
    so the Jacobian is (p - y)/N rather than a separately differentiated
    softmax.

What alternative method could have been used?
    Sum reduction; sparse vs dense labels; sigmoid binary cross-entropy;
    a temperature in the softmax.

What can go wrong?
    Naive ``exp`` without a max-shift overflows. Ill-scaled logits make
    probabilities underflow to 0/1, so finite-difference checks become
    uninformative. Class imbalance makes unweighted CE a majority-class fit.

How is correctness independently checked?
    MSE gradient matches 2(yhat-y)/N. Softmax-CE gradient matches
    (softmax(z) - onehot)/N. Both are compared to finite differences on
    a tiny well-scaled example.

What can legitimately be concluded?
    The implemented gradients are the derivatives of the implemented scalars,
    under the stated reduction.

What cannot be concluded?
    That a low training loss is a good predictor of out-of-sample risk, or
    that cross-entropy is a proper scoring rule demonstration beyond the
    algebra used here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mse(yhat: ArrayLike, y: ArrayLike) -> float:
    """Mean of squared residuals over every element."""
    a = np.asarray(yhat, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"yhat shape {a.shape} does not match y shape {b.shape}")
    if a.size == 0:
        raise ValueError("yhat must be non-empty")
    return float(np.mean((a - b) ** 2))


def mse_grad(yhat: ArrayLike, y: ArrayLike) -> NDArray[np.floating]:
    """dL/dyhat for L = mean((yhat-y)^2), i.e. 2(yhat-y)/N."""
    a = np.asarray(yhat, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"yhat shape {a.shape} does not match y shape {b.shape}")
    if a.size == 0:
        raise ValueError("yhat must be non-empty")
    return 2.0 * (a - b) / float(a.size)


def log_softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.floating]:
    """Stable log-softmax: subtract the max before exp."""
    z = np.asarray(logits, dtype=float)
    zmax = np.max(z, axis=axis, keepdims=True)
    shifted = z - zmax
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.floating]:
    """Stable softmax. Not used as a separate layer in the fused CE loss."""
    z = np.asarray(logits, dtype=float)
    zmax = np.max(z, axis=axis, keepdims=True)
    expz = np.exp(z - zmax)
    return expz / np.sum(expz, axis=axis, keepdims=True)


def naive_softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.floating]:
    """Softmax without the max-shift. Overflows for large logits. Teaching only."""
    z = np.asarray(logits, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        expz = np.exp(z)
        denom = np.sum(expz, axis=axis, keepdims=True)
        return expz / denom


def _one_hot(labels: NDArray[np.integer], n_classes: int) -> NDArray[np.floating]:
    n = labels.shape[0]
    out = np.zeros((n, n_classes), dtype=float)
    out[np.arange(n), labels] = 1.0
    return out


def softmax_cross_entropy(logits: ArrayLike, labels: ArrayLike, class_weight: ArrayLike | None = None) -> float:
    """Mean fused softmax cross-entropy.

    If ``class_weight`` is given, it must have length C. The scalar is
    ``sum_i w_{y_i} ce_i / sum_i w_{y_i}``, the weighted mean used by
    ``torch.nn.CrossEntropyLoss`` with a class-weight vector and
    ``reduction='mean'``. Unweighted CE divides by the batch size N.
    """
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    y = np.asarray(labels, dtype=int).reshape(-1)
    n, c = z.shape
    if y.size != n:
        raise ValueError("labels must have length N")
    if np.any(y < 0) or np.any(y >= c):
        raise ValueError("labels must lie in {0, ..., C-1}")
    log_p = log_softmax(z, axis=1)
    per = -log_p[np.arange(n), y]
    if class_weight is None:
        return float(np.mean(per))
    w = np.asarray(class_weight, dtype=float).reshape(-1)
    if w.size != c:
        raise ValueError("class_weight must have length C")
    if np.any(w < 0):
        raise ValueError("class_weight must be non-negative")
    sample_w = w[y]
    total = float(sample_w.sum())
    if total <= 0.0:
        raise ValueError("sum of class weights on the batch must be positive")
    return float(np.dot(sample_w, per) / total)


def softmax_cross_entropy_grad(
    logits: ArrayLike,
    labels: ArrayLike,
    class_weight: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """dL/dlogits for mean softmax CE: (p - onehot)/N, or the weighted analogue.

    With class weights w, L = sum_i w_{y_i} ce_i / sum_j w_{y_j} and
    dL/dz_i = (w_{y_i} / sum_j w_{y_j}) (p_i - e_{y_i}).
    """
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    y = np.asarray(labels, dtype=int).reshape(-1)
    n, c = z.shape
    if y.size != n:
        raise ValueError("labels must have length N")
    if np.any(y < 0) or np.any(y >= c):
        raise ValueError("labels must lie in {0, ..., C-1}")
    p = softmax(z, axis=1)
    onehot = _one_hot(y, c)
    delta = p - onehot
    if class_weight is None:
        return delta / float(n)
    w = np.asarray(class_weight, dtype=float).reshape(-1)
    if w.size != c:
        raise ValueError("class_weight must have length C")
    if np.any(w < 0):
        raise ValueError("class_weight must be non-negative")
    sample_w = w[y]
    total = float(sample_w.sum())
    if total <= 0.0:
        raise ValueError("sum of class weights on the batch must be positive")
    return delta * (sample_w / total)[:, None]
