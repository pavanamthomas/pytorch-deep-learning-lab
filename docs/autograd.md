# Autograd

Reverse-mode automatic differentiation records a graph of elementary
operations and applies the chain rule from the scalar loss back to
parameters. For the maps in this laboratory, that is the same algebra
as `dllab.numpy_net`, executed by PyTorch.

I compare the two on one controlled batch with a shared initialisation
(`dllab.torch_net.compare`). The estimand is the pair
(forward tensor, parameter gradient). The DGP is a tiny Gaussian batch.
A small max-abs difference is a numerical identity, not a model-quality
claim.

Autograd conventions that matter here:

- ReLU'(0) = 0.
- `MSELoss` with mean reduction averages over every element.
- `CrossEntropyLoss` with mean reduction averages over the batch.
  With a class-weight vector it uses a weighted mean
  \(\sum_i w_{y_i}\mathrm{ce}_i / \sum_i w_{y_i}\).
- `BatchNorm1d` in train uses batch moments; in eval it uses running
  moments. Those are different functions. See `docs/failure_modes.md`.

Autograd is not a substitute for a finite-difference check when the
question is "did I implement the scalar I think I implemented?" It is
the right tool when the question is "does this NumPy backward match
PyTorch on this graph?"

GPU nondeterminism is a different issue. See `docs/reproducibility.md`.
