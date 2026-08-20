# Optimisation

The maps in this laboratory are trained with first-order methods on a
minibatch scalar. The estimand of an optimiser comparison is the path
\(\theta_t\) on a stated objective, not a ranking of methods for an
application.

## SGD

\[
\theta \leftarrow \theta - \eta\, g, \qquad
g = \nabla_\theta L_{\mathrm{batch}}(+ \lambda\theta \text{ if coupled L2}).
\]

That is `dllab.numpy_net.sgd.sgd_step` and `torch.optim.SGD` with
momentum 0.

## Momentum

Velocity accumulates \(g\). On an ill-conditioned quadratic
\(f(w)=\tfrac12(w_0^2/a + a w_1^2)\), SGD crawls along the long valley;
momentum cuts across more. The figure from `scripts/run_all.py` is that
path. It is two parameters. It is not ImageNet.

## Adam and AdamW

Adam scales each coordinate by an exponential moving RMS of gradients.
L2 regularisation *inside* that update is coupled: the decay term
\(\lambda\theta\) is treated as a gradient and then divided by
\(\sqrt{v}+\epsilon\). Coordinates with large historical \(|g|\) are
decayed less.

AdamW applies a decay \(\theta\leftarrow \theta-\eta\lambda\theta\)
outside the adaptive scaling (decoupled weight decay). On the same
quadratic with \(\lambda>0\), the two final \(\theta\) disagree, and
the ratio of coordinate-wise shrinkage disagrees. That is the point of
`run_adam_vs_adamw`.

I do not claim AdamW is uniformly better. I claim the updates are
different objects that share a name in older APIs.

## Gradient clipping

`clip_grad_norm` rescales the *concatenation* of all gradients so that
its Euclidean norm is at most `max_norm`. It is not a per-parameter
clip unless the parameter is the only one. Tests lock the (3,4)→(0.6,0.8)
example.

Clipping bounds the step. It does not fix an exploding architecture.
