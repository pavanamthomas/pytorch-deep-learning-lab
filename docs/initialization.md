# Initialisation

The estimand is the law of hidden pre-activations at step 0, and the
symmetry of hidden units under SGD, on a Gaussian input \(x\sim N(0,I)\).
That is a probe of the parameter draw, not a training result on data.

## Zero

If \(W=0\) and \(b=0\), every unit in a layer computes the zero map.
If two units start equal and receive the same incoming gradient formula,
SGD keeps them equal. The network cannot break that permutation symmetry
by itself. `run_init_study` records identical hidden units.

## Large Gaussian

A large scale on \(W\) drives sigmoid/tanh into saturation (\(\sigma'\approx 0\))
or, with unbounded maps, produces huge products of Jacobians. Either
way the useful signal in \(\partial L/\partial W_{\mathrm{early}}\) dies
or explodes. The laboratory uses scale 8 as a visible failure, not as a
tuned hyperparameter.

## Xavier / Glorot

Variance \(2/(d_{\mathrm{in}}+d_{\mathrm{out}})\) is the tanh-oriented
choice that tries to keep Var(pre-activation) of order one when fan-in
and fan-out are comparable and the activation is odd and approximately
linear near 0. It is not a theorem for ReLU.

## He

Variance \(2/d_{\mathrm{in}}\) compensates for ReLU dropping half of
the mass. It is the default I use for ReLU MLPs in this repository.

None of these draws identifies a good architecture. They change the
starting Jacobian. `docs/failure_modes.md` records what still fails
after a "correct" scheme.
