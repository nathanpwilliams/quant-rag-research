# Labeling and cross-validation for financial ML: triple-barrier, purging, embargo

Personal reference notes (Nathan), distilled from the ideas in López de Prado,
*Advances in Financial Machine Learning*, chapters 3 and 7. Written in my own
words for AlphaPulse; nothing here is quoted from the book. Purpose: pin down the
mechanics and vocabulary I rely on for labeling and evaluation.

## 1. Why fixed-horizon labels are a poor default

The obvious way to label a price series for supervised learning is to pick a
horizon of `h` bars and label each observation with the sign of the return over
the next `h` bars, possibly after a fixed threshold. This has two defects.
Volatility is not constant, so a fixed threshold means "large move" in a calm
regime and "noise" in a turbulent one. And a fixed horizon ignores the path: a
real position with a stop-loss is closed when the stop is hit, not at an
arbitrary calendar endpoint, so an endpoint-only label describes a trade
nobody would run.

## 2. Triple-barrier labeling

The triple-barrier method labels each observation by simulating a trade opened
at that bar and recording which of three barriers the subsequent price path
touches first.

- **Upper barrier** (profit-taking). Entry price plus a multiple of a current
  volatility estimate, for example an exponentially weighted standard deviation
  of recent returns.
- **Lower barrier** (stop-loss). Entry price minus a multiple of the same
  volatility estimate. The multiples can be asymmetric.
- **Vertical barrier** (time limit). A maximum holding period. If neither
  horizontal barrier is reached before it, the simulated trade expires.

Label assignment:

- Upper barrier touched first: label `+1`. The position would have reached its
  profit target.
- Lower barrier touched first: label `-1`. The position would have been stopped
  out.
- Vertical barrier touched first: either label `0` (a genuine "nothing
  happened" class) or the sign of the return at expiry, depending on whether a
  three-class or two-class target is wanted.

Because the horizontal barriers scale with volatility, a label means the same
thing in every regime: "a move of roughly `k` sigma within the holding window."
Fixing the side from a primary model and applying the barriers gives a binary
"did this bet pay off?" label, which is the meta-labeling setup.

Practical points. The barrier multiples, the volatility lookback, and the
maximum holding period are hyperparameters and belong in config. Always inspect
the label distribution: if nearly every observation expires on the vertical
barrier, the horizontal barriers are too wide for the chosen horizon.

The detail that matters for evaluation: every label now carries two timestamps,
`t0` (the bar at which the observation is made) and `t1` (the bar at which the
first barrier is touched). The interval `[t0, t1]` is the label's information
window. Everything in the next sections follows from the fact that these
windows overlap.

## 3. Why ordinary k-fold leaks on this data

Standard k-fold cross-validation assumes samples are independent and
identically distributed. Triple-barrier labels violate that in a specific way:
consecutive observations have overlapping label windows. Observation `i`'s
label is determined by prices up to `t1(i)`, and observation `i+1`'s window
opens before `t1(i)` closes. If `i` lands in the test fold and `i+1` in the
training fold, the model is trained on price action that also decides the test
label. The score looks good and does not survive out of sample. Shuffling makes
it worse by scattering near-duplicate adjacent samples across train and test.

## 4. Purging

Purged k-fold keeps folds contiguous in time and, for each test fold spanning
`[T_start, T_end]`, removes from the training set every observation whose label
window overlaps that span. Concretely, a training sample is purged if
`t0 <= T_end` and `t1 >= T_start`. This mostly bites on samples just before the
test fold, whose windows reach forward into it, and on any sample whose window
straddles the boundary. Purging removes the direct channel: no training label
is computed from prices inside the test period.

## 5. Embargo

Purging closes the label-window leak but not a second, subtler channel.
Features are built from lookback windows (a 20-bar volatility, a moving
average, a rolling z-score), so a training sample that starts a few bars after
`T_end` has features computed partly from test-period prices, and its label is
serially correlated with the final test labels. The embargo adds a gap: after
each test fold, drop training samples for an additional `h` bars immediately
following `T_end`, where `h` is on the order of the longest feature lookback. The
embargo is only needed after the test fold. Samples before it cannot see future
prices through their features, and purging already handled their labels.

## 6. Checklist for AlphaPulse

- Store `t1` alongside every label. The cross-validation code needs it, and a
  missing or wrong `t1` silently turns purging into a no-op.
- Use purged k-fold with an embargo for model selection. Report final numbers
  from a walk-forward split, never from the folds used for tuning.
- Sanity check: the purged score should be noticeably lower than the naive
  k-fold score on the same data. If it is not, suspect the `t1` column or the
  feature lookbacks before believing the result.
- Combinatorial purged cross-validation extends the same idea to many backtest
  paths; revisit once the single-path evaluation is trustworthy.
