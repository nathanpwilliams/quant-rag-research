# AlphaPulse notes: backtest setup, walk-forward split, transaction cost assumptions

<!-- Own-notes corpus document. Fill in from project knowledge; this enables corpus-vs-own-practice comparison questions. -->

## 1. Data and universe

<!-- Which venue and instruments, what bar frequency, and what date range? -->
<!-- Where does the data come from and how is it cleaned (gaps, outliers, funding events)? -->
<!-- Are there any known data-quality problems in the range used? -->

## 2. Walk-forward split

<!-- How are train / validation / test windows sized and rolled forward? -->
<!-- Is there a gap (embargo) between train and test windows, and how wide? -->
<!-- How many walk-forward folds are there, and are results aggregated or reported per fold? -->

## 3. Transaction cost assumptions

<!-- What fee schedule is assumed (maker/taker bps) and does it match the venue's actual tiers? -->
<!-- Is slippage modeled, and if so as a fixed bps, spread-based, or volume-based estimate? -->
<!-- Are costs applied during training, evaluation, or both, and why? -->

## 4. Execution model

<!-- Are fills assumed at close, next open, or mid-price plus half-spread? -->
<!-- How is partial fill / latency handled in simulation, if at all? -->
<!-- What position sizing rule is used and is leverage allowed? -->

## 5. Metrics reported

<!-- Which metrics are headline (Sharpe, max drawdown, turnover, hit rate) and over what period? -->
<!-- How is the benchmark defined (buy-and-hold, flat, a simple rule)? -->
<!-- Are confidence intervals or multiple seeds reported? -->

## 6. Known caveats

<!-- Which assumptions would most change the result if wrong? -->
<!-- What has already been observed to differ between backtest and testnet? -->
