# AlphaPulse notes: MDP formulation, state/action/reward design

<!-- Own-notes corpus document. Fill in from project knowledge; keep it prose, not bullets of numbers only. -->

## 1. Scope and objective

<!-- What market, instrument(s), and horizon does the agent trade? -->
<!-- What is the agent optimizing for in one sentence: return, risk-adjusted return, PnL net of costs? -->
<!-- What is explicitly out of scope (execution, portfolio allocation, multi-asset)? -->

## 2. State space

<!-- Which features enter the observation: price/return lookbacks, order-book features, position, unrealized PnL, time-of-day? -->
<!-- How are features normalized, and over what lookback window? -->
<!-- What information is deliberately excluded to avoid lookahead? -->

## 3. Action space

<!-- Discrete (long/flat/short) or continuous (target position in [-1, 1])? Why that choice? -->
<!-- How does an action map to an order: target position, delta from current, or fixed-size trade? -->
<!-- Any action constraints: max position, cooldown, no-trade zones? -->

## 4. Reward design

<!-- What is the per-step reward: raw PnL, log return, volatility-scaled return, differential Sharpe? -->
<!-- How are transaction costs and slippage charged inside the reward? -->
<!-- What reward variants were tried and what went wrong with the rejected ones? -->

## 5. Episode structure and transitions

<!-- What defines an episode: fixed length, one trading session, until drawdown limit? -->
<!-- How is the environment stepped: bar-by-bar on historical data, or event-driven? -->
<!-- Is there any stochasticity injected (random start offsets, noise) to reduce overfitting? -->

## 6. Open questions and known weaknesses

<!-- Which parts of the formulation am I least confident in? -->
<!-- What would I change first if the agent underperforms on the testnet? -->
