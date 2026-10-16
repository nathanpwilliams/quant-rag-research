# AlphaPulse notes: failure modes and lessons learned on Kraken Futures testnet

<!-- Own-notes corpus document with zero overlap with the papers. Fill in from project knowledge; concrete incidents beat generalities. -->

## 1. Deployment setup

<!-- What ran where: process layout, data feed, order gateway, how the model was loaded? -->
<!-- How was testnet different from the backtest environment (fees, liquidity, symbols)? -->
<!-- What monitoring existed on day one? -->

## 2. Incident log

<!-- One subsection per incident: what happened, when, how it was detected, root cause, fix. -->
<!-- Which incidents were infrastructure (connectivity, auth, rate limits) versus strategy (bad signals, sizing)? -->

### 2.1 <incident title>

<!-- Symptom, detection, root cause, fix, and what would have caught it earlier. -->

## 3. Infrastructure and connectivity lessons

<!-- What broke around websockets, reconnects, clock skew, order acknowledgements? -->
<!-- What did the testnet API do that the docs did not lead me to expect? -->
<!-- What guards (kill switch, max position, heartbeat) were added as a result? -->

## 4. Strategy and model lessons

<!-- Where did live behavior diverge from backtest behavior, and what explained the gap? -->
<!-- Which features or reward terms turned out to be fragile under real fills? -->
<!-- Did anything in the state/action design cause pathological trading (churn, stuck positions)? -->

## 5. Process lessons

<!-- What should be checked before every deployment from now on? -->
<!-- What logging or replay capability would have shortened debugging? -->

## 6. What I would do differently

<!-- Top three changes, in priority order, for the next iteration. -->
