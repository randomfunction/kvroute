# analysis/

Python-side analysis and plotting of harness output (design phase 1c).

Phase-1 `kvroute bench` prints a comparison table to stdout. The next step
(tracked in the design doc) is to have the harness also emit per-request JSONL,
and to plot TTFT distributions and cache-hit curves here with pandas +
matplotlib, and to drive λ sweeps through the PyO3 bindings (`kvroute-py`).

Nothing to run yet — this directory is a placeholder so the layout matches the
design doc's workspace tree.
