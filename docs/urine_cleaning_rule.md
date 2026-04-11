# Urine Raw Cleaning Rule

## Why This Step Exists

`Urine` is not treated like an ordinary lab value.

In the raw ICU data, urine values are charted repeatedly over time and are later used for AKI labeling. Before converting the raw table into an hourly wide table, urine rows need a small amount of event-level cleanup.

The main issue observed in this dataset is:

- exact same `stay_id + charttime` having multiple urine values
- some duplicate groups contain extreme conflicts such as `500` and `19990`

If these conflicting rows are summed directly inside the same hour, hourly urine output can be inflated badly.

## Practical Interpretation

The observed pattern suggests that urine rows are mostly charted as output events or short-interval totals, not as a single cumulative running counter.

So:

- values from different timestamps inside the same hour may reasonably be summed
- values at the exact same timestamp should first be deduplicated

## First-Pass Cleaning Rule

This project uses a conservative first-pass rule for `Urine` only.

For each exact `stay_id + charttime` duplicate group:

1. If all values are identical:
   - keep one value
2. If values conflict:
   - keep the smallest positive value
   - if every value is zero, keep zero

## Why This Rule

This is not meant to be a perfect physiological truth.

It is a conservative data cleaning rule for exact timestamp conflicts only:

- it prevents obvious blow-ups from values like `19990`
- it avoids hard-coded urine-volume thresholds with weak justification
- it avoids summing clearly conflicting duplicate rows
- it still preserves ordinary repeated charting across different timestamps

## What This Rule Does Not Do

- it does not clip all large urine values globally
- it does not change values from different timestamps
- it does not solve every documentation ambiguity in ICU charting

## Future Refinement

Possible future improvements:

1. interval-aware urine rate cleaning using time since previous chart
2. review of clinically implausible `mL/hour` values
3. sensitivity analysis with and without urine-based AKI labels
