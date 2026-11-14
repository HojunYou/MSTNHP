# Revision implementation plan

Status: source inspection and visualization path repair completed. Quantitative
evaluation and experiments below are proposed, not implemented or run.

## Findings that determine the first experiment

1. `easy_tpp/model/torch_model/torch_stnhp.py` already exposes `forward`,
   `compute_states_at_sample_points`, and `layer_intensity`. Reuse this model
   rather than copying neural equations into an independent evaluator.
   `n_comps` controls distances to recent event locations; record it per run.
2. `train_mstnhp/stpp_simulator/mstppg.py:BivariateHawkesLam.value`
   evaluates generating intensities given a supplied history. The caller
   must restrict that history to events strictly before the query time.
3. `simulate_mstpp_parallel.py` contains successive active assignments to
   kernel variables. The final power/exponential/mixture/sine assignment
   determines its current simulation. Names of earlier commented cases do
   not define selectable experiments. Its domain is [-1,1]^2, with area 4.
4. The existing `SeparableGaussianKernel` has Gaussian decay in both time
   and space. Do not substitute it for an exponential-in-time,
   normalized-Gaussian-in-space generating kernel merely because of its name.
   Establish the exact Biv 1--4 parameter and coordinate mapping first.
5. `aggregate_dataset.py` makes 90/5/5 splits but writes empty `args` metadata.
   The simulator returns lengths, but its current serialization omits them.
   Preserve parameters, bounds, horizon, lengths and seeds explicitly.
6. The simulator rejects nonpositive candidate intensities and retries
   sequences with too few events or a violated upper bound. Audit these
   choices before replication: do not silently change the generating model
   by clipping negative intensities or conditioning on successful runs.
7. `viz/mstnhp_manuscript.py` and related plotting scripts select a training
   batch, select one sequence, and shift some query times by 0.01. Their
   normalized/transformed displays must not become the metric input.
8. `torch_stnhp.py` uses a spatial grid with weights 4/ns^2. The evaluator
   must make domain area and any coordinate-density transformation explicit.
9. The current equal-time plotting helper assembles intervals between events;
   the model's default forward loop stops before the last event update.
   Add an explicit arbitrary-query evaluator that handles the post-last-event
   interval through T. State the initial-history convention consistently.
10. The runner selects the saved model by validation likelihood and also
    writes a final checkpoint. Use the validation-selected checkpoint for
    reporting; do not accidentally load `last` or select using test metrics.
    Several plotting scripts load with `strict=False`; the metric evaluator
    should instead check checkpoint compatibility strictly.

## First deliverable: one checkpoint, one verified generating setting

Use a saved run config, its checkpoint, held-out test sequences, and the
actual generating parameters. These inputs were excluded from the uploaded
archive. No historical numerical results can be verified without them.

Implement under `train_mstnhp/evaluation/`:

| Proposed file | Responsibility |
| --- | --- |
| `settings.py` | Explicit named generating settings, kernels, bounds, units and metadata |
| `true_intensity.py` | Reference intensities using strictly past events |
| `model_intensity.py` | Strict checkpoint load, event-state cache, arbitrary space--time queries |
| `intensity_metrics.py` | Weighted temporal and full space--time errors |
| `evaluate_intensity.py` | CLI for config, checkpoint, split, grid and outputs |
| `summarize_replicates.py` | Per-dataset aggregation and mean/SD across independent fits |

At each query time, select the correct post-event state from the same observed
test history used by the reference intensity. Build spatial distances in the
same oldest-to-newest order and with the same early-history padding as the
training implementation. Never update a state with a future event. Use eval
mode and no-grad; chunk spatial queries to control memory.

Start with explicit conditioning on the first observed event, evaluating
the subsequent interval through the known horizon T for both models. For
zero/one-event sequences, define support and reporting rather than silently
dropping them. A later full-window evaluation requires an explicit initial
intensity convention.

## Metrics and quadrature

Let lambda* be the known intensity, lambda_hat the fitted intensity, and
Lambda_k(t) = integral_S lambda_k(s,t) ds. Compute per event type:

- Temporal RMSE: sqrt(integral (Lambda_hat-Lambda*)^2 dt / window_length).
- Temporal relative L2: sqrt(integral (Lambda_hat-Lambda*)^2 dt /
  integral Lambda*^2 dt).
- Space--time RMSE: sqrt(integral integral (lambda_hat-lambda*)^2 ds dt /
  (window_length * area)).
- Space--time relative L2: sqrt(integral integral (lambda_hat-lambda*)^2 ds dt /
  integral integral lambda*^2 ds dt).

Also retain spatial errors at individual times and raw intensity snapshots
for the simultaneous-centre experiment. Compare raw intensities, not logged,
per-type-normalized or time-averaged display values. A zero reference norm
must produce a documented undefined relative error, not an arbitrary epsilon
that hides the issue; retain absolute error in that case.

Use equal-area spatial midpoint cells on a rectangle and time quadrature
split at event times, with explicit weights. Avoid exact jump times or define
their left limits consistently. Refine spatial and temporal resolution
separately on a pilot and record convergence; do not assume a 40x40 grid is
sufficient for narrow kernels. For polygon domains account for cell areas
or document and refine the boundary approximation.

Validate the metric implementation with identical arrays (zero error), a
known constant offset, and a constant intensity integrated over area 4.
Validate state timing with a short hand-built event sequence, including
queries after the last event. Confirm recent-centre ordering for J=1 and J=3.
Check the reference kernel values independently against their formulas.

Outputs: one CSV row per test sequence and type, a run-summary JSON with
config/checkpoint identification, setting parameters, seeds, split, window,
quadrature, and counts. Optionally save selected untransformed snapshots.

## Replication experiment

After the single-run evaluator is validated:

1. Generate an independently seeded train/validation/test dataset per replica.
2. Fit a fresh model and select by validation likelihood.
3. Evaluate all test sequences, not just the first batch.
4. Average sequence errors within each replica with a documented weighting.
5. Report mean and sample SD of replica-level errors. If reporting confidence
   intervals for the mean, distinguish these from SD.

Run a three-replica pilot for cost and failure diagnosis; choose the final
replication count afterward. Repeated optimizer seeds on one dataset assess
training variability, not dataset replication. Keep dataset seed and model
seed separate. For method comparisons reuse the same datasets and evaluate
paired replica-level differences. Record failures rather than discarding
unfavourable runs.

## Remaining editor requests

| Comment | Next work after the evaluator |
| --- | --- |
| 1: spatial baselines | Choose neural and parametric competitors; reuse data, splits, grids and metrics; document comparable tuning and runtime |
| 2: replicated accuracy | Complete the pipeline above, with per-type errors and variability |
| 3: simultaneous centres | Reconcile Equation (5) with actual J; test separated active hotspots at fixed times; compare J=1 and J=3 without assuming multi-distance decay guarantees recovery |
| 4: common events | Preserve all events; establish timestamp precision and common tie handling; compare temporal outcomes on the same observations, not raw spatial versus temporal likelihoods |
| 5: application holdout | Freeze chronological splits and model selection; add held-out prediction and temporal/spatial calibration, respecting timestamp resolution |

First practical milestone: a validated error table from one existing synthetic
checkpoint. Then add replication orchestration. Avoid launching a large sweep
until the generating setting, spatial scaling and evaluator agree.
