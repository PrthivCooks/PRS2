# Position 2 — Simulation-Informed AI and Multimodal EZ Localisation

**Candidate:** Prithivraj S  
**Contact:** theprithivrajs@gmail.com

## 1. Executive summary

I treat this as a subject-grouped, highly imbalanced node-ranking task. All model, missing-modality and fusion decisions are frozen on train/validation before one final test reporting batch. The final model uses one transparent message-passing architecture and whichever missing-B strategy/fusion family validation selected.

**Final test:** AUPRC **0.729** versus prevalence **0.072**, AUROC **0.942**, mean Top-k Dice **0.669**. Frozen missing strategy: **A_fallback**; base: **ensemble10**; fusion: **stacked_lr**.

## 2. Data and preprocessing

140 subjects × 68 nodes; 35 subjects lack B. A uses train-median imputation, missingness indicators and train standardisation. B imputer/scaler is fit only on B-present train subjects. Restricted intervention/outcome fields are never model inputs. A strong non-graph audit compared balanced LR, ElasticNet LR, HistGradientBoosting and MLP before retaining the parsimonious baseline.

## 3. Main results

| experiment                                             |   auprc |   auroc |   prevalence |   topk_dice |   auprc_sd |   auroc_sd |   topk_dice_sd |   prevalence_baseline |
|:-------------------------------------------------------|--------:|--------:|-------------:|------------:|-----------:|-----------:|---------------:|----------------------:|
| Non-graph A+B baseline                                 |  0.463  |  0.8839 |        0.072 |      0.4795 |   —      |   —      |       —      |                 0.072 |
| Graph A only                                           |  0.4203 |  0.8735 |        0.072 |      0.402  |   —      |   —      |       —      |                 0.072 |
| Graph B only (all test; missing B handled)             |  0.3835 |  0.8311 |        0.072 |      0.386  |   —      |   —      |       —      |                 0.072 |
| Graph A+B chosen learned base (A_fallback, ensemble10) |  0.5369 |  0.9071 |        0.072 |      0.5618 |   —      |   —      |       —      |                 0.072 |
| Graph real — 10 training-resample runs                 |  0.5059 |  0.9004 |        0.072 |      0.5465 |     0.0145 |     0.0032 |         0.0183 |                 0.072 |
| Graph shuffled — 10 training-resample runs             |  0.4409 |  0.8692 |        0.072 |      0.466  |     0.013  |     0.0088 |         0.0182 |                 0.072 |
| Graph identity — 10 training-resample runs             |  0.454  |  0.8786 |        0.072 |      0.4742 |     0.0103 |     0.004  |         0.0124 |                 0.072 |
| Simulation alone                                       |  0.4048 |  0.7726 |        0.072 |      0.4563 |   —      |   —      |       —      |                 0.072 |
| Naive raw 50/50 average                                |  0.6797 |  0.9335 |        0.072 |      0.6282 |   —      |   —      |       —      |                 0.072 |
| Frozen convex fusion                                   |  0.6884 |  0.9346 |        0.072 |      0.6006 |   —      |   —      |       —      |                 0.072 |
| Frozen confidence-power fusion                         |  0.7099 |  0.9387 |        0.072 |      0.6545 |   —      |   —      |       —      |                 0.072 |
| Frozen train-OOF logistic stacker                      |  0.7286 |  0.9419 |        0.072 |      0.6685 |   —      |   —      |       —      |                 0.072 |
| FINAL validation-selected fusion (stacked_lr)          |  0.7286 |  0.9419 |        0.072 |      0.6685 |   —      |   —      |       —      |                 0.072 |

Mandatory B-present/B-absent reporting for the chosen learned base:

| experiment                         |   n_subjects |   auprc |   auroc |   prevalence |   topk_dice |
|:-----------------------------------|-------------:|--------:|--------:|-------------:|------------:|
| Chosen A+B graph model — B-present |           21 |  0.5638 |  0.9189 |       0.0728 |      0.6045 |
| Chosen A+B graph model — B-absent  |            7 |  0.4782 |  0.8671 |       0.0693 |      0.4337 |

## 4. Graph evidence and ablations

The model uses `[X,A_NX]` with symmetric normalisation. Real/shuffled/identity are repeated over 10 seeded subject-bootstrap training resamples, with genuine SGD random-state runs as a separate stochasticity check. Real topology is directionally stronger, but test subject-bootstrap CIs for real-minus-identity and real-minus-shuffled include zero, so I do **not** claim definitive graph superiority.

Bonus graph robustness includes zero-message, topology-fixed weight randomisation, binarisation, 0/1/2/3-hop propagation, and edge-threshold sparsification:

|   retain_fraction |   mean_edges_retained_pct |   auprc_mean |   auprc_sd |   auroc_mean |   topk_dice_mean |
|------------------:|--------------------------:|-------------:|-----------:|-------------:|-----------------:|
|              0.25 |                        25 |       0.3942 |     0.0134 |       0.843  |           0.4404 |
|              0.5  |                        50 |       0.4531 |     0.011  |       0.8545 |           0.4426 |
|              0.75 |                        75 |       0.4584 |     0.0122 |       0.8692 |           0.4692 |
|              1    |                       100 |       0.4556 |     0.0127 |       0.8731 |           0.4689 |

Graph error analysis identifies **22 fixed** and **11 harmed** abnormal top-k nodes; 11 subjects improve, 3 worsen and 14 are unchanged. Pattern variables are reported descriptively, not turned into post-hoc mechanistic claims.

## 5. Missing modality B

Three strategies are compared head-to-head on validation; the predeclared 0.005 AUPRC parsimony rule prevents tiny fluctuations from replacing the current strategy:

| strategy       |   auprc |   auroc |   prevalence |   topk_dice |   B_present_auprc |   B_present_dice |   B_absent_auprc |   B_absent_dice |
|:---------------|--------:|--------:|-------------:|------------:|------------------:|-----------------:|-----------------:|----------------:|
| A_fallback     |  0.4815 |  0.8835 |       0.0714 |      0.4921 |            0.5245 |           0.5623 |           0.4156 |          0.3167 |
| zero_plus_flag |  0.4682 |  0.8807 |       0.0714 |      0.4849 |            0.5245 |           0.5623 |           0.4032 |          0.2917 |
| zero_no_flag   |  0.451  |  0.8721 |       0.0714 |      0.4668 |            0.5086 |           0.5268 |           0.4175 |          0.3167 |

Test results for all strategies are reporting-only and never used to choose the strategy:

| strategy       | subgroup   |   n_subjects |   auprc |   auroc |   prevalence |   topk_dice |
|:---------------|:-----------|-------------:|--------:|--------:|-------------:|------------:|
| zero_plus_flag | all        |           28 |  0.5209 |  0.9056 |       0.072  |      0.5567 |
| zero_plus_flag | B-present  |           21 |  0.5589 |  0.9189 |       0.0728 |      0.5966 |
| zero_plus_flag | B-absent   |            7 |  0.4264 |  0.8629 |       0.0693 |      0.4371 |
| zero_no_flag   | all        |           28 |  0.497  |  0.8967 |       0.072  |      0.5728 |
| zero_no_flag   | B-present  |           21 |  0.5374 |  0.9121 |       0.0728 |      0.6112 |
| zero_no_flag   | B-absent   |            7 |  0.4475 |  0.8709 |       0.0693 |      0.4575 |
| A_fallback     | all        |           28 |  0.5328 |  0.9065 |       0.072  |      0.5559 |
| A_fallback     | B-present  |           21 |  0.5589 |  0.9189 |       0.0728 |      0.5966 |
| A_fallback     | B-absent   |            7 |  0.4945 |  0.8648 |       0.0693 |      0.4337 |

## 6. Simulation and fusion

Simulation is evaluated standalone before fusion. Concordance is non-redundant (pooled model-simulation Spearman 0.277; mean subject Spearman 0.343; mean top-k Jaccard 0.267). The most discordant validation case is sub-075. Simulation confidence predicts simulation Dice (Spearman 0.701, p=3.22e-05), motivating confidence-aware fusion.

Fusion comparison includes model alone, simulation alone, naive average, validation-tuned convex fusion, confidence-power fusion, and a **learned logistic stacker trained on train-only subject-grouped OOF base predictions whose base construction mirrors the validation-selected `ensemble10` deployment stream**. The stacker must beat the best justified parametric fusion by at least 0.005 validation AUPRC without >0.02 Dice loss before added complexity is accepted. Validation selects `stacked_lr`; stacker train-grouped-CV details are saved separately.

| family           | scale          |   auprc |   auroc |   prevalence |   topk_dice |   w_model |   alpha |   gamma |   stack_C |
|:-----------------|:---------------|--------:|--------:|-------------:|------------:|----------:|--------:|--------:|----------:|
| stacked_lr       | train_oof_meta |  0.7107 |  0.9377 |       0.0714 |      0.6701 |    —    |   —   |     — |       0.1 |
| confidence_power | val_minmax     |  0.6952 |  0.9347 |       0.0714 |      0.6701 |    —    |     0.7 |       1 |     —   |
| confidence_power | raw            |  0.695  |  0.9347 |       0.0714 |      0.6701 |    —    |     0.7 |       1 |     —   |
| convex           | val_minmax     |  0.66   |  0.9232 |       0.0714 |      0.6679 |      0.55 |   —   |     — |     —   |
| convex           | raw            |  0.6599 |  0.9232 |       0.0714 |      0.6679 |      0.55 |   —   |     — |     —   |
| naive_50_50      | val_minmax     |  0.6586 |  0.9222 |       0.0714 |      0.6588 |    —    |   —   |     — |     —   |
| naive_50_50      | raw            |  0.6585 |  0.9222 |       0.0714 |      0.6588 |    —    |   —   |     — |     —   |

## 7. Calibration, uncertainty, outcome and limitations

Grouped OOF calibration is diagnostic only; no external calibration cohort exists. Per-node uncertainty is ensemble spread, while inference is clustered by subject. The outcome bonus is observational: Seizure-free mean resection-overlap Dice 0.367 vs 0.317; rank-biserial +0.219, Mann-Whitney p=0.387, bootstrap mean-difference CI [-0.075,+0.167].

Limitations: only 28 test subjects; one atlas/order; B absent in 25%; Top-k uses oracle true-k for evaluation only; simulation internals are a supplied black box; no external/prospective cohort; site-stratified counts are small.

## 8. What I would do with more time

1. Nested subject-level CV over train+validation for model-selection variance.
2. External/prospective site-shift validation.
3. Validate calibrated probability/abstention policies.
4. Integrate actual TVB/VEP outputs and compare with SEEG/outcomes prospectively.
5. Pre-register any deeper graph or learned fusion refinement on a fresh cohort.
