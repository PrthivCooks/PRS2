# IIIT-Delhi Position 2 Take-Home — Reproducible Submission

**Candidate:** Prithivraj S  
**Contact:** theprithivrajs@gmail.com

## Run
1. Put the provided `candidate_package/data` contents under `data/`, or set `DATA_P2_DIR`.
2. Install `requirements.txt`.
3. Run `python src/run_pipeline.py` from repository root.

## Stack choice
I intentionally use **NumPy/scikit-learn rather than PyTorch/PyG** because the brief explicitly permits a hand-written propagation layer, the graphs are small, and the transparent implementation makes graph controls easier to audit and defend.

## Final pipeline
- A: train-median imputation + missing indicators + train standardisation.
- B: three missing-modality strategies are compared on validation; `A_fallback` is frozen.
- Graph: one-hop symmetric normalisation with residual `[X,A_NX]`.
- Classifier: balanced logistic regression.
- Fusion candidates: naive, convex, confidence-power, and train-OOF learned logistic stacker. The stacker's train OOF `model_score` mirrors the frozen `ensemble10` base construction; it must clear a 0.005 validation-AUPRC materiality margin with no >0.02 Dice harm; `stacked_lr` is frozen by validation.
- Final test AUPRC/AUROC/Top-k Dice: `0.728639` / `0.941945` / `0.668537`.

## Assumptions/tradeoffs
- Tables aligned explicitly on `(subject_id,node_id)`; node IDs 0–67 asserted.
- Restricted resection/outcome fields are post-hoc only.
- Primary repeated graph analysis uses subject-bootstrap training resamples; genuine stochastic seeds are checked separately.
- `prob_abnormal` is a bounded score; external calibration is not established.
- Top-k true k is evaluation-only.
- Stacker meta-training uses subject-OOF base scores built with the same single/ensemble base kind selected on validation.
- Test is a final reporting batch; reporting-only optional test tables never feed back into model selection.
