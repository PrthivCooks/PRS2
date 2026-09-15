# Required analysis — does graph structure genuinely contribute?

**Candidate:** Prithivraj S  
**Contact:** theprithivrajs@gmail.com

## 1. Controls
Real adjacency uses the actual weighted topology. Shuffled adjacency applies a node permutation to destroy feature-topology correspondence while preserving graph-level structure. Identity removes neighbour mixing. `[X,0]` verifies the ordinary non-graph baseline in the doubled representation. Weight-shuffling preserves binary topology but destroys edge-strength assignment; binarisation removes edge magnitudes; sparsification tests dependence on weak edges.

## 2. Repeated runs and uncertainty
The primary graph comparison uses 10 seeded subject-bootstrap training resamples and reports mean±SD. A separate SGD experiment changes genuine optimizer/shuffle random states. On test, subject-bootstrap AUPRC differences are real-identity +0.062 [-0.008,+0.143] and real-shuffled +0.061 [-0.006,+0.137]. Both CIs cross zero, so evidence is suggestive rather than conclusive.

## 3. Error analysis
Using true k only for evaluation, the canonical graph model fixes 22 abnormal nodes and harms 11; 11 subjects improve, 3 worsen, 14 are unchanged. Specific subject/node/region records plus degree, abnormal-neighbour exposure, simulation evidence and B availability are saved in `results/graph_error_nodes.csv` and summaries.

## 4. Falsification criterion
I would conclude the graph does not help if shuffled/identity matched real within repeated-run variability, signs were unstable, confidence intervals were centred at or below zero alongside inconsistent validation results, or gains were concentrated in one/two subjects with no recoveries. I observe consistent validation directionality and more fixed than harmed nodes, but wide test CIs. Therefore I reject a strong causal/topological claim and state only that topology shows **directionally consistent, not definitive** value in this sample.

## 5. Bonus graph robustness
Propagation depth, fixed-topology weight randomisation, binarised topology and edge-threshold sparsification are validation-only robustness checks. They do not reopen test selection.
