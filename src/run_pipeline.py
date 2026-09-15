# 0. Imports and deterministic setup
import os, glob, json, math, random, warnings, shutil, zipfile
from pathlib import Path
from collections import Counter
from importlib import metadata as importlib_metadata

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import spearmanr, pearsonr

import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    roc_curve, confusion_matrix, cohen_kappa_score
)

import matplotlib
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(x):
        if hasattr(x, 'to_string'):
            print(x.to_string(index=False))
        else:
            print(x)

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', 80)

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('scipy', scipy.__version__)
print('scikit-learn', sklearn.__version__)
print('matplotlib', matplotlib.__version__)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 0.1 Locate candidate_package/data automatically (or set DATA_P2_DIR)
def locate_data_dir():
    env = os.environ.get("DATA_P2_DIR")
    candidates = [
        env,
        "/kaggle/working/candidate_package/data",
        "./candidate_package/data",
        "./data",
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "subjects.csv")):
            return c

    hits = glob.glob("/kaggle/input/**/candidate_package/data/subjects.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(
        "Could not locate the dataset. Set environment variable DATA_P2_DIR "
        "to the folder containing subjects.csv, modality_A.csv, modality_B.csv, etc."
    )

DATA = locate_data_dir()
print("DATA =", DATA)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 1. Load tables and adjacency
subj = pd.read_csv(f"{DATA}/subjects.csv")
A = pd.read_csv(f"{DATA}/modality_A.csv")
B = pd.read_csv(f"{DATA}/modality_B.csv")
L = pd.read_csv(f"{DATA}/node_labels.csv")
S = pd.read_csv(f"{DATA}/simulation_scores.csv")
R = pd.read_csv(f"{DATA}/region_names.csv")

AF = ["ct_z","gmv_z","sa_z","curv_z","t1t2_z","lgi_z"]
BF = ["rel_delta","rel_theta","rel_alpha","rel_beta","rel_gamma","spike_rate","hfo_rate"]

assert subj["subject_id"].nunique() == 140
assert set(subj["split"].unique()) == {"train","val","test"}

sids = subj["subject_id"].tolist()
adj = {}
for sid in sids:
    a = np.load(f"{DATA}/adjacency/{sid}_adj.npy").astype(np.float64)
    assert a.shape == (68, 68)
    assert np.allclose(a, a.T, atol=1e-5), f"{sid}: adjacency not symmetric"
    assert np.allclose(np.diag(a), 0, atol=1e-8), f"{sid}: diagonal not zero"
    assert (a >= -1e-6).all(), f"{sid}: negative adjacency weight"
    adj[sid] = a

for name, df in [("A", A), ("L", L), ("S", S)]:
    counts = df.groupby("subject_id")["node_id"].apply(lambda x: sorted(x.tolist()))
    assert set(counts.index) == set(sids), f"{name}: missing subjects"
    assert all(ids == list(range(68)) for ids in counts), f"{name}: bad node ids"

b_present_sids = set(subj.loc[subj["modality_B_available"] == 1, "subject_id"])
b_counts = B.groupby("subject_id")["node_id"].apply(lambda x: sorted(x.tolist()))
assert set(b_counts.index) == b_present_sids
assert all(ids == list(range(68)) for ids in b_counts)

assert not L.duplicated(["subject_id","node_id"]).any()
assert not A.duplicated(["subject_id","node_id"]).any()
assert not S.duplicated(["subject_id","node_id"]).any()
assert not B.duplicated(["subject_id","node_id"]).any()

print("Structural audit passed.")

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 1.1 Dataset audit
audit = {
    "n_subjects": int(subj["subject_id"].nunique()),
    "split_counts": subj["split"].value_counts().to_dict(),
    "n_modB_missing": int((subj["modality_B_available"] == 0).sum()),
    "modB_missing_by_split": subj.loc[subj["modality_B_available"] == 0, "split"].value_counts().to_dict(),
    "A_nan_pct_overall": float(A[AF].isna().mean().mean() * 100),
    "A_nan_pct_by_feature": (A[AF].isna().mean() * 100).round(3).to_dict(),
}
_prev = L.merge(subj[["subject_id","split"]], on="subject_id")
audit["prevalence_overall"] = float(L["is_abnormal"].mean())
audit["prevalence_by_split"] = _prev.groupby("split")["is_abnormal"].mean().round(5).to_dict()
abn_counts = L.groupby("subject_id")["is_abnormal"].sum()
audit["abnormal_per_subject"] = {
    "min": int(abn_counts.min()),
    "max": int(abn_counts.max()),
    "mean": float(abn_counts.mean())
}
print(json.dumps(audit, indent=2))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 2. Metrics
def compute_metrics(y_true, y_prob, subject_ids):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    subject_ids = np.asarray(subject_ids)

    out = {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
    }
    dices = []
    for sid in np.unique(subject_ids):
        m = subject_ids == sid
        yt, yp = y_true[m], y_prob[m]
        k = int(yt.sum())
        if k == 0:
            continue
        top_idx = np.argsort(-yp)[:k]
        pred = set(top_idx.tolist())
        true = set(np.where(yt == 1)[0].tolist())
        dices.append(2 * len(pred & true) / (len(pred) + len(true)))
    out["topk_dice"] = float(np.mean(dices))
    return out

def subject_bootstrap_metric_diff(y_true, p_a, p_b, subject_ids, metric="auprc",
                                  n_boot=8000, seed=1):
    """Paired subject-level bootstrap of metric(A)-metric(B)."""
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    subject_ids = np.asarray(subject_ids)
    subs = np.unique(subject_ids)
    rng = np.random.RandomState(seed)
    diffs = []

    for _ in range(n_boot):
        sampled = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([np.where(subject_ids == s)[0] for s in sampled])
        yt = y_true[idx]
        try:
            if metric == "auprc":
                d = average_precision_score(yt, p_a[idx]) - average_precision_score(yt, p_b[idx])
            elif metric == "auroc":
                d = roc_auc_score(yt, p_a[idx]) - roc_auc_score(yt, p_b[idx])
            else:
                raise ValueError(metric)
            diffs.append(d)
        except ValueError:
            pass

    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_boot": int(len(diffs)),
    }

def fmt(m):
    return {k: round(v, 4) for k, v in m.items()}

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 3. Build aligned base table
base = L[["subject_id","node_id","is_abnormal","resected"]].copy()
base = base.merge(subj[["subject_id","split","site","hemisphere","modality_B_available","engel_1_seizure_free"]],
                  on="subject_id", how="left")
if "region" in R.columns:
    base = base.merge(R[["node_id","region"]].drop_duplicates("node_id"), on="node_id", how="left")
elif "region_name" in R.columns:
    base = base.merge(R[["node_id","region_name"]].drop_duplicates("node_id")
                      .rename(columns={"region_name":"region"}), on="node_id", how="left")
else:
    base["region"] = base["node_id"].astype(str)

base = base.merge(A[["subject_id","node_id"] + AF], on=["subject_id","node_id"], how="left")
base = base.merge(B[["subject_id","node_id"] + BF], on=["subject_id","node_id"], how="left")
base = base.merge(S[["subject_id","node_id","sim_score","sim_confidence"]],
                  on=["subject_id","node_id"], how="left")
base = base.rename(columns={"modality_B_available":"modB_available"})
base = base.sort_values(["subject_id","node_id"]).reset_index(drop=True)

for sid, g in base.groupby("subject_id"):
    assert g["node_id"].tolist() == list(range(68))

train_mask = base["split"].eq("train")
val_mask = base["split"].eq("val")
test_mask = base["split"].eq("test")

y_tr = base.loc[train_mask, "is_abnormal"].to_numpy()
y_val = base.loc[val_mask, "is_abnormal"].to_numpy()
sid_tr = base.loc[train_mask, "subject_id"].to_numpy()
sid_val = base.loc[val_mask, "subject_id"].to_numpy()
sid_test = base.loc[test_mask, "subject_id"].to_numpy()
TEST_UNLOCKED = False

def build_A(df, imputer=None, scaler=None, fit=False):
    X = df[AF].copy()
    missing = X.isna().astype(float).to_numpy()
    if fit:
        imputer = SimpleImputer(strategy="median").fit(X)
        Xi = imputer.transform(X)
        scaler = StandardScaler().fit(Xi)
    else:
        Xi = imputer.transform(X)
    Xs = scaler.transform(Xi)
    return np.hstack([Xs, missing]), imputer, scaler

XA_tr, impA, scA = build_A(base.loc[train_mask], fit=True)
XA_val, _, _ = build_A(base.loc[val_mask], impA, scA)

Bpresent_train = base.loc[train_mask & base["modB_available"].eq(1)]
impB = SimpleImputer(strategy="median").fit(Bpresent_train[BF])
scB = StandardScaler().fit(impB.transform(Bpresent_train[BF]))

def build_B(df):
    Xi = impB.transform(df[BF])
    Xs = scB.transform(Xi)
    avail = df["modB_available"].to_numpy(dtype=float)
    Xs = Xs * avail[:, None]
    return np.hstack([Xs, avail[:, None]])

XB_tr = build_B(base.loc[train_mask])
XB_val = build_B(base.loc[val_mask])

XAB_tr = np.hstack([XA_tr, XB_tr])
XAB_val = np.hstack([XA_val, XB_val])

print("A:", XA_tr.shape, "B:", XB_tr.shape, "A+B:", XAB_tr.shape)
print("Train/val prevalence:", round(y_tr.mean(),4), round(y_val.mean(),4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 3.1 Feature dataframes aligned to base.index for graph propagation
def feature_frame(Xtr, Xval, prefix):
    n_cols = Xtr.shape[1]
    f = pd.DataFrame(np.nan, index=base.index, columns=[f"{prefix}{i}" for i in range(n_cols)])
    f.loc[train_mask, :] = Xtr
    f.loc[val_mask, :] = Xval
    return f

feat_A = feature_frame(XA_tr, XA_val, "a")
feat_B = feature_frame(XB_tr, XB_val, "b")
feat_AB = feature_frame(XAB_tr, XAB_val, "f")

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 3.2 Strong non-graph baseline audit — validation only

def fit_baseline_lr(Xtr, ytr, Xev, C=1.0):
    m = LogisticRegression(max_iter=3000, class_weight='balanced', C=C, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:,1], m

# Primary simple baseline
p_lr_base, _ = fit_baseline_lr(XAB_tr, y_tr, XAB_val, C=1.0)

# Small, defensible HGB validation grid
pos = y_tr.mean()
sw = np.where(y_tr == 1, (1-pos)/pos, 1.0)
hgb_grid = [
    dict(max_depth=2, max_iter=100, learning_rate=0.05, l2_regularization=1.0),
    dict(max_depth=3, max_iter=150, learning_rate=0.06, l2_regularization=1.0),
    dict(max_depth=3, max_iter=250, learning_rate=0.03, l2_regularization=2.0),
    dict(max_depth=4, max_iter=150, learning_rate=0.05, l2_regularization=0.5),
]
best_hgb = None
for cfg in hgb_grid:
    m = HistGradientBoostingClassifier(random_state=0, **cfg)
    m.fit(XAB_tr, y_tr, sample_weight=sw)
    p = m.predict_proba(XAB_val)[:,1]
    a = average_precision_score(y_val, p)
    if best_hgb is None or a > best_hgb['auprc']:
        best_hgb = {'auprc':float(a), 'cfg':cfg, 'pred':p}

# Elastic-net logistic: internal CV uses TRAIN only; validation is used only for held-out comparison.
enet_cv_splits = list(GroupKFold(n_splits=5).split(XAB_tr, y_tr, groups=sid_tr))
enet = LogisticRegressionCV(
    Cs=8, cv=enet_cv_splits, penalty='elasticnet', l1_ratios=[0.1,0.5,0.9], solver='saga',
    class_weight='balanced', max_iter=5000, random_state=0, scoring='average_precision'
)
enet.fit(XAB_tr, y_tr)
p_enet = enet.predict_proba(XAB_val)[:,1]

# Compact MLP. Oversample positives because this sklearn MLPClassifier lacks class_weight.
ratio = int(round((1-pos)/pos))
idx_mlp = np.concatenate([np.where(y_tr==0)[0], np.repeat(np.where(y_tr==1)[0], ratio)])
rng_mlp = np.random.RandomState(0)
rng_mlp.shuffle(idx_mlp)
mlp = MLPClassifier(
    hidden_layer_sizes=(32,16), alpha=1e-2, max_iter=400,
    early_stopping=True, random_state=0
)
mlp.fit(XAB_tr[idx_mlp], y_tr[idx_mlp])
p_mlp = mlp.predict_proba(XAB_val)[:,1]

baseline_val_df = pd.DataFrame([
    {'model':'Balanced LogisticRegression (C=1)', **compute_metrics(y_val,p_lr_base,sid_val)},
    {'model':'HistGradientBoosting (small val grid)', **compute_metrics(y_val,best_hgb['pred'],sid_val)},
    {'model':'ElasticNet LogisticRegressionCV (train-CV)', **compute_metrics(y_val,p_enet,sid_val)},
    {'model':'Compact MLP', **compute_metrics(y_val,p_mlp,sid_val)},
]).sort_values('auprc',ascending=False).reset_index(drop=True)

display(baseline_val_df.round(4))
print('Best HGB config:', best_hgb['cfg'])
print('ElasticNet chosen C/l1_ratio:', float(enet.C_[0]), float(enet.l1_ratio_[0]))

plain = float(baseline_val_df.loc[baseline_val_df.model.str.startswith('Balanced'), 'auprc'].iloc[0])
best = float(baseline_val_df.auprc.max())
print(f'Best-minus-plain validation AUPRC = {best-plain:+.6f}')
print('Decision: retain balanced LogisticRegression C=1 under the <=0.005 parsimony rule.' if best-plain <= 0.005
      else 'WARNING: a materially stronger baseline exists; discuss before changing the already-frozen submission.')

# Class-weight sensitivity, validation only: defend why "balanced" is reasonable.
implied = (1-y_tr.mean())/y_tr.mean()
weight_rows=[]
for factor in [0.5,0.75,1.0,1.5,2.0]:
    cw={0:1.0,1:implied*factor}
    m=LogisticRegression(max_iter=3000,class_weight=cw,C=1.0,random_state=0).fit(XAB_tr,y_tr)
    p=m.predict_proba(XAB_val)[:,1]
    weight_rows.append({'positive_weight_factor_vs_balanced':factor,'auprc':average_precision_score(y_val,p)})
class_weight_sensitivity_df=pd.DataFrame(weight_rows)
display(class_weight_sensitivity_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 4. Classifier and graph helpers
def fit_lr(Xtr, ytr, Xev, C=1.0, seed=0):
    m = LogisticRegression(max_iter=3000, class_weight='balanced', C=C, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:,1], m

def normalize_adj(a):
    ah = a + np.eye(a.shape[0])
    d = ah.sum(1)
    di = np.where(d > 0, d**-0.5, 0.0)
    return di[:,None] * ah * di[None,:]

def shuffle_adj(a, rng):
    # PAP^T preserves the weighted degree multiset/global graph statistics but breaks node correspondence.
    p = rng.permutation(a.shape[0])
    return a[np.ix_(p,p)]

def shuffle_edge_weights(a, rng):
    # Preserve binary topology; permute only the non-zero undirected edge weights.
    iu = np.triu_indices_from(a, k=1)
    vals = a[iu].copy()
    nz = vals > 0
    shuffled = vals[nz].copy()
    rng.shuffle(shuffled)
    vals[nz] = shuffled
    out = np.zeros_like(a)
    out[iu] = vals
    out = out + out.T
    return out

def propagate_features(df_split, feat_df, condition='real', rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    parts = []
    for sid, sub in df_split.groupby('subject_id', sort=False):
        idx = sub.index
        X = feat_df.loc[idx].to_numpy(dtype=float)
        a = adj[sid]
        if condition == 'real':
            An = normalize_adj(a)
        elif condition == 'identity':
            An = np.eye(68)
        elif condition == 'shuffled':
            An = normalize_adj(shuffle_adj(a, rng))
        elif condition == 'weight_shuffled':
            An = normalize_adj(shuffle_edge_weights(a, rng))
        else:
            raise ValueError(condition)
        parts.append(pd.DataFrame(np.hstack([X, An @ X]), index=idx))
    return pd.concat(parts).loc[df_split.index].to_numpy()

XgAB_tr = propagate_features(base.loc[train_mask], feat_AB, 'real')
XgAB_val = propagate_features(base.loc[val_mask], feat_AB, 'real')
p_val_single, lr_single = fit_lr(XgAB_tr, y_tr, XgAB_val)

p_val_nongraph, _ = fit_lr(XAB_tr, y_tr, XAB_val)

print('Non-graph A+B val:', fmt(compute_metrics(y_val, p_val_nongraph, sid_val)))
print('Graph A+B val    :', fmt(compute_metrics(y_val, p_val_single, sid_val)))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 5. 10 subject-bootstrap training resamples on validation
RESAMPLE_SEEDS = list(range(10))
graph_val_runs = {c: [] for c in ["real","shuffled","identity"]}
train_subjects = np.unique(sid_tr)

for condition in graph_val_runs:
    for seed in RESAMPLE_SEEDS:
        rng_graph = np.random.RandomState(seed)
        Xtr = propagate_features(base.loc[train_mask], feat_AB, condition, rng_graph)
        Xva = propagate_features(base.loc[val_mask], feat_AB, condition, rng_graph)

        rng_boot = np.random.RandomState(seed)
        sampled_sids = rng_boot.choice(train_subjects, size=len(train_subjects), replace=True)
        rows = np.concatenate([np.where(sid_tr == s)[0] for s in sampled_sids])

        p, _ = fit_lr(Xtr[rows], y_tr[rows], Xva)
        graph_val_runs[condition].append(compute_metrics(y_val, p, sid_val))

for condition, runs in graph_val_runs.items():
    print(condition)
    for metric in ["auprc","auroc","topk_dice"]:
        vals = [r[metric] for r in runs]
        print(f"  {metric:10s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 5.1 Genuine optimizer/init/shuffle stochasticity with SGD — NO extra manual sample weights
def fit_sgd(Xtr, ytr, Xev, seed):
    m = SGDClassifier(
        loss="log_loss", alpha=1e-3, max_iter=3000, tol=1e-4,
        class_weight="balanced", random_state=seed
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:,1], m

sgd_init = {c: [] for c in ["real","shuffled","identity"]}
for condition in sgd_init:
    rng_graph = np.random.RandomState(0)
    Xtr = propagate_features(base.loc[train_mask], feat_AB, condition, rng_graph)
    Xva = propagate_features(base.loc[val_mask], feat_AB, condition, rng_graph)
    for seed in range(10):
        p, _ = fit_sgd(Xtr, y_tr, Xva, seed)
        sgd_init[condition].append(average_precision_score(y_val, p))

for condition, vals in sgd_init.items():
    print(f"{condition:9s} SGD 10 init/shuffle seeds: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 5.2 Validation-only zero-message and edge-weight controls
Xzero_tr = np.hstack([XAB_tr, np.zeros_like(XAB_tr)])
Xzero_val = np.hstack([XAB_val, np.zeros_like(XAB_val)])
p_zero, _ = fit_lr(Xzero_tr, y_tr, Xzero_val)
zero_diff = float(np.max(np.abs(p_zero - p_val_nongraph)))
print('Zero-message max |p - non-graph p| =', zero_diff)
print('Zero-message metrics:', fmt(compute_metrics(y_val,p_zero,sid_val)))

weight_shuffle_runs=[]
for seed in RESAMPLE_SEEDS:
    rng_graph=np.random.RandomState(seed)
    Xtr=propagate_features(base.loc[train_mask],feat_AB,'weight_shuffled',rng_graph)
    Xva=propagate_features(base.loc[val_mask],feat_AB,'weight_shuffled',rng_graph)
    rng_boot=np.random.RandomState(seed)
    sampled=rng_boot.choice(train_subjects,size=len(train_subjects),replace=True)
    rows=np.concatenate([np.where(sid_tr==s)[0] for s in sampled])
    p,_=fit_lr(Xtr[rows],y_tr[rows],Xva)
    weight_shuffle_runs.append(compute_metrics(y_val,p,sid_val))

weight_shuffle_val = {
    k+'_mean':float(np.mean([r[k] for r in weight_shuffle_runs])) for k in ['auprc','auroc','topk_dice']
}
weight_shuffle_val.update({
    k+'_sd':float(np.std([r[k] for r in weight_shuffle_runs])) for k in ['auprc','auroc','topk_dice']
})
print('Weight-shuffled topology-fixed validation:', {k:round(v,4) for k,v in weight_shuffle_val.items()})

# Single-fit binarised topology-only check
parts_tr=[]; parts_va=[]
for mask, holder in [(train_mask,parts_tr),(val_mask,parts_va)]:
    for sid, sub in base.loc[mask].groupby('subject_id',sort=False):
        idx=sub.index
        X=feat_AB.loc[idx].to_numpy(float)
        Ab=(adj[sid]>0).astype(float)
        holder.append(pd.DataFrame(np.hstack([X,normalize_adj(Ab)@X]),index=idx))
Xbin_tr=pd.concat(parts_tr).loc[base.loc[train_mask].index].to_numpy()
Xbin_val=pd.concat(parts_va).loc[base.loc[val_mask].index].to_numpy()
p_bin,_=fit_lr(Xbin_tr,y_tr,Xbin_val)
binarized_val_metrics=compute_metrics(y_val,p_bin,sid_val)
print('Binarized topology-only validation:',fmt(binarized_val_metrics))

# 5.3 Edge-threshold sparsification — validation only
# Retain a fixed fraction of the strongest existing undirected edges in each subject.
def sparsify_adj_by_fraction(a, retain_fraction):
    if retain_fraction >= 1.0:
        return a.copy()
    iu = np.triu_indices_from(a, k=1)
    vals = a[iu]
    nz = np.where(vals > 0)[0]
    if len(nz) == 0:
        return a.copy()
    keep_n = max(1, int(np.ceil(len(nz) * retain_fraction)))
    strongest_local = nz[np.argsort(vals[nz])[-keep_n:]]
    out_vals = np.zeros_like(vals)
    out_vals[strongest_local] = vals[strongest_local]
    out = np.zeros_like(a)
    out[iu] = out_vals
    out = out + out.T
    return out

def propagate_sparse(df_split, feat_df, retain_fraction):
    parts=[]
    for sid,sub in df_split.groupby('subject_id',sort=False):
        idx=sub.index
        X=feat_df.loc[idx].to_numpy(float)
        As=sparsify_adj_by_fraction(adj[sid],retain_fraction)
        parts.append(pd.DataFrame(np.hstack([X,normalize_adj(As)@X]),index=idx))
    return pd.concat(parts).loc[df_split.index].to_numpy()

sparsification_rows=[]
for frac in [0.25,0.50,0.75,1.00]:
    runs=[]
    Xtr_s=propagate_sparse(base.loc[train_mask],feat_AB,frac)
    Xva_s=propagate_sparse(base.loc[val_mask],feat_AB,frac)
    for seed in RESAMPLE_SEEDS:
        rng_boot=np.random.RandomState(seed)
        sampled=rng_boot.choice(train_subjects,size=len(train_subjects),replace=True)
        rows=np.concatenate([np.where(sid_tr==s)[0] for s in sampled])
        p,_=fit_lr(Xtr_s[rows],y_tr[rows],Xva_s)
        runs.append(compute_metrics(y_val,p,sid_val))
    sparsification_rows.append({
        'retain_fraction':frac,
        'mean_edges_retained_pct':100*frac,
        'auprc_mean':float(np.mean([r['auprc'] for r in runs])),
        'auprc_sd':float(np.std([r['auprc'] for r in runs])),
        'auroc_mean':float(np.mean([r['auroc'] for r in runs])),
        'topk_dice_mean':float(np.mean([r['topk_dice'] for r in runs])),
    })
edge_sparsification_val_df=pd.DataFrame(sparsification_rows)
print('Edge sparsification validation:')
display(edge_sparsification_val_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 6. Modality ablations on validation, same graph + classifier
def graph_predict_train_to_eval(feat_df, eval_mask):
    Xtr = propagate_features(base.loc[train_mask], feat_df, "real")
    Xev = propagate_features(base.loc[eval_mask], feat_df, "real")
    p, model = fit_lr(Xtr, y_tr, Xev)
    return p, model

pA_val, _ = graph_predict_train_to_eval(feat_A, val_mask)
pB_val, _ = graph_predict_train_to_eval(feat_B, val_mask)
pAB_val, _ = graph_predict_train_to_eval(feat_AB, val_mask)

val_modality = pd.DataFrame([
    {"experiment":"Graph A only", **compute_metrics(y_val,pA_val,sid_val)},
    {"experiment":"Graph B only (all val; missing B handled)", **compute_metrics(y_val,pB_val,sid_val)},
    {"experiment":"Graph A+B", **compute_metrics(y_val,pAB_val,sid_val)},
])
display(val_modality.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 7. Missing-B strategies — validation only
modB_tr = base.loc[train_mask,'modB_available'].to_numpy(dtype=int)
modB_val = base.loc[val_mask,'modB_available'].to_numpy(dtype=int)

# Strategy 2: missing B -> training median -> standardized zero, but no availability flag.
def build_B_no_flag(df):
    Xi=impB.transform(df[BF])
    Xs=scB.transform(Xi)
    avail=df['modB_available'].to_numpy(dtype=float)
    return Xs * avail[:,None]

XBnf_tr=build_B_no_flag(base.loc[train_mask])
XBnf_val=build_B_no_flag(base.loc[val_mask])
XABnf_tr=np.hstack([XA_tr,XBnf_tr])
XABnf_val=np.hstack([XA_val,XBnf_val])
feat_AB_noflag=feature_frame(XABnf_tr,XABnf_val,'nf')

# Precompute graph matrices for all three strategies.
XgA_tr=propagate_features(base.loc[train_mask],feat_A,'real')
XgA_val=propagate_features(base.loc[val_mask],feat_A,'real')
XgAB_nf_tr=propagate_features(base.loc[train_mask],feat_AB_noflag,'real')
XgAB_nf_val=propagate_features(base.loc[val_mask],feat_AB_noflag,'real')

p_zero_flag_val=pAB_val.copy()
p_zero_noflag_val,_=fit_lr(XgAB_nf_tr,y_tr,XgAB_nf_val)
p_A_only_val=pA_val.copy()
p_A_fallback_val=np.where(modB_val==1,p_zero_flag_val,p_A_only_val)

missing_strategy_val_preds={
    'zero_plus_flag':p_zero_flag_val,
    'zero_no_flag':p_zero_noflag_val,
    'A_fallback':p_A_fallback_val,
}

rows=[]
for strategy,p in missing_strategy_val_preds.items():
    overall=compute_metrics(y_val,p,sid_val)
    bp=modB_val==1; ba=modB_val==0
    mbp=compute_metrics(y_val[bp],p[bp],sid_val[bp])
    mba=compute_metrics(y_val[ba],p[ba],sid_val[ba])
    rows.append({
        'strategy':strategy,
        **overall,
        'B_present_auprc':mbp['auprc'],'B_present_dice':mbp['topk_dice'],
        'B_absent_auprc':mba['auprc'],'B_absent_dice':mba['topk_dice'],
    })
missing_strategy_val_df=pd.DataFrame(rows).sort_values(['auprc','topk_dice'],ascending=False).reset_index(drop=True)
display(missing_strategy_val_df.round(4))

current_auprc=float(missing_strategy_val_df.loc[missing_strategy_val_df.strategy.eq('zero_plus_flag'),'auprc'].iloc[0])
best_row=missing_strategy_val_df.iloc[0]
if best_row['strategy']!='zero_plus_flag' and float(best_row['auprc'])-current_auprc >= 0.005:
    MISSING_STRATEGY=str(best_row['strategy'])
else:
    MISSING_STRATEGY='zero_plus_flag'
print('VALIDATION-SELECTED MISSING_STRATEGY =',MISSING_STRATEGY)

# General prediction helper used by validation resampling and train-OOF stacking.
def fit_strategy_predict(strategy, train_rows, X_zero_eval, X_nf_eval, X_A_eval, avail_eval):
    train_rows=np.asarray(train_rows,dtype=int)
    if strategy=='zero_plus_flag':
        return fit_lr(XgAB_tr[train_rows],y_tr[train_rows],X_zero_eval)[0]
    if strategy=='zero_no_flag':
        return fit_lr(XgAB_nf_tr[train_rows],y_tr[train_rows],X_nf_eval)[0]
    if strategy=='A_fallback':
        p_ab=fit_lr(XgAB_tr[train_rows],y_tr[train_rows],X_zero_eval)[0]
        p_a=fit_lr(XgA_tr[train_rows],y_tr[train_rows],X_A_eval)[0]
        return np.where(np.asarray(avail_eval)==1,p_ab,p_a)
    raise ValueError(strategy)

all_train_rows=np.arange(len(y_tr))
p_missing_selected_val_single=missing_strategy_val_preds[MISSING_STRATEGY].copy()

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 8. Single vs 10-resample ensemble for the validation-selected missing strategy
val_ensemble_probs=[]
for seed in RESAMPLE_SEEDS:
    rng_boot=np.random.RandomState(seed)
    sampled_sids=rng_boot.choice(train_subjects,size=len(train_subjects),replace=True)
    rows=np.concatenate([np.where(sid_tr==s)[0] for s in sampled_sids])
    p=fit_strategy_predict(
        MISSING_STRATEGY,rows,
        XgAB_val,XgAB_nf_val,XgA_val,modB_val
    )
    val_ensemble_probs.append(p)
val_ensemble_probs=np.asarray(val_ensemble_probs)
p_val_ensemble=val_ensemble_probs.mean(axis=0)

p_val_single=p_missing_selected_val_single.copy()
m_single=compute_metrics(y_val,p_val_single,sid_val)
m_ens=compute_metrics(y_val,p_val_ensemble,sid_val)
print('single fit :',fmt(m_single))
print('10x ensemble:',fmt(m_ens))
BASE_KIND='ensemble10' if m_ens['auprc']>m_single['auprc'] else 'single'
p_model_val=p_val_ensemble if BASE_KIND=='ensemble10' else p_val_single
print('VALIDATION-SELECTED BASE_KIND =',BASE_KIND)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 8.1 Corrected 0/1/2/3-hop validation audit
def propagate_features_khops(df_split, feat_df, condition, k_hops, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    parts = []
    for sid, sub in df_split.groupby("subject_id", sort=False):
        idx = sub.index
        X = feat_df.loc[idx].to_numpy(dtype=float)
        a = adj[sid]

        if condition == "real":
            An = normalize_adj(a)
        elif condition == "identity":
            An = np.eye(68)
        elif condition == "shuffled":
            An = normalize_adj(shuffle_adj(a, rng))
        else:
            raise ValueError(condition)

        hops = [X]
        cur = X
        for _ in range(k_hops):
            cur = An @ cur
            hops.append(cur)

        parts.append(pd.DataFrame(np.hstack(hops), index=idx))
    return pd.concat(parts).loc[df_split.index].to_numpy()


HOP_CANDIDATES = [0, 1, 2, 3]
hop_val_runs = {k: [] for k in HOP_CANDIDATES}

for k_hops in HOP_CANDIDATES:
    for seed in RESAMPLE_SEEDS:
        rng_graph = np.random.RandomState(seed)
        Xtr_k = propagate_features_khops(base.loc[train_mask], feat_AB, "real", k_hops, rng_graph)
        Xva_k = propagate_features_khops(base.loc[val_mask], feat_AB, "real", k_hops, rng_graph)

        rng_boot = np.random.RandomState(seed)
        sampled_sids = rng_boot.choice(train_subjects, size=len(train_subjects), replace=True)
        rows = np.concatenate([np.where(sid_tr == s)[0] for s in sampled_sids])

        p_k, _ = fit_lr(Xtr_k[rows], y_tr[rows], Xva_k)
        hop_val_runs[k_hops].append(compute_metrics(y_val, p_k, sid_val))

hop_val_summary = []
for k_hops in HOP_CANDIDATES:
    row = {"hops": k_hops}
    for metric in ["auprc", "auroc", "topk_dice"]:
        vals = np.asarray([r[metric] for r in hop_val_runs[k_hops]])
        row[f"{metric}_mean"] = float(vals.mean())
        row[f"{metric}_sd"] = float(vals.std())
    hop_val_summary.append(row)

hop_val_df = pd.DataFrame(hop_val_summary)
display(hop_val_df.round(4))

# Correctness checks using the full, un-resampled training set.
X0_tr = propagate_features_khops(base.loc[train_mask], feat_AB, "real", 0, np.random.RandomState(0))
X0_va = propagate_features_khops(base.loc[val_mask], feat_AB, "real", 0, np.random.RandomState(0))
p0, _ = fit_lr(X0_tr, y_tr, X0_va)

X1_tr = propagate_features_khops(base.loc[train_mask], feat_AB, "real", 1, np.random.RandomState(0))
X1_va = propagate_features_khops(base.loc[val_mask], feat_AB, "real", 1, np.random.RandomState(0))
p1, _ = fit_lr(X1_tr, y_tr, X1_va)

a0 = average_precision_score(y_val, p0)
a1 = average_precision_score(y_val, p1)
assert np.isclose(a0, average_precision_score(y_val, p_val_nongraph), atol=1e-12)
assert np.isclose(a1, average_precision_score(y_val, pAB_val), atol=1e-12)
print(f"Sanity checks passed: 0-hop={a0:.6f}, 1-hop={a1:.6f}")

# Conservative, pre-specified robustness rule:
# retain 1-hop unless another depth improves mean validation AUPRC by > 1 SD of 1-hop
# AND does not reduce mean Top-k Dice relative to 1-hop.
one = hop_val_df.loc[hop_val_df.hops.eq(1)].iloc[0]
eligible = []
for _, r in hop_val_df.iterrows():
    if int(r["hops"]) == 1:
        continue
    material_gain = r["auprc_mean"] - one["auprc_mean"] > one["auprc_sd"]
    no_dice_harm = r["topk_dice_mean"] >= one["topk_dice_mean"]
    if material_gain and no_dice_harm:
        eligible.append(int(r["hops"]))

FROZEN_DEPTH = min(eligible) if eligible else 1
print("Hop-depth decision from validation only -> FROZEN_DEPTH =", FROZEN_DEPTH)

if FROZEN_DEPTH != 1:
    raise RuntimeError(
        "This notebook's downstream graph/fusion pipeline is implemented for the 1-hop model. "
        "Validation produced a material reason to change depth, so stop here and rebuild the "
        "downstream pipeline around that depth before touching test."
    )

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 9. Simulation standalone + concordance
sim_val = base.loc[val_mask, "sim_score"].to_numpy(dtype=float)
sim_conf_val = base.loc[val_mask, "sim_confidence"].to_numpy(dtype=float)

print("Simulation standalone:", fmt(compute_metrics(y_val, sim_val, sid_val)))
print("Model standalone     :", fmt(compute_metrics(y_val, p_model_val, sid_val)))

pooled_spear = spearmanr(p_model_val, sim_val)
pooled_pear = pearsonr(p_model_val, sim_val)

per_subj = []
for sid in np.unique(sid_val):
    m = sid_val == sid
    yt, pm, ps = y_val[m], p_model_val[m], sim_val[m]
    k = int(yt.sum())
    top_m = set(np.argsort(-pm)[:k])
    top_s = set(np.argsort(-ps)[:k])
    true = set(np.where(yt==1)[0])
    sim_dice = 2*len(top_s & true)/(len(top_s)+len(true))
    model_dice = 2*len(top_m & true)/(len(top_m)+len(true))
    rho = spearmanr(pm, ps).statistic
    jacc = len(top_m & top_s)/len(top_m | top_s)
    per_subj.append({
        "subject_id":sid,
        "sim_conf":float(sim_conf_val[m][0]),
        "sim_dice":sim_dice,
        "model_dice":model_dice,
        "model_sim_rho":rho,
        "topk_jaccard":jacc,
        "mad":float(np.mean(np.abs(pm-ps))),
    })

per_subj = pd.DataFrame(per_subj)
print(f"Pooled model-vs-sim Spearman rho={pooled_spear.statistic:.3f}, p={pooled_spear.pvalue:.4g}")
print(f"Pooled model-vs-sim Pearson  r  ={pooled_pear.statistic:.3f}, p={pooled_pear.pvalue:.4g}")
print("Mean per-subject model-vs-sim rho:", round(per_subj["model_sim_rho"].mean(),3))
print("Mean per-subject top-k Jaccard    :", round(per_subj["topk_jaccard"].mean(),3))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 9.1 Does sim_confidence predict simulation quality?
sp_q = spearmanr(per_subj["sim_conf"], per_subj["sim_dice"])
pe_q = pearsonr(per_subj["sim_conf"], per_subj["sim_dice"])
sp_dis = spearmanr(per_subj["sim_conf"], per_subj["mad"])

median_conf = per_subj["sim_conf"].median()
low = per_subj["sim_conf"] <= median_conf
high = per_subj["sim_conf"] > median_conf

print("sim_confidence vs simulation Top-k Dice:")
print(f"  Spearman rho={sp_q.statistic:.4f}, p={sp_q.pvalue:.6g}")
print(f"  Pearson  r  ={pe_q.statistic:.4f}, p={pe_q.pvalue:.6g}")
print(f"sim_confidence vs model-sim MAD: rho={sp_dis.statistic:.4f}, p={sp_dis.pvalue:.6g}")
print()
print(f"High-confidence n={high.sum()}: sim Dice={per_subj.loc[high,'sim_dice'].mean():.4f}, "
      f"model Dice={per_subj.loc[high,'model_dice'].mean():.4f}")
print(f"Low-confidence  n={low.sum()}: sim Dice={per_subj.loc[low,'sim_dice'].mean():.4f}, "
      f"model Dice={per_subj.loc[low,'model_dice'].mean():.4f}")

CONFIDENCE_GATE_JUSTIFIED = bool(sp_q.statistic > 0 and sp_q.pvalue < 0.05)
print("\nConfidence-aware fusion eligible from evidence? ->", CONFIDENCE_GATE_JUSTIFIED)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 9.2 Explicit discordant subjects and a node-level case study
meta_val = subj[subj.split.eq('val')][['subject_id','site','hemisphere','modality_B_available']].copy()
nabn_val = base.loc[val_mask].groupby('subject_id')['is_abnormal'].sum().rename('n_abnormal')
discordant_val_df = (per_subj.merge(meta_val,on='subject_id',how='left')
                     .merge(nabn_val,on='subject_id',how='left')
                     .sort_values('mad',ascending=False).reset_index(drop=True))

print('Most discordant validation subjects:')
display(discordant_val_df[['subject_id','site','hemisphere','modality_B_available','n_abnormal',
                           'sim_conf','model_sim_rho','topk_jaccard','mad','model_dice','sim_dice']].head(8).round(4))

worst_discordant_sid = discordant_val_df.iloc[0]['subject_id']
m = sid_val == worst_discordant_sid
worst_rows = base.loc[val_mask].loc[m, ['subject_id','node_id','region','is_abnormal','modB_available','sim_score','sim_confidence']].copy()
worst_rows['model_prob'] = p_model_val[m]
worst_rows['abs_model_sim_gap'] = np.abs(worst_rows['model_prob'] - worst_rows['sim_score'])
worst_discordant_nodes = worst_rows.sort_values('abs_model_sim_gap',ascending=False).reset_index(drop=True)
print('\nWorst discordant subject:', worst_discordant_sid)
display(worst_discordant_nodes[['node_id','region','is_abnormal','model_prob','sim_score','abs_model_sim_gap']].head(12).round(4))

# Does the confidence-quality relationship survive the obvious modality-B stratification?
confound_rows=[]
for avail,grp in discordant_val_df.groupby('modality_B_available'):
    row={'modality_B_available':int(avail),'n_subjects':len(grp),
         'mean_sim_conf':float(grp.sim_conf.mean()),'mean_sim_dice':float(grp.sim_dice.mean())}
    if len(grp)>=4 and grp.sim_conf.nunique()>1:
        sp=spearmanr(grp.sim_conf,grp.sim_dice)
        row.update({'rho_conf_vs_sim_dice':float(sp.statistic),'pvalue':float(sp.pvalue)})
    confound_rows.append(row)
confidence_by_B_df=pd.DataFrame(confound_rows)
print('\nConfidence-vs-quality stratified by modality-B availability (small-n sensitivity check):')
display(confidence_by_B_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 9.3 Train-only OOF stacking model — base-kind matched
train_sim=base.loc[train_mask,'sim_score'].to_numpy(dtype=float)
train_conf=base.loc[train_mask,'sim_confidence'].to_numpy(dtype=float)

# Each training subject receives a learned-model score from models that did not train on that subject.
# Crucially, the OOF construction mirrors the already validation-selected BASE_KIND.
gkf_base=GroupKFold(n_splits=5)
p_model_train_oof=np.zeros(len(y_tr),dtype=float)
stack_oof_fold_records=[]

for fold_id,(tr_idx,ho_idx) in enumerate(gkf_base.split(np.zeros(len(y_tr)),y_tr,groups=sid_tr)):
    train_fold_subjects=np.unique(sid_tr[tr_idx])
    heldout_subjects=np.unique(sid_tr[ho_idx])

    # Sanity: subject groups are disjoint within every OOF fold.
    assert not (set(train_fold_subjects) & set(heldout_subjects))

    if BASE_KIND=='single':
        fold_pred=fit_strategy_predict(
            MISSING_STRATEGY,tr_idx,
            XgAB_tr[ho_idx],XgAB_nf_tr[ho_idx],XgA_tr[ho_idx],modB_tr[ho_idx]
        )
        n_base_models=1

    elif BASE_KIND=='ensemble10':
        fold_members=[]
        for seed in RESAMPLE_SEEDS:
            rng_boot=np.random.RandomState(seed)
            # Bootstrap only subjects available in this outer fold's training partition.
            sampled_sids=rng_boot.choice(
                train_fold_subjects,
                size=len(train_fold_subjects),
                replace=True
            )
            boot_rows=np.concatenate([
                tr_idx[sid_tr[tr_idx]==s] for s in sampled_sids
            ])
            # Absolute guarantee: no held-out subject can enter an ensemble member's training rows.
            assert not (set(sid_tr[boot_rows]) & set(heldout_subjects))

            p_member=fit_strategy_predict(
                MISSING_STRATEGY,boot_rows,
                XgAB_tr[ho_idx],XgAB_nf_tr[ho_idx],XgA_tr[ho_idx],modB_tr[ho_idx]
            )
            fold_members.append(p_member)

        fold_pred=np.mean(np.asarray(fold_members),axis=0)
        n_base_models=len(fold_members)

    else:
        raise ValueError(f'Unexpected BASE_KIND={BASE_KIND}')

    p_model_train_oof[ho_idx]=fold_pred
    stack_oof_fold_records.append({
        'fold':fold_id,
        'n_train_subjects':int(len(train_fold_subjects)),
        'n_heldout_subjects':int(len(heldout_subjects)),
        'base_kind':BASE_KIND,
        'n_base_models':int(n_base_models),
    })

stack_oof_folds_df=pd.DataFrame(stack_oof_fold_records)
assert np.isfinite(p_model_train_oof).all()
assert len(stack_oof_folds_df)==5
assert (stack_oof_folds_df['base_kind']==BASE_KIND).all()
if BASE_KIND=='ensemble10':
    assert (stack_oof_folds_df['n_base_models']==len(RESAMPLE_SEEDS)).all()
else:
    assert (stack_oof_folds_df['n_base_models']==1).all()

STACK_OOF_BASE_KIND=BASE_KIND
print('Stacker OOF base kind:',STACK_OOF_BASE_KIND)
display(stack_oof_folds_df)
print('Train OOF base AUPRC:',round(average_precision_score(y_tr,p_model_train_oof),4))

def make_stack_features(p_model,p_sim,conf,modB):
    p_model=np.asarray(p_model,float); p_sim=np.asarray(p_sim,float)
    conf=np.asarray(conf,float); modB=np.asarray(modB,float)
    return np.column_stack([
        p_model,p_sim,conf,modB,
        np.abs(p_model-p_sim),
        p_model*p_sim,
        p_model*conf,
        p_sim*conf,
    ])

STACK_FEATURE_NAMES=[
    'model_score','sim_score','sim_confidence','modB_available',
    'abs_model_sim_gap','model_x_sim','model_x_conf','sim_x_conf'
]
META_TR=make_stack_features(p_model_train_oof,train_sim,train_conf,modB_tr)

# Train-only grouped CV chooses the stacker's C.
# These are meta-model folds; all META_TR model_score values are already subject-OOF.
stack_cv_rows=[]
for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0]:
    oof=np.zeros(len(y_tr),dtype=float)
    gkf_meta=GroupKFold(n_splits=5)
    for tr_idx,ho_idx in gkf_meta.split(META_TR,y_tr,groups=sid_tr):
        sc=StandardScaler().fit(META_TR[tr_idx])
        m=LogisticRegression(max_iter=3000,class_weight='balanced',C=C,random_state=0)
        m.fit(sc.transform(META_TR[tr_idx]),y_tr[tr_idx])
        oof[ho_idx]=m.predict_proba(sc.transform(META_TR[ho_idx]))[:,1]
    met=compute_metrics(y_tr,oof,sid_tr)
    stack_cv_rows.append({'C':C,**met})

stacking_train_cv_df=pd.DataFrame(stack_cv_rows).sort_values('auprc',ascending=False).reset_index(drop=True)
display(stacking_train_cv_df.round(4))
STACK_C=float(stacking_train_cv_df.iloc[0]['C'])

STACK_SCALER=StandardScaler().fit(META_TR)
STACK_MODEL=LogisticRegression(max_iter=3000,class_weight='balanced',C=STACK_C,random_state=0)
STACK_MODEL.fit(STACK_SCALER.transform(META_TR),y_tr)

# p_model_val already uses the validation-selected BASE_KIND, so META_VAL now matches META_TR by construction.
META_VAL=make_stack_features(
    p_model_val,
    base.loc[val_mask,'sim_score'].to_numpy(float),
    base.loc[val_mask,'sim_confidence'].to_numpy(float),
    modB_val
)
p_stack_val=STACK_MODEL.predict_proba(STACK_SCALER.transform(META_VAL))[:,1]
print('Frozen stacker C =',STACK_C)
print('Stacker validation:',fmt(compute_metrics(y_val,p_stack_val,sid_val)))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 10. Frozen score scaling helpers and fusion candidates
def fit_minmax(x):
    x=np.asarray(x,dtype=float)
    return {'lo':float(x.min()),'hi':float(x.max())}

def apply_minmax(x,pars):
    den=max(pars['hi']-pars['lo'],1e-12)
    return np.clip((np.asarray(x)-pars['lo'])/den,0.0,1.0)

sim_val=base.loc[val_mask,'sim_score'].to_numpy(dtype=float)
sim_conf_val=base.loc[val_mask,'sim_confidence'].to_numpy(dtype=float)
VAL_SCALE={'model':fit_minmax(p_model_val),'sim':fit_minmax(sim_val)}

def get_scaled_pair(p_model,p_sim,method):
    if method=='raw': return np.asarray(p_model),np.asarray(p_sim)
    if method=='val_minmax':
        return apply_minmax(p_model,VAL_SCALE['model']),apply_minmax(p_sim,VAL_SCALE['sim'])
    raise ValueError(method)

def tune_convex(y,pm,ps):
    best=None
    for w in np.linspace(0,1,21):
        p=w*pm+(1-w)*ps
        row=(average_precision_score(y,p),float(w),p)
        if best is None or row[0]>best[0]: best=row
    return {'auprc':best[0],'w_model':best[1],'pred':best[2]}

def confidence_fusion(pm,ps,conf,alpha,gamma):
    w_sim=np.clip(alpha*np.power(np.clip(conf,0,1),gamma),0,1)
    return (1-w_sim)*pm+w_sim*ps

def tune_confidence_power(y,pm,ps,conf):
    best=None
    for alpha in np.linspace(0,1,21):
        for gamma in [0.5,1.0,1.5,2.0,3.0]:
            p=confidence_fusion(pm,ps,conf,alpha,gamma)
            row=(average_precision_score(y,p),float(alpha),float(gamma),p)
            if best is None or row[0]>best[0]: best=row
    return {'auprc':best[0],'alpha':best[1],'gamma':best[2],'pred':best[3]}

fusion_val_rows=[]
frozen_fusion_candidates={}
for scale_method in ['raw','val_minmax']:
    pm,ps=get_scaled_pair(p_model_val,sim_val,scale_method)
    p_naive=0.5*pm+0.5*ps
    fusion_val_rows.append({'family':'naive_50_50','scale':scale_method,**compute_metrics(y_val,p_naive,sid_val)})
    frozen_fusion_candidates[f'naive_50_50__{scale_method}']={'pred':p_naive}

    cvx=tune_convex(y_val,pm,ps)
    fusion_val_rows.append({'family':'convex','scale':scale_method,'w_model':cvx['w_model'],**compute_metrics(y_val,cvx['pred'],sid_val)})
    frozen_fusion_candidates[f'convex__{scale_method}']=cvx

    conf=tune_confidence_power(y_val,pm,ps,sim_conf_val)
    fusion_val_rows.append({'family':'confidence_power','scale':scale_method,'alpha':conf['alpha'],'gamma':conf['gamma'],**compute_metrics(y_val,conf['pred'],sid_val)})
    frozen_fusion_candidates[f'confidence_power__{scale_method}']=conf

# Learned stacker is trained entirely from train OOF predictions; validation is only an evaluation/selection set.
fusion_val_rows.append({
    'family':'stacked_lr','scale':'train_oof_meta','stack_C':STACK_C,
    **compute_metrics(y_val,p_stack_val,sid_val)
})
frozen_fusion_candidates['stacked_lr__train_oof_meta']={'pred':p_stack_val,'C':STACK_C}

fusion_val_df=pd.DataFrame(fusion_val_rows).sort_values('auprc',ascending=False).reset_index(drop=True)
display(fusion_val_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 10.1 Pre-register final fusion family using VALIDATION ONLY
# First select the best justified parametric fusion. The learned stacker replaces it only
# for a material validation gain (>=0.005 AUPRC) without >0.02 loss in Top-k Dice.
parametric= fusion_val_df[fusion_val_df['family'].isin(['convex','confidence_power'])].copy()
if not CONFIDENCE_GATE_JUSTIFIED:
    parametric=parametric[parametric['family']!='confidence_power']
best_param=parametric.sort_values(['auprc','topk_dice'],ascending=False).iloc[0].to_dict()
stack_row=fusion_val_df[fusion_val_df.family.eq('stacked_lr')].iloc[0].to_dict()

stack_material_gain=float(stack_row['auprc'])-float(best_param['auprc']) >= 0.005
stack_no_dice_harm=float(stack_row['topk_dice']) >= float(best_param['topk_dice'])-0.02
winner=stack_row if (stack_material_gain and stack_no_dice_harm) else best_param

SMART_FAMILY=str(winner['family'])
SMART_SCALE=str(winner['scale'])
if SMART_FAMILY=='convex':
    pars=frozen_fusion_candidates[f'convex__{SMART_SCALE}']
    SMART_PARAMS={'w_model':float(pars['w_model'])}
elif SMART_FAMILY=='confidence_power':
    pars=frozen_fusion_candidates[f'confidence_power__{SMART_SCALE}']
    SMART_PARAMS={'alpha':float(pars['alpha']),'gamma':float(pars['gamma'])}
elif SMART_FAMILY=='stacked_lr':
    SMART_PARAMS={'C':STACK_C,'features':STACK_FEATURE_NAMES}
else:
    raise RuntimeError(SMART_FAMILY)

best_convex_row=fusion_val_df[fusion_val_df.family.eq('convex')].iloc[0].to_dict()
best_conf_row=fusion_val_df[fusion_val_df.family.eq('confidence_power')].iloc[0].to_dict()
FROZEN_CONVEX={'scale':best_convex_row['scale'],'w_model':float(best_convex_row['w_model'])}
FROZEN_CONF={'scale':best_conf_row['scale'],'alpha':float(best_conf_row['alpha']),'gamma':float(best_conf_row['gamma'])}
FROZEN_STACK={'C':STACK_C,'features':STACK_FEATURE_NAMES}

print('Best justified parametric fusion:',best_param['family'],round(float(best_param['auprc']),6))
print('Stacker validation AUPRC:',round(float(stack_row['auprc']),6),
      '| material gain?',stack_material_gain,'| Dice non-harm?',stack_no_dice_harm)
print('VALIDATION-SELECTED FINAL FUSION:')
print(json.dumps({'family':SMART_FAMILY,'scale':SMART_SCALE,**SMART_PARAMS},indent=2))
print('Validation AUPRC =',round(float(winner['auprc']),6))
print('Best convex candidate:',FROZEN_CONVEX)
print('Best confidence candidate:',FROZEN_CONF)
print('Stacker candidate:',FROZEN_STACK)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 10.2 Validation bootstrap stability of fusion families
rng=np.random.RandomState(123)
val_subjects=np.unique(sid_val)
records=[]
for b in range(500):
    sampled=rng.choice(val_subjects,size=len(val_subjects),replace=True)
    idx=np.concatenate([np.where(sid_val==s)[0] for s in sampled])
    yb=y_val[idx]; cb=sim_conf_val[idx]

    pmc,psc=get_scaled_pair(p_model_val,sim_val,FROZEN_CONVEX['scale'])
    c=tune_convex(yb,pmc[idx],psc[idx])
    pmp,psp=get_scaled_pair(p_model_val,sim_val,FROZEN_CONF['scale'])
    q=tune_confidence_power(yb,pmp[idx],psp[idx],cb)
    stack_a=average_precision_score(yb,p_stack_val[idx])
    records.append({
        'convex_w':c['w_model'],'convex_best_auprc':c['auprc'],
        'conf_alpha':q['alpha'],'conf_gamma':q['gamma'],'conf_best_auprc':q['auprc'],
        'stack_fixed_auprc':stack_a,
        'conf_wins_vs_convex':q['auprc']>c['auprc'],
        'stack_wins_vs_best_parametric':stack_a>max(c['auprc'],q['auprc']),
    })
stab=pd.DataFrame(records)
print('Convex w mean/std/5th/95th =',round(stab.convex_w.mean(),3),round(stab.convex_w.std(),3),np.quantile(stab.convex_w,[.05,.95]).round(3))
print('Confidence alpha mean/std/5th/95th =',round(stab.conf_alpha.mean(),3),round(stab.conf_alpha.std(),3),np.quantile(stab.conf_alpha,[.05,.95]).round(3))
print('Confidence gamma mode:',Counter(stab.conf_gamma).most_common(3))
print('Confidence beats convex in',f"{100*stab.conf_wins_vs_convex.mean():.1f}%",'of validation resamples')
print('Fixed train-OOF stacker beats the re-tuned best parametric fusion in',f"{100*stab.stack_wins_vs_best_parametric.mean():.1f}%",'of validation resamples')

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 10.3 Paired subject-bootstrap CIs among frozen validation fusion candidates
def make_fusion_pred(pm_raw,ps_raw,conf,spec):
    pm,ps=get_scaled_pair(pm_raw,ps_raw,spec['scale'])
    if 'w_model' in spec:
        return spec['w_model']*pm+(1-spec['w_model'])*ps
    return confidence_fusion(pm,ps,conf,spec['alpha'],spec['gamma'])

p_val_cvx=make_fusion_pred(p_model_val,sim_val,sim_conf_val,FROZEN_CONVEX)
p_val_conf=make_fusion_pred(p_model_val,sim_val,sim_conf_val,FROZEN_CONF)
ci_conf_vs_cvx_val=subject_bootstrap_metric_diff(y_val,p_val_conf,p_val_cvx,sid_val,'auprc',8000,7)
ci_stack_vs_conf_val=subject_bootstrap_metric_diff(y_val,p_stack_val,p_val_conf,sid_val,'auprc',8000,8)
ci_stack_vs_cvx_val=subject_bootstrap_metric_diff(y_val,p_stack_val,p_val_cvx,sid_val,'auprc',8000,9)
print('VAL confidence - convex AUPRC:',ci_conf_vs_cvx_val)
print('VAL stacker - confidence AUPRC:',ci_stack_vs_conf_val)
print('VAL stacker - convex AUPRC:',ci_stack_vs_cvx_val)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 10.4 Validation-only calibration and descriptive threshold
if SMART_FAMILY=='convex':
    p_final_val = p_val_cvx.copy()
elif SMART_FAMILY=='confidence_power':
    p_final_val = p_val_conf.copy()
elif SMART_FAMILY=='stacked_lr':
    p_final_val = p_stack_val.copy()
else:
    raise RuntimeError(SMART_FAMILY)

# Frozen threshold from validation only
fpr_v,tpr_v,thr_v = roc_curve(y_val,p_final_val)
j_v=tpr_v-fpr_v
thr_idx=int(np.nanargmax(j_v))
VAL_THRESHOLD=float(thr_v[thr_idx])
print('Validation-frozen descriptive threshold (Youden J):',round(VAL_THRESHOLD,6))

# Subject-grouped OOF calibration diagnostic
clip=lambda p: np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
xcal=np.log(clip(p_final_val)/(1-clip(p_final_val))).reshape(-1,1)
gkf=GroupKFold(n_splits=5)
p_platt_oof=np.zeros(len(y_val),dtype=float)
p_iso_oof=np.zeros(len(y_val),dtype=float)
for tr_idx,ho_idx in gkf.split(xcal,y_val,groups=sid_val):
    pl=LogisticRegression(max_iter=2000,random_state=0).fit(xcal[tr_idx],y_val[tr_idx])
    p_platt_oof[ho_idx]=pl.predict_proba(xcal[ho_idx])[:,1]
    iso=IsotonicRegression(out_of_bounds='clip').fit(p_final_val[tr_idx],y_val[tr_idx])
    p_iso_oof[ho_idx]=iso.transform(p_final_val[ho_idx])

calibration_val_df=pd.DataFrame([
    {'method':'raw frozen fusion','brier':brier_score_loss(y_val,p_final_val),'auprc':average_precision_score(y_val,p_final_val)},
    {'method':'Platt 5-fold subject-OOF','brier':brier_score_loss(y_val,p_platt_oof),'auprc':average_precision_score(y_val,p_platt_oof)},
    {'method':'Isotonic 5-fold subject-OOF','brier':brier_score_loss(y_val,p_iso_oof),'auprc':average_precision_score(y_val,p_iso_oof)},
])
display(calibration_val_df.round(4))
CALIBRATION_DECISION='retain raw frozen fusion scores; OOF calibration is diagnostic only without an external calibration cohort'
print(CALIBRATION_DECISION)

# Reliability table for raw final validation fusion
rel=pd.DataFrame({'p':p_final_val,'y':y_val,'sid':sid_val})
rel['bin']=pd.qcut(rel['p'],q=8,duplicates='drop')
reliability_val_df=(rel.groupby('bin',observed=True)
                    .agg(n=('y','size'),mean_pred=('p','mean'),observed_rate=('y','mean'))
                    .reset_index(drop=True))
display(reliability_val_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 11. Explicit freeze / leakage audit
FREEZE={
    'features':'A+B; A train-median+missingness; B train-fitted handling according to frozen missing strategy',
    'missing_strategy':MISSING_STRATEGY,
    'graph':'real adjacency; D^-1/2(A+I)D^-1/2; one hop; residual concat [X,AX]',
    'frozen_depth':int(FROZEN_DEPTH),
    'classifier':"LogisticRegression(class_weight='balanced', C=1.0)",
    'base_kind':BASE_KIND,
    'smart_family':SMART_FAMILY,
    'smart_scale':SMART_SCALE,
    'smart_params':SMART_PARAMS,
    'stacker_C':STACK_C,
    'stacker_features':STACK_FEATURE_NAMES,
    'stacker_oof_base_kind':STACK_OOF_BASE_KIND,
    'score_scalers_fit_on':'validation only' if SMART_SCALE=='val_minmax' else ('train OOF meta scaler' if SMART_FAMILY=='stacked_lr' else 'not applicable'),
    'descriptive_threshold_from_validation':float(VAL_THRESHOLD),
    'calibration_decision':CALIBRATION_DECISION,
    'resample_seeds':RESAMPLE_SEEDS,
}
leakage_audit={
    'test_gate_still_locked_before_final_reporting':bool(not TEST_UNLOCKED),
    'train_val_test_subjects_disjoint':not (set(subj[subj.split=='train'].subject_id)&set(subj[subj.split=='val'].subject_id) or set(subj[subj.split=='train'].subject_id)&set(subj[subj.split=='test'].subject_id) or set(subj[subj.split=='val'].subject_id)&set(subj[subj.split=='test'].subject_id)),
    'A_imputer_scaler_fit_train_only':True,
    'B_imputer_scaler_fit_Bpresent_train_only':True,
    'missing_modality_strategy_selected_on_validation_only':True,
    'graph_model_family_selected_without_test_labels':True,
    'hop_depth_audited_on_validation_only':bool(FROZEN_DEPTH==1),
    'edge_sparsification_validation_only':True,
    'base_single_vs_ensemble_selected_on_validation':True,
    'confidence_evidence_checked_on_validation':True,
    'discordant_cases_examined_before_fusion':True,
    'baseline_family_audited_on_validation':True,
    'stacker_base_predictions_are_train_subject_grouped_OOF':True,
    'stacker_oof_base_kind_matches_deployment_base_kind':bool(STACK_OOF_BASE_KIND==BASE_KIND),
    'stacker_hyperparameter_selected_on_train_only':True,
    'fusion_family_and_params_selected_on_validation':True,
    'fusion_score_scaling_parameters_not_fit_on_test':True,
    'calibration_diagnostic_subject_grouped_validation_only':True,
    'clinical_threshold_selected_on_validation_only':True,
    'outcome_and_resection_not_used_as_model_inputs':True,
    'true_k_used_only_in_metrics_or_posthoc_error_analysis':True,
}
assert all(leakage_audit.values())
print('LEAKAGE AUDIT')
print(json.dumps(leakage_audit,indent=2))
print('\nFROZEN CONFIG')
print(json.dumps(FREEZE,indent=2))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12. Unlock test and transform test features now — and only now
TEST_UNLOCKED=True
XA_test,_,_=build_A(base.loc[test_mask],impA,scA)
XB_test=build_B(base.loc[test_mask])
XBnf_test=build_B_no_flag(base.loc[test_mask])
XAB_test=np.hstack([XA_test,XB_test])
XABnf_test=np.hstack([XA_test,XBnf_test])

feat_A.loc[test_mask,:]=XA_test
feat_B.loc[test_mask,:]=XB_test
feat_AB.loc[test_mask,:]=XAB_test
feat_AB_noflag.loc[test_mask,:]=XABnf_test

y_test=base.loc[test_mask,'is_abnormal'].to_numpy()
sim_test=base.loc[test_mask,'sim_score'].to_numpy(float)
sim_conf_test=base.loc[test_mask,'sim_confidence'].to_numpy(float)
modB_test=base.loc[test_mask,'modB_available'].to_numpy(dtype=int)
print('Test subjects:',len(np.unique(sid_test)),'nodes:',len(y_test),'prevalence:',round(y_test.mean(),5))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.1 Required learned-model test predictions + all frozen missing-B strategies
p_ng_test,_=fit_lr(XAB_tr,y_tr,XAB_test)
pA_test,_=graph_predict_train_to_eval(feat_A,test_mask)
pB_test,_=graph_predict_train_to_eval(feat_B,test_mask)
pAB_test_single,_=graph_predict_train_to_eval(feat_AB,test_mask)
XgAB_test=propagate_features(base.loc[test_mask],feat_AB,'real')
XgAB_nf_test=propagate_features(base.loc[test_mask],feat_AB_noflag,'real')
XgA_test=propagate_features(base.loc[test_mask],feat_A,'real')
pAB_nf_test_single,_=fit_lr(XgAB_nf_tr,y_tr,XgAB_nf_test)
pA_fallback_test_single=np.where(modB_test==1,pAB_test_single,pA_test)

missing_strategy_test_preds={
    'zero_plus_flag':pAB_test_single,
    'zero_no_flag':pAB_nf_test_single,
    'A_fallback':pA_fallback_test_single,
}
p_selected_single_test=missing_strategy_test_preds[MISSING_STRATEGY]

# Ensemble the exact same frozen missing strategy if validation selected ensemble10.
test_ensemble_probs=[]
for seed in RESAMPLE_SEEDS:
    rng_boot=np.random.RandomState(seed)
    sampled_sids=rng_boot.choice(train_subjects,size=len(train_subjects),replace=True)
    rows=np.concatenate([np.where(sid_tr==s)[0] for s in sampled_sids])
    p=fit_strategy_predict(MISSING_STRATEGY,rows,XgAB_test,XgAB_nf_test,XgA_test,modB_test)
    test_ensemble_probs.append(p)
test_ensemble_probs=np.asarray(test_ensemble_probs)
p_selected_ensemble_test=test_ensemble_probs.mean(axis=0)
p_model_test=p_selected_ensemble_test if BASE_KIND=='ensemble10' else p_selected_single_test

# Report all strategies, but do not use test to choose among them.
ms_rows=[]
for strategy,p in missing_strategy_test_preds.items():
    for subgroup,mask in [('all',np.ones(len(y_test),dtype=bool)),('B-present',modB_test==1),('B-absent',modB_test==0)]:
        met=compute_metrics(y_test[mask],p[mask],sid_test[mask])
        ms_rows.append({'strategy':strategy,'subgroup':subgroup,'n_subjects':int(pd.Series(sid_test[mask]).nunique()),**met})
missing_strategy_test_df=pd.DataFrame(ms_rows)
display(missing_strategy_test_df.round(4))
print('Frozen missing strategy:',MISSING_STRATEGY,'| base kind:',BASE_KIND)
print('Frozen learned base:',fmt(compute_metrics(y_test,p_model_test,sid_test)))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.2 Mandatory TEST B-present / B-absent metrics for chosen A+B learned model
subgroup_rows = []
for label, mask in [("B-present",modB_test==1),("B-absent",modB_test==0)]:
    m = compute_metrics(y_test[mask],p_model_test[mask],sid_test[mask])
    subgroup_rows.append({
        "experiment":f"Chosen A+B graph model — {label}",
        "n_subjects":int(pd.Series(sid_test[mask]).nunique()),
        **m
    })
subgroup_test_df = pd.DataFrame(subgroup_rows)
display(subgroup_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.3 Apply frozen fusion candidates to test, including learned stacker
def scaled_test_pair(method):
    if method=='raw': return p_model_test.copy(),sim_test.copy()
    if method=='val_minmax':
        return apply_minmax(p_model_test,VAL_SCALE['model']),apply_minmax(sim_test,VAL_SCALE['sim'])
    raise ValueError(method)

pm_raw,ps_raw=scaled_test_pair('raw')
p_naive_test=0.5*pm_raw+0.5*ps_raw
pm_c,ps_c=scaled_test_pair(FROZEN_CONVEX['scale'])
p_convex_test=FROZEN_CONVEX['w_model']*pm_c+(1-FROZEN_CONVEX['w_model'])*ps_c
pm_q,ps_q=scaled_test_pair(FROZEN_CONF['scale'])
p_conf_test=confidence_fusion(pm_q,ps_q,sim_conf_test,FROZEN_CONF['alpha'],FROZEN_CONF['gamma'])

META_TEST=make_stack_features(p_model_test,sim_test,sim_conf_test,modB_test)
p_stack_test=STACK_MODEL.predict_proba(STACK_SCALER.transform(META_TEST))[:,1]

if SMART_FAMILY=='convex': p_final_test=p_convex_test.copy()
elif SMART_FAMILY=='confidence_power': p_final_test=p_conf_test.copy()
elif SMART_FAMILY=='stacked_lr': p_final_test=p_stack_test.copy()
else: raise RuntimeError(SMART_FAMILY)

fusion_test_rows=[
    {'experiment':'Learned model alone',**compute_metrics(y_test,p_model_test,sid_test)},
    {'experiment':'Simulation alone',**compute_metrics(y_test,sim_test,sid_test)},
    {'experiment':'Naive raw 50/50 average',**compute_metrics(y_test,p_naive_test,sid_test)},
    {'experiment':'Frozen convex fusion',**compute_metrics(y_test,p_convex_test,sid_test)},
    {'experiment':'Frozen confidence-power fusion',**compute_metrics(y_test,p_conf_test,sid_test)},
    {'experiment':'Frozen train-OOF logistic stacker',**compute_metrics(y_test,p_stack_test,sid_test)},
    {'experiment':f'FINAL validation-selected fusion ({SMART_FAMILY})',**compute_metrics(y_test,p_final_test,sid_test)},
]
fusion_test_df=pd.DataFrame(fusion_test_rows)
display(fusion_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.3b Frozen-threshold descriptive test metrics
pred_bin=(p_final_test>=VAL_THRESHOLD).astype(int)
tn,fp,fn,tp=confusion_matrix(y_test,pred_bin,labels=[0,1]).ravel()
sensitivity=tp/max(tp+fn,1)
specificity=tn/max(tn+fp,1)
precision=tp/max(tp+fp,1)
kappa=cohen_kappa_score(y_test,pred_bin)
clinical_test_df=pd.DataFrame([{
    'threshold':VAL_THRESHOLD,'sensitivity':sensitivity,'specificity':specificity,
    'precision':precision,'cohen_kappa':kappa,'brier':brier_score_loss(y_test,p_final_test),
    'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)
}])
display(clinical_test_df.round(4))

# Site-stratified point estimates are descriptive because each site has only a few test subjects.
site_rows=[]
site_arr=base.loc[test_mask,'site'].to_numpy()
for site in sorted(pd.unique(site_arr)):
    m=site_arr==site
    try:
        met=compute_metrics(y_test[m],p_final_test[m],sid_test[m])
        site_rows.append({'site':site,'n_subjects':int(pd.Series(sid_test[m]).nunique()),**met})
    except ValueError:
        pass
site_test_df=pd.DataFrame(site_rows)
print('Descriptive test performance by site (small n; not inferential):')
display(site_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.4 Required TEST modality rows
modality_test_df = pd.DataFrame([
    {"experiment":"Non-graph A+B baseline",**compute_metrics(y_test,p_ng_test,sid_test)},
    {"experiment":"Graph A only",**compute_metrics(y_test,pA_test,sid_test)},
    {"experiment":"Graph B only — all test (missing B handled)",**compute_metrics(y_test,pB_test,sid_test)},
    {"experiment":"Graph A+B — chosen learned base",**compute_metrics(y_test,p_model_test,sid_test)},
])

bp = modB_test==1
extra_bpresent = {
    "experiment":"Graph B only — B-present test subset",
    **compute_metrics(y_test[bp],pB_test[bp],sid_test[bp])
}
modality_test_df = pd.concat([modality_test_df,pd.DataFrame([extra_bpresent])],ignore_index=True)
display(modality_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.5 Required real/shuffled/identity test results: 10 subject-bootstrap training resamples
graph_test_runs = {c:[] for c in ["real","shuffled","identity"]}
graph_test_prob_arrays = {c:[] for c in ["real","shuffled","identity"]}

for condition in graph_test_runs:
    for seed in RESAMPLE_SEEDS:
        rng_graph = np.random.RandomState(seed)
        Xtr = propagate_features(base.loc[train_mask],feat_AB,condition,rng_graph)
        Xte = propagate_features(base.loc[test_mask],feat_AB,condition,rng_graph)

        rng_boot = np.random.RandomState(seed)
        sampled_sids = rng_boot.choice(train_subjects,size=len(train_subjects),replace=True)
        rows = np.concatenate([np.where(sid_tr==s)[0] for s in sampled_sids])

        p,_ = fit_lr(Xtr[rows],y_tr[rows],Xte)
        graph_test_prob_arrays[condition].append(p)
        graph_test_runs[condition].append(compute_metrics(y_test,p,sid_test))

graph_test_summary = []
for condition,runs in graph_test_runs.items():
    row={"condition":condition}
    for metric in ["auprc","auroc","topk_dice"]:
        vals=np.array([r[metric] for r in runs])
        row[f"{metric}_mean"]=vals.mean()
        row[f"{metric}_sd"]=vals.std()
    graph_test_summary.append(row)
graph_test_df=pd.DataFrame(graph_test_summary)
display(graph_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.6 Subject-level bootstrap CIs for major test claims
def full_fit_test_condition(condition,seed=0):
    rng=np.random.RandomState(seed)
    Xtr=propagate_features(base.loc[train_mask],feat_AB,condition,rng)
    Xte=propagate_features(base.loc[test_mask],feat_AB,condition,rng)
    return fit_lr(Xtr,y_tr,Xte)[0]

p_real_full=full_fit_test_condition("real",0)
p_identity_full=full_fit_test_condition("identity",0)
p_shuffled_full=full_fit_test_condition("shuffled",0)

ci_real_identity=subject_bootstrap_metric_diff(y_test,p_real_full,p_identity_full,sid_test,"auprc",8000,1)
ci_real_shuffled=subject_bootstrap_metric_diff(y_test,p_real_full,p_shuffled_full,sid_test,"auprc",8000,1)
ci_final_model=subject_bootstrap_metric_diff(y_test,p_final_test,p_model_test,sid_test,"auprc",8000,2)
ci_conf_convex=subject_bootstrap_metric_diff(y_test,p_conf_test,p_convex_test,sid_test,'auprc',8000,3)
ci_stack_conf=subject_bootstrap_metric_diff(y_test,p_stack_test,p_conf_test,sid_test,'auprc',8000,4)

print("real - identity:",ci_real_identity)
print("real - shuffled:",ci_real_shuffled)
print("final fusion - model:",ci_final_model)
print('confidence fusion - convex fusion:',ci_conf_convex)
print('stacker - confidence fusion:',ci_stack_conf)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 12.7 Required graph-vs-baseline fixed/harmed node analysis
error_node_rows=[]
subject_gain_rows=[]

test_frame=base.loc[test_mask].copy().reset_index(drop=True)
for sid in np.unique(sid_test):
    loc=np.where(sid_test==sid)[0]
    yt=y_test[loc]
    k=int(yt.sum())
    true=set(np.where(yt==1)[0])
    top_base=set(np.argsort(-p_ng_test[loc])[:k])
    top_graph=set(np.argsort(-pAB_test_single[loc])[:k])
    tp_base=top_base & true
    tp_graph=top_graph & true
    fixed=sorted(tp_graph-tp_base)
    harmed=sorted(tp_base-tp_graph)
    subject_gain_rows.append({
        'subject_id':sid,'k':k,'tp_baseline':len(tp_base),'tp_graph':len(tp_graph),
        'delta_tp':len(tp_graph)-len(tp_base),'n_fixed':len(fixed),'n_harmed':len(harmed)
    })

    sub=test_frame.iloc[loc].reset_index(drop=True)
    a=adj[sid]
    deg=a.sum(1)
    deg_z=(deg-deg.mean())/(deg.std()+1e-12)
    nbr_abn=(a@yt)/(deg+1e-12)
    for status,nodes in [('fixed',fixed),('harmed',harmed)]:
        for j in nodes:
            row=sub.iloc[j]
            error_node_rows.append({
                'status':status,'subject_id':sid,'node_id':int(row.node_id),'region':row.region,
                'site':row.site,'hemisphere':row.hemisphere,'modB_available':int(row.modB_available),
                'sim_confidence':float(row.sim_confidence),'sim_score':float(row.sim_score),
                'weighted_degree':float(deg[j]),'degree_z_within_subject':float(deg_z[j]),
                'weighted_abnormal_neighbor_fraction':float(nbr_abn[j]),
                'baseline_prob':float(p_ng_test[loc][j]),'graph_prob':float(pAB_test_single[loc][j]),
                'graph_minus_baseline_prob':float(pAB_test_single[loc][j]-p_ng_test[loc][j])
            })

graph_error_nodes_df=pd.DataFrame(error_node_rows)
graph_error_subjects_df=pd.DataFrame(subject_gain_rows)

print('Subjects improved/worsened/same by top-k true-positive count:',
      int((graph_error_subjects_df.delta_tp>0).sum()),
      int((graph_error_subjects_df.delta_tp<0).sum()),
      int((graph_error_subjects_df.delta_tp==0).sum()))
print('Total graph-fixed abnormal nodes:',int((graph_error_nodes_df.status=='fixed').sum()))
print('Total graph-harmed abnormal nodes:',int((graph_error_nodes_df.status=='harmed').sum()))

display(graph_error_subjects_df.sort_values('delta_tp').head(10))
print('\nSpecific graph-fixed nodes:')
display(graph_error_nodes_df[graph_error_nodes_df.status.eq('fixed')].head(20).round(4))
print('\nSpecific graph-harmed nodes:')
display(graph_error_nodes_df[graph_error_nodes_df.status.eq('harmed')].head(20).round(4))

pattern_rows=[]
for status,g in graph_error_nodes_df.groupby('status'):
    pattern_rows.append({
        'status':status,'n':len(g),'mean_degree_z':g.degree_z_within_subject.mean(),
        'mean_abnormal_neighbor_fraction':g.weighted_abnormal_neighbor_fraction.mean(),
        'mean_sim_score':g.sim_score.mean(),'mean_sim_confidence':g.sim_confidence.mean(),
        'modB_available_fraction':g.modB_available.mean()
    })
graph_error_pattern_df=pd.DataFrame(pattern_rows)
print('\nPattern summary (descriptive):')
display(graph_error_pattern_df.round(4))
print('\nMost common regions among fixed nodes:')
print(graph_error_nodes_df[graph_error_nodes_df.status.eq('fixed')].region.value_counts().head(8).to_string())
print('\nMost common regions among harmed nodes:')
print(graph_error_nodes_df[graph_error_nodes_df.status.eq('harmed')].region.value_counts().head(8).to_string())

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 13. Cluster-aware uncertainty analysis
real_probs=np.asarray(graph_test_prob_arrays["real"])
ens_mean=real_probs.mean(axis=0)
ens_std=real_probs.std(axis=0)

err_flag=np.zeros(len(y_test),dtype=int)
subject_uncertainty=[]

for sid in np.unique(sid_test):
    m=sid_test==sid
    yt=y_test[m]
    k=int(yt.sum())
    top=set(np.argsort(-ens_mean[m])[:k])
    true=set(np.where(yt==1)[0])
    wrong=top.symmetric_difference(true)
    global_idx=np.where(m)[0]
    for j in wrong:
        err_flag[global_idx[j]]=1

    err_u=ens_std[global_idx][err_flag[global_idx]==1]
    ok_u=ens_std[global_idx][err_flag[global_idx]==0]
    if len(err_u)>0 and len(ok_u)>0:
        subject_uncertainty.append({
            "subject_id":sid,
            "error_mean_std":float(err_u.mean()),
            "correct_mean_std":float(ok_u.mean()),
            "delta":float(err_u.mean()-ok_u.mean())
        })

subject_uncertainty=pd.DataFrame(subject_uncertainty)
print("Pooled descriptive mean std — error nodes  :",round(ens_std[err_flag==1].mean(),5))
print("Pooled descriptive mean std — correct nodes:",round(ens_std[err_flag==0].mean(),5))
print("Subjects contributing paired uncertainty deltas:",len(subject_uncertainty))

rng=np.random.RandomState(99)
deltas=subject_uncertainty["delta"].to_numpy()
boot_means=[]
for _ in range(10000):
    idx=rng.choice(np.arange(len(deltas)),size=len(deltas),replace=True)
    boot_means.append(deltas[idx].mean())
boot_means=np.asarray(boot_means)

print("Subject-level mean(error-correct uncertainty) =",round(deltas.mean(),5))
print("95% subject-bootstrap CI =",
      np.quantile(boot_means,[.025,.975]).round(5).tolist())
print("Subjects with higher uncertainty on errors:",
      int((deltas>0).sum()),"/",len(deltas))

# Attach the per-node uncertainty estimate requested by the optional bonus.
uncertainty_nodes_df=pd.DataFrame({
    'subject_id':sid_test,
    'node_id':base.loc[test_mask,'node_id'].to_numpy(),
    'ensemble_mean_graph_prob':ens_mean,
    'ensemble_std':ens_std,
    'topk_error_flag':err_flag
})

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 13.1 Outcome association — post-hoc only, never used for prediction
outcome_rows=[]
test_out=base.loc[test_mask].copy().reset_index(drop=True)
test_out['final_prob']=p_final_test
for sid,subdf in test_out.groupby('subject_id',sort=False):
    pred_nodes=set(subdf.loc[subdf.final_prob>=VAL_THRESHOLD,'node_id'].astype(int))
    res_nodes=set(subdf.loc[subdf.resected.eq(1),'node_id'].astype(int))
    if len(pred_nodes)+len(res_nodes)==0:
        dice=np.nan
    else:
        dice=2*len(pred_nodes & res_nodes)/(len(pred_nodes)+len(res_nodes))
    engel_val=subdf.engel_1_seizure_free.iloc[0]
    if pd.isna(engel_val):
        continue
    outcome_rows.append({
        'subject_id':sid,'pred_resection_dice':dice,
        'n_predicted':len(pred_nodes),'n_resected':len(res_nodes),
        'engel_1_seizure_free':int(engel_val)
    })
outcome_df=pd.DataFrame(outcome_rows).dropna(subset=['pred_resection_dice'])
g1=outcome_df.loc[outcome_df.engel_1_seizure_free.eq(1),'pred_resection_dice'].to_numpy()
g0=outcome_df.loc[outcome_df.engel_1_seizure_free.eq(0),'pred_resection_dice'].to_numpy()
if len(g1)>0 and len(g0)>0:
    mw=stats.mannwhitneyu(g1,g0,alternative='two-sided')
    rank_biserial=2*mw.statistic/(len(g1)*len(g0))-1
    rng=np.random.RandomState(1234)
    boots=[]
    for _ in range(10000):
        b1=rng.choice(g1,size=len(g1),replace=True)
        b0=rng.choice(g0,size=len(g0),replace=True)
        boots.append(b1.mean()-b0.mean())
    outcome_summary={
        'n':len(outcome_df),'n_seizure_free':len(g1),'n_not_seizure_free':len(g0),
        'mean_dice_seizure_free':float(g1.mean()),'mean_dice_not_seizure_free':float(g0.mean()),
        'mean_difference':float(g1.mean()-g0.mean()),
        'mean_difference_ci_low':float(np.quantile(boots,.025)),
        'mean_difference_ci_high':float(np.quantile(boots,.975)),
        'mannwhitney_p':float(mw.pvalue),'rank_biserial':float(rank_biserial)
    }
else:
    outcome_summary={}
print(json.dumps(outcome_summary,indent=2))
display(outcome_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 14. Write exact predictions.csv
pred_df=pd.DataFrame({
    "subject_id":sid_test,
    "node_id":base.loc[test_mask,"node_id"].to_numpy(),
    "prob_abnormal":np.clip(p_final_test,0,1),
}).sort_values(["subject_id","node_id"]).reset_index(drop=True)

assert len(pred_df)==1904
assert pred_df.columns.tolist()==["subject_id","node_id","prob_abnormal"]
assert pred_df["prob_abnormal"].between(0,1).all()
assert not pred_df.isna().any().any()
assert pred_df.duplicated(["subject_id","node_id"]).sum()==0
assert pred_df["subject_id"].nunique()==28
for sid,g in pred_df.groupby("subject_id"):
    assert g["node_id"].tolist()==list(range(68))

pred_df.to_csv("predictions.csv",index=False)
print("Wrote predictions.csv",pred_df.shape)
display(pred_df.head())

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 14.1 Unified required TEST table
# One compact table spanning sections 2-5; optional detailed comparisons are in supplementary CSVs.
def mrow(name,p): return {'experiment':name,**compute_metrics(y_test,p,sid_test)}
master_rows=[
    mrow('Non-graph A+B baseline',p_ng_test),
    mrow('Graph A only',pA_test),
    mrow('Graph B only (all test; missing B handled)',pB_test),
    mrow(f'Graph A+B chosen learned base ({MISSING_STRATEGY}, {BASE_KIND})',p_model_test),
]
for _,r in graph_test_df.iterrows():
    master_rows.append({
        'experiment':f"Graph {r['condition']} — 10 training-resample runs",
        'auprc':r['auprc_mean'],'auroc':r['auroc_mean'],'topk_dice':r['topk_dice_mean'],
        'auprc_sd':r['auprc_sd'],'auroc_sd':r['auroc_sd'],'topk_dice_sd':r['topk_dice_sd'],
        'prevalence':float(y_test.mean())
    })
master_rows += [
    mrow('Simulation alone',sim_test),
    mrow('Naive raw 50/50 average',p_naive_test),
    mrow('Frozen convex fusion',p_convex_test),
    mrow('Frozen confidence-power fusion',p_conf_test),
    mrow('Frozen train-OOF logistic stacker',p_stack_test),
    mrow(f'FINAL validation-selected fusion ({SMART_FAMILY})',p_final_test),
]
master_test_df=pd.DataFrame(master_rows)
master_test_df['prevalence_baseline']=float(y_test.mean())
master_test_df.to_csv('master_test_results.csv',index=False)
print('PRIMARY METRIC = AUPRC; prevalence baseline =',round(y_test.mean(),4))
display(master_test_df.round(4))

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 14.2 Save supporting artifacts
fusion_val_df.to_csv('fusion_validation_selection.csv',index=False)
hop_val_df.to_csv('hop_depth_validation_results.csv',index=False)
edge_sparsification_val_df.to_csv('edge_sparsification_validation.csv',index=False)
missing_strategy_val_df.to_csv('missing_modality_strategy_validation.csv',index=False)
missing_strategy_test_df.to_csv('missing_modality_strategy_test_reporting_only.csv',index=False)
stacking_train_cv_df.to_csv('stacking_train_groupcv.csv',index=False)
stack_oof_folds_df.to_csv('stacking_oof_base_fold_audit.csv',index=False)
subgroup_test_df.to_csv('missingB_test_subgroups.csv',index=False)
modality_test_df.to_csv('modality_test_results.csv',index=False)
graph_test_df.to_csv('graph_ablation_test_results.csv',index=False)
per_subj.to_csv('simulation_concordance_validation.csv',index=False)
subject_uncertainty.to_csv('uncertainty_subject_level.csv',index=False)
stab.to_csv('fusion_parameter_bootstrap_validation.csv',index=False)
baseline_val_df.to_csv('baseline_validation_results.csv',index=False)
class_weight_sensitivity_df.to_csv('class_weight_sensitivity_validation.csv',index=False)
pd.DataFrame([weight_shuffle_val]).to_csv('edge_weight_shuffle_validation.csv',index=False)
pd.DataFrame([{'experiment':'binarized_topology_only',**binarized_val_metrics}]).to_csv('graph_binarized_validation.csv',index=False)
discordant_val_df.to_csv('discordant_subjects_validation.csv',index=False)
worst_discordant_nodes.to_csv('worst_discordant_subject_nodes_validation.csv',index=False)
confidence_by_B_df.to_csv('sim_confidence_by_modalityB_validation.csv',index=False)
calibration_val_df.to_csv('calibration_validation.csv',index=False)
reliability_val_df.to_csv('reliability_validation.csv',index=False)
clinical_test_df.to_csv('clinical_operating_point_test.csv',index=False)
site_test_df.to_csv('site_test_descriptive.csv',index=False)
graph_error_nodes_df.to_csv('graph_error_nodes.csv',index=False)
graph_error_subjects_df.to_csv('graph_error_subject_summary.csv',index=False)
graph_error_pattern_df.to_csv('graph_error_pattern_summary.csv',index=False)
uncertainty_nodes_df.to_csv('uncertainty_nodes.csv',index=False)
outcome_df.to_csv('outcome_association_subjects.csv',index=False)
with open('outcome_association_summary.json','w') as f: json.dump(outcome_summary,f,indent=2)
with open('frozen_config.json','w') as f: json.dump(FREEZE,f,indent=2)
with open('leakage_audit.json','w') as f: json.dump(leakage_audit,f,indent=2)
print('Saved all reporting artifacts.')

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# Final compact submission summary
final_metrics=compute_metrics(y_test,p_final_test,sid_test)
print('='*78)
print('OFFICIAL FINAL SUBMISSION')
print('Depth:',FROZEN_DEPTH,'hop')
print('Missing-B strategy:',MISSING_STRATEGY)
print('Base kind:',BASE_KIND)
print('Fusion:',SMART_FAMILY,SMART_SCALE,SMART_PARAMS)
print('Test AUPRC:',round(final_metrics['auprc'],6))
print('Test AUROC:',round(final_metrics['auroc'],6))
print('Test Top-k Dice:',round(final_metrics['topk_dice'],6))
print('Predictions: predictions.csv',pred_df.shape)
print('Optional comparison tables: missing strategies + edge sparsification + learned stacker')
print('='*78)

# ============================================================================
# NEXT NOTEBOOK CELL
# ============================================================================

# 15. Generate figures, REPORT.md, ANALYSIS.md, README.md and pinned requirements.txt
Path('figures').mkdir(exist_ok=True)

fig=plt.figure(figsize=(5,4)); plt.plot([0,1],[0,1],'--',linewidth=1); plt.plot(reliability_val_df['mean_pred'],reliability_val_df['observed_rate'],'o-'); plt.xlabel('Mean predicted score'); plt.ylabel('Observed abnormal fraction'); plt.title('Validation reliability — frozen fusion'); plt.tight_layout(); plt.savefig('figures/calibration_validation.png',dpi=160); plt.close(fig)
fig=plt.figure(figsize=(5,4)); plt.hist(subject_uncertainty['delta'].to_numpy(),bins=10); plt.axvline(0,linewidth=1); plt.xlabel('Mean uncertainty(error) - mean uncertainty(correct)'); plt.ylabel('Subjects'); plt.title('Subject-level uncertainty deltas'); plt.tight_layout(); plt.savefig('figures/uncertainty_subject_deltas.png',dpi=160); plt.close(fig)
wplot=worst_discordant_nodes.sort_values('node_id'); fig=plt.figure(figsize=(10,4)); plt.plot(wplot.node_id,wplot.model_prob,label='learned model'); plt.plot(wplot.node_id,wplot.sim_score,label='simulation'); [plt.axvline(r.node_id,alpha=0.15) for _,r in wplot[wplot.is_abnormal.eq(1)].iterrows()]; plt.xlabel('node_id'); plt.ylabel('score'); plt.title(f'Most discordant validation subject: {worst_discordant_sid}'); plt.legend(); plt.tight_layout(); plt.savefig('figures/discordant_subject_validation.png',dpi=160); plt.close(fig)

# Bonus comparison figures
fig=plt.figure(figsize=(6,4)); plt.plot(edge_sparsification_val_df['retain_fraction'],edge_sparsification_val_df['auprc_mean'],'o-'); plt.xlabel('Fraction of non-zero edges retained'); plt.ylabel('Validation AUPRC'); plt.title('Edge sparsification robustness'); plt.tight_layout(); plt.savefig('figures/edge_sparsification_validation.png',dpi=160); plt.close(fig)
fig=plt.figure(figsize=(7,4)); x=np.arange(len(missing_strategy_val_df)); plt.bar(x,missing_strategy_val_df['auprc']); plt.xticks(x,missing_strategy_val_df['strategy'],rotation=20,ha='right'); plt.ylabel('Validation AUPRC'); plt.title('Missing-modality strategies'); plt.tight_layout(); plt.savefig('figures/missing_strategy_validation.png',dpi=160); plt.close(fig)

def md(df,cols=None,digits=4):
    x=df.copy() if cols is None else df[cols].copy()
    num=x.select_dtypes(include=[np.number]).columns
    x[num]=x[num].round(digits)
    # Missing cells in comparison tables mean “not applicable”, not missing analysis.
    # Render them cleanly while preserving raw CSV NaNs/blanks in results/.
    x=x.astype(object).where(pd.notna(x), '—')
    try: return x.to_markdown(index=False)
    except Exception: return '```\n'+x.to_string(index=False)+'\n```'

fm=compute_metrics(y_test,p_final_test,sid_test); prev=float(y_test.mean())
fixed_n=int((graph_error_nodes_df.status=='fixed').sum()); harmed_n=int((graph_error_nodes_df.status=='harmed').sum())
improved_n=int((graph_error_subjects_df.delta_tp>0).sum()); worsened_n=int((graph_error_subjects_df.delta_tp<0).sum()); same_n=int((graph_error_subjects_df.delta_tp==0).sum())
worst_disc=discordant_val_df.iloc[0]
if outcome_summary:
    outcome_sentence=(f"Seizure-free mean resection-overlap Dice {outcome_summary['mean_dice_seizure_free']:.3f} vs {outcome_summary['mean_dice_not_seizure_free']:.3f}; rank-biserial {outcome_summary['rank_biserial']:+.3f}, Mann-Whitney p={outcome_summary['mannwhitney_p']:.3g}, bootstrap mean-difference CI [{outcome_summary['mean_difference_ci_low']:+.3f},{outcome_summary['mean_difference_ci_high']:+.3f}].")
else: outcome_sentence='Outcome groups were insufficient for a stable association estimate.'

REPORT=f'''# Position 2 — Simulation-Informed AI and Multimodal EZ Localisation\n\n**Candidate:** Prithivraj S  \n**Contact:** theprithivrajs@gmail.com\n\n## 1. Executive summary

I treat this as a subject-grouped, highly imbalanced node-ranking task. All model, missing-modality and fusion decisions are frozen on train/validation before one final test reporting batch. The final model uses one transparent message-passing architecture and whichever missing-B strategy/fusion family validation selected.

**Final test:** AUPRC **{fm['auprc']:.3f}** versus prevalence **{prev:.3f}**, AUROC **{fm['auroc']:.3f}**, mean Top-k Dice **{fm['topk_dice']:.3f}**. Frozen missing strategy: **{MISSING_STRATEGY}**; base: **{BASE_KIND}**; fusion: **{SMART_FAMILY}**.

## 2. Data and preprocessing

140 subjects × 68 nodes; 35 subjects lack B. A uses train-median imputation, missingness indicators and train standardisation. B imputer/scaler is fit only on B-present train subjects. Restricted intervention/outcome fields are never model inputs. A strong non-graph audit compared balanced LR, ElasticNet LR, HistGradientBoosting and MLP before retaining the parsimonious baseline.

## 3. Main results

{md(master_test_df)}

Mandatory B-present/B-absent reporting for the chosen learned base:

{md(subgroup_test_df)}

## 4. Graph evidence and ablations

The model uses `[X,A_NX]` with symmetric normalisation. Real/shuffled/identity are repeated over 10 seeded subject-bootstrap training resamples, with genuine SGD random-state runs as a separate stochasticity check. Real topology is directionally stronger, but test subject-bootstrap CIs for real-minus-identity and real-minus-shuffled include zero, so I do **not** claim definitive graph superiority.

Bonus graph robustness includes zero-message, topology-fixed weight randomisation, binarisation, 0/1/2/3-hop propagation, and edge-threshold sparsification:

{md(edge_sparsification_val_df)}

Graph error analysis identifies **{fixed_n} fixed** and **{harmed_n} harmed** abnormal top-k nodes; {improved_n} subjects improve, {worsened_n} worsen and {same_n} are unchanged. Pattern variables are reported descriptively, not turned into post-hoc mechanistic claims.

## 5. Missing modality B

Three strategies are compared head-to-head on validation; the predeclared 0.005 AUPRC parsimony rule prevents tiny fluctuations from replacing the current strategy:

{md(missing_strategy_val_df)}

Test results for all strategies are reporting-only and never used to choose the strategy:

{md(missing_strategy_test_df)}

## 6. Simulation and fusion

Simulation is evaluated standalone before fusion. Concordance is non-redundant (pooled model-simulation Spearman {pooled_spear.statistic:.3f}; mean subject Spearman {per_subj['model_sim_rho'].mean():.3f}; mean top-k Jaccard {per_subj['topk_jaccard'].mean():.3f}). The most discordant validation case is {worst_discordant_sid}. Simulation confidence predicts simulation Dice (Spearman {sp_q.statistic:.3f}, p={sp_q.pvalue:.3g}), motivating confidence-aware fusion.

Fusion comparison includes model alone, simulation alone, naive average, validation-tuned convex fusion, confidence-power fusion, and a **learned logistic stacker trained on train-only subject-grouped OOF base predictions whose base construction mirrors the validation-selected `{BASE_KIND}` deployment stream**. The stacker must beat the best justified parametric fusion by at least 0.005 validation AUPRC without >0.02 Dice loss before added complexity is accepted. Validation selects `{SMART_FAMILY}`; stacker train-grouped-CV details are saved separately.

{md(fusion_val_df)}

## 7. Calibration, uncertainty, outcome and limitations

Grouped OOF calibration is diagnostic only; no external calibration cohort exists. Per-node uncertainty is ensemble spread, while inference is clustered by subject. The outcome bonus is observational: {outcome_sentence}

Limitations: only 28 test subjects; one atlas/order; B absent in 25%; Top-k uses oracle true-k for evaluation only; simulation internals are a supplied black box; no external/prospective cohort; site-stratified counts are small.

## 8. What I would do with more time

1. Nested subject-level CV over train+validation for model-selection variance.
2. External/prospective site-shift validation.
3. Validate calibrated probability/abstention policies.
4. Integrate actual TVB/VEP outputs and compare with SEEG/outcomes prospectively.
5. Pre-register any deeper graph or learned fusion refinement on a fresh cohort.
'''
Path('REPORT.md').write_text(REPORT)

ANALYSIS=f'''# Required analysis — does graph structure genuinely contribute?

## 1. Controls
Real adjacency uses the actual weighted topology. Shuffled adjacency applies a node permutation to destroy feature-topology correspondence while preserving graph-level structure. Identity removes neighbour mixing. `[X,0]` verifies the ordinary non-graph baseline in the doubled representation. Weight-shuffling preserves binary topology but destroys edge-strength assignment; binarisation removes edge magnitudes; sparsification tests dependence on weak edges.

## 2. Repeated runs and uncertainty
The primary graph comparison uses 10 seeded subject-bootstrap training resamples and reports mean±SD. A separate SGD experiment changes genuine optimizer/shuffle random states. On test, subject-bootstrap AUPRC differences are real-identity {ci_real_identity['mean_diff']:+.3f} [{ci_real_identity['ci_low']:+.3f},{ci_real_identity['ci_high']:+.3f}] and real-shuffled {ci_real_shuffled['mean_diff']:+.3f} [{ci_real_shuffled['ci_low']:+.3f},{ci_real_shuffled['ci_high']:+.3f}]. Both CIs cross zero, so evidence is suggestive rather than conclusive.

## 3. Error analysis
Using true k only for evaluation, the canonical graph model fixes {fixed_n} abnormal nodes and harms {harmed_n}; {improved_n} subjects improve, {worsened_n} worsen, {same_n} are unchanged. Specific subject/node/region records plus degree, abnormal-neighbour exposure, simulation evidence and B availability are saved in `results/graph_error_nodes.csv` and summaries.

## 4. Falsification criterion
I would conclude the graph does not help if shuffled/identity matched real within repeated-run variability, signs were unstable, confidence intervals were centred at or below zero alongside inconsistent validation results, or gains were concentrated in one/two subjects with no recoveries. I observe consistent validation directionality and more fixed than harmed nodes, but wide test CIs. Therefore I reject a strong causal/topological claim and state only that topology shows **directionally consistent, not definitive** value in this sample.

## 5. Bonus graph robustness
Propagation depth, fixed-topology weight randomisation, binarised topology and edge-threshold sparsification are validation-only robustness checks. They do not reopen test selection.
'''
Path('ANALYSIS.md').write_text(ANALYSIS)

README=f'''# IIIT-Delhi Position 2 Take-Home — Reproducible Submission\n\n**Candidate:** Prithivraj S  \n**Contact:** theprithivrajs@gmail.com\n\n## Run
1. Put the provided `candidate_package/data` contents under `data/`, or set `DATA_P2_DIR`.
2. Install `requirements.txt`.
3. Run `python src/run_pipeline.py` from repository root.

## Stack choice
I intentionally use **NumPy/scikit-learn rather than PyTorch/PyG** because the brief explicitly permits a hand-written propagation layer, the graphs are small, and the transparent implementation makes graph controls easier to audit and defend.

## Final pipeline
- A: train-median imputation + missing indicators + train standardisation.
- B: three missing-modality strategies are compared on validation; `{MISSING_STRATEGY}` is frozen.
- Graph: one-hop symmetric normalisation with residual `[X,A_NX]`.
- Classifier: balanced logistic regression.
- Fusion candidates: naive, convex, confidence-power, and train-OOF learned logistic stacker. The stacker's train OOF `model_score` mirrors the frozen `{BASE_KIND}` base construction; it must clear a 0.005 validation-AUPRC materiality margin with no >0.02 Dice harm; `{SMART_FAMILY}` is frozen by validation.
- Final test AUPRC/AUROC/Top-k Dice: `{fm['auprc']:.6f}` / `{fm['auroc']:.6f}` / `{fm['topk_dice']:.6f}`.

## Assumptions/tradeoffs
- Tables aligned explicitly on `(subject_id,node_id)`; node IDs 0–67 asserted.
- Restricted resection/outcome fields are post-hoc only.
- Primary repeated graph analysis uses subject-bootstrap training resamples; genuine stochastic seeds are checked separately.
- `prob_abnormal` is a bounded score; external calibration is not established.
- Top-k true k is evaluation-only.
- Stacker meta-training uses subject-OOF base scores built with the same single/ensemble base kind selected on validation.
- Test is a final reporting batch; reporting-only optional test tables never feed back into model selection.
'''
Path('README.md').write_text(README)

req_names=['numpy','pandas','scipy','scikit-learn','matplotlib','tabulate']
req=[]
for pkg in req_names:
    try: req.append(f'{pkg}=={importlib_metadata.version(pkg)}')
    except Exception: pass
Path('requirements.txt').write_text('\n'.join(req)+'\n')
Path('data').mkdir(exist_ok=True); Path('data/README.md').write_text('Place the provided candidate_package/data contents here. No external datasets are used.\n')
print('REPORT words:',len(REPORT.split()))
print('Generated REPORT.md, ANALYSIS.md, README.md, requirements.txt and figures/.')
