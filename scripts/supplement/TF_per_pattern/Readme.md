



```python

import pickle
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycisTopic.topic_binarization import binarize_topics


def topic_name_to_model_index_organoid(t: str) -> int:
    cell_type = t.split("_Topic")[0]
    topic = int(t.split("_")[-1])
    if cell_type == "neuron":
        return topic - 1
    elif cell_type == "progenitor":
        return topic - 1 + 25
    elif cell_type == "neural_crest":
        return topic - 1 + 55
    else:
        raise ValueError(f"{t} unknown.")


def topic_name_to_model_index_embryo(t) -> int:
    cell_type = t.split("_Topic")[0]
    topic = int(t.split("_")[-1])
    if cell_type == "neuron":
        return topic - 1
    elif cell_type == "progenitor":
        return topic - 1 + 30
    elif cell_type == "neural_crest":
        return topic - 1 + 90
    else:
        raise ValueError(f"{t} unknown.")


def model_index_to_topic_name_organoid(i: int) -> str:
    if i >= 0 and i < 25:
        cell_type = "neuron"
        topic_number = i + 1
    elif i >= 25 and i < 55:
        cell_type = "progenitor"
        topic_number = i + 1 - 25
    elif i >= 55 and i < 65:
        cell_type = "neural_crest"
        topic_number = i + 1 - 55
    elif i >= 65 and i < 75:
        cell_type = "pluripotent"
        index_to_topic_mapping = {
            65: 4,
            66: 5,
            67: 12,
            68: 18,
            69: 20,
            70: 23,
            71: 35,
            72: 37,
            73: 45,
            74: 47,
        }
        topic_number = index_to_topic_mapping[i]
    else:
        raise ValueError("Unknown index")
    return f"{cell_type}_Topic_{topic_number}"


def model_index_to_topic_name_embryo(i: int) -> str:
    if i >= 0 and i < 30:
        cell_type = "neuron"
        topic_number = i + 1
    elif i >= 30 and i < 90:
        cell_type = "progenitor"
        topic_number = i + 1 - 30
    elif i >= 90 and i < 120:
        cell_type = "neural_crest"
        topic_number = i + 1 - 90
    else:
        raise ValueError("Unknown index")
    return f"{cell_type}_Topic_{topic_number}"


def atac_to_rna(l):
    bc, sample_id = l.strip().split("-1", 1)
    sample_id = sample_id.split("___")[-1]
    return bc + "-1" + "-" + sample_id_to_num[sample_id]


def allign_patterns_of_cluster_for_topic(
    pattern_to_topic_to_grad: dict[str, dict[int, np.ndarray]],
    pattern_metadata: pd.DataFrame,
    cluster_id: int,
    topic: int,
):
    P = []
    O = []
    cluster_patterns = pattern_metadata.query(
        "hier_cluster == @cluster_id"
    ).index.to_list()
    for pattern_name in cluster_patterns:
        pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
        ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
            pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
        ]
        pattern_grads_ic = [p[ic_start:ic_end] for p in pattern_grads]
        pattern_ohs_ic = [o[ic_start:ic_end] for o in pattern_ohs]
        if is_rc_to_root:
            pattern_grads_ic = [p[::-1, ::-1] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[::-1, ::-1] for o in pattern_ohs_ic]
        if offset_to_root > 0:
            pattern_grads_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), p_ic])
                for p_ic in pattern_grads_ic
            ]
            pattern_ohs_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), o_ic])
                for o_ic in pattern_ohs_ic
            ]
        elif offset_to_root < 0:
            pattern_grads_ic = [p[abs(offset_to_root) :, :] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[abs(offset_to_root) :, :] for o in pattern_ohs_ic]
        P.extend(pattern_grads_ic)
        O.extend(pattern_ohs_ic)
    max_len = max([p.shape[0] for p in P])
    P = [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P]
    O = [np.concatenate([o, np.zeros((max_len - o.shape[0], 4))]) for o in O]
    return np.array(P), np.array(O)



lambert_human_tfs = pd.read_csv(
  "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
  index_col = 0
)

adata_organoid = sc.read_h5ad("../../data_prep_new/organoid_data/RNA/GEX_adata.h5ad")

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

organoid_atac_cell_names = []
with open("../../data_prep_new/organoid_data/ATAC/cell_names.txt") as f:
    for l in f:
        bc, sample_id = l.strip().split("-1", 1)
        sample_id = sample_id.split("___")[-1]
        organoid_atac_cell_names.append(bc + "-1" + "-" + sample_id_to_num[sample_id])

adata_organoid = adata_organoid[
    list(set(adata_organoid.obs_names) & set(organoid_atac_cell_names))
].copy()

# load embryo data and subset for ATAC cells
adata_embryo = sc.read_h5ad("../../data_prep_new/embryo_data/RNA/adata_raw_filtered.h5ad")
cell_data_embryo = pd.read_csv(
    "../../data_prep_new/embryo_data/RNA/cell_data.csv", index_col=0
)

adata_embryo = adata_embryo[cell_data_embryo.index].copy()
adata_embryo.obs = cell_data_embryo.loc[adata_embryo.obs_names]

embryo_atac_cell_names = []
with open("../../data_prep_new/embryo_data/ATAC/cell_names.txt") as f:
    for l in f:
        bc, sample_id = l.strip().split("-1-", 1)
        embryo_atac_cell_names.append(bc + "-1___" + sample_id)

adata_embryo = adata_embryo[
    list(set(adata_embryo.obs_names) & set(embryo_atac_cell_names))
].copy()


###############################################################################################################
#                                             SIMPLE PREPROCESSING                                            #
###############################################################################################################

print("PREPROCEESING DATA")

def normalize(a):
    a.layers["count"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)


def dim_red_2d(a, n_pcs=None):
    sc.tl.pca(a)
    k = "X_pca_harmony" if "X_pca_harmony" in a.obsm else "X_pca"
    sc.pp.neighbors(a, use_rep=k, n_pcs=n_pcs)
    sc.tl.umap(a)


normalize(adata_organoid)
normalize(adata_embryo)

sc.external.pp.harmony_integrate(
    adata_organoid,
    "experimental_batch_line",
    max_iter_harmony=15,
)

dim_red_2d(adata_organoid)
dim_red_2d(adata_embryo)

###############################################################################################################
#                                             ENSURE COMMON ANNOTATION                                        #
###############################################################################################################

print("SETTING COMMON ANNOTATION")

embryo_common_annotation_d = {
    "Endo": "endo",
    "Mesoderm": "mesenchyme",
    "NC": "neural crest",
    "NP": "neuronal progenitor",
    "Neuron": {
        0: "neuron",
        1: "neuron",
        4: "neuron",
        6: "peripheral neuron",
        10: "peripheral neuron",
    },
    "OP": "otic placode",
    "PVM/MIC": "PVM/MIC",
}

def label_embryo(annotation, cluster):
    new_a = embryo_common_annotation_d[annotation]
    if isinstance(new_a, str):
        return new_a
    else:
        return new_a[cluster]


adata_embryo.obs["COMMON_ANNOTATION"] = [
    label_embryo(a, c)
    for a, c in zip(adata_embryo.obs["Classes"], adata_embryo.obs["leiden_0.1"])
]

organoid_common_annotation_d = {
    "Mesoderm": "mesenchyme",
    "Neural crest": "neural crest",
    "Peripheral neuron": "peripheral neuron",
    "Pluritpotent stem cell": "pluripotent stem cell",
    "Progenitor": "neuronal progenitor",
    "Neuron": "neuron",
}

adata_organoid.obs["COMMON_ANNOTATION"] = [
    organoid_common_annotation_d[a] for a in adata_organoid.obs["Annotation_Type_1"]
]


###############################################################################################################
#                                             subset progenitor                                               #
###############################################################################################################

print("GETTING PROGENITORS")


def dim_red_3d(a):
    sc.tl.pca(a)
    k = "X_pca_harmony" if "X_pca_harmony" in a.obsm else "X_pca"
    sc.pp.neighbors(a, use_rep=k)
    sc.tl.umap(a, n_components=3)


adata_embryo_progenitor = adata_embryo[
    adata_embryo.obs.COMMON_ANNOTATION == "neuronal progenitor"
].copy()

dim_red_3d(adata_embryo_progenitor)

adata_organoid_progenitor = adata_organoid[
    adata_organoid.obs.Annotation_progenitor_step_2 == "Progenitor"
].copy()

dim_red_2d(adata_organoid_progenitor)

exp_embryo_progenitor = adata_embryo_progenitor.to_df(layer="log_cpm")
exp_organoid_progenitor = adata_organoid_progenitor.to_df(layer="log_cpm")


cell_topic_bin_organoid = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)

embryo_progenitor_cell_topic = pd.read_table(
    "../../data_prep_new/embryo_data/ATAC/progenitor_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_progenitor_cell_topic.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in embryo_progenitor_cell_topic
]

embryo_progenitor_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_progenitor_cell_topic.index
]


cells, scores, thresholds = binarize_topics(
    embryo_progenitor_cell_topic.to_numpy(), embryo_progenitor_cell_topic.index, "li"
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(
        np.repeat("progenitor", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)


## 
# progenitor
##

organoid_progenitor_topics_to_show = [33, 38, 36, 54, 48]
embryo_progenitor_topics_to_show = [34, 38, 79, 88, 58]

avg_expr_organoid_per_topic_progenitor = {}
for topic in organoid_progenitor_topics_to_show:
    cell_names = list(
        set(
            [
                atac_to_rna(x)
                for x in cell_topic_bin_organoid.set_index("topic_name")
                .loc[
                    model_index_to_topic_name_organoid(topic - 1)
                    .replace("progenitor", "")
                    .replace("_", ""),
                    "cell_barcode",
                ]
                .values
            ]
        )
        & set(adata_organoid_progenitor.obs_names)
    )
    avg_expr_organoid_per_topic_progenitor[topic] = exp_organoid_progenitor.loc[cell_names].mean()

avg_expr_embryo_per_topic_progenitor = {}
for topic in embryo_progenitor_topics_to_show:
    cell_names = list(
        set(
            cell_topic_bin_embryo.set_index("topic_name")
            .loc[
                model_index_to_topic_name_embryo(topic - 1)
                .replace("progenitor", "")
                .replace("_", ""),
                "cell_barcode",
            ]
            .values
        )
        & set(adata_embryo_progenitor.obs_names)
    )
    avg_expr_embryo_per_topic_progenitor[topic] = exp_embryo_progenitor.loc[cell_names].mean()

expr_per_topic_organoid_progenitor = pd.DataFrame(avg_expr_organoid_per_topic_progenitor)
expr_per_topic_embryo_progenitor = pd.DataFrame(avg_expr_embryo_per_topic_progenitor)

pattern_to_topic_to_grad_organoid_progenitor = pickle.load(
  open("../../figure_3/pattern_to_topic_to_grad_organoid.pkl", "rb")
)

pattern_to_topic_to_grad_embryo_progenitor = pickle.load(
  open("../../figure_3/pattern_to_topic_to_grad_embryo.pkl", "rb")
)

progenitor_pattern_metadata = pd.read_table(
  "../../figure_3/draft/motif_metadata.tsv",
  index_col = 0
)

cluster_to_topic_to_avg_pattern_organoid_progenitor = {}
for cluster in set(progenitor_pattern_metadata["hier_cluster"]):
    cluster_to_topic_to_avg_pattern_organoid_progenitor[cluster] = {}
    for topic in organoid_progenitor_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid_progenitor,
            pattern_metadata=progenitor_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        cluster_to_topic_to_avg_pattern_organoid_progenitor[cluster][topic] = (P * O).mean(0)

cluster_to_topic_to_avg_pattern_embryo_progenitor = {}
for cluster in set(progenitor_pattern_metadata["hier_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo_progenitor[cluster] = {}
    for topic in embryo_progenitor_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo_progenitor,
            pattern_metadata=progenitor_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        cluster_to_topic_to_avg_pattern_embryo_progenitor[cluster][topic] = (P * O).mean(0)

cluster_to_topic_to_avg_pattern_organoid_progenitor = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid_progenitor).applymap(np.mean).T

cluster_to_topic_to_avg_pattern_embryo_progenitor = pd.DataFrame(cluster_to_topic_to_avg_pattern_embryo_progenitor).applymap(np.mean).T


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def tau(x):
  x_hat = x / max(x)
  return (1 - x_hat).sum() / (len(x) - 1)

selected_clusters = [6, 7, 12, 15, 10, 18, 1, 19]

cluster_to_DBD = {
  6:  ["Forkhead"],
  7:  ["TEA"],
  12: ["RFX"],
  15: ["Homeodomain"],
  10: ["Homeodomain; Paired box", "Paired box"],
  18: ["C2H2 ZF"],
  1:  ["HMG/Sox"],
  19: ["C2H2 ZF; Homeodomain"]
}

candidate_TFs_per_cluster_progenitor = []
for cluster in selected_clusters:
  dbd = cluster_to_DBD[cluster]
  assert len(set(lambert_human_tfs["DBD"]) & set(dbd)) == len(dbd)
  candidate_TFs = lambert_human_tfs.query("DBD in @dbd")["HGNC symbol"].to_list()
  corr_coef_organoid = [
    pearsonr(
        expr_per_topic_organoid_progenitor.loc[g, organoid_progenitor_topics_to_show],
        cluster_to_topic_to_avg_pattern_organoid_progenitor.loc[cluster, organoid_progenitor_topics_to_show]
    ).statistic if g in expr_per_topic_organoid_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  corr_coef_embryo = [
    pearsonr(
        expr_per_topic_embryo_progenitor.loc[g, embryo_progenitor_topics_to_show],
        cluster_to_topic_to_avg_pattern_embryo_progenitor.loc[cluster, embryo_progenitor_topics_to_show]
    ).statistic if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  gini_organoid = [
      gini_coefficient(expr_per_topic_organoid_progenitor.loc[g])
      if g in expr_per_topic_organoid_progenitor.index else np.nan
      for g in candidate_TFs
  ]
  gini_embryo = [
    gini_coefficient(expr_per_topic_embryo_progenitor.loc[g])
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  tau_organoid = [
    tau(expr_per_topic_organoid_progenitor.loc[g])
    if g in expr_per_topic_organoid_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  tau_embryo = [
    tau(expr_per_topic_embryo_progenitor.loc[g])
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_organoid = [
    expr_per_topic_organoid_progenitor.loc[g].mean()
    if g in expr_per_topic_organoid_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_organoid = [
    expr_per_topic_organoid_progenitor.loc[g].max()
    if g in expr_per_topic_organoid_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_embryo = [
    expr_per_topic_embryo_progenitor.loc[g].mean()
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_embryo = [
    expr_per_topic_embryo_progenitor.loc[g].max()
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  data = pd.DataFrame(
    dict(
      cluster=np.repeat(cluster, len(candidate_TFs)),
      candidate_TFs=candidate_TFs,
      corr_coef_organoid=corr_coef_organoid,
      corr_coef_embryo=corr_coef_embryo,
      gini_organoid=gini_organoid,
      gini_embryo=gini_embryo,
      tau_organoid=tau_organoid,
      tau_embryo=tau_embryo,
      avg_exp_organoid=avg_exp_organoid,
      avg_exp_embryo=avg_exp_embryo,
      max_exp_organoid=max_exp_organoid,
      max_exp_embryo=max_exp_embryo
    )
  )
  candidate_TFs_per_cluster_progenitor.append(data)

candidate_TFs_per_cluster_progenitor = pd.concat(candidate_TFs_per_cluster_progenitor)

from adjustText import adjust_text
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

for cluster in tqdm(selected_clusters):
  fig, axs = plt.subplots(ncols = 2, figsize = (8, 4), sharex = True, sharey = True)
  x = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["corr_coef_organoid"]
  y = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["max_exp_organoid"]
  n = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["tau_organoid"] * 30 + 1
  axs[0].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[0], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  x = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["corr_coef_embryo"]
  y = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["max_exp_embryo"]
  n = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["tau_embryo"] * 30 + 1
  axs[1].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[1], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  for ax in axs:
    _ = ax.set_xlabel("Pearson correlation coef.")
    _ = ax.set_ylabel("Max exp")
    ax.grid(True)
    ax.set_axisbelow(True)
  _ = axs[0].set_title("Organoid")
  _ = axs[1].set_title("Embryo")
  fig.tight_layout()
  fig.savefig(f"progenitor_{cluster}.pdf")
  plt.close(fig)

##
# Neural Crest
##

selected_clusters = [3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2]

neural_crest_topics_organoid = [62, 60, 65, 59, 58]
neural_crest_topics_embryo = [103, 105, 94, 91]

adata_embryo_neural_crest = adata_embryo[
    [
        x in ["neural crest", "peripheral neuron", "mesenchyme", "otic placode"]
        for x in adata_embryo.obs.COMMON_ANNOTATION
    ]
].copy()

dim_red_2d(adata_embryo_neural_crest, n_pcs=10)

adata_organoid_neural_crest = adata_organoid[
    [
        x in ["neural crest", "peripheral neuron", "mesenchyme", "otic placode"]
        for x in adata_organoid.obs.COMMON_ANNOTATION
    ]
].copy()

adata_organoid_neural_crest.obsm["X_pca"] = adata_organoid_neural_crest.obsm[
    "X_pca_harmony"
].copy()
del adata_organoid_neural_crest.obsm["X_pca_harmony"]

dim_red_2d(adata_organoid_neural_crest, n_pcs=10)

cell_topic_bin_organoid = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)


def rename_organoid_atac_cell(l):
    bc, sample_id = l.strip().split("-1", 1)
    sample_id = sample_id.split("___")[-1]
    return bc + "-1" + "-" + sample_id_to_num[sample_id]



cell_topic_bin_organoid.cell_barcode = [
    rename_organoid_atac_cell(x) for x in cell_topic_bin_organoid.cell_barcode
]

exp_neural_crest_organoid = adata_organoid_neural_crest.to_df(layer="log_cpm")

exp_per_topic_neural_crest_organoid = pd.DataFrame(
    index=[
        model_index_to_topic_name_organoid(t - 1)
        .replace("neural_crest_", "")
        .replace("_", "")
        for t in neural_crest_topics_organoid
    ],
    columns=exp_neural_crest_organoid.columns,
)

for topic in tqdm(exp_per_topic_neural_crest_organoid.index):
    cells = list(
        set(exp_neural_crest_organoid.index)
        & set(
            cell_topic_bin_organoid.query(
                "group == 'neural_crest' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_neural_crest_organoid.loc[topic] = exp_neural_crest_organoid.loc[cells].mean()

embryo_neural_crest_cell_topic = pd.read_table(
    "../../data_prep_new/embryo_data/ATAC/neural_crest_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_neural_crest_cell_topic.columns = [
    f"neural_crest_Topic_{c.replace('Topic', '')}"
    for c in embryo_neural_crest_cell_topic
]

embryo_neural_crest_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_neural_crest_cell_topic.index
]

cells, scores, thresholds = binarize_topics(
    embryo_neural_crest_cell_topic.to_numpy(),
    embryo_neural_crest_cell_topic.index,
    "li",
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(
        np.repeat("neural_crest", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_neural_crest_embryo = adata_embryo_neural_crest.to_df(layer="log_cpm")

exp_per_topic_neural_crest_embryo = pd.DataFrame(
    index=[
        model_index_to_topic_name_embryo(t - 1)
        .replace("neural_crest_", "")
        .replace("_", "")
        for t in neural_crest_topics_embryo
    ],
    columns=exp_neural_crest_embryo.columns,
)

for topic in tqdm(exp_per_topic_neural_crest_embryo.index):
    cells = list(
        set(exp_neural_crest_embryo.index)
        & set(
            cell_topic_bin_embryo.query(
                "group == 'neural_crest' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_neural_crest_embryo.loc[topic] = exp_neural_crest_embryo.loc[cells].mean()

exp_per_topic_neural_crest_organoid = exp_per_topic_neural_crest_organoid.T

exp_per_topic_neural_crest_embryo = exp_per_topic_neural_crest_embryo.T



pattern_to_topic_to_grad_organoid_neural_crest = pickle.load(
  open("../../figure_5/pattern_to_topic_to_grad_organoid.pkl", "rb")
)

pattern_to_topic_to_grad_embryo_neural_crest = pickle.load(
  open("../../figure_5/pattern_to_topic_to_grad_embryo.pkl", "rb")
)

neural_crest_pattern_metadata = pd.read_table(
  "../../figure_5/draft/pattern_metadata.tsv",
  index_col = 0
)

def allign_patterns_of_cluster_for_topic(
    pattern_to_topic_to_grad: dict[str, dict[int, np.ndarray]],
    pattern_metadata: pd.DataFrame,
    cluster_id: int,
    topic: int,
):
    P = []
    O = []
    cluster_patterns = pattern_metadata.query(
        "cluster_sub_cluster == @cluster_id"
    ).index.to_list()
    for pattern_name in cluster_patterns:
        pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
        ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
            pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
        ]
        pattern_grads_ic = [p[ic_start:ic_end] for p in pattern_grads]
        pattern_ohs_ic = [o[ic_start:ic_end] for o in pattern_ohs]
        if is_rc_to_root:
            pattern_grads_ic = [p[::-1, ::-1] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[::-1, ::-1] for o in pattern_ohs_ic]
        if offset_to_root > 0:
            pattern_grads_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), p_ic])
                for p_ic in pattern_grads_ic
            ]
            pattern_ohs_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), o_ic])
                for o_ic in pattern_ohs_ic
            ]
        elif offset_to_root < 0:
            pattern_grads_ic = [p[abs(offset_to_root) :, :] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[abs(offset_to_root) :, :] for o in pattern_ohs_ic]
        P.extend(pattern_grads_ic)
        O.extend(pattern_ohs_ic)
    max_len = max([p.shape[0] for p in P])
    P = [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P]
    O = [np.concatenate([o, np.zeros((max_len - o.shape[0], 4))]) for o in O]
    return np.array(P), np.array(O)


neural_crest_pattern_metadata["ic_start"] = 0
neural_crest_pattern_metadata["ic_stop"] = 30


def ic(ppm, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)


def ic_trim(ic, min_v: float) -> tuple[int, int]:
    delta = np.where(np.diff((ic > min_v) * 1))[0]
    if len(delta) == 0:
        return 0, 0
    start_index = min(delta)
    end_index = max(delta)
    return start_index, end_index + 1


cluster_to_topic_to_avg_pattern_organoid_neural_crest = {}
for cluster in set(neural_crest_pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_organoid_neural_crest[cluster] = {}
    for topic in organoid_neural_crest_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid_neural_crest,
            pattern_metadata=neural_crest_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_organoid_neural_crest[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_embryo_neural_crest = {}
for cluster in set(neural_crest_pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo_neural_crest[cluster] = {}
    for topic in embryo_neural_crest_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo_neural_crest,
            pattern_metadata=neural_crest_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_embryo_neural_crest[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_organoid_neural_crest = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid_neural_crest).applymap(np.mean).T

cluster_to_topic_to_avg_pattern_embryo_neural_crest = pd.DataFrame(cluster_to_topic_to_avg_pattern_embryo_neural_crest).applymap(np.mean).T

cluster_to_DBD = {
  3.0: ["TEA"],
  13.1: ["Grainyhead"],
  9.2: ["C2H2 ZF; Homeodomain"],
  14.0: ["RFX"],
  11.1: ["Nuclear receptor"],
  9.1: ["AP-2"],
  10.2: ["C2H2 ZF"],
  2.2:  ["HMG/Sox"],
  2.1:  ["Forkhead"],
  13.2: ["bHLH"]
}

exp_per_topic_neural_crest_organoid.columns = [
  topic_name_to_model_index_organoid("neural_crest_" + x.replace("Topic", "Topic_")) + 1
  for x in exp_per_topic_neural_crest_organoid.columns
]

exp_per_topic_neural_crest_embryo.columns = [
  topic_name_to_model_index_embryo("neural_crest_" + x.replace("Topic", "Topic_")) + 1
  for x in exp_per_topic_neural_crest_embryo.columns
]

exp_per_topic_neural_crest_organoid = exp_per_topic_neural_crest_organoid.astype(float)

exp_per_topic_neural_crest_embryo = exp_per_topic_neural_crest_embryo.astype(float)

candidate_TFs_per_cluster_neural_crest = []
for cluster in selected_clusters:
  dbd = cluster_to_DBD[cluster]
  assert len(set(lambert_human_tfs["DBD"]) & set(dbd)) == len(dbd)
  candidate_TFs = lambert_human_tfs.query("DBD in @dbd")["HGNC symbol"].to_list()
  corr_coef_organoid = [
    pearsonr(
        exp_per_topic_neural_crest_organoid.loc[g, organoid_neural_crest_topics_to_show],
        cluster_to_topic_to_avg_pattern_organoid_neural_crest.loc[cluster, organoid_neural_crest_topics_to_show]
    ).statistic if g in exp_per_topic_neural_crest_organoid.index else np.nan
    for g in candidate_TFs
  ]
  corr_coef_embryo = [
    pearsonr(
        exp_per_topic_neural_crest_embryo.loc[g, embryo_neural_crest_topics_to_show],
        cluster_to_topic_to_avg_pattern_embryo_neural_crest.loc[cluster, embryo_neural_crest_topics_to_show]
    ).statistic if g in exp_per_topic_neural_crest_embryo.index else np.nan
    for g in candidate_TFs
  ]
  gini_organoid = [
      gini_coefficient(exp_per_topic_neural_crest_organoid.loc[g])
      if g in exp_per_topic_neural_crest_organoid.index else np.nan
      for g in candidate_TFs
  ]
  gini_embryo = [
    gini_coefficient(exp_per_topic_neural_crest_embryo.loc[g])
    if g in exp_per_topic_neural_crest_embryo.index else np.nan
    for g in candidate_TFs
  ]
  tau_organoid = [
    tau(exp_per_topic_neural_crest_organoid.loc[g])
    if g in exp_per_topic_neural_crest_organoid.index else np.nan
    for g in candidate_TFs
  ]
  tau_embryo = [
    tau(exp_per_topic_neural_crest_embryo.loc[g])
    if g in exp_per_topic_neural_crest_embryo.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_organoid = [
    exp_per_topic_neural_crest_organoid.loc[g].mean()
    if g in exp_per_topic_neural_crest_organoid.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_organoid = [
    exp_per_topic_neural_crest_organoid.loc[g].max()
    if g in exp_per_topic_neural_crest_organoid.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_embryo = [
    exp_per_topic_neural_crest_embryo.loc[g].mean()
    if g in exp_per_topic_neural_crest_embryo.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_embryo = [
    exp_per_topic_neural_crest_embryo.loc[g].max()
    if g in exp_per_topic_neural_crest_embryo.index else np.nan
    for g in candidate_TFs
  ]
  data = pd.DataFrame(
    dict(
      cluster=np.repeat(cluster, len(candidate_TFs)),
      candidate_TFs=candidate_TFs,
      corr_coef_organoid=corr_coef_organoid,
      corr_coef_embryo=corr_coef_embryo,
      gini_organoid=gini_organoid,
      gini_embryo=gini_embryo,
      tau_organoid=tau_organoid,
      tau_embryo=tau_embryo,
      avg_exp_organoid=avg_exp_organoid,
      avg_exp_embryo=avg_exp_embryo,
      max_exp_organoid=max_exp_organoid,
      max_exp_embryo=max_exp_embryo
    )
  )
  candidate_TFs_per_cluster_neural_crest.append(data)

candidate_TFs_per_cluster_neural_crest = pd.concat(candidate_TFs_per_cluster_neural_crest)

for cluster in tqdm(selected_clusters):
  fig, axs = plt.subplots(ncols = 2, figsize = (8, 4), sharex = True, sharey = True)
  x = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["corr_coef_organoid"]
  y = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["max_exp_organoid"]
  n = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["tau_organoid"] * 30 + 1
  axs[0].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[0], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  x = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["corr_coef_embryo"]
  y = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["max_exp_embryo"]
  n = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_neural_crest.dropna().query("cluster == @cluster")["tau_embryo"] * 30 + 1
  axs[1].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[1], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  for ax in axs:
    _ = ax.set_xlabel("Pearson correlation coef.")
    _ = ax.set_ylabel("Max exp")
    ax.grid(True)
    ax.set_axisbelow(True)
  _ = axs[0].set_title("Organoid")
  _ = axs[1].set_title("Embryo")
  fig.tight_layout()
  fig.savefig(f"neural_crest_{cluster}.pdf")
  plt.close(fig)

##
# Neuron
##


adata_embryo_neuron = adata_embryo[
    [x in ["neuron"] for x in adata_embryo.obs.COMMON_ANNOTATION]
].copy()

dim_red_2d(adata_embryo_neuron, n_pcs=10)

adata_organoid_neuron = adata_organoid[
    [x in ["neuron"] for x in adata_organoid.obs.COMMON_ANNOTATION]
].copy()


dim_red_2d(adata_organoid_neuron)


organoid_neuron_topics_to_show = [6, 4, 23, 24, 13, 2]
embryo_neuron_topics_to_show = [10, 8, 13, 24, 18, 29]

neuron_topics_organoid = organoid_neuron_topics_to_show
neuron_topics_embryo = embryo_neuron_topics_to_show

cell_topic_bin_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)

cell_topic_bin_organoid.cell_barcode = [
    rename_organoid_atac_cell(x) for x in cell_topic_bin_organoid.cell_barcode
]

exp_neuron_organoid = adata_organoid_neuron.to_df(layer="log_cpm")

exp_per_topic_neuron_organoid = pd.DataFrame(
    index=[
        model_index_to_topic_name_organoid(t - 1)
        .replace("neuron_", "")
        .replace("_", "")
        for t in neuron_topics_organoid
    ],
    columns=exp_neuron_organoid.columns,
)

for topic in tqdm(exp_per_topic_neuron_organoid.index):
    cells = list(
        set(exp_neuron_organoid.index)
        & set(
            cell_topic_bin_organoid.query(
                "group == 'neuron' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_neuron_organoid.loc[topic] = exp_neuron_organoid.loc[cells].mean()

embryo_neuron_cell_topic = pd.read_table(
    "../../data_prep_new/embryo_data/ATAC/neuron_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_neuron_cell_topic.columns = [
    f"neuron_Topic_{c.replace('Topic', '')}" for c in embryo_neuron_cell_topic
]

embryo_neuron_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_neuron_cell_topic.index
]

#

cells, scores, thresholds = binarize_topics(
    embryo_neuron_cell_topic.to_numpy(),
    embryo_neuron_cell_topic.index,
    "li",
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(np.repeat("neuron", len(cells[topic_idx])))
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_neuron_embryo = adata_embryo_neuron.to_df(layer="log_cpm")

exp_per_topic_neuron_embryo = pd.DataFrame(
    index=[
        model_index_to_topic_name_embryo(t - 1).replace("neuron_", "").replace("_", "")
        for t in neuron_topics_embryo
    ],
    columns=exp_neuron_embryo.columns,
)

for topic in tqdm(exp_per_topic_neuron_embryo.index):
    cells = list(
        set(exp_neuron_embryo.index)
        & set(
            cell_topic_bin_embryo.query(
                "group == 'neuron' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_neuron_embryo.loc[topic] = exp_neuron_embryo.loc[cells].mean()


exp_per_topic_neuron_organoid = exp_per_topic_neuron_organoid.T

exp_per_topic_neuron_embryo = exp_per_topic_neuron_embryo.T



pattern_to_topic_to_grad_organoid_neuron = pickle.load(
  open("../../figure_6/pattern_to_topic_to_grad_organoid.pkl", "rb")
)

pattern_to_topic_to_grad_embryo_neuron = pickle.load(
  open("../../figure_6/pattern_to_topic_to_grad_embryo.pkl", "rb")
)

neuron_pattern_metadata = pd.read_table(
  "../../figure_6/draft/pattern_metadata.tsv",
  index_col = 0
)


neuron_pattern_metadata["ic_start"] = 0
neuron_pattern_metadata["ic_stop"] = 30

cluster_to_topic_to_avg_pattern_organoid_neuron = {}
for cluster in set(neuron_pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_organoid_neuron[cluster] = {}
    for topic in organoid_neuron_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid_neuron,
            pattern_metadata=neuron_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_organoid_neuron[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_embryo_neuron = {}
for cluster in set(neuron_pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo_neuron[cluster] = {}
    for topic in embryo_neuron_topics_to_show :
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo_neuron,
            pattern_metadata=neuron_pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_embryo_neuron[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_organoid_neuron = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid_neuron).applymap(np.mean).T

cluster_to_topic_to_avg_pattern_embryo_neuron = pd.DataFrame(cluster_to_topic_to_avg_pattern_embryo_neuron).applymap(np.mean).T


selected_clusters = [1.1, 2.1, 8.0, 2.2, 4.0, 5.2, 6.0, 7.3, 3.1, 7.5, 5.1]

cluster_to_DBD = {
    1.1: ["TEA"],
    1.3: ["Homeodomain; Paired box", "Paired box"],
    2.1: ["Forkhead"],
    2.2: ["HMG/Sox"],
    3.1: ["CUT; Homeodomain"],
    4.0: ["Nuclear receptor"],
    5.1: ["C2H2 ZF; Homeodomain"],
    5.2: ["bHLH"],
    6.0: ["EBF1"],
    7.3: ["Homeodomain"],
    7.5: ["GATA"],
    8.0: ["RFX"],
}

exp_per_topic_neuron_organoid.columns = [
  topic_name_to_model_index_organoid("neuron_" + x.replace("Topic", "Topic_")) + 1
  for x in exp_per_topic_neuron_organoid.columns
]

exp_per_topic_neuron_embryo.columns = [
  topic_name_to_model_index_embryo("neuron_" + x.replace("Topic", "Topic_")) + 1
  for x in exp_per_topic_neuron_embryo.columns
]

exp_per_topic_neuron_organoid = exp_per_topic_neuron_organoid.astype(float)

exp_per_topic_neuron_embryo = exp_per_topic_neuron_embryo.astype(float)

candidate_TFs_per_cluster_neuron = []
for cluster in selected_clusters:
  dbd = cluster_to_DBD[cluster]
  assert len(set(lambert_human_tfs["DBD"]) & set(dbd)) == len(dbd)
  candidate_TFs = lambert_human_tfs.query("DBD in @dbd")["HGNC symbol"].to_list()
  corr_coef_organoid = [
    pearsonr(
        exp_per_topic_neuron_organoid.loc[g, organoid_neuron_topics_to_show],
        cluster_to_topic_to_avg_pattern_organoid_neuron.loc[cluster, organoid_neuron_topics_to_show]
    ).statistic if g in exp_per_topic_neuron_organoid.index else np.nan
    for g in candidate_TFs
  ]
  corr_coef_embryo = [
    pearsonr(
        exp_per_topic_neuron_embryo.loc[g, embryo_neuron_topics_to_show],
        cluster_to_topic_to_avg_pattern_embryo_neuron.loc[cluster, embryo_neuron_topics_to_show]
    ).statistic if g in exp_per_topic_neuron_embryo.index else np.nan
    for g in candidate_TFs
  ]
  gini_organoid = [
      gini_coefficient(exp_per_topic_neuron_organoid.loc[g])
      if g in exp_per_topic_neuron_organoid.index else np.nan
      for g in candidate_TFs
  ]
  gini_embryo = [
    gini_coefficient(exp_per_topic_neuron_embryo.loc[g])
    if g in exp_per_topic_neuron_embryo.index else np.nan
    for g in candidate_TFs
  ]
  tau_organoid = [
    tau(exp_per_topic_neuron_organoid.loc[g])
    if g in exp_per_topic_neuron_organoid.index else np.nan
    for g in candidate_TFs
  ]
  tau_embryo = [
    tau(exp_per_topic_neuron_embryo.loc[g])
    if g in exp_per_topic_neuron_embryo.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_organoid = [
    exp_per_topic_neuron_organoid.loc[g].mean()
    if g in exp_per_topic_neuron_organoid.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_organoid = [
    exp_per_topic_neuron_organoid.loc[g].max()
    if g in exp_per_topic_neuron_organoid.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_embryo = [
    exp_per_topic_neuron_embryo.loc[g].mean()
    if g in exp_per_topic_neuron_embryo.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_embryo = [
    exp_per_topic_neuron_embryo.loc[g].max()
    if g in exp_per_topic_neuron_embryo.index else np.nan
    for g in candidate_TFs
  ]
  data = pd.DataFrame(
    dict(
      cluster=np.repeat(cluster, len(candidate_TFs)),
      candidate_TFs=candidate_TFs,
      corr_coef_organoid=corr_coef_organoid,
      corr_coef_embryo=corr_coef_embryo,
      gini_organoid=gini_organoid,
      gini_embryo=gini_embryo,
      tau_organoid=tau_organoid,
      tau_embryo=tau_embryo,
      avg_exp_organoid=avg_exp_organoid,
      avg_exp_embryo=avg_exp_embryo,
      max_exp_organoid=max_exp_organoid,
      max_exp_embryo=max_exp_embryo
    )
  )
  candidate_TFs_per_cluster_neuron.append(data)

candidate_TFs_per_cluster_neuron = pd.concat(candidate_TFs_per_cluster_neuron)

for cluster in tqdm(selected_clusters):
  fig, axs = plt.subplots(ncols = 2, figsize = (8, 4), sharex = True, sharey = True)
  x = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["corr_coef_organoid"]
  y = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["max_exp_organoid"]
  n = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["tau_organoid"] * 30 + 1
  axs[0].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[0].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[0], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  x = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["corr_coef_embryo"]
  y = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["max_exp_embryo"]
  n = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_neuron.dropna().query("cluster == @cluster")["tau_embryo"] * 30 + 1
  axs[1].scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [axs[1].text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = axs[1], x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  for ax in axs:
    _ = ax.set_xlabel("Pearson correlation coef.")
    _ = ax.set_ylabel("Max exp")
    ax.grid(True)
    ax.set_axisbelow(True)
  _ = axs[0].set_title("Organoid")
  _ = axs[1].set_title("Embryo")
  fig.tight_layout()
  fig.savefig(f"neuron_{cluster}.pdf")
  plt.close(fig)


candidate_TFs_per_cluster_progenitor.to_csv(
  "candidate_TFs_per_cluster_progenitor.tsv", sep = "\t", index = False
)

candidate_TFs_per_cluster_neural_crest.to_csv(
  "candidate_TFs_per_cluster_neural_crest.tsv", sep = "\t", index = False
)

candidate_TFs_per_cluster_neuron.to_csv(
  "candidate_TFs_per_cluster_neuron.tsv", sep = "\t", index = False
)

```

```python

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from dataclasses import dataclass
from typing import Self
import os
from pycisTopic.topic_binarization import binarize_topics
from scipy.stats import pearsonr


@dataclass
class Seqlet:
    contrib_scores: np.ndarray
    hypothetical_contrib_scores: np.ndarray
    ppm: np.ndarray
    start: int
    end: int
    region_name: str
    region_one_hot: np.ndarray
    is_revcomp: bool
    def __init__(
        self,
        p: h5py._hl.group.Group,
        seqlet_idx: int,
        ohs: np.ndarray,
        region_names: list[str],
    ):
        self.contrib_scores = p["contrib_scores"][seqlet_idx]
        self.hypothetical_contrib_scores = p["hypothetical_contribs"][seqlet_idx]
        self.ppm = p["sequence"][seqlet_idx]
        self.start = p["start"][seqlet_idx]
        self.end = p["end"][seqlet_idx]
        self.is_revcomp = p["is_revcomp"][seqlet_idx]
        region_idx = p["example_idx"][seqlet_idx]
        self.region_name = region_names[region_idx]
        self.region_one_hot = ohs[region_idx]
        if (
            not np.all(self.ppm == self.region_one_hot[self.start : self.end])
            and not self.is_revcomp
        ) or (
            not np.all(
                self.ppm[::-1, ::-1] == self.region_one_hot[self.start : self.end]
            )
            and self.is_revcomp
        ):
            raise ValueError(
                f"ppm does not match onehot\n"
                + f"region_idx\t{region_idx}\n"
                + f"start\t\t{self.start}\n"
                + f"end\t\t{self.end}\n"
                + f"is_revcomp\t{self.is_revcomp}\n"
                + f"{self.ppm.argmax(1)}\n"
                + f"{self.region_one_hot[self.start: self.end].argmax(1)}"
            )
    def __repr__(self):
        return f"Seqlet on {self.region_name} {self.start}:{self.end}"


@dataclass
class ModiscoPattern:
    contrib_scores: np.ndarray
    hypothetical_contrib_scores: np.ndarray
    ppm: np.ndarray
    is_pos: bool
    seqlets: list[Seqlet]
    subpatterns: list[Self] | None = None
    def __init__(
        self,
        p: h5py._hl.group.Group,
        is_pos: bool,
        ohs: np.ndarray,
        region_names: list[str],
    ):
        self.contrib_scores = p["contrib_scores"][:]
        self.hypothetical_contrib_scores = p["hypothetical_contribs"][:]
        self.ppm = p["sequence"][:]
        self.is_pos = is_pos
        self.seqlets = [
            Seqlet(p["seqlets"], i, ohs, region_names)
            for i in range(p["seqlets"]["n_seqlets"][0])
        ]
        self.subpatterns = [
            ModiscoPattern(p[sub], is_pos, ohs, region_names)
            for sub in p.keys()
            if sub.startswith("subpattern_")
        ]
    def __repr__(self):
        return f"ModiscoPattern with {len(self.seqlets)} seqlets"
    def ic(self, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
        return (
            self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)
        ).sum(1)
    def ic_trim(self, min_v: float, **kwargs) -> tuple[int, int]:
        delta = np.where(np.diff((self.ic(**kwargs) > min_v) * 1))[0]
        if len(delta) == 0:
            return 0, 0
        start_index = min(delta)
        end_index = max(delta)
        return start_index, end_index + 1


def load_pattern_from_modisco(filename, ohs, region_names):
    with h5py.File(filename) as f:
        for pos_neg in ["pos_patterns", "neg_patterns"]:
            if pos_neg not in f.keys():
                continue
            for pattern in f[pos_neg].keys():
                yield (
                    filename.split("/")[-1].rsplit(".", 1)[0]
                    + "_"
                    + pos_neg.split("_")[0]
                    + "_"
                    + pattern,
                    ModiscoPattern(
                        f[pos_neg][pattern],
                        pos_neg == "pos_patterns",
                        ohs,
                        region_names,
                    ),
                )

def model_index_to_topic_name_embryo(i: int) -> str:
    if i >= 0 and i < 30:
        cell_type = "neuron"
        topic_number = i + 1
    elif i >= 30 and i < 90:
        cell_type = "progenitor"
        topic_number = i + 1 - 30
    elif i >= 90 and i < 120:
        cell_type = "neural_crest"
        topic_number = i + 1 - 90
    else:
        raise ValueError("Unknown index")
    return f"{cell_type}_Topic_{topic_number}"




adata_embryo = sc.read_h5ad("../../data_prep_new/embryo_data/RNA/adata_raw_filtered.h5ad")
cell_data_embryo = pd.read_csv(
    "../../data_prep_new/embryo_data/RNA/cell_data.csv", index_col=0
)

adata_embryo = adata_embryo[cell_data_embryo.index].copy()
adata_embryo.obs = cell_data_embryo.loc[adata_embryo.obs_names]

embryo_atac_cell_names = []
with open("../../data_prep_new/embryo_data/ATAC/cell_names.txt") as f:
    for l in f:
        bc, sample_id = l.strip().split("-1-", 1)
        embryo_atac_cell_names.append(bc + "-1___" + sample_id)

adata_embryo = adata_embryo[
    list(set(adata_embryo.obs_names) & set(embryo_atac_cell_names))
].copy()


def normalize(a):
    a.layers["count"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)


def dim_red_2d(a, n_pcs=None):
    sc.tl.pca(a)
    k = "X_pca_harmony" if "X_pca_harmony" in a.obsm else "X_pca"
    sc.pp.neighbors(a, use_rep=k, n_pcs=n_pcs)
    sc.tl.umap(a)


normalize(adata_embryo)

dim_red_2d(adata_embryo)

embryo_common_annotation_d = {
    "Endo": "endo",
    "Mesoderm": "mesenchyme",
    "NC": "neural crest",
    "NP": "neuronal progenitor",
    "Neuron": {
        0: "neuron",
        1: "neuron",
        4: "neuron",
        6: "peripheral neuron",
        10: "peripheral neuron",
    },
    "OP": "otic placode",
    "PVM/MIC": "PVM/MIC",
}


def label_embryo(annotation, cluster):
    new_a = embryo_common_annotation_d[annotation]
    if isinstance(new_a, str):
        return new_a
    else:
        return new_a[cluster]


adata_embryo.obs["COMMON_ANNOTATION"] = [
    label_embryo(a, c)
    for a, c in zip(adata_embryo.obs["Classes"], adata_embryo.obs["leiden_0.1"])
]


def dim_red_3d(a):
    sc.tl.pca(a)
    k = "X_pca_harmony" if "X_pca_harmony" in a.obsm else "X_pca"
    sc.pp.neighbors(a, use_rep=k)
    sc.tl.umap(a, n_components=3)


adata_embryo_progenitor = adata_embryo[
    adata_embryo.obs.COMMON_ANNOTATION == "neuronal progenitor"
].copy()

dim_red_3d(adata_embryo_progenitor)

# load cell topic

embryo_progenitor_cell_topic = pd.read_table(
    "../../data_prep_new/embryo_data/ATAC/progenitor_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_progenitor_cell_topic.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in embryo_progenitor_cell_topic
]

embryo_progenitor_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_progenitor_cell_topic.index
]

ap_topics = np.array([31, 29, 1, 32, 40, 22, 41]) + 30

embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(ap_topics):
    ohs = np.load(os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz"))[
        "oh"
    ]
    region_names = np.load(
        os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz")
    )["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            embryo_dl_motif_dir,
            f"patterns_Topic_{topic}.hdf5",
        ),
        ohs=ohs,
        region_names=region_names,
    ):
        pattern_names_dl_embryo.append("embryo_" + name)
        patterns_dl_embryo.append(pattern)

pattern_topic = np.array([x.split("_")[3] for x in pattern_names_dl_embryo])

pattern_metadata = pd.read_table("../../figure_4/motif_metadata.tsv", index_col = 0)

patterns_to_cluster = [
  p for p, n in zip(patterns_dl_embryo, pattern_names_dl_embryo)
  if n in pattern_metadata.index
]

patterns_to_cluster_names = [
  n for n in pattern_names_dl_embryo if n in pattern_metadata.index
]

clusters = pattern_metadata.loc[patterns_to_cluster_names, "hier_cluster"].tolist()

clusters_to_show = np.array([1, 4, 8, 7, 6])

seqlet_count_per_cluster = np.zeros((10, len(ap_topics)))
for pattern, cluster, name in zip(
    patterns_to_cluster, clusters, patterns_to_cluster_names
):
    topic = int(name.split("_")[3])
    j = np.where(ap_topics == topic)[0][0]
    seqlet_count_per_cluster[cluster - 1, j] += len(pattern.seqlets)

seqlet_count_per_cluster = pd.DataFrame(
    np.log10(seqlet_count_per_cluster[np.array(clusters_to_show) - 1] + 1),
    columns = ap_topics,
    index = clusters_to_show)

cells, scores, thresholds = binarize_topics(
    embryo_progenitor_cell_topic.to_numpy(),
    embryo_progenitor_cell_topic.index,
    "li",
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(
        np.repeat("progenitor", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_progenitor_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")

exp_per_topic_progenitor_embryo = pd.DataFrame(
    index=[
        model_index_to_topic_name_embryo(t - 1)
        .replace("progenitor_", "")
        .replace("_", "")
        for t in ap_topics
    ],
    columns=exp_progenitor_embryo.columns,
)

for topic in tqdm(exp_per_topic_progenitor_embryo.index):
    cells = list(
        set(exp_progenitor_embryo.index)
        & set(
            cell_topic_bin_embryo.query(
                "group == 'progenitor' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_progenitor_embryo.loc[topic] = exp_progenitor_embryo.loc[cells].mean()


exp_per_topic_progenitor_embryo = exp_per_topic_progenitor_embryo.T

seqlet_count_per_cluster.columns = [
        model_index_to_topic_name_embryo(t - 1)
        .replace("progenitor_", "")
        .replace("_", "")
        for t in seqlet_count_per_cluster.columns
    ]


lambert_human_tfs = pd.read_csv(
  "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
  index_col = 0
)

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def tau(x):
  x_hat = x / max(x)
  return (1 - x_hat).sum() / (len(x) - 1)

expr_per_topic_embryo_progenitor = exp_per_topic_progenitor_embryo.astype(float)
cluster_to_topic_to_avg_pattern_embryo_progenitor = seqlet_count_per_cluster.astype(float)
embryo_progenitor_topics_to_show = expr_per_topic_embryo_progenitor.columns

candidate_TFs_per_cluster_progenitor = []
for cluster in clusters_to_show:
  dbd = ["Homeodomain", "Homeodomain; Paired box", "Paired box"]
  assert len(set(lambert_human_tfs["DBD"]) & set(dbd)) == len(dbd)
  candidate_TFs = lambert_human_tfs.query("DBD in @dbd")["HGNC symbol"].to_list()
  corr_coef_embryo = [
    pearsonr(
        expr_per_topic_embryo_progenitor.loc[g, embryo_progenitor_topics_to_show],
        cluster_to_topic_to_avg_pattern_embryo_progenitor.loc[cluster, embryo_progenitor_topics_to_show]
    ).statistic if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  gini_embryo = [
    gini_coefficient(expr_per_topic_embryo_progenitor.loc[g])
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  tau_embryo = [
    tau(expr_per_topic_embryo_progenitor.loc[g])
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  avg_exp_embryo = [
    expr_per_topic_embryo_progenitor.loc[g].mean()
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  max_exp_embryo = [
    expr_per_topic_embryo_progenitor.loc[g].max()
    if g in expr_per_topic_embryo_progenitor.index else np.nan
    for g in candidate_TFs
  ]
  data = pd.DataFrame(
    dict(
      cluster=np.repeat(cluster, len(candidate_TFs)),
      candidate_TFs=candidate_TFs,
      corr_coef_embryo=corr_coef_embryo,
      gini_embryo=gini_embryo,
      tau_embryo=tau_embryo,
      avg_exp_embryo=avg_exp_embryo,
      max_exp_embryo=max_exp_embryo
    )
  )
  candidate_TFs_per_cluster_progenitor.append(data)

candidate_TFs_per_cluster_progenitor = pd.concat(candidate_TFs_per_cluster_progenitor)

from adjustText import adjust_text
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

for cluster in tqdm(clusters_to_show):
  fig, ax = plt.subplots(figsize = (4, 4))
  x = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["corr_coef_embryo"]
  y = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["max_exp_embryo"]
  n = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["candidate_TFs"]
  s = candidate_TFs_per_cluster_progenitor.dropna().query("cluster == @cluster")["tau_embryo"] * 30 + 1
  ax.scatter(
    x = x,
    y = y,
    s = s,
    color = "black"
  )
  if len(x) < 20:
    texts = [ax.text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) ]
  else:
    texts = [ax.text(x =_x, y=_y, s=_n, alpha = 0.5) for _x, _y, _n in zip(x, y, n) if abs(_x) > 0.6 and _y > 0.1]
  adjust_text(texts, ax = ax, x=x, y=y, arrowprops=dict(arrowstyle="-", color="black"))
  _ = ax.set_xlabel("Pearson correlation coef.")
  _ = ax.set_ylabel("Max exp")
  ax.grid(True)
  ax.set_axisbelow(True)
  _ = ax.set_title("Embryo")
  fig.tight_layout()
  fig.savefig(f"progenitor_ap_{cluster}.pdf")
  plt.close(fig)

candidate_TFs_per_cluster_progenitor.to_csv(
  "candidate_TFs_per_cluster_progenitor_ap.tsv", sep = "\t", index = False
)


```
