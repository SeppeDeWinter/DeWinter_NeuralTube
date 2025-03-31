```python

import scanpy as sc
import matplotlib.pyplot as plt
import json
import matplotlib
from scipy import stats
from itertools import combinations
import numpy as np

adata_organoid = sc.read_h5ad("../../figure_1/adata_organoid.h5ad")
adata_embryo = sc.read_h5ad("../../figure_1/adata_embryo.h5ad")

matplotlib.rcParams['pdf.fonttype'] = 42

avg_exp_per_experiment = {
  "organoid": adata_organoid.to_df(layer = "log_cpm") \
    .groupby(adata_organoid.obs.COMMON_ANNOTATION).mean(),
  "embryo": adata_embryo.to_df(layer = "log_cpm") \
    .groupby(adata_embryo.obs.COMMON_ANNOTATION).mean(),
}

line_combos = list(combinations(avg_exp_per_experiment.keys(), 2))

annotations = list(set(avg_exp_per_experiment["organoid"].index) & set(avg_exp_per_experiment["embryo"].index))

genes = list(set(avg_exp_per_experiment["organoid"].columns) & set(avg_exp_per_experiment["embryo"].columns))

genes = list(
  set(sc.pp.highly_variable_genes(adata_organoid, layer = "log_cpm", inplace = False).query("highly_variable").index) & \
  set(sc.pp.highly_variable_genes(adata_embryo, layer = "log_cpm", inplace = False).query("highly_variable").index)
)

line_1 = "organoid"
line_2 = "embryo"

fig, axs = plt.subplots(
  ncols = len(annotations), nrows = len(annotations),
  figsize = (len(annotations) * 2, len(annotations) * 2 ),
  sharex = True, sharey = True
)
for i, cell_type_1 in enumerate(annotations):
  for j, cell_type_2 in enumerate(annotations):
    ax = axs[j, i]
    if i == 0:
      _ = ax.set_ylabel(f"{cell_type_2} {line_2[0:3]}")
    if j == 0:
      _ = ax.set_title(f"{cell_type_1} {line_1[0:3]}")
    _ = ax.scatter(
      x = avg_exp_per_experiment[line_1].loc[cell_type_1, genes],
      y = avg_exp_per_experiment[line_2].loc[cell_type_2, genes],
      color = "black" if cell_type_1 == cell_type_2 else "darkgray",
      s = 0.5
    )
    _ = ax.text(
      x = 0.1,
      y = 0.8,
      s = np.round(
        stats.pearsonr(
          avg_exp_per_experiment[line_1].loc[cell_type_1, genes],
          avg_exp_per_experiment[line_2].loc[cell_type_2, genes]
        ).statistic,
        2
      ),
      transform = ax.transAxes
    )
fig.tight_layout()
fig.savefig("organoid_embryo_corrs.png")
fig.savefig("organoid_embryo_corrs.pdf")
plt.close(fig)


```


```python


import scanpy as sc
import pandas as pd
from scipy import sparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams['pdf.fonttype'] = 42

topics_to_show = [
    "neural_crest_Topic_1",
    "neural_crest_Topic_3",
    "neural_crest_Topic_4",
    "neural_crest_Topic_5",
    "neural_crest_Topic_7",
    "neural_crest_Topic_9",
    "neural_crest_Topic_10",
    "neuron_Topic_1",
    "neuron_Topic_2",
    "neuron_Topic_3",
    "neuron_Topic_4",
    "neuron_Topic_6",
    "neuron_Topic_10",
    "neuron_Topic_11",
    "neuron_Topic_12",
    "neuron_Topic_13",
    "neuron_Topic_15",
    "neuron_Topic_16",
    "neuron_Topic_18",
    "neuron_Topic_19",
    "neuron_Topic_20",
    "neuron_Topic_21",
    "neuron_Topic_23",
    "neuron_Topic_24",
    "neuron_Topic_25",
    "progenitor_Topic_1",
    "progenitor_Topic_3",
    "progenitor_Topic_8",
    "progenitor_Topic_9",
    "progenitor_Topic_11",
    "progenitor_Topic_13",
    "progenitor_Topic_14",
    "progenitor_Topic_16",
    "progenitor_Topic_19",
    "progenitor_Topic_21",
    "progenitor_Topic_23",
    "progenitor_Topic_24",
    "progenitor_Topic_25",
    "progenitor_Topic_29",
    "progenitor_Topic_30",
]

topics_order = [
    "progenitor_Topic_23",
    "progenitor_Topic_30",
    "progenitor_Topic_3",
    "progenitor_Topic_1",
    "progenitor_Topic_11",
    "progenitor_Topic_9",
    # "progenitor_Topic_21",
    "progenitor_Topic_13",
    "progenitor_Topic_29",
    "progenitor_Topic_19",
    "progenitor_Topic_14",
    "progenitor_Topic_24",
    "progenitor_Topic_25",
    "progenitor_Topic_16",
    "progenitor_Topic_8",
    "neuron_Topic_12",
    # "neuron_Topic_19",
    "neuron_Topic_15",
    # "neuron_Topic_21",
    "neuron_Topic_6",
    # "neuron_Topic_3",
    "neuron_Topic_4",
    "neuron_Topic_23",
    # "neuron_Topic_10",
    "neuron_Topic_24",
    "neuron_Topic_11",
    # "neuron_Topic_20",
    # "neuron_Topic_18",
    # "neuron_Topic_2",
    "neuron_Topic_13",
    "neuron_Topic_25",
    # "neuron_Topic_16",
    "neural_crest_Topic_7",
    "neural_crest_Topic_5",
    "neural_crest_Topic_4",
    # "neural_crest_Topic_10",
    "neural_crest_Topic_9",
    # "neural_crest_Topic_1",
    "neural_crest_Topic_3",
]



adata_organoid = sc.read_h5ad("../../figure_1/adata_organoid.h5ad")

adata_embryo = sc.read_h5ad("../../figure_1/adata_embryo.h5ad")


region_topic_nc = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
    index_col=0,
)
region_topic_prog = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
    index_col=0,
)
region_topic_neu = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv", index_col=0
)

region_topic_nc.columns = [
    f"neural_crest_Topic_{c.replace('Topic', '')}" for c in region_topic_nc
]

region_topic_prog.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in region_topic_prog
]

region_topic_neu.columns = [
    f"neuron_Topic_{c.replace('Topic', '')}" for c in region_topic_neu
]

region_topic = pd.concat(
    [region_topic_nc, region_topic_prog, region_topic_neu], axis=1
).dropna()


embryo_bin_matrix = sparse.load_npz(
    os.path.join("../../data_prep_new/embryo_data/ATAC/embryo_atac_merged_bin_matrix.npz")
)

embryo_region_names = []
with open(
    os.path.join("../../data_prep_new/embryo_data/ATAC/embry_atac_region_names.txt")
) as f:
    for l in f:
        embryo_region_names.append(l.strip())

embryo_cell_names = []
with open(
    os.path.join("../../data_prep_new/embryo_data/ATAC/embry_atac_cell_names.tx")
) as f:
    for l in f:
        embryo_cell_names.append(l.strip())

common_regions = list(set(region_topic.index) & set(embryo_region_names))

X = pd.DataFrame(
    embryo_bin_matrix.todense(), index=embryo_region_names, columns=embryo_cell_names
).loc[common_regions]

embryo_cell_topic = region_topic.loc[common_regions].T @ X

embryo_cell_topic = embryo_cell_topic.T

def smooth_topics_distributions(
    topic_region_distributions: pd.DataFrame,
) -> pd.DataFrame:
    def smooth_topic_distribution(x: np.ndarray) -> np.ndarray:
       return x * (np.log(x + 1e-100) - np.sum(np.log(x + 1e-100)) / x.shape[0])
    smoothed_topic_region_distributions = pd.DataFrame(
        np.apply_along_axis(
            smooth_topic_distribution,
            1,
            topic_region_distributions.values,
        ),
        index=topic_region_distributions.index,
        columns=topic_region_distributions.columns,
    )
    return smoothed_topic_region_distributions

def bin_cell_topic_ntop(ct, ntop=1_000):
    topic_dist = pd.DataFrame(
        smooth_topics_distributions(ct.copy()), index=ct.index, columns=ct.columns
    )
    binarized_topics = {}
    for topic in topic_dist.columns:
        l = np.asarray(topic_dist[topic])
        binarized_topics[topic] = pd.DataFrame(
            topic_dist.sort_values(topic, ascending=False)[topic].head(ntop)
        )
    return binarized_topics

embryo_bin_cell_topic = bin_cell_topic_ntop(embryo_cell_topic)

embryo_cell_topic = []
for topic in embryo_bin_cell_topic:
    tmp = (
        embryo_bin_cell_topic[topic]
        .copy()
        .reset_index()
        .rename({"index": "bc", topic: "prob"}, axis=1)
    )
    tmp["topic_grp"] = topic
    embryo_cell_topic.append(tmp)

embryo_cell_topic = pd.concat(embryo_cell_topic)

bc_to_topic = (
    embryo_cell_topic.pivot(index="bc", columns="topic_grp", values="prob"
    )[topics_to_show]
    .fillna(0)
    .T.idxmax()
    .to_dict()
)

bc_to_topic = {x.replace("-1-", "-1___"): bc_to_topic[x] for x in bc_to_topic}

common_cells = list(set(bc_to_topic.keys()) & set(adata_embryo.obs_names))

adata_embryo_bin_topic = adata_embryo[common_cells].copy()

adata_embryo_bin_topic.X = adata_embryo_bin_topic.layers["count"]

def normalize(a):
    a.layers["count"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)


normalize(adata_embryo_bin_topic)

adata_embryo_bin_topic.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_embryo_bin_topic.obs_names
]

cell_topic_organoid = pd.read_table(
    os.path.join("../../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv")
)

cell_topic_organoid["topic_grp"] = (
    cell_topic_organoid["group"]
    + "_"
    + "Topic"
    + "_"
    + cell_topic_organoid["topic_name"].str.replace("Topic", "")
)

bc_to_topic = (
    cell_topic_organoid.pivot(
        index="cell_barcode", columns="topic_grp", values="topic_prob"
    )[topics_to_show]
    .fillna(0)
    .T.idxmax()
    .to_dict()
)

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

bc_to_topic = {
    x.split("-1")[0] + "-1" + "-" + sample_id_to_num[x.split("___")[-1]]: bc_to_topic[x]
    for x in bc_to_topic
}

adata_organoid.obs["topic"] = [bc_to_topic[bc] for bc in adata_organoid.obs_names]

avg_organoid = (
    pd.DataFrame(
        adata_organoid.layers["log_cpm"].todense(),
        index=adata_organoid.obs_names,
        columns=adata_organoid.var_names,
    )
    .groupby(adata_organoid.obs.topic)
    .mean()
)

raw_cts_organoid = pd.DataFrame(
    adata_organoid.layers["count"].todense(),
    index=adata_organoid.obs_names,
    columns=adata_organoid.var_names,
)

pct_cell_organoid = pd.DataFrame(
    np.zeros_like(avg_organoid),
    index=avg_organoid.index,
    columns=adata_organoid.var_names,
    dtype=np.float64,
)

for topic in pct_cell_organoid.index:
    cell_per_topic = adata_organoid.obs.query("topic == @topic").index
    pct_cell_organoid.loc[topic] = (raw_cts_organoid.loc[cell_per_topic] > 0).sum(
        0
    ) / len(cell_per_topic)

df_to_plot_avg_organoid = avg_organoid.copy().T.loc[:, topics_order]
df_to_plot_pct_organoid = pct_cell_organoid.copy().T.loc[:, topics_order]

avg_embryo = (
    pd.DataFrame(
        adata_embryo_bin_topic.layers["log_cpm"].todense(),
        index=adata_embryo_bin_topic.obs_names,
        columns=adata_embryo_bin_topic.var_names,
    )
    .groupby(adata_embryo_bin_topic.obs.topic)
    .mean()
)

raw_cts_embryo = pd.DataFrame(
    adata_embryo_bin_topic.layers["count"].todense(),
    index=adata_embryo_bin_topic.obs_names,
    columns=adata_embryo_bin_topic.var_names,
)

pct_cell_embryo = pd.DataFrame(
    np.zeros_like(avg_embryo),
    index=avg_embryo.index,
    columns=adata_embryo_bin_topic.var_names,
    dtype=np.float64,
)

for topic in pct_cell_embryo.index:
    cell_per_topic = adata_embryo_bin_topic.obs.query("topic == @topic").index
    pct_cell_embryo.loc[topic] = (raw_cts_embryo.loc[cell_per_topic] > 0).sum(0) / len(
        cell_per_topic
    )

df_to_plot_avg_embryo = avg_embryo.copy().T.loc[:, topics_order]
df_to_plot_pct_embryo = pct_cell_embryo.copy().T.loc[:, topics_order]

marker_genes = [
    "NKX2-2",
    "PAX6",
    "LMX1A",
    "SHH",
    "FOXA2",
    "RFX4",
    "KRT19",
    "NFE2L3",
    "SOX10",
    "TFAP2A",
    "PRRX1",
    "GRHL2",
    "NEUROD1",
    "STMN2",
    "GATA2",
    "ZEB2",
    "SOX2",
    "ASCL1",
    "UNCX",
    "GAD1",
]

hvg = list(
  set(sc.pp.highly_variable_genes(adata_organoid, layer = "log_cpm", inplace = False).query("highly_variable").index) & \
  set(sc.pp.highly_variable_genes(adata_embryo, layer = "log_cpm", inplace = False).query("highly_variable").index)
)

corrs = np.zeros(len(topics_order))
for i, topic in enumerate(topics_order):
  corrs[i] = stats.pearsonr(
    df_to_plot_avg_embryo.loc[hvg, topic],
    df_to_plot_avg_organoid.loc[hvg, topic]
  ).statistic

corrs_non_diag = np.zeros( len(topics_order) * len(topics_order) - len(topics_order) )
i = 0
for topic_1 in topics_order:
  for topic_2 in topics_order:
    if topic_1 == topic_2:
      continue
    corrs_non_diag[i] = stats.pearsonr(
      df_to_plot_avg_embryo.loc[hvg, topic_1],
      df_to_plot_avg_organoid.loc[hvg, topic_2]
    ).statistic
    i += 1

fig, ax = plt.subplots()
_ = ax.hist(
  corrs,
  color = "cyan",
  density = True,
  label = f"matching topics (u = {np.round(corrs.mean(), 3)})",
  zorder = 3,
  alpha = 0.7,
  bins = np.arange(0, 1, 0.05)
)
_ = ax.hist(
  corrs_non_diag,
  color = "magenta",
  density = True,
  label = f"non-matching topics (u = {np.round(corrs_non_diag.mean(), 3)})",
  zorder = 2,
  bins = np.arange(0, 1, 0.05)
)
ax.legend()
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xlabel("Pearsonr(organoid, embryo)")
_ = ax.set_ylabel("density")
fig.tight_layout()
fig.savefig("hist_org_v_embryo_topics_hvg.png")
fig.savefig("hist_org_v_embryo_topics_hvg.pdf")
plt.close(fig)






```
