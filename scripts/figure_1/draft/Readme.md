
```python

import scanpy as sc
import os
import pandas as pd

plot_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_1/draft/plots"
data_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data"

adata_organoid = sc.read_h5ad(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/RNA_DOWNSTREAM/analysis_w_20221021/GEX_adata.h5ad"
)

human_tfs = pd.read_csv("https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv")

TF_names = human_tfs["HGNC symbol"].tolist()

cell_topic_organoid = pd.read_table(
    os.path.join(data_dir, "cell_bin_topic.tsv")
)

cell_topic_organoid["topic_grp"] = cell_topic_organoid["group"] + "_" + cell_topic_organoid["topic_name"] 

# for each cell get the topic with highest probability

topics_to_show = [
    "neural_crest_Topic1",
    "neural_crest_Topic3",
    "neural_crest_Topic4",
    "neural_crest_Topic5",
    "neural_crest_Topic7",
    "neural_crest_Topic9",
    "neural_crest_Topic10",
    "neuron_Topic1",
    "neuron_Topic2",
    "neuron_Topic3",
    "neuron_Topic4",
    "neuron_Topic6",
    "neuron_Topic10",
    "neuron_Topic11",
    "neuron_Topic12",
    "neuron_Topic13",
    "neuron_Topic15",
    "neuron_Topic16",
    "neuron_Topic18",
    "neuron_Topic19",
    "neuron_Topic20",
    "neuron_Topic21",
    "neuron_Topic23",
    "neuron_Topic24",
    "neuron_Topic25",
    "progenitor_Topic1",
    "progenitor_Topic3",
    "progenitor_Topic8",
    "progenitor_Topic9",
    "progenitor_Topic11",
    "progenitor_Topic13",
    "progenitor_Topic14",
    "progenitor_Topic16",
    "progenitor_Topic19",
    "progenitor_Topic21",
    "progenitor_Topic23",
    "progenitor_Topic24",
    "progenitor_Topic25",
    "progenitor_Topic29",
    "progenitor_Topic30",
    "all_Topic4",
    "all_Topic12",
    "all_Topic18",
    "all_Topic20",
    "all_Topic23",
    "all_Topic35",
    "all_Topic37",
    "all_Topic45",
    "all_Topic47",
]

bc_to_topic = cell_topic_organoid \
    .pivot(index = "cell_barcode", columns = "topic_grp", values = "topic_prob")[topics_to_show] \
    .fillna(0).T.idxmax().to_dict()

def normalize(a):
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

bc_to_topic = {
    x.split("-1")[0] + "-1" + "-" + sample_id_to_num[x.split("___")[-1]]: bc_to_topic[x]
    for x in bc_to_topic
}

common_cells = list(set(bc_to_topic.keys()) & set(adata_organoid.obs_names))

adata_organoid = adata_organoid[common_cells].copy()

normalize(adata_organoid)

adata_organoid.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_organoid.obs_names
]

sc.tl.rank_genes_groups(
    adata_organoid,
    groupby = "topic",
    method = "wilcoxon",
    key_added = "wilcoxon_topic",
    pts = True
)

eRegulon_metadata = pd.read_table(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/COMBINED_ANALYSIS_w_20221004/outs/eRegulon_metadata.tsv"
)

TF_names = eRegulon_metadata.groupby("TF")["triplet_score"].min().sort_values().head(300).index

Q = "pvals_adj < 0.05 & logfoldchanges > 1.5 & names in @TF_names & pct_ratio > 1.5 & pct_nz_group > 0.5"
marker_TFs = set()
for topic in adata_organoid.obs["topic"].drop_duplicates():
    tmp = sc.get.rank_genes_groups_df(adata_organoid, group = topic, key = "wilcoxon_topic")
    tmp["pct_ratio"] = tmp["pct_nz_group"] / tmp["pct_nz_reference"]
    marker_TFs.update(
        tmp.query(Q).names    
    )

avg_organoid = adata_organoid.to_df().groupby(adata_organoid.obs.topic).mean()

import seaborn as sns

fig = sns.clustermap(
    avg_organoid[marker_TFs].T,
    robust = True,
    figsize = (10, 10),
)
fig.savefig(os.path.join(plot_dir, "clustermap_marker_TFs.png"))

```


```python

from pycistarget.input_output import read_hdf5

menr = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data/cistarget_topics.hdf5"
)

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
# something wrong with "all" topics .. rerun

import numpy as np
import os
def load_motif(n, d):
    motifs = []
    names = []
    with open(os.path.join(d, f"{n}.cb")) as f:
        tmp = None
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                names.append(l.replace(">", ""))
                if tmp is not None:
                    tmp = np.array(tmp)
                    tmp = (tmp.T / tmp.sum(1)).T
                    motifs.append(tmp)
                tmp = []
            else:
                tmp.append([float(x) for x in l.split()])
        tmp = np.array(tmp)
        tmp = (tmp.T / tmp.sum(1)).T
        motifs.append(tmp)
    return motifs, names

from tqdm import tqdm
import scanpy as sc
from tangermeme.tools.tomtom import tomtom
import torch
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns

plot_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_1/draft/plots"

sc.settings.figdir = plot_dir

all_motif_enrichment_res = []
for topic in topics_to_show:
    all_motif_enrichment_res.append(
        menr[topic].motif_enrichment
    )

all_motif_enrichment_res = pd.concat(
    all_motif_enrichment_res
)

motif_name_to_max_NES = all_motif_enrichment_res.reset_index().pivot_table(index = "index", columns = "Region_set", values = "NES").fillna(0).max(1)

all_motifs = []
motif_sub_names = []
motif_names = []
for topic in tqdm(topics_to_show):
    for motif_name in menr[topic].motif_enrichment.index:
        if motif_name in motif_names:
            continue
        _motifs, _m_sub_names = load_motif(
            motif_name,
            "/staging/leuven/stg_00002/lcb/cbravo/cluster_motif_collection_V10_no_desso_no_factorbook/cluster_buster/"
        )
        all_motifs.extend(_motifs)
        motif_sub_names.extend(_m_sub_names)
        motif_names.extend(np.repeat(motif_name, len(_motifs)))

t_all_motifs = [
    torch.from_numpy(m).T for m in tqdm(all_motifs)
]

pvals, scores, offsets, overlaps, strands = tomtom(
    t_all_motifs,
    t_all_motifs
)

evals = pvals.numpy() * len(all_motifs)

adata_motifs = sc.AnnData(
    evals,
    obs = pd.DataFrame(
        index = motif_sub_names,
        data = {
            "motif_name": motif_names,
            "max_NES": motif_name_to_max_NES.loc[motif_names].values
        }
    )
)

sc.tl.pca(adata_motifs)
sc.pp.neighbors(adata_motifs)
sc.tl.tsne(adata_motifs)

sc.tl.leiden(adata_motifs, resolution = 2)

sc.pl.tsne(adata_motifs, color = ["leiden"], save = "_leiden_motifs.pdf")

adata_motifs.obs["max_NES_leiden"] = adata_motifs.obs.groupby("leiden", observed = True)["max_NES"].transform("max")

leiden_to_best_motif = adata_motifs.obs.query("max_NES == max_NES_leiden").reset_index(drop = True).drop_duplicates().reset_index(drop = True).sort_values("leiden")

leiden_to_best_motif["Logo"] = [
    f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{m}.png" width="200" >'
    for m in leiden_to_best_motif["motif_name"]
]

leiden_to_best_motif[[
    "motif_name",
    "leiden",
    "max_NES",
    "Logo"]
].to_html(
    os.path.join(plot_dir, "leiden_to_best_motif.html"),
    escape = False,
    col_space = 80
)

motif_to_leiden = {}
for m in adata_motifs.obs.motif_name:
    clusters, count = np.unique(adata_motifs.obs.query("motif_name == @m")["leiden"], return_counts = True)
    motif_to_leiden[m] = clusters[count.argmax()]

leiden_to_NES = all_motif_enrichment_res \
    .reset_index() \
    .pivot_table(index = "index", columns = "Region_set", values = "NES") \
    .fillna(0) \
    .groupby(motif_to_leiden) \
    .max()

leiden_to_n_hits = all_motif_enrichment_res \
    .reset_index() \
    .pivot_table(index = "index", columns = "Region_set", values = "Motif_hits") \
    .fillna(0) \
    .groupby(motif_to_leiden) \
    .max()


def calc_IC_per_pos(pwm, bg = np.repeat(0.25, 4), pc = 1e-4):
    return (np.log2((pwm + pc) / bg) * pwm).sum(1)
 
def get_consensus_sequence(pwm, letter_order = ["A", "C", "G", "T"], min_ic = 0.8):
    ic_above_min = np.where(calc_IC_per_pos(pwm) > min_ic)[0]
    to_keep = np.zeros(pwm.shape[0], dtype = bool)
    to_keep[ic_above_min.min():ic_above_min.max() + 1] = True
    consensus = [letter_order[x] for x in pwm.argmax(1)[to_keep]]
    score = (pwm.max(1) * calc_IC_per_pos(pwm))[to_keep]
    return consensus, score

def tex_font(l, s, color='black'):
    return r"{\fontsize{" + f"{s}" + r"pt}{3em}\selectfont \color{" + color + r"} " + l + r"}"

letter_to_color = {
    "G": "orange",
    "T": "red",
    "A": "green",
    "C": "blue"
}


adata_motifs.obs["IC"] = [
    calc_IC_per_pos(all_motifs[motif_sub_names.index(m)]).sum()
    for m in adata_motifs.obs_names
]

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{color}\usepackage{xcolor}')

motif_name = "metacluster_43.1"
motif_sub_name = adata_motifs.obs.query("motif_name == @motif_name").sort_values("IC", ascending = False).index[0]
letter_order = "ACGT"

fig, axs = plt.subplots(nrows = 2)
pwm = all_motifs[motif_sub_names.index(motif_sub_name)]
ic = calc_IC_per_pos(pwm)
_ = logomaker.Logo(
    pd.DataFrame(
        pwm,
        columns = list(letter_order)
    ),
    ax = axs[0]
)
_ = logomaker.Logo(
    pd.DataFrame(
        pwm * ic[:, None],
        columns = list(letter_order)
    ),
    ax = axs[1]
)
_ = axs[1].set_ylim((0, 2))
consensus, score = get_consensus_sequence(pwm)
title = r"".join(
    [
        tex_font(
            c, s * 10, letter_to_color[c]
        )
        for c,s in zip(consensus, score)
    ]
)
_ = axs[1].set_title(title)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "test.pdf"), backend = "pgf")

def get_consensus_logo_for_motif_name(motif_name):
    motif_sub_name = adata_motifs.obs.query("motif_name == @motif_name").sort_values("IC", ascending = False).index[0]
    pwm = all_motifs[motif_sub_names.index(motif_sub_name)]
    ic = calc_IC_per_pos(pwm)
    consensus, score = get_consensus_sequence(pwm)
    title = r"".join(
        [
            tex_font(
                c, s * 5, letter_to_color[c]
            )
            for c,s in zip(consensus, score)
        ]
    )
    return title


leiden_to_best_motif_d = leiden_to_best_motif.set_index("leiden")["motif_name"].to_dict()

df_to_plot = leiden_to_NES.copy()

df_to_plot.index = [
    get_consensus_logo_for_motif_name(leiden_to_best_motif_d[l])
    for l in df_to_plot.index
]

m = df_to_plot.reset_index().melt(id_vars = "index")

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(
    x = m["Region_set"],
    y = m["index"],
    s = m["value"] * 10,
    c = m["value"],
    cmap='viridis'
)
ax.set_xticklabels([])
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "clustermap_nes_per_leiden.pdf"))

topics_order = [
    "progenitor_Topic_23",
    "progenitor_Topic_30",
    "progenitor_Topic_3",
    "progenitor_Topic_1",
    "progenitor_Topic_11",
    "progenitor_Topic_9",
    "progenitor_Topic_21",
    "progenitor_Topic_13",
    "progenitor_Topic_29",
    "progenitor_Topic_19",
    "progenitor_Topic_14",
    "progenitor_Topic_24",
    "progenitor_Topic_25",
    "progenitor_Topic_16",
    "progenitor_Topic_8",
    "neuron_Topic_12",
    "neuron_Topic_19",
    "neuron_Topic_15",
    "neuron_Topic_21",
    "neuron_Topic_6",
    "neuron_Topic_3",
    "neuron_Topic_4",
    "neuron_Topic_23",
    "neuron_Topic_10",
    "neuron_Topic_24",
    "neuron_Topic_11",
    #"neuron_Topic_20",
    "neuron_Topic_18",
    "neuron_Topic_2",
    "neuron_Topic_13",
    "neuron_Topic_25",
    "neuron_Topic_16",
    "neural_crest_Topic_7",
    "neural_crest_Topic_5",
    "neural_crest_Topic_4",
    "neural_crest_Topic_10",
    "neural_crest_Topic_9",
    "neural_crest_Topic_1",
    "neural_crest_Topic_3",
    ]


leiden_to_show = [
        "0",
        "3",
        "4",
        "6",
        "8",
        "11",
        "17",
        "18",
        "19",
        "21",
        "23",
        "24",
        "26",
        "30",
        "31",
        "32",
        "35",
        "39",
        "41",
        "42",
        "45",
        "47",
        "50",
        "52",
]

df_to_plot_NES = leiden_to_NES.copy().loc[leiden_to_show, topics_order]
df_to_plot_hits = leiden_to_n_hits.copy().loc[leiden_to_show, topics_order]

leiden_to_idx_topic_order = {
    l: topics_order.index(v) for l, v in zip(df_to_plot_NES.T.idxmax().index, df_to_plot_NES.T.idxmax().values)}
leiden_order = sorted(leiden_to_idx_topic_order, key = lambda l: leiden_to_idx_topic_order[l])

df_to_plot_NES = df_to_plot_NES.loc[leiden_order]
df_to_plot_hits = df_to_plot_hits.loc[leiden_order]

df_to_plot_NES.index = [
    l + "-" + get_consensus_logo_for_motif_name(leiden_to_best_motif_d[l])
    for l in df_to_plot_NES.index
]

m_NES = df_to_plot_NES.reset_index().melt(id_vars = "index")
m_hits = df_to_plot_hits.reset_index().melt(id_vars = "index")

fig, ax = plt.subplots(figsize = (10, 5))
_ = ax.scatter(
    x = m_NES["Region_set"],
    y = m_NES["index"],
    s = m_hits["value"] / 25,
    c = m_NES["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 3, vmax = 10
)
ax.set_xticklabels(
    df_to_plot_NES.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "dotplot_nes_per_leiden_sorted.pdf"))


from pycistarget.utils import get_TF_list

TF_names = get_TF_list(
    all_motif_enrichment_res.loc[
       list(adata_motifs.obs.query("leiden in @leiden_to_show")["motif_name"].drop_duplicates().values) 
    ],
    annotation = ["Direct_annot"])


###

data_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data"

adata_organoid = sc.read_h5ad(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/RNA_DOWNSTREAM/analysis_w_20221021/GEX_adata.h5ad"
)

cell_topic_organoid = pd.read_table(
    os.path.join(data_dir, "cell_bin_topic.tsv")
)

cell_topic_organoid["topic_grp"] = cell_topic_organoid["group"] \
    + "_" + "Topic" + "_" + cell_topic_organoid["topic_name"].str.replace("Topic", "")

bc_to_topic = cell_topic_organoid \
    .pivot(index = "cell_barcode", columns = "topic_grp", values = "topic_prob")[topics_to_show] \
    .fillna(0).T.idxmax().to_dict()

def normalize(a):
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

bc_to_topic = {
    x.split("-1")[0] + "-1" + "-" + sample_id_to_num[x.split("___")[-1]]: bc_to_topic[x]
    for x in bc_to_topic
}

common_cells = list(set(bc_to_topic.keys()) & set(adata_organoid.obs_names))

adata_organoid = adata_organoid[common_cells].copy()

normalize(adata_organoid)

adata_organoid.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_organoid.obs_names
]

sc.tl.rank_genes_groups(
    adata_organoid,
    groupby = "topic",
    method = "wilcoxon",
    key_added = "wilcoxon_topic",
    pts = True
)

Q = "pvals_adj < 0.05 & logfoldchanges > 1.5 & names in @TF_names & pct_ratio > 1 & pct_nz_group > 0.5"
marker_TFs = set()
marker_TFs_to_logfc = {}
for topic in set(topics_order) & set(adata_organoid.obs.topic):
    tmp = sc.get.rank_genes_groups_df(adata_organoid, group = topic, key = "wilcoxon_topic")
    tmp["pct_ratio"] = tmp["pct_nz_group"] / tmp["pct_nz_reference"]
    selected_TFs = tmp.query(Q).names  
    marker_TFs.update(
        selected_TFs
    )
    tmp = tmp.set_index("names")["pct_nz_group"].to_dict()
    for tf in selected_TFs:
        marker_TFs_to_logfc[tf] = max(tmp[tf], marker_TFs_to_logfc.get(tf, 0))    

eRegulon_metadata = pd.read_table(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/COMBINED_ANALYSIS_w_20221004/outs/eRegulon_metadata.tsv"
)

SCENIC_TF_names = eRegulon_metadata.groupby("TF")["triplet_score"].min().sort_values().head(300).index

marker_TFs = marker_TFs & set(SCENIC_TF_names)

marker_genes = [
    "NKX2-8",
    "NKX2-2",
    "PAX6",
    "ZIC1",
    "IRX3",
    "OLIG3",
    "WNT1",
    "LMX1A",
    "ARX",
    "SHH",
    "FOXA2",
    "RFX4",
    "FOXG1",
    "SOX9",
    "KRT19",
    "NFE2L3",
    "POU5F1",
    "ONECUT2",
    "XACT",
    "SOX10",
    "TFAP2A",
    "FOXD3",
    "PRRX1",
    "GRHL1",
    "FOXC2",
    "TWIST1",
    "NEUROD1",
    "SIX1",
    "STMN2",
    "GATA2",
    "ISL1",
    "ZEB1",
    "SST",
    "ELAVL4",
    "SOX2",
    "ASCL1",
    "NKX6-2",
    "PHOX2B",
    "UNCX",
    "GAD1"
]

marker_TFs = marker_genes

avg_organoid = pd.DataFrame(
    adata_organoid.layers["log_cpm"].todense(),
    index = adata_organoid.obs_names,
    columns = adata_organoid.var_names).groupby(adata_organoid.obs.topic).mean()

raw_cts = pd.DataFrame(
    adata_organoid.layers["counts"].todense(),
    index = adata_organoid.obs_names,
    columns = adata_organoid.var_names)

pct_cell = pd.DataFrame(
    np.zeros_like(avg_organoid),
    index = avg_organoid.index,
    columns = adata_organoid.var_names,
    dtype = np.float64
)

for topic in pct_cell.index:
    cell_per_topic = adata_organoid.obs.query("topic == @topic").index
    pct_cell.loc[topic] = (raw_cts.loc[cell_per_topic] > 0).sum(0) / len(cell_per_topic)

df_to_plot_avg = avg_organoid.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct = pct_cell.copy().T.loc[list(marker_TFs), topics_order]

marker_to_idx_topic_order = {
    m: topics_order.index(v) for m, v in zip(df_to_plot_avg.T.idxmax().index, df_to_plot_avg.T.idxmax().values)}
marker_order = sorted(marker_to_idx_topic_order, key = lambda l: marker_to_idx_topic_order[l])

df_to_plot_avg = df_to_plot_avg.loc[marker_order]
df_to_plot_pct = df_to_plot_pct.loc[marker_order]

m_avg = df_to_plot_avg.reset_index().melt(id_vars = "index")
m_pct = df_to_plot_pct.reset_index().melt(id_vars = "index")

fig, ax = plt.subplots(figsize = (10, 5))
_ = ax.scatter(
    x = m_avg["topic"],
    y = m_avg["index"],
    s = m_pct["value"] * 50,
    c = m_avg["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 0, vmax = 1.5
)
ax.set_xticklabels(
    df_to_plot_avg.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.tick_params(axis='y', which='major', labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "dotplot_avg_per_topic_sorted.pdf"))

fig, axs = plt.subplots(figsize = (10, 10), sharex = True, nrows = 2, height_ratios = [2, 1])
ax = axs[0]
_ = ax.scatter(
    x = m_avg["topic"],
    y = m_avg["index"],
    s = m_pct["value"] * 50,
    c = m_avg["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 0, vmax = 1.5
)
ax.set_xticklabels(
    df_to_plot_avg.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.tick_params(axis='y', which='major', labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
ax = axs[1]
_ = ax.scatter(
    x = m_NES["Region_set"],
    y = m_NES["index"],
    s = m_hits["value"] / 50,
    c = m_NES["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 3, vmax = 10
)
ax.set_xticklabels(
    df_to_plot_NES.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "dotplot_exp_nes.pdf"))

```


```python

topics_order = [
    "progenitor_Topic_23",
    "progenitor_Topic_30",
    "progenitor_Topic_3",
    "progenitor_Topic_1",
    "progenitor_Topic_11",
    "progenitor_Topic_9",
    #"progenitor_Topic_21",
    "progenitor_Topic_13",
    "progenitor_Topic_29",
    "progenitor_Topic_19",
    "progenitor_Topic_14",
    "progenitor_Topic_24",
    "progenitor_Topic_25",
    "progenitor_Topic_16",
    "progenitor_Topic_8",
    "neuron_Topic_12",
    #"neuron_Topic_19",
    "neuron_Topic_15",
    #"neuron_Topic_21",
    "neuron_Topic_6",
    #"neuron_Topic_3",
    "neuron_Topic_4",
    "neuron_Topic_23",
    #"neuron_Topic_10",
    "neuron_Topic_24",
    "neuron_Topic_11",
    #"neuron_Topic_20",
    #"neuron_Topic_18",
    #"neuron_Topic_2",
    "neuron_Topic_13",
    "neuron_Topic_25",
    #"neuron_Topic_16",
    "neural_crest_Topic_7",
    "neural_crest_Topic_5",
    "neural_crest_Topic_4",
    #"neural_crest_Topic_10",
    "neural_crest_Topic_9",
    #"neural_crest_Topic_1",
    "neural_crest_Topic_3",
]



import pandas as pd

region_topic_nc = pd.read_table(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/ATAC_DOWNSTREAM/pycisTopic_per_cell_type/neural_crest/region_topic_contrib.tsv",
    index_col = 0
)
region_topic_prog = pd.read_table(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/ATAC_DOWNSTREAM/pycisTopic_per_cell_type/progenitors/region_topic_contrib.tsv",
    index_col = 0
)
region_topic_neu = pd.read_table(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/ATAC_DOWNSTREAM/pycisTopic_per_cell_type/neurons/region_topic_contrib.tsv",
    index_col = 0
)

region_topic_nc.columns = [
    f"neural_crest_Topic_{c.replace('Topic', '')}"
    for c in region_topic_nc
]

region_topic_prog.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}"
    for c in region_topic_prog
]

region_topic_neu.columns = [
    f"neuron_Topic_{c.replace('Topic', '')}"
    for c in region_topic_neu
]

region_topic = pd.concat(
    [
        region_topic_nc,
        region_topic_prog,
        region_topic_neu
    ],
    axis = 1
).dropna()

data_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data"

from scipy import sparse
import os

embryo_bin_matrix = sparse.load_npz(os.path.join(data_dir, "embryo_atac_merged_bin_matrix.npz"))

embryo_region_names = []
with open(os.path.join(data_dir, "embry_atac_region_names.txt")) as f:
    for l in f:
        embryo_region_names.append(l.strip())

embryo_cell_names = []
with open(os.path.join(data_dir, "embry_atac_cell_names.tx")) as f:
    for l in f:
        embryo_cell_names.append(l.strip())

common_regions = list(set(region_topic.index) & set(embryo_region_names))

X = pd.DataFrame(
    embryo_bin_matrix.todense(),
    index = embryo_region_names,
    columns = embryo_cell_names
).loc[common_regions]

embryo_cell_topic = region_topic.loc[common_regions].T@X

embryo_cell_topic = embryo_cell_topic.T

from pycisTopic.topic_binarization import smooth_topics_distributions, cross_entropy
import numpy as np
def bin_cell_topic_li(ct, nbins = 100):
    topic_dist = smooth_topics_distributions(ct.copy())
    binarized_topics = {}
    for topic in topic_dist.columns:
        l = np.asarray(topic_dist[topic])
        l_norm = (l - np.min(l)) / np.ptp(l)
        thresholds = np.arange(np.min(l_norm) + 0.01, np.max(l_norm) - 0.01, 0.01)
        entropies = [cross_entropy(l_norm, t, nbins=nbins) for t in thresholds]
        thr = thresholds[np.argmin(entropies)]
        binarized_topics[topic] = pd.DataFrame(
            topic_dist.iloc[l_norm > thr][topic]
        ).sort_values(topic, ascending=False)
    return binarized_topics

def bin_cell_topic_ntop(ct, ntop = 1_000):
    topic_dist = smooth_topics_distributions(ct.copy())
    binarized_topics = {}
    for topic in topic_dist.columns:
        l = np.asarray(topic_dist[topic])
        binarized_topics[topic]  = pd.DataFrame(
            topic_dist.sort_values(topic, ascending = False)[topic].head(ntop)
        )
    return binarized_topics

#embryo_bin_cell_topic = bin_cell_topic_li(embryo_cell_topic)

embryo_bin_cell_topic = bin_cell_topic_ntop(embryo_cell_topic)

embryo_cell_topic = []
for topic in embryo_bin_cell_topic:
    tmp = embryo_bin_cell_topic[topic].copy() \
        .reset_index() \
        .rename(
            {
                "index": "bc",
                topic: "prob"
            },
            axis = 1
        )
    tmp["topic_grp"] = topic
    embryo_cell_topic.append(tmp)

embryo_cell_topic = pd.concat(embryo_cell_topic)

import scanpy as sc
adata_embryo = sc.read_h5ad(
    "/staging/leuven/stg_00002/lcb/cmannens/floor_plate/analysis/4w_head/RNA/adata_raw_filtered.h5ad"
)

adata_embryo_p = sc.read_h5ad(
    "/staging/leuven/stg_00002/lcb/cmannens/floor_plate/analysis/4w_head/RNA/adata_annotated.h5ad"
)

cell_data = pd.read_csv(
    "/staging/leuven/stg_00002/lcb/cmannens/floor_plate/analysis/4w_head/RNA/cell_data.csv",
    index_col = 0
)

adata_embryo = adata_embryo[cell_data.index].copy()
adata_embryo.obs = cell_data.loc[adata_embryo.obs_names]

bc_to_topic = embryo_cell_topic \
    .pivot(index = "bc", columns = "topic_grp", values = "prob")[topics_order] \
    .fillna(0).T.idxmax().to_dict()

def normalize(a):
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)

bc_to_topic = {
    x.replace("-1-", "-1___"): bc_to_topic[x]
    for x in bc_to_topic
}

common_cells = list(set(bc_to_topic.keys()) & set(adata_embryo.obs_names))

adata_embryo = adata_embryo[common_cells].copy()

normalize(adata_embryo)

adata_embryo.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_embryo.obs_names
]

marker_genes = [
    "NKX2-8",
    "NKX2-2",
    "PAX6",
    "ZIC1",
    "IRX3",
    "OLIG3",
    "WNT1",
    "LMX1A",
    "ARX",
    "SHH",
    "FOXA2",
    "RFX4",
    "FOXG1",
    "SOX9",
    "KRT19",
    "NFE2L3",
    "POU5F1",
    "ONECUT2",
    "XACT",
    "SOX10",
    "TFAP2A",
    "FOXD3",
    "PRRX1",
    "GRHL2",
    "FOXC2",
    "TWIST1",
    "NEUROD1",
    "SIX1",
    "STMN2",
    "GATA2",
    "ISL1",
    "ZEB1",
    "SST",
    "ELAVL4",
    "SOX2",
    "ASCL1",
    "NKX6-2",
    "PHOX2B",
    "UNCX",
    "GAD1"
]

marker_TFs = marker_genes

avg_embryo = pd.DataFrame(
    adata_embryo.layers["log_cpm"].todense(),
    index = adata_embryo.obs_names,
    columns = adata_embryo.var_names).groupby(adata_embryo.obs.topic).mean()

raw_cts = pd.DataFrame(
    adata_embryo.layers["counts"].todense(),
    index = adata_embryo.obs_names,
    columns = adata_embryo.var_names)

pct_cell = pd.DataFrame(
    np.zeros_like(avg_embryo),
    index = avg_embryo.index,
    columns = adata_embryo.var_names,
    dtype = np.float64
)

for topic in pct_cell.index:
    cell_per_topic = adata_embryo.obs.query("topic == @topic").index
    pct_cell.loc[topic] = (raw_cts.loc[cell_per_topic] > 0).sum(0) / len(cell_per_topic)

df_to_plot_avg = avg_embryo.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct = pct_cell.copy().T.loc[list(marker_TFs), topics_order]

marker_to_idx_topic_order = {
    m: topics_order.index(v) for m, v in zip(df_to_plot_avg.T.idxmax().index, df_to_plot_avg.T.idxmax().values)}
marker_order = sorted(marker_to_idx_topic_order, key = lambda l: marker_to_idx_topic_order[l])

df_to_plot_avg = df_to_plot_avg.loc[marker_order]
df_to_plot_pct = df_to_plot_pct.loc[marker_order]

m_avg = df_to_plot_avg.reset_index().melt(id_vars = "index")
m_pct = df_to_plot_pct.reset_index().melt(id_vars = "index")

import matplotlib.pyplot as plt
plot_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_1/draft/plots"

fig, ax = plt.subplots(figsize = (10, 5))
_ = ax.scatter(
    x = m_avg["topic"],
    y = m_avg["index"],
    s = m_pct["value"] * 50,
    c = m_avg["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 0, vmax = 1
)
ax.set_xticklabels(
    df_to_plot_avg.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.tick_params(axis='y', which='major', labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "dotplot_avg_per_topic_sorted_embryo.pdf"))

from pycistarget.input_output import read_hdf5

menr = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data/cistarget_topics.hdf5"
)

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
# something wrong with "all" topics .. rerun

import numpy as np
import os
def load_motif(n, d):
    motifs = []
    names = []
    with open(os.path.join(d, f"{n}.cb")) as f:
        tmp = None
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                names.append(l.replace(">", ""))
                if tmp is not None:
                    tmp = np.array(tmp)
                    tmp = (tmp.T / tmp.sum(1)).T
                    motifs.append(tmp)
                tmp = []
            else:
                tmp.append([float(x) for x in l.split()])
        tmp = np.array(tmp)
        tmp = (tmp.T / tmp.sum(1)).T
        motifs.append(tmp)
    return motifs, names

from tqdm import tqdm
import scanpy as sc
from tangermeme.tools.tomtom import tomtom
import torch
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns

plot_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_1/draft/plots"

sc.settings.figdir = plot_dir

all_motif_enrichment_res = []
for topic in topics_to_show:
    all_motif_enrichment_res.append(
        menr[topic].motif_enrichment
    )

all_motif_enrichment_res = pd.concat(
    all_motif_enrichment_res
)

motif_name_to_max_NES = all_motif_enrichment_res.reset_index().pivot_table(index = "index", columns = "Region_set", values = "NES").fillna(0).max(1)

all_motifs = []
motif_sub_names = []
motif_names = []
for topic in tqdm(topics_to_show):
    for motif_name in menr[topic].motif_enrichment.index:
        if motif_name in motif_names:
            continue
        _motifs, _m_sub_names = load_motif(
            motif_name,
            "/staging/leuven/stg_00002/lcb/cbravo/cluster_motif_collection_V10_no_desso_no_factorbook/cluster_buster/"
        )
        all_motifs.extend(_motifs)
        motif_sub_names.extend(_m_sub_names)
        motif_names.extend(np.repeat(motif_name, len(_motifs)))

t_all_motifs = [
    torch.from_numpy(m).T for m in tqdm(all_motifs)
]

pvals, scores, offsets, overlaps, strands = tomtom(
    t_all_motifs,
    t_all_motifs
)

evals = pvals.numpy() * len(all_motifs)

adata_motifs = sc.AnnData(
    evals,
    obs = pd.DataFrame(
        index = motif_sub_names,
        data = {
            "motif_name": motif_names,
            "max_NES": motif_name_to_max_NES.loc[motif_names].values
        }
    )
)

sc.tl.pca(adata_motifs)
sc.pp.neighbors(adata_motifs)
sc.tl.tsne(adata_motifs)

sc.tl.leiden(adata_motifs, resolution = 2)

sc.pl.tsne(adata_motifs, color = ["leiden"], save = "_leiden_motifs.pdf")

adata_motifs.obs["max_NES_leiden"] = adata_motifs.obs.groupby("leiden", observed = True)["max_NES"].transform("max")

leiden_to_best_motif = adata_motifs.obs.query("max_NES == max_NES_leiden").reset_index(drop = True).drop_duplicates().reset_index(drop = True).sort_values("leiden")

leiden_to_best_motif["Logo"] = [
    f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{m}.png" width="200" >'
    for m in leiden_to_best_motif["motif_name"]
]

leiden_to_best_motif[[
    "motif_name",
    "leiden",
    "max_NES",
    "Logo"]
].to_html(
    os.path.join(plot_dir, "leiden_to_best_motif.html"),
    escape = False,
    col_space = 80
)

motif_to_leiden = {}
for m in adata_motifs.obs.motif_name:
    clusters, count = np.unique(adata_motifs.obs.query("motif_name == @m")["leiden"], return_counts = True)
    motif_to_leiden[m] = clusters[count.argmax()]

leiden_to_NES = all_motif_enrichment_res \
    .reset_index() \
    .pivot_table(index = "index", columns = "Region_set", values = "NES") \
    .fillna(0) \
    .groupby(motif_to_leiden) \
    .max()

leiden_to_n_hits = all_motif_enrichment_res \
    .reset_index() \
    .pivot_table(index = "index", columns = "Region_set", values = "Motif_hits") \
    .fillna(0) \
    .groupby(motif_to_leiden) \
    .max()


def calc_IC_per_pos(pwm, bg = np.repeat(0.25, 4), pc = 1e-4):
    return (np.log2((pwm + pc) / bg) * pwm).sum(1)
 
def get_consensus_sequence(pwm, letter_order = ["A", "C", "G", "T"], min_ic = 0.8):
    ic_above_min = np.where(calc_IC_per_pos(pwm) > min_ic)[0]
    to_keep = np.zeros(pwm.shape[0], dtype = bool)
    to_keep[ic_above_min.min():ic_above_min.max() + 1] = True
    consensus = [letter_order[x] for x in pwm.argmax(1)[to_keep]]
    score = (pwm.max(1) * calc_IC_per_pos(pwm))[to_keep]
    return consensus, score

def tex_font(l, s, color='black'):
    return r"{\fontsize{" + f"{s}" + r"pt}{3em}\selectfont \color{" + color + r"} " + l + r"}"

letter_to_color = {
    "G": "orange",
    "T": "red",
    "A": "green",
    "C": "blue"
}

adata_motifs.obs["IC"] = [
    calc_IC_per_pos(all_motifs[motif_sub_names.index(m)]).sum()
    for m in adata_motifs.obs_names
]

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{color}\usepackage{xcolor}')

def get_consensus_logo_for_motif_name(motif_name):
    motif_sub_name = adata_motifs.obs.query("motif_name == @motif_name").sort_values("IC", ascending = False).index[0]
    pwm = all_motifs[motif_sub_names.index(motif_sub_name)]
    ic = calc_IC_per_pos(pwm)
    consensus, score = get_consensus_sequence(pwm)
    title = r"".join(
        [
            tex_font(
                c, s * 5, letter_to_color[c]
            )
            for c,s in zip(consensus, score)
        ]
    )
    return title


leiden_to_best_motif_d = leiden_to_best_motif.set_index("leiden")["motif_name"].to_dict()

leiden_to_show = [
        "0",
        "3",
        "4",
        "6",
        "8",
        "11",
        "17",
        "18",
        "19",
        "21",
        "23",
        "24",
        "26",
        "30",
        "31",
        "32",
        "35",
        "39",
        "41",
        "42",
        "45",
        "47",
        "50",
        "52",
]

###

data_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data"

adata_organoid = sc.read_h5ad(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/RNA_DOWNSTREAM/analysis_w_20221021/GEX_adata.h5ad"
)

cell_topic_organoid = pd.read_table(
    os.path.join(data_dir, "cell_bin_topic.tsv")
)

cell_topic_organoid["topic_grp"] = cell_topic_organoid["group"] \
    + "_" + "Topic" + "_" + cell_topic_organoid["topic_name"].str.replace("Topic", "")

bc_to_topic = cell_topic_organoid \
    .pivot(index = "cell_barcode", columns = "topic_grp", values = "topic_prob")[topics_to_show] \
    .fillna(0).T.idxmax().to_dict()

def normalize(a):
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers["log_cpm"] = a.X.copy()
    sc.pp.scale(a)

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

bc_to_topic = {
    x.split("-1")[0] + "-1" + "-" + sample_id_to_num[x.split("___")[-1]]: bc_to_topic[x]
    for x in bc_to_topic
}

common_cells = list(set(bc_to_topic.keys()) & set(adata_organoid.obs_names))

adata_organoid = adata_organoid[common_cells].copy()

normalize(adata_organoid)

adata_organoid.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_organoid.obs_names
]

marker_genes = [
    "NKX2-8",
    "NKX2-2",
    "PAX6",
    "ZIC1",
    "IRX3",
    "OLIG3",
    "WNT1",
    "LMX1A",
    "ARX",
    "SHH",
    "FOXA2",
    "RFX4",
    "FOXG1",
    "SOX9",
    "KRT19",
    "NFE2L3",
    "POU5F1",
    "ONECUT2",
    "XACT",
    "SOX10",
    "TFAP2A",
    "FOXD3",
    "PRRX1",
    "GRHL1",
    "FOXC2",
    "TWIST1",
    "NEUROD1",
    "SIX1",
    "STMN2",
    "GATA2",
    "ISL1",
    "ZEB1",
    "SST",
    "ELAVL4",
    "SOX2",
    "ASCL1",
    "NKX6-2",
    "PHOX2B",
    "UNCX",
    "GAD1"
]

marker_TFs = marker_genes

avg_organoid = pd.DataFrame(
    adata_organoid.layers["log_cpm"].todense(),
    index = adata_organoid.obs_names,
    columns = adata_organoid.var_names).groupby(adata_organoid.obs.topic).mean()

raw_cts_organoid = pd.DataFrame(
    adata_organoid.layers["counts"].todense(),
    index = adata_organoid.obs_names,
    columns = adata_organoid.var_names)

pct_cell_organoid = pd.DataFrame(
    np.zeros_like(avg_organoid),
    index = avg_organoid.index,
    columns = adata_organoid.var_names,
    dtype = np.float64
)

for topic in pct_cell.index:
    cell_per_topic = adata_organoid.obs.query("topic == @topic").index
    pct_cell_organoid.loc[topic] = (raw_cts_organoid.loc[cell_per_topic] > 0).sum(0) / len(cell_per_topic)

df_to_plot_avg_organoid = avg_organoid.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct_organoid = pct_cell_organoid.copy().T.loc[list(marker_TFs), topics_order]

marker_to_idx_topic_order = {
    m: topics_order.index(v) for m, v in zip(df_to_plot_avg_organoid.T.idxmax().index, df_to_plot_avg_organoid.T.idxmax().values)}
marker_order = sorted(marker_to_idx_topic_order, key = lambda l: marker_to_idx_topic_order[l])

df_to_plot_avg_organoid = df_to_plot_avg_organoid.loc[marker_order]
df_to_plot_pct_organoid = df_to_plot_pct_organoid.loc[marker_order]

m_avg_organoid = df_to_plot_avg_organoid.reset_index().melt(id_vars = "index")
m_pct_organoid = df_to_plot_pct_organoid.reset_index().melt(id_vars = "index")

avg_embryo = pd.DataFrame(
    adata_embryo.layers["log_cpm"].todense(),
    index = adata_embryo.obs_names,
    columns = adata_embryo.var_names).groupby(adata_embryo.obs.topic).mean()

raw_cts_embryo = pd.DataFrame(
    adata_embryo.layers["counts"].todense(),
    index = adata_embryo.obs_names,
    columns = adata_embryo.var_names)

pct_cell_embryo = pd.DataFrame(
    np.zeros_like(avg_embryo),
    index = avg_embryo.index,
    columns = adata_embryo.var_names,
    dtype = np.float64
)

for topic in pct_cell.index:
    cell_per_topic = adata_embryo.obs.query("topic == @topic").index
    pct_cell_embryo.loc[topic] = (raw_cts_embryo.loc[cell_per_topic] > 0).sum(0) / len(cell_per_topic)

df_to_plot_avg_embryo = avg_embryo.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct_embryo = pct_cell_embryo.copy().T.loc[list(marker_TFs), topics_order]

df_to_plot_avg_embryo = df_to_plot_avg_embryo.loc[marker_order]
df_to_plot_pct_embryo = df_to_plot_pct_embryo.loc[marker_order]

m_avg_embryo = df_to_plot_avg_embryo.reset_index().melt(id_vars = "index")
m_pct_embryo = df_to_plot_pct_embryo.reset_index().melt(id_vars = "index")

df_to_plot_NES = leiden_to_NES.copy().loc[leiden_to_show, topics_order]
df_to_plot_hits = leiden_to_n_hits.copy().loc[leiden_to_show, topics_order]

leiden_to_idx_topic_order = {
    l: topics_order.index(v) for l, v in zip(df_to_plot_NES.T.idxmax().index, df_to_plot_NES.T.idxmax().values)}
leiden_order = sorted(leiden_to_idx_topic_order, key = lambda l: leiden_to_idx_topic_order[l])

df_to_plot_NES = df_to_plot_NES.loc[leiden_order]
df_to_plot_hits = df_to_plot_hits.loc[leiden_order]

df_to_plot_NES.index = [
    l + "-" + get_consensus_logo_for_motif_name(leiden_to_best_motif_d[l])
    for l in df_to_plot_NES.index
]

m_NES = df_to_plot_NES.reset_index().melt(id_vars = "index")
m_hits = df_to_plot_hits.reset_index().melt(id_vars = "index")

fig, axs = plt.subplots(figsize = (10, 12), sharex = True, nrows = 3, height_ratios = [2, 2, 2])
ax = axs[0]
m_avg = m_avg_organoid
m_pct = m_pct_organoid
_ = ax.scatter(
    x = m_avg["topic"],
    y = m_avg["index"],
    s = m_pct["value"] * 50,
    c = m_avg["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 0, vmax = 1.5
)
ax.set_xticklabels(
    df_to_plot_avg.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.tick_params(axis='y', which='major', labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
ax.set_ylabel("Organoid")
m_avg = m_avg_embryo
m_pct = m_pct_embryo
ax = axs[1]
_ = ax.scatter(
    x = m_avg["topic"],
    y = m_avg["index"],
    s = m_pct["value"] * 50,
    c = m_avg["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 0, vmax = 1
)
ax.set_xticklabels(
    df_to_plot_avg.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.tick_params(axis='y', which='major', labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
ax.set_ylabel("Embryo")
ax = axs[2]
_ = ax.scatter(
    x = m_NES["Region_set"],
    y = m_NES["index"],
    s = m_hits["value"] / 50,
    c = m_NES["value"],
    lw = 1, edgecolor = "black",
    cmap='viridis',
    vmin = 3, vmax = 10
)
ax.set_xticklabels(
    df_to_plot_NES.columns,
    rotation=45, ha='right', rotation_mode='anchor'
)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "dotplot_exp_nes_organoid_embryo.pdf"))

```

```python


from pycistarget.input_output import read_hdf5

menr_org = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/data_prep/data/cistarget_topics.hdf5"
)

menr_embr_progenitor = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/PYCISTOPIC/outs/topic_modeling_progenitor/ctx_topics_otsu/cistarget_topics.hdf5"
)

menr_embr_neuron = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/PYCISTOPIC/outs/topic_modeling_neuron/ctx_topics_otsu/cistarget_topics.hdf5"
)

menr_embr_neural_crest = read_hdf5(
    "/staging/leuven/stg_00002/lcb/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/PYCISTOPIC/outs/topic_modeling_neural_crest/ctx_topics_otsu/cistarget_topics.hdf5"
)

menr_d = {
    "organoid": menr_org,
    "embr_prog": menr_embr_progenitor,
    "embr_neu": menr_embr_neuron,
    "embr_ncc": menr_embr_neural_crest
}


org_topics_to_show = [
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
# something wrong with "all" topics .. rerun

embr_progenitor_topics_to_show = [
    "topics_region_Topic_1_binarized",
    "topics_region_Topic_3_binarized",
    "topics_region_Topic_4_binarized",
    "topics_region_Topic_8_binarized",
    "topics_region_Topic_10_binarized",
    "topics_region_Topic_11_binarized",
    "topics_region_Topic_16_binarized",
    "topics_region_Topic_17_binarized",
    "topics_region_Topic_21_binarized",
    "topics_region_Topic_22_binarized",
    "topics_region_Topic_28_binarized",
    "topics_region_Topic_29_binarized",
    "topics_region_Topic_31_binarized",
    "topics_region_Topic_32_binarized",
    "topics_region_Topic_36_binarized",
    "topics_region_Topic_40_binarized",
    "topics_region_Topic_41_binarized",
    "topics_region_Topic_44_binarized",
    "topics_region_Topic_45_binarized",
    "topics_region_Topic_49_binarized",
    "topics_region_Topic_51_binarized",
    "topics_region_Topic_57_binarized",
    "topics_region_Topic_58_binarized",
    "topics_region_Topic_59_binarized"
]

embr_neuron_topics_to_show = [
    "topics_region_Topic_1_binarized",
    "topics_region_Topic_3_binarized",
    "topics_region_Topic_5_binarized",
    "topics_region_Topic_6_binarized",
    "topics_region_Topic_7_binarized",
    "topics_region_Topic_8_binarized",
    "topics_region_Topic_9_binarized",
    "topics_region_Topic_10_binarized",
    "topics_region_Topic_11_binarized",
    "topics_region_Topic_12_binarized",
    "topics_region_Topic_13_binarized",
    "topics_region_Topic_14_binarized",
    "topics_region_Topic_15_binarized",
    "topics_region_Topic_17_binarized",
    "topics_region_Topic_18_binarized",
    "topics_region_Topic_19_binarized",
    "topics_region_Topic_22_binarized",
    "topics_region_Topic_24_binarized",
    "topics_region_Topic_26_binarized",
    "topics_region_Topic_27_binarized",
    "topics_region_Topic_29_binarized",
    "topics_region_Topic_30_binarized"
]

embr_neural_crest_topics_to_show = [
    "topics_region_Topic_1_binarized",
    "topics_region_Topic_4_binarized",
    "topics_region_Topic_7_binarized",
    "topics_region_Topic_10_binarized",
    "topics_region_Topic_11_binarized",
    "topics_region_Topic_12_binarized",
    "topics_region_Topic_13_binarized",
    "topics_region_Topic_15_binarized",
    "topics_region_Topic_19_binarized",
    "topics_region_Topic_21_binarized",
    "topics_region_Topic_22_binarized",
    "topics_region_Topic_29_binarized"
]

topics_to_show_d = {
    "organoid": org_topics_to_show,
    "embr_prog": embr_progenitor_topics_to_show,
    "embr_neu": embr_neuron_topics_to_show,
    "embr_ncc": embr_neural_crest_topics_to_show
}

import numpy as np
import os
def load_motif(n, d):
    motifs = []
    names = []
    with open(os.path.join(d, f"{n}.cb")) as f:
        tmp = None
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                names.append(l.replace(">", ""))
                if tmp is not None:
                    tmp = np.array(tmp)
                    tmp = (tmp.T / tmp.sum(1)).T
                    motifs.append(tmp)
                tmp = []
            else:
                tmp.append([float(x) for x in l.split()])
        tmp = np.array(tmp)
        tmp = (tmp.T / tmp.sum(1)).T
        motifs.append(tmp)
    return motifs, names

from tqdm import tqdm
import scanpy as sc
from tangermeme.tools.tomtom import tomtom
import torch
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import seaborn as sns

plot_dir = "/staging/leuven/stg_00002/lcb/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_1/draft/plots"

sc.settings.figdir = plot_dir

samples = [
    "organoid",
    "embr_prog",
    "embr_neu",
    "embr_ncc"
]

all_motifs = []
motif_sub_names = []
motif_names = []
for sample in tqdm(samples):
    for topic in tqdm(topics_to_show_d[sample], total = len(topics_to_show_d[sample])):
        for motif_name in menr_d[sample][topic].motif_enrichment.index:
            if motif_name in motif_names:
                continue
            _motifs, _m_sub_names = load_motif(
                motif_name,
                "/staging/leuven/stg_00002/lcb/cbravo/cluster_motif_collection_V10_no_desso_no_factorbook/cluster_buster/"
            )
            all_motifs.extend(_motifs)
            motif_sub_names.extend(_m_sub_names)
            motif_names.extend(np.repeat(motif_name, len(_motifs)))


sample_to_motif_enrichment = []
for sample in tqdm(samples):
    tmp = pd.concat(
        [
            menr_d[sample][topic].motif_enrichment
            for topic in topics_to_show_d[sample]
        ]
    )
    tmp["sample"] = sample
    sample_to_motif_enrichment.append(
        tmp
    )

all_motif_enrichment_res = pd.concat(sample_to_motif_enrichment)

motif_name_to_max_NES = all_motif_enrichment_res \
    .reset_index() \
    .groupby(["sample", "index"])["NES"].max() \
    .reset_index() \
    .pivot_table(index = "index", columns = "sample", values = "NES") \
    .fillna(0) \
    .loc[motif_names]
    
motif_name_to_max_NES.insert(4, "motif_sub_name", motif_sub_names)

motif_name_to_max_NES = motif_name_to_max_NES.reset_index().rename({"index": "motif_name"}, axis = 1).set_index("motif_sub_name")

t_all_motifs = [
    torch.from_numpy(m).T for m in tqdm(all_motifs)
]

pvals, scores, offsets, overlaps, strands = tomtom(
    t_all_motifs,
    t_all_motifs
)

evals = pvals.numpy() * len(all_motifs)

adata_motifs = sc.AnnData(
    evals,
    obs = motif_name_to_max_NES.loc[motif_sub_names]   
)

sc.tl.pca(adata_motifs)
sc.pp.neighbors(adata_motifs)
sc.tl.tsne(adata_motifs)

sc.tl.leiden(adata_motifs, resolution = 2)

sc.pl.tsne(adata_motifs, color = ["leiden"], save = "_leiden_motifs_org_and_embr.pdf")


adata_motifs.obs = adata_motifs.obs \
    .merge(
        adata_motifs.obs.groupby("leiden", observed = True)[["embr_ncc", "embr_neu", "embr_prog", "organoid"]].transform("max"),
        left_index = True, right_index = True, suffixes = ("", "_max")
    )

leiden_to_best_motif = pd.DataFrame(
    index = adata_motifs.obs.leiden.unique(),
    columns = ["embr_ncc", "embr_neu", "embr_prog", "organoid"]
)
for s in ["embr_ncc", "embr_neu", "embr_prog", "organoid"]:    
    tmp = adata_motifs.obs.loc[adata_motifs.obs[s] != 0].loc[
        adata_motifs.obs[s] == adata_motifs.obs[f"{s}_max"], ["motif_name", "leiden"]].drop_duplicates().set_index("leiden")
    leiden_to_best_motif.loc[tmp.index, s] = tmp["motif_name"]

def to_logo(x):
    if not isinstance(x, str):
        return np.nan
    else:
        return f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{x}.png" width="200" >'

for x in leiden_to_best_motif.columns:
    leiden_to_best_motif[x] = leiden_to_best_motif[x].apply(to_logo)

leiden_to_best_motif.sort_index().to_html(
    os.path.join(plot_dir, "leiden_to_best_motif_embr_and_org.html"),
    escape = False,
    col_space = 80
)

leiden_to_region_set = adata_motifs.obs[["motif_name", "leiden"]].merge(all_motif_enrichment_res[["sample", "Region_set"]], left_on = "motif_name", right_index= True)[["leiden", "sample", "Region_set"]].drop_duplicates()

leiden_to_region_set["region_set_name"] = leiden_to_region_set["sample"] + "___" + leiden_to_region_set.Region_set

leiden_to_region_set_np = np.zeros((len(set(leiden_to_region_set["region_set_name"])), len(set(leiden_to_region_set["leiden"]))))

region_sets = list(set(leiden_to_region_set["region_set_name"]))

for i,region_set in enumerate(region_sets):
    leiden_to_region_set_np[
        i, leiden_to_region_set.loc[leiden_to_region_set["region_set_name"] == region_set, "leiden"
    ].values.astype(int)] = 1

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=20, random_state=0).fit(leiden_to_region_set_np) # apply kmeans on Z
labels=kmeans.labels_

```
