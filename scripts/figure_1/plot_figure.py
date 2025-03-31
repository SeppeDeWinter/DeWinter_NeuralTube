# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "matplotlib",
#   "scanpy",
#   "numpy",
#   "pandas",
#   "tqdm",
#   "tangermeme",
#   "torch",
#   "scipy",
#   "harmonypy",
#   "pycistarget @ git+https://github.com/aertslab/pycistarget",
#   "setuptools",
#   "tables",
#   "h5py",
#   "igraph",
#   "leidenalg"
# ]
# ///

print("importing modules...")

import matplotlib.pyplot as plt
import matplotlib
import scanpy as sc
import numpy as np
import pandas as pd
import json
from pycistarget.input_output import read_hdf5
import os
from tqdm import tqdm
from tangermeme.tools.tomtom import tomtom
import torch
from scipy import sparse


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


###############################################################################################################
#                                             LOAD DATA                                                       #
###############################################################################################################

print("LOADING DATA")

# load organoid RNA data and subset for ATAC cells
adata_organoid = sc.read_h5ad("../data_prep_new/organoid_data/RNA/GEX_adata.h5ad")

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

organoid_atac_cell_names = []
with open("../data_prep_new/organoid_data/ATAC/cell_names.txt") as f:
    for l in f:
        bc, sample_id = l.strip().split("-1", 1)
        sample_id = sample_id.split("___")[-1]
        organoid_atac_cell_names.append(bc + "-1" + "-" + sample_id_to_num[sample_id])

adata_organoid = adata_organoid[
    list(set(adata_organoid.obs_names) & set(organoid_atac_cell_names))
].copy()

# load embryo data and subset for ATAC cells
adata_embryo = sc.read_h5ad("../data_prep_new/embryo_data/RNA/adata_raw_filtered.h5ad")
cell_data_embryo = pd.read_csv(
    "../data_prep_new/embryo_data/RNA/cell_data.csv", index_col=0
)

adata_embryo = adata_embryo[cell_data_embryo.index].copy()
adata_embryo.obs = cell_data_embryo.loc[adata_embryo.obs_names]

embryo_atac_cell_names = []
with open("../data_prep_new/embryo_data/ATAC/cell_names.txt") as f:
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

sc.settings.figdir = "test_figs"

sc.pl.umap(adata_organoid, color=["Annotation_Type_1"], save="_organoid_annotation.pdf")

adata_embryo.obs["leiden_0.1"] = pd.Categorical(adata_embryo.obs["leiden_0.1"])
sc.pl.umap(adata_embryo, color=["Classes", "leiden_0.1"], save="_embryo_annotation.pdf")
sc.pl.umap(adata_embryo, color=["SIX1", "NEUROD1"], save="_embryo_periph_neu_mark.pdf")


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

sc.pl.umap(
    adata_embryo_progenitor,
    color=["FOXA2", "NKX2-2", "PAX6"],
    s=50,
    layer="log_cpm",
    vmax=1.5,
    save="_dv_embr.pdf",
    components=["2, 3"],
)

sc.pl.umap(
    adata_embryo_progenitor,
    color=["SIX6", "SIX3", "FOXG1", "EMX2", "BARHL1", "EN1", "HOXB3", "PAX8"],
    s=50,
    layer="log_cpm",
    vmax=1.5,
    save="_ap_embr.pdf",
    components=["1, 2"],
)

adata_organoid_progenitor = adata_organoid[
    adata_organoid.obs.Annotation_progenitor_step_2 == "Progenitor"
].copy()

dim_red_2d(adata_organoid_progenitor)

sc.pl.umap(
    adata_organoid_progenitor,
    color=[
        "FOXA2",
        "NKX2-2",
        "PAX6",
        "PAX7",
        "ZIC1",
        "WNT1",
        "GDF7",
        "SIX6",
        "SIX3",
        "FOXG1",
        "EMX2",
        "BARHL1",
        "EN1",
        "HOXB3",
    ],
    s=50,
    layer="log_cpm",
    vmax=1.5,
    save="_dv_org.pdf",
)

###############################################################################################################
#                                             subset neuron                                                   #
###############################################################################################################

print("GETTING NEURONS")

adata_embryo_neuron = adata_embryo[
    adata_embryo.obs.COMMON_ANNOTATION == "neuron"
].copy()

dim_red_2d(adata_embryo_neuron, n_pcs=10)

sc.pl.umap(
    adata_embryo_neuron,
    color=[
        "SOX2",
        "ASCL1",
        "GAD1",
        "UNCX",
        "SST",
        "GATA2",
        "ISL1",
        "ONECUT2",
        "EBF1",
        "SIX6",
        "SIX3",
        "FOXG1",
        "EMX2",
        "BARHL1",
        "EN1",
        "HOXB3",
    ],
    s=200,
    layer="log_cpm",
    vmax=1.5,
    save="_neu_embr.pdf",
)

adata_organoid_neuron = adata_organoid[
    adata_organoid.obs.COMMON_ANNOTATION == "neuron"
].copy()

dim_red_2d(adata_organoid_neuron)

sc.pl.umap(
    adata_organoid_neuron,
    color=[
        "SOX2",
        "ASCL1",
        "GAD1",
        "UNCX",
        "SST",
        "GATA2",
        "ISL1",
        "ONECUT2",
        "EBF1",
        "annotation_most_detailed",
        "phase",
    ],
    s=100,
    layer="log_cpm",
    vmax=1.5,
    save="_neu_org.pdf",
)


###############################################################################################################
#                                             subset neural crest                                             #
###############################################################################################################

print("GETTING NEURAL CREST")

adata_embryo_neural_crest = adata_embryo[
    [
        x in ["neural crest", "peripheral neuron", "mesenchyme", "otic placode"]
        for x in adata_embryo.obs.COMMON_ANNOTATION
    ]
].copy()

dim_red_2d(adata_embryo_neural_crest, n_pcs=20)

sc.pl.umap(
    adata_embryo_neural_crest,
    color=[
        "GRHL2",
        "ZEB2",
        "ZEB1",
        "SOX9",
        "SOX10",
        "TFAP2A",
        "TFAP2B",
        "FOXD3",
        "TWIST1",
        "FOXC1",
        "PAX2",
        "TP63",
        "NEUROD1",
    ],
    s=200,
    layer="log_cpm",
    vmax=2.0,
    save="_nc_embr.pdf",
)


adata_organoid_neural_crest = adata_organoid[
    [
        x in ["neural crest", "peripheral neuron", "mesenchyme", "otic placode"]
        for x in adata_organoid.obs.COMMON_ANNOTATION
    ]
].copy()


dim_red_2d(adata_organoid_neural_crest)

sc.pl.umap(
    adata_organoid_neural_crest,
    color=[
        "GRHL2",
        "ZEB2",
        "ZEB1",
        "SOX9",
        "SOX10",
        "TFAP2A",
        "TFAP2B",
        "FOXD3",
        "TWIST1",
        "FOXC1",
        "PAX2",
        "TP63",
        "NEUROD1",
    ],
    s=100,
    layer="log_cpm",
    vmax=1.5,
    save="_nc_org.pdf",
)

adata_organoid.write("adata_organoid.h5ad")
adata_organoid_progenitor.write("adata_organoid_progenitor.h5ad")
adata_organoid_neural_crest.write("adata_organoid_neural_crest.h5ad")
adata_organoid_neuron.write("adata_organoid_neuron.h5ad")

adata_embryo.write("adata_embryo.h5ad")
adata_embryo_progenitor.write("adata_embryo_progenitor.h5ad")
adata_embryo_neural_crest.write("adata_embryo_neural_crest.h5ad")
adata_embryo_neuron.write("adata_embryo_neuron.h5ad")

exp_progenitor_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")
exp_progenitor_organoid = adata_organoid_progenitor.to_df(layer="log_cpm")

exp_neuron_embryo = adata_embryo_neuron.to_df(layer="log_cpm")
exp_neuron_organoid = adata_organoid_neuron.to_df(layer="log_cpm")

exp_neural_crest_embryo = adata_embryo_neural_crest.to_df(layer="log_cpm")
exp_neural_crest_organoid = adata_organoid_neural_crest.to_df(layer="log_cpm")

###############################################################################################################
#                                             cell cycle scoring                                              #
###############################################################################################################

print("SCORING CELL CYCLE")

cell_cycle_genes = [
    x.strip() for x in open("../data_prep_new/regev_lab_cell_cycle_genes.txt")
]
s_genes = cell_cycle_genes[0:43]
g2m_genes = cell_cycle_genes[43:]

sc.tl.score_genes_cell_cycle(
    adata_embryo,
    s_genes=list(set(s_genes) & set(adata_embryo.var_names)),
    g2m_genes=list(set(g2m_genes) & set(adata_embryo.var_names)),
)

sc.tl.score_genes_cell_cycle(
    adata_organoid,
    s_genes=list(set(s_genes) & set(adata_organoid.var_names)),
    g2m_genes=list(set(g2m_genes) & set(adata_organoid.var_names)),
)

df_cc_org = pd.DataFrame(
    {
        p: (
            adata_organoid.obs.query("phase == @p")
            .groupby("COMMON_ANNOTATION")["phase"]
            .count()
            / adata_organoid.obs.groupby("COMMON_ANNOTATION")["phase"].count()
        )
        for p in ["G1", "G2M", "S"]
    }
)

df_cc_embr = pd.DataFrame(
    {
        p: (
            adata_embryo.obs.query("phase == @p")
            .groupby("COMMON_ANNOTATION")["phase"]
            .count()
            / adata_embryo.obs.groupby("COMMON_ANNOTATION")["phase"].count()
        )
        for p in ["G1", "G2M", "S"]
    }
)

sorted_idx = (df_cc_org["G1"] + df_cc_embr["G1"]).dropna().sort_values().index

df_cc_org = df_cc_org.loc[sorted_idx]
df_cc_embr = df_cc_embr.loc[sorted_idx]

###############################################################################################################
#                                             dot plot and menr                                               #
###############################################################################################################

print("GENERATING MENR DOTPLOT")


region_topic_nc = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
    index_col=0,
)
region_topic_prog = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
    index_col=0,
)
region_topic_neu = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv", index_col=0
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
    os.path.join("../data_prep_new/embryo_data/ATAC/embryo_atac_merged_bin_matrix.npz")
)

embryo_region_names = []
with open(
    os.path.join("../data_prep_new/embryo_data/ATAC/embry_atac_region_names.txt")
) as f:
    for l in f:
        embryo_region_names.append(l.strip())

embryo_cell_names = []
with open(
    os.path.join("../data_prep_new/embryo_data/ATAC/embry_atac_cell_names.tx")
) as f:
    for l in f:
        embryo_cell_names.append(l.strip())

common_regions = list(set(region_topic.index) & set(embryo_region_names))

X = pd.DataFrame(
    embryo_bin_matrix.todense(), index=embryo_region_names, columns=embryo_cell_names
).loc[common_regions]

embryo_cell_topic = region_topic.loc[common_regions].T @ X

embryo_cell_topic = embryo_cell_topic.T


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

menr = read_hdf5("../data_prep/data/cistarget_topics.hdf5")

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

bc_to_topic = (
    embryo_cell_topic.pivot(index="bc", columns="topic_grp", values="prob")[
        topics_order
    ]
    .fillna(0)
    .T.idxmax()
    .to_dict()
)

bc_to_topic = {x.replace("-1-", "-1___"): bc_to_topic[x] for x in bc_to_topic}

common_cells = list(set(bc_to_topic.keys()) & set(adata_embryo.obs_names))

adata_embryo_bin_topic = adata_embryo[common_cells].copy()

adata_embryo_bin_topic.X = adata_embryo_bin_topic.layers["count"]

normalize(adata_embryo_bin_topic)

adata_embryo_bin_topic.obs["topic"] = [
    bc_to_topic[bc] for bc in adata_embryo_bin_topic.obs_names
]


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


all_motif_enrichment_res = []
for topic in topics_to_show:
    all_motif_enrichment_res.append(menr[topic].motif_enrichment)

all_motif_enrichment_res = pd.concat(all_motif_enrichment_res)

motif_name_to_max_NES = (
    all_motif_enrichment_res.reset_index()
    .pivot_table(index="index", columns="Region_set", values="NES")
    .fillna(0)
    .max(1)
)

all_motifs = []
motif_sub_names = []
motif_names = []
for topic in tqdm(topics_to_show):
    for motif_name in menr[topic].motif_enrichment.index:
        if motif_name in motif_names:
            continue
        _motifs, _m_sub_names = load_motif(
            motif_name, "../../../../motif_collection/cluster_buster"
        )
        all_motifs.extend(_motifs)
        motif_sub_names.extend(_m_sub_names)
        motif_names.extend(np.repeat(motif_name, len(_motifs)))

t_all_motifs = [torch.from_numpy(m).T for m in tqdm(all_motifs)]

pvals, scores, offsets, overlaps, strands = tomtom(t_all_motifs, t_all_motifs)

evals = pvals.numpy() * len(all_motifs)

adata_motifs = sc.AnnData(
    evals,
    obs=pd.DataFrame(
        index=motif_sub_names,
        data={
            "motif_name": motif_names,
            "max_NES": motif_name_to_max_NES.loc[motif_names].values,
        },
    ),
)

sc.tl.pca(adata_motifs)
sc.pp.neighbors(adata_motifs)
sc.tl.tsne(adata_motifs)

sc.tl.leiden(adata_motifs, resolution=2)
sc.pl.tsne(adata_motifs, color=["leiden"], save="_leiden_motifs.pdf")

adata_motifs.obs["max_NES_leiden"] = adata_motifs.obs.groupby("leiden", observed=True)[
    "max_NES"
].transform("max")

leiden_to_best_motif = (
    adata_motifs.obs.query("max_NES == max_NES_leiden")
    .reset_index(drop=True)
    .drop_duplicates()
    .reset_index(drop=True)
    .sort_values("leiden")
)

leiden_to_best_motif["Logo"] = [
    f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{m}.png" width="200" >'
    for m in leiden_to_best_motif["motif_name"]
]

leiden_to_best_motif[["motif_name", "leiden", "max_NES", "Logo"]].to_html(
    os.path.join("test_figs", "leiden_to_best_motif.html"), escape=False, col_space=80
)

motif_to_leiden = {}
for m in adata_motifs.obs.motif_name:
    clusters, count = np.unique(
        adata_motifs.obs.query("motif_name == @m")["leiden"], return_counts=True
    )
    motif_to_leiden[m] = clusters[count.argmax()]

leiden_to_NES = (
    all_motif_enrichment_res.reset_index()
    .pivot_table(index="index", columns="Region_set", values="NES")
    .fillna(0)
    .groupby(motif_to_leiden)
    .max()
)

leiden_to_n_hits = (
    all_motif_enrichment_res.reset_index()
    .pivot_table(index="index", columns="Region_set", values="Motif_hits")
    .fillna(0)
    .groupby(motif_to_leiden)
    .max()
)


def calc_IC_per_pos(pwm, bg=np.repeat(0.25, 4), pc=1e-4):
    return (np.log2((pwm + pc) / bg) * pwm).sum(1)


def get_consensus_sequence(pwm, letter_order=["A", "C", "G", "T"], min_ic=0.8):
    ic_above_min = np.where(calc_IC_per_pos(pwm) > min_ic)[0]
    to_keep = np.zeros(pwm.shape[0], dtype=bool)
    to_keep[ic_above_min.min() : ic_above_min.max() + 1] = True
    consensus = [letter_order[x] for x in pwm.argmax(1)[to_keep]]
    score = (pwm.max(1) * calc_IC_per_pos(pwm))[to_keep]
    return consensus, score


def tex_font(l, s, color="black"):
    return (
        r"{\fontsize{"
        + f"{s}"
        + r"pt}{3em}\selectfont \color{"
        + color
        + r"} "
        + l
        + r"}"
    )


letter_to_color = {"G": "orange", "T": "red", "A": "green", "C": "blue"}

adata_motifs.obs["IC"] = [
    calc_IC_per_pos(all_motifs[motif_sub_names.index(m)]).sum()
    for m in adata_motifs.obs_names
]


def get_consensus_logo_for_motif_name(motif_name):
    motif_sub_name = (
        adata_motifs.obs.query("motif_name == @motif_name")
        .sort_values("IC", ascending=False)
        .index[0]
    )
    pwm = all_motifs[motif_sub_names.index(motif_sub_name)]
    ic = calc_IC_per_pos(pwm)
    consensus, score = get_consensus_sequence(pwm)
    title = r"".join(
        [tex_font(c, s * 2.5, letter_to_color[c]) for c, s in zip(consensus, score)]
    )
    return title


leiden_to_best_motif_d = leiden_to_best_motif.set_index("leiden")[
    "motif_name"
].to_dict()

leiden_to_show = [
    "0",
    "3",
    # "4",
    # "6",
    "8",
    "11",
    # "17",
    "18",
    "19",
    "21",
    "23",
    "24",
    "26",
    "30",
    "31",
    # "32",
    # "35",
    # "39",
    "41",
    # "42",
    "45",
    "47",
    # "50",
    # "52",
]

###


cell_topic_organoid = pd.read_table(
    os.path.join("../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv")
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

marker_TFs = marker_genes

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

df_to_plot_avg_organoid = avg_organoid.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct_organoid = pct_cell_organoid.copy().T.loc[list(marker_TFs), topics_order]

marker_to_idx_topic_order = {
    m: topics_order.index(v)
    for m, v in zip(
        df_to_plot_avg_organoid.T.idxmax().index,
        df_to_plot_avg_organoid.T.idxmax().values,
    )
}
marker_order = sorted(
    marker_to_idx_topic_order, key=lambda l: marker_to_idx_topic_order[l]
)

df_to_plot_avg_organoid = df_to_plot_avg_organoid.loc[marker_order]
df_to_plot_pct_organoid = df_to_plot_pct_organoid.loc[marker_order]

m_avg_organoid = df_to_plot_avg_organoid.reset_index().melt(id_vars="index")
m_pct_organoid = df_to_plot_pct_organoid.reset_index().melt(id_vars="index")

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

df_to_plot_avg_embryo = avg_embryo.copy().T.loc[list(marker_TFs), topics_order]
df_to_plot_pct_embryo = pct_cell_embryo.copy().T.loc[list(marker_TFs), topics_order]

df_to_plot_avg_embryo = df_to_plot_avg_embryo.loc[marker_order]
df_to_plot_pct_embryo = df_to_plot_pct_embryo.loc[marker_order]

m_avg_embryo = df_to_plot_avg_embryo.reset_index().melt(id_vars="index")
m_pct_embryo = df_to_plot_pct_embryo.reset_index().melt(id_vars="index")

df_to_plot_NES = leiden_to_NES.copy().loc[leiden_to_show, topics_order]
df_to_plot_hits = leiden_to_n_hits.copy().loc[leiden_to_show, topics_order]

leiden_to_idx_topic_order = {
    l: topics_order.index(v)
    for l, v in zip(df_to_plot_NES.T.idxmax().index, df_to_plot_NES.T.idxmax().values)
}
leiden_order = sorted(
    leiden_to_idx_topic_order, key=lambda l: leiden_to_idx_topic_order[l]
)

df_to_plot_NES = df_to_plot_NES.loc[leiden_order]
df_to_plot_hits = df_to_plot_hits.loc[leiden_order]

df_to_plot_NES.index = [
    l + "-" + get_consensus_logo_for_motif_name(leiden_to_best_motif_d[l])
    for l in df_to_plot_NES.index
]

m_NES = df_to_plot_NES.reset_index().melt(id_vars="index")
m_hits = df_to_plot_hits.reset_index().melt(id_vars="index")

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{color}\usepackage{xcolor}")

pgf_with_latex = {
    "text.usetex": True,  # use LaTeX to write all text
    "pgf.rcfonts": False,  # Ignore Matplotlibrc
    "pgf.preamble": r"\usepackage{xcolor}\definecolor{orange}{RGB}{255,165,3}\definecolor{blue}{RGB}{28,0,255}\definecolor{red}{RGB}{255,0,0}\definecolor{green}{RGB}{0,128,0}",  # xcolor for colours
}
matplotlib.rcParams.update(pgf_with_latex)

fig, axs = plt.subplots(figsize=(10, 12), sharex=True, nrows=3, height_ratios=[2, 2, 2])
ax = axs[0]
m_avg = m_avg_organoid
m_pct = m_pct_organoid
_ = ax.scatter(
    x=m_avg["topic"],
    y=m_avg["index"],
    s=m_pct["value"] * 50,
    c=m_avg["value"],
    lw=1,
    edgecolor="black",
    cmap="viridis",
    vmin=0,
    vmax=1.5,
)
ax.set_xticklabels(
    df_to_plot_avg_organoid.columns, rotation=45, ha="right", rotation_mode="anchor"
)
ax.tick_params(axis="y", which="major", labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
ax.set_ylabel("Organoid")
m_avg = m_avg_embryo
m_pct = m_pct_embryo
ax = axs[1]
_ = ax.scatter(
    x=m_avg["topic"],
    y=m_avg["index"],
    s=m_pct["value"] * 50,
    c=m_avg["value"],
    lw=1,
    edgecolor="black",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
ax.set_xticklabels(
    df_to_plot_avg_embryo.columns, rotation=45, ha="right", rotation_mode="anchor"
)
ax.tick_params(axis="y", which="major", labelsize=6)
ax.grid("gray")
ax.set_axisbelow(True)
ax.set_ylabel("Embryo")
ax = axs[2]
_ = ax.scatter(
    x=m_NES["Region_set"],
    y=m_NES["index"],
    s=m_hits["value"] / 50,
    c=m_NES["value"],
    lw=1,
    edgecolor="black",
    cmap="viridis",
    vmin=3,
    vmax=10,
)
ax.set_xticklabels(
    df_to_plot_NES.columns, rotation=45, ha="right", rotation_mode="anchor"
)
ax.grid("gray")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(
    os.path.join("test_figs", "dotplot_exp_nes_organoid_embryo.pdf"), backend="pgf"
)


###############################################################################################################
#                                             plotting functions                                              #
###############################################################################################################


def plot_umap_w_annotation(a, key, color_dict, ax):
    ax.set_axis_off()
    ax.scatter(
        a.obsm["X_umap"][:, 0],
        a.obsm["X_umap"][:, 1],
        c=[color_dict[x] for x in a.obs[key]],
        s=0.5,
    )


def plot_umap_legend(color_dict, ax, ncols, bbox_to_anchor=(0, 3), loc="upper left"):
    ax.set_axis_off()
    legend_elements = [
        ax.scatter([], [], color=color_dict[x], label=x, s=30) for x in color_dict
    ]
    ax.legend(
        handles=legend_elements,
        ncols=ncols,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
    )


def rgb_scatter_plot(
    x,
    y,
    r_values,
    g_values,
    b_values,
    ax,
    g_cut=0,
    e_thr=0.4,
    r_name="",
    g_name="",
    b_name="",
    r_vmin=None,
    r_vmax=None,
    g_vmin=None,
    g_vmax=None,
    b_vmin=None,
    b_vmax=None,
):
    def normalize_channel(values, vmin=None, vmax=None):
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        return np.clip((values - vmin) / (vmax - vmin), 0, 1)
    ax.set_axis_off()
    r_normalized = normalize_channel(r_values, r_vmin, r_vmax)
    g_normalized = normalize_channel(g_values, g_vmin, g_vmax)
    b_normalized = normalize_channel(b_values, b_vmin, b_vmax)
    colors = np.column_stack((r_normalized, g_normalized, b_normalized))
    greens = (colors[:, 1] / (colors.sum(1) + 1e-5)) > e_thr
    no_expressors = colors.max(1) <= g_cut
    ax.scatter(x[no_expressors], y[no_expressors], color="lightgray", s=1)
    s = np.argsort(colors.sum(1))[~no_expressors]
    ax.scatter(
        x[s],
        y[s],
        c=colors[s, :],
        edgecolors=[colors[x] if not greens[x] else "black" for x in s],
        s=[3 if not greens[x] else 6 for x in s],
        lw=0.5,
    )
    ax.text(0.8, 0.98, r_name, color="red", ha="left", va="top", transform=ax.transAxes)
    ax.text(
        0.8,
        0.88,
        g_name,
        color="green",
        ha="left",
        va="top",
        transform=ax.transAxes,
        path_effects=[
            matplotlib.patheffects.withStroke(linewidth=1, foreground="black")
        ],
    )
    ax.text(
        0.8, 0.78, b_name, color="blue", ha="left", va="top", transform=ax.transAxes
    )


def plot_cc_bar(ax, df, mapping_names=None):
    prev_phase = []
    for phase in ["G1", "G2M", "S"]:
        _ = ax.barh(
            y=np.arange(len(df)),
            width=df[phase] * 100,
            left=df[prev_phase].sum(1) * 100 if len(prev_phase) > 0 else None,
            color=color_dict["cell_cycle"][phase],
            label=phase,
        )
        prev_phase.append(phase)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(
        [mapping_names[x] if mapping_names is not None else x for x in df.index]
    )
    ax.spines[["right", "top"]].set_visible(False)


def plot_NES_EXP_dotplot(
    axs: list[plt.Axes],
    m_avg_organoid,
    m_pct_organoid,
    df_to_plot_avg_organoid,
    m_avg_embryo,
    m_pct_embryo,
    df_to_plot_avg_embryo,
    m_NES,
    m_hits,
):
    def rename(x):
        d = {"progenitor": "prog.", "neuron": "neu.", "neural_crest": "n.c."}
        cell_type, topic_num = x.rsplit("_", 1)
        cell_type = cell_type.replace("_Topic", "")
        return d[cell_type] + " " + "$T_{" + topic_num + "}$"
    ax = axs[0]
    m_avg = m_avg_organoid
    m_pct = m_pct_organoid
    _ = ax.scatter(
        x=m_avg["topic"],
        y=m_avg["index"],
        s=m_pct["value"] * 25,
        c=m_avg["value"],
        lw=1,
        edgecolor="black",
        cmap="OrRd",
        vmin=0,
        vmax=1.5,
    )
    ax.tick_params(axis="y", which="major", labelsize=4)
    ax.grid("gray")
    ax.set_axisbelow(True)
    m_avg = m_avg_embryo
    m_pct = m_pct_embryo
    ax = axs[1]
    _ = ax.scatter(
        x=m_avg["topic"],
        y=m_avg["index"],
        s=m_pct["value"] * 25,
        c=m_avg["value"],
        lw=1,
        edgecolor="black",
        cmap="OrRd",
        vmin=0,
        vmax=1,
    )
    ax.tick_params(axis="y", which="major", labelsize=4)
    ax.grid("gray")
    ax.set_axisbelow(True)
    ax = axs[2]
    _ = ax.scatter(
        x=m_NES["Region_set"],
        y=m_NES["index"],
        s=m_hits["value"] / 100,
        c=m_NES["value"],
        lw=1,
        edgecolor="black",
        cmap="GnBu",
        vmin=3,
        vmax=10,
    )
    ax.set_xticklabels(
        [rename(x) for x in df_to_plot_NES.columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.grid("gray")
    ax.set_axisbelow(True)
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])


color_dict = json.load(open("../color_maps.json"))

fig, ax = plt.subplots()
exp = adata_embryo_progenitor.to_df(layer="log_cpm")
r_values = exp["FOXA2"].values
g_values = exp["NKX2-2"].values
b_values = exp["PAX6"].values
rgb_scatter_plot(
    x=adata_embryo_progenitor.obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor.obsm["X_umap"][:, 2],
    ax=ax,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
)
fig.savefig("test_figs/rgb_test.png")

###############################################################################################################
#                                             plotting figure                                                 #
###############################################################################################################

print("GENERATING FIGURE 1")

N_PIXELS_PER_GRID = 50

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

plt.style.use(
    "/data/projects/c20/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/paper.mplstyle"
)

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{color}\usepackage{xcolor}")

pgf_with_latex = {
    "text.usetex": True,  # use LaTeX to write all text
    "pgf.rcfonts": False,  # Ignore Matplotlibrc
    "font.family": "sans-serif",
    "pgf.preamble": r"\usepackage{xcolor}\definecolor{orange}{RGB}{255,165,3}\definecolor{blue}{RGB}{28,0,255}\definecolor{red}{RGB}{255,0,0}\definecolor{green}{RGB}{0,128,0}",  # xcolor for colours
}
matplotlib.rcParams.update(pgf_with_latex)

fig = plt.figure()
width, height = fig.get_size_inches()
n_w_pixels = fig.get_dpi() * width
n_h_pixels = fig.get_dpi() * height
ncols = int((n_w_pixels) // N_PIXELS_PER_GRID)
nrows = int((n_h_pixels) // N_PIXELS_PER_GRID)
gs = fig.add_gridspec(
    nrows, ncols, wspace=0.05, hspace=0.1, left=0.05, right=0.97, bottom=0.05, top=0.95
)
# axes for schematic
current_x = 0
current_y = 0
ax_panel_a = fig.add_subplot(
    gs[current_y : current_y + nrows // 4, current_x : current_x + nrows // 4]
)
ax_panel_a.set_axis_off()
current_x += nrows // 4
ax_panel_b_org = fig.add_subplot(
    gs[current_y : nrows // 4 - 1, current_x + 2 : current_x + 2 + nrows // 4]
)
current_x += nrows // 4 + 2
ax_panel_b_embr = fig.add_subplot(
    gs[current_y : nrows // 4 - 1, current_x + 1 : current_x + 1 + nrows // 4]
)
ax_panel_b_legend = fig.add_subplot(gs[nrows // 4, nrows // 4 - 1 :])
plot_umap_w_annotation(
    adata_organoid, "COMMON_ANNOTATION", color_dict["all_cells"], ax_panel_b_org
)
plot_umap_w_annotation(
    adata_embryo, "COMMON_ANNOTATION", color_dict["all_cells"], ax_panel_b_embr
)
plot_umap_legend(
    color_dict["all_cells"], ax_panel_b_legend, ncols=len(color_dict["all_cells"]) // 2
)
_ = ax_panel_b_org.set_title(f"Organoid ({len(adata_organoid)} cells)")
_ = ax_panel_b_embr.set_title(f"Embryo ({len(adata_embryo)} cells)")
# axes for progenitor DV
current_y = nrows // 4 + 1
current_x = 0
ax_pannel_c_org_prog_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_prog_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y = nrows // 4 + 1
current_x += nrows // 8
ax_pannel_c_org_prog_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_prog_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
# axes for neuron markers
current_y = nrows // 4 + 1
current_x = (nrows // 8) * 2 + 1
ax_pannel_c_org_neu_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_neu_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y = nrows // 4 + 1
current_x += nrows // 8
ax_pannel_c_org_neu_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_neu_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
# Axes for neural crest markers
current_y = nrows // 4 + 1
current_x = (nrows // 8) * 4 + 2
ax_pannel_c_org_nc_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_nc_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y = nrows // 4 + 1
current_x += nrows // 8
ax_pannel_c_org_nc_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_c_embr_nc_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
# Axes for progenitor AP
current_y = 2 * nrows // 4 + 2
current_x = 0
ax_pannel_d_org_prog_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_d_embr_prog_1 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y = 2 * nrows // 4 + 2
current_x += nrows // 8
ax_pannel_d_org_prog_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
ax_pannel_d_embr_prog_2 = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
# axes for cell cycle
current_y = 3 * nrows // 4 + 2
current_x = 0
ax_pannel_e_org_cc = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_x += nrows // 8
ax_pannel_e_embr_cc = fig.add_subplot(
    gs[current_y : current_y + nrows // 8, current_x : current_x + nrows // 8]
)
current_y += nrows // 8
current_x = 2
ax_pannel_e_org_cc_bar = fig.add_subplot(
    gs[current_y : current_y + nrows // 8 - 3, current_x : current_x + nrows // 8 - 2]
)
current_x += nrows // 8 - 1
ax_pannel_e_embr_cc_bar = fig.add_subplot(
    gs[current_y : current_y + nrows // 8 - 3, current_x : current_x + nrows // 8 - 2],
)
current_y = current_y + nrows // 8 - 3
current_x = 0
ax_pannel_e_legend = fig.add_subplot(
    gs[current_y : current_y + 3, current_x : current_x + 2 * nrows // 8]
)
# ax for dotplot
current_y = 2 * nrows // 4 + 1
current_x = 2 * nrows // 8 + 5
ax_pannel_f_dotplot_1 = fig.add_subplot(gs[current_y : current_y + 7, current_x:])
current_y += 7
ax_pannel_f_dotplot_2 = fig.add_subplot(
    gs[current_y : current_y + 7, current_x:],
)
current_y += 7
ax_pannel_f_dotplot_3 = fig.add_subplot(
    gs[current_y : current_y + 8, current_x:],
)
# progenitor DV
rgb_scatter_plot(
    x=adata_embryo_progenitor.obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor.obsm["X_umap"][:, 2],
    r_values=exp_progenitor_embryo["ARX"].values,
    g_values=exp_progenitor_embryo["NKX2-2"].values,
    b_values=exp_progenitor_embryo["PAX6"].values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_c_embr_prog_1,
)
rgb_scatter_plot(
    x=adata_organoid_progenitor.obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_organoid["ARX"].values,
    g_values=exp_progenitor_organoid["NKX2-2"].values,
    b_values=exp_progenitor_organoid["PAX6"].values,
    r_name="ARX",
    g_name="NKX2-2",
    b_name="PAX6",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_c_org_prog_1,
)
rgb_scatter_plot(
    x=adata_embryo_progenitor.obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor.obsm["X_umap"][:, 2],
    r_values=exp_progenitor_embryo["IRX3"].values,
    g_values=exp_progenitor_embryo["ZIC1"].values,
    b_values=exp_progenitor_embryo["GDF7"].values,
    r_vmin=0,
    r_vmax=1.0,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_c_embr_prog_2,
)
rgb_scatter_plot(
    x=adata_organoid_progenitor.obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_organoid["IRX3"].values,
    g_values=exp_progenitor_organoid["ZIC1"].values,
    b_values=exp_progenitor_organoid["GDF7"].values,
    r_name="IRX3",
    g_name="ZIC1",
    b_name="GDF7",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_c_org_prog_2,
)
# progenitor ap
rgb_scatter_plot(
    x=adata_embryo_progenitor.obsm["X_umap"][:, 0],
    y=adata_embryo_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_embryo["SIX3"].values,
    g_values=exp_progenitor_embryo["EMX2"].values,
    b_values=exp_progenitor_embryo["FOXG1"].values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=0.8,
    ax=ax_pannel_d_embr_prog_1,
)
rgb_scatter_plot(
    x=adata_organoid_progenitor.obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_organoid["SIX3"].values,
    g_values=exp_progenitor_organoid["EMX2"].values,
    b_values=exp_progenitor_organoid["FOXG1"].values,
    r_name="SIX3",
    g_name="EMX2",
    b_name="FOXG1",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=0.8,
    ax=ax_pannel_d_org_prog_1,
)
rgb_scatter_plot(
    x=adata_embryo_progenitor.obsm["X_umap"][:, 0],
    y=adata_embryo_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_embryo["BARHL1"].values,
    g_values=exp_progenitor_embryo["EN1"].values,
    b_values=exp_progenitor_embryo["HOXB3"].values,
    r_vmin=0,
    r_vmax=1.0,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_d_embr_prog_2,
)
rgb_scatter_plot(
    x=adata_organoid_progenitor.obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor.obsm["X_umap"][:, 1],
    r_values=exp_progenitor_organoid["BARHL1"].values,
    g_values=exp_progenitor_organoid["EN1"].values,
    b_values=exp_progenitor_organoid["HOXB3"].values,
    r_name="BARHL1",
    g_name="EN1",
    b_name="HOXB3",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_d_org_prog_2,
)
# neuron
rgb_scatter_plot(
    x=adata_embryo_neuron.obsm["X_umap"][:, 0],
    y=adata_embryo_neuron.obsm["X_umap"][:, 1],
    r_values=exp_neuron_embryo["SOX2"].values,
    g_values=exp_neuron_embryo["ASCL1"].values,
    b_values=exp_neuron_embryo["ISL1"].values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.5,
    ax=ax_pannel_c_embr_neu_1,
)
rgb_scatter_plot(
    x=adata_organoid_neuron.obsm["X_umap"][:, 0],
    y=adata_organoid_neuron.obsm["X_umap"][:, 1],
    r_values=exp_neuron_organoid["SOX2"].values,
    g_values=exp_neuron_organoid["ASCL1"].values,
    b_values=exp_neuron_organoid["ISL1"].values,
    r_name="SOX2",
    g_name="ASCL1",
    b_name="ISL1",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=1.0,
    ax=ax_pannel_c_org_neu_1,
)
rgb_scatter_plot(
    x=adata_embryo_neuron.obsm["X_umap"][:, 0],
    y=adata_embryo_neuron.obsm["X_umap"][:, 1],
    r_values=exp_neuron_embryo["GAD1"].values,
    g_values=exp_neuron_embryo["UNCX"].values,
    b_values=exp_neuron_embryo["GATA2"].values,
    g_cut=-1,
    r_vmin=0,
    r_vmax=0.8,
    g_vmin=0,
    g_vmax=0.8,
    b_vmin=0,
    b_vmax=0.8,
    ax=ax_pannel_c_embr_neu_2,
)
rgb_scatter_plot(
    x=adata_organoid_neuron.obsm["X_umap"][:, 0],
    y=adata_organoid_neuron.obsm["X_umap"][:, 1],
    r_values=exp_neuron_organoid["GAD1"].values,
    g_values=exp_neuron_organoid["UNCX"].values,
    b_values=exp_neuron_organoid["GATA2"].values,
    r_name="GAD1",
    g_name="UNCX",
    b_name="GATA2",
    g_cut=-1,
    r_vmin=0,
    r_vmax=0.8,
    g_vmin=0,
    g_vmax=0.8,
    b_vmin=0,
    b_vmax=0.8,
    ax=ax_pannel_c_org_neu_2,
)
# neural_crest
rgb_scatter_plot(
    x=adata_embryo_neural_crest.obsm["X_umap"][:, 0],
    y=adata_embryo_neural_crest.obsm["X_umap"][:, 1],
    r_values=exp_neural_crest_embryo["GRHL2"].values,
    g_values=exp_neural_crest_embryo["ZEB2"].values,
    b_values=exp_neural_crest_embryo["SOX10"].values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=3.0,
    b_vmin=0,
    b_vmax=0.8,
    e_thr=0.8,
    ax=ax_pannel_c_embr_nc_1,
)
rgb_scatter_plot(
    x=adata_organoid_neural_crest.obsm["X_umap"][:, 0],
    y=adata_organoid_neural_crest.obsm["X_umap"][:, 1],
    r_values=exp_neural_crest_organoid["GRHL2"].values,
    g_values=exp_neural_crest_organoid["ZEB2"].values,
    b_values=exp_neural_crest_organoid["SOX10"].values,
    r_name="GRHL2",
    g_name="ZEB2",
    b_name="SOX10",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=3.0,
    b_vmin=0,
    b_vmax=0.8,
    e_thr=0.8,
    ax=ax_pannel_c_org_nc_1,
)
rgb_scatter_plot(
    x=adata_embryo_neural_crest.obsm["X_umap"][:, 0],
    y=adata_embryo_neural_crest.obsm["X_umap"][:, 1],
    r_values=exp_neural_crest_embryo["NEUROD1"].values,
    g_values=exp_neural_crest_embryo["PAX2"].values,
    b_values=exp_neural_crest_embryo["TWIST1"].values,
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=2.0,
    ax=ax_pannel_c_embr_nc_2,
)
rgb_scatter_plot(
    x=adata_organoid_neural_crest.obsm["X_umap"][:, 0],
    y=adata_organoid_neural_crest.obsm["X_umap"][:, 1],
    g_values=exp_neural_crest_organoid["PAX2"].values,
    r_values=exp_neural_crest_organoid["NEUROD1"].values,
    b_values=exp_neural_crest_organoid["TWIST1"].values,
    g_name="PAX2",
    r_name="NEUROD1",
    b_name="TWIST1",
    r_vmin=0,
    r_vmax=1.5,
    g_vmin=0,
    g_vmax=1.5,
    b_vmin=0,
    b_vmax=2.0,
    ax=ax_pannel_c_org_nc_2,
)
#
plot_umap_w_annotation(
    adata_embryo, "phase", color_dict["cell_cycle"], ax_pannel_e_embr_cc
)
plot_umap_w_annotation(
    adata_organoid, "phase", color_dict["cell_cycle"], ax_pannel_e_org_cc
)
plot_cc_bar(
    ax=ax_pannel_e_org_cc_bar,
    df=df_cc_org,
    mapping_names={
        "neuron": "neu.",
        "peripheral neuron": "periph. n.",
        "neural crest": "nc",
        "mesenchyme": "mes",
        "neuronal progenitor": "neu. prog.",
    },
)
plot_cc_bar(ax=ax_pannel_e_embr_cc_bar, df=df_cc_embr)
plot_umap_legend(
    color_dict=color_dict["cell_cycle"],
    ax=ax_pannel_e_legend,
    ncols=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -2),
)
ax_pannel_e_embr_cc_bar.set_yticklabels([])
plot_NES_EXP_dotplot(
    axs=[ax_pannel_f_dotplot_1, ax_pannel_f_dotplot_2, ax_pannel_f_dotplot_3],
    m_avg_organoid=m_avg_organoid,
    m_pct_organoid=m_pct_organoid,
    df_to_plot_avg_organoid=df_to_plot_avg_organoid,
    m_avg_embryo=m_avg_embryo,
    m_pct_embryo=m_pct_embryo,
    df_to_plot_avg_embryo=df_to_plot_avg_embryo,
    m_NES=m_NES,
    m_hits=m_hits,
)
fig.tight_layout()
fig.savefig("Figure_1.png", transparent=False)
try:
    fig.savefig("Figure_1.pdf", backend="pgf")
except Exception as e:
    print(e)
    print("PGF BACKEND NOT WORKING - USING DEFAULT")
    fig.savefig("Figure_1.pdf")
plt.close(fig)
