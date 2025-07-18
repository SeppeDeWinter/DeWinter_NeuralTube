import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import seaborn as sns
from tqdm import tqdm
import os
import scanpy as sc
import scipy
import logomaker

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def region_name_to_chrom_start_end(r: str) -> tuple[str, int, int]:
    chrom, start, end = r.replace("-", ":").split(":")
    return (chrom, int(start), int(end))

def seqlets_to_bed(
    df: pd.DataFrame,
    out_f: str,
    c_region_name: str = "region_name",
    c_start: str = "start",
    c_end: str = "end",
    c_name: str = "dbd_per_leiden",
    c_score: str = "mean_contrib"
) -> None:
    g_chroms = []
    g_starts = []
    g_ends   = []
    scores   = []
    names    = []
    for _, (region_name, start, end, name, score) in df[
        [c_region_name, c_start, c_end, c_name, c_score]
    ].iterrows():
        g_chrom, g_start, _ = region_name_to_chrom_start_end(region_name)
        g_chroms.append(g_chrom)
        g_starts.append(g_start + start)
        g_ends.append(g_start + end)
        scores.append(score)
        names.append(name)
    pd.DataFrame(
        data = {
            "chrom": g_chroms,
            "start": g_starts,
            "end": g_ends,
            "name": names,
            "score": scores
        }
    ).sort_values(["chrom", "start", "end"]) \
    .to_csv(
        path_or_buf=out_f,
        sep="\t",
        header=False,
        index=False
    )

def sanitize(s):
    return s.replace(' ', '_').replace('/', '_').replace(';', '')

def get_intersect(
        a: str,
        b: str
) -> int:
    r = subprocess.run(
        f"bedtools intersect -a {a} -b {b} -wa | sort -u | wc -l",
        shell=True, capture_output=True, text=True
    )
    return int(r.stdout.strip())


#####
# tSNE
####

seq_organoid = anndata.read_h5ad("../data_prep_new/organoid_data/mindi/organoid_seqlet_adata_no_na.h5ad")
seq_embryo = anndata.read_h5ad("../data_prep_new/embryo_data/mindi/embryo_seqlet_adata_no_na.h5ad")

a = seq_organoid.obs.groupby("dbd_per_leiden", observed = True).size()
b = seq_embryo.obs.groupby("dbd_per_leiden", observed = True).size()

common_dbd = list(set(a.index) & set(b.index))

scipy.stats.pearsonr(a.loc[common_dbd], b.loc[common_dbd])

dbd, count = np.unique(
    [*seq_organoid.obs["dbd_per_leiden"], *seq_embryo.obs["dbd_per_leiden"]],
    return_counts=True
)

dbd_to_color = {
    dbd: plt.cm.tab20b(i) if i % 2 else plt.cm.tab20(i)
    for i, dbd in enumerate(dbd[np.argsort(-count)])
}

seq = seq_organoid

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    seq.obsm["X_tsne"][:, 0], seq.obsm["X_tsne"][:, 1],
    c = [dbd_to_color[dbd] for dbd in seq.obs["dbd_per_leiden"]],
    s = 1
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "tSNE_organoid.png",
    dpi = 500,
    transparent=True
)

seq = seq_embryo

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    seq.obsm["X_tsne"][:, 0], seq.obsm["X_tsne"][:, 1],
    c = [dbd_to_color[dbd] for dbd in seq.obs["dbd_per_leiden"]],
    s = 1
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "tSNE_embryo.png",
    dpi = 500,
    transparent=True
)

fig, ax = plt.subplots()
for d in dbd[np.argsort(-count)]:
    color = dbd_to_color[d]
    ax.scatter([], [], color = color, label = d)
ax.legend()
ax.set_axis_off()
fig.tight_layout()
fig.savefig("dbd_legend.pdf")
fig.savefig("dbd_legend.png")

###
# jaccard seqlets
##


common_regions = list(
    set(seq_organoid.obs["region_name"]) & \
    set(seq_embryo.obs["region_name"])
)

seq_org_sub = seq_organoid.obs.query("region_name in @common_regions")
seq_emb_sub = seq_embryo.obs.query("region_name in @common_regions")

for seq, d_out in zip(
    [seq_org_sub, seq_emb_sub],
    ["organoid_bed_sub", "embryo_bed_sub"]
):
    if not os.path.exists(d_out):
        os.makedirs(d_out)
    for dbd in tqdm(seq["dbd_per_leiden"].unique(), desc=d_out):
        seqlets_to_bed(
            df=seq.loc[seq["dbd_per_leiden"] == dbd],
            out_f=os.path.join(d_out, f"{sanitize(dbd)}.bed")
        )


org_dbd = seq_org_sub["dbd_per_leiden"].unique()
emb_dbd = seq_emb_sub["dbd_per_leiden"].unique()

common_dbd = list(set(org_dbd) & set(emb_dbd))

intersect_count = np.zeros(
    (len(org_dbd), len(emb_dbd)),
    dtype=int
)

for i in tqdm(range(len(org_dbd))):
    for j in range(len(emb_dbd)):
        a = f"organoid_bed_sub/{sanitize(org_dbd[i])}.bed"
        b = f"embryo_bed_sub/{sanitize(emb_dbd[j])}.bed"
        intersect_count[i, j] = get_intersect(a, b)

union_count = np.zeros_like(intersect_count)

for i in tqdm(range(len(org_dbd))):
    for j in range(len(emb_dbd)):
        dbd1 = org_dbd[i]
        dbd2 = emb_dbd[j]
        union_count[i, j] = (
            sum(seq_org_sub["dbd_per_leiden"] == dbd1) \
            + sum(seq_emb_sub["dbd_per_leiden"] == dbd2) \
            - intersect_count[i, j]
        )

df_jaccard= pd.DataFrame(
    np.divide(intersect_count, union_count),
    index = org_dbd,
    columns = emb_dbd
).loc[common_dbd, common_dbd]



annot_labels = np.empty(
    (df_jaccard.shape[0], df_jaccard.shape[1]),
    dtype="<U4"
)
for i in range(df_jaccard.shape[0]):
    for j in range(df_jaccard.shape[1]):
        if df_jaccard.iloc[i, j] > 0.07:
            annot_labels[i, j] = str(np.round(df_jaccard.iloc[i, j], 2))


fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(
    df_jaccard,
    vmin = 0, vmax = 0.35,
    ax = ax,
    xticklabels=True, yticklabels=True,
    annot = annot_labels, fmt = "",
    square=True, cbar_kws = dict(label = "Jaccard"),
    linewidths=1, linecolor="white",
    cmap = "viridis"
)
fig.tight_layout()
fig.savefig("jaccard_seqlets_common_regions.pdf")
fig.savefig("jaccard_seqlets_common_regions.png", dpi = 500)

##
# DOTPLOT
##


seq_organoid.obs["model_class"] = seq_organoid.obs["model_class"].str.split(",")
seq_embryo.obs["model_class"] = seq_embryo.obs["model_class"].str.split(",")

seq_org = seq_organoid.obs.explode("model_class")
seq_emb = seq_embryo.obs.explode("model_class")

org_topics_to_show = [
    33, 38, 36, 54, 48,
    62, 60, 65, 59, 58,
    6, 4, 23, 24, 13, 2
]

org_topics_to_show = [
    f"Topic_{x}" for x in org_topics_to_show
]

emb_topics_to_show = [
    34, 38, 79, 88, 58,
    61, 59, 31, 62, 70, 52, 71,
    103, 105, 94, 91,
    10, 8, 13, 24, 18, 29
]

emb_topics_to_show = [
    f"Topic_{x}" for x in emb_topics_to_show
]


org_count = pd.crosstab(
    seq_org["model_class"].values,
    seq_org["dbd_per_leiden"].values
).loc[org_topics_to_show].T

org_count = org_count / org_count.sum()

dbd_order = org_count.T.idxmax().sort_values(
    key = lambda X: [org_topics_to_show.index(x) for x in X],
    ascending=False
).index

org_count = org_count.loc[dbd_order]

org_avg_count_per_seq = seq_org \
    .groupby(['region_name', 'model_class', 'dbd_per_leiden']).size().reset_index(name='count') \
    .query("count != 0") \
    .groupby(["model_class", "dbd_per_leiden"])["count"].mean() \
    .reset_index() \
    .pivot(index = "dbd_per_leiden", columns = "model_class")["count"] \
    .fillna(0) \
    .round() \
    .astype(int)

org_avg_count_per_seq = org_avg_count_per_seq.loc[dbd_order, org_topics_to_show]

cmap = matplotlib.cm.ScalarMappable(
    norm = matplotlib.colors.Normalize(vmin = 1, vmax = org_avg_count_per_seq.max().max() + 1),
    cmap = matplotlib.cm.gnuplot2
)

x, y = np.meshgrid(
    np.arange(org_count.shape[1]),
    np.arange(org_count.shape[0])
)

x_flat = x.flatten()
y_flat = y.flatten()
values = org_count.to_numpy().flatten()
values_c = org_avg_count_per_seq.to_numpy().flatten()

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    x_flat, y_flat,
    s = values * 500,
    c = [cmap.to_rgba(x) for x in values_c],
    edgecolors="black", lw = 2
)
ax.set_xticks(
    np.arange(x_flat.max() + 1),
    org_count.columns,
    rotation = 90
)
ax.set_yticks(
    np.arange(y_flat.max() + 1),
    org_count.index
)
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("organoid_dotplot.pdf")
fig.savefig("organoid_dotplot.png")


emb_count = pd.crosstab(
    seq_emb["model_class"].values,
    seq_emb["dbd_per_leiden"].values
).loc[emb_topics_to_show].T

emb_count = emb_count / emb_count.sum()

emb_count = emb_count.loc[dbd_order]

emb_avg_count_per_seq = seq_emb \
    .groupby(['region_name', 'model_class', 'dbd_per_leiden']).size().reset_index(name='count') \
    .query("count != 0") \
    .groupby(["model_class", "dbd_per_leiden"])["count"].mean() \
    .reset_index() \
    .pivot(index = "dbd_per_leiden", columns = "model_class")["count"] \
    .fillna(0) \
    .round() \
    .astype(int)

emb_avg_count_per_seq = emb_avg_count_per_seq.loc[dbd_order, emb_topics_to_show]

cmap = matplotlib.cm.ScalarMappable(
    norm = matplotlib.colors.Normalize(vmin = 1, vmax = emb_avg_count_per_seq.max().max() + 1),
    cmap = matplotlib.cm.gnuplot2
)

x, y = np.meshgrid(
    np.arange(emb_count.shape[1]),
    np.arange(emb_count.shape[0])
)

x_flat = x.flatten()
y_flat = y.flatten()
values = emb_count.to_numpy().flatten()
values_c = emb_avg_count_per_seq.to_numpy().flatten()

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    x_flat, y_flat,
    s = values * 500,
    c = [cmap.to_rgba(x) for x in values_c],
    edgecolors="black", lw = 2
)
ax.set_xticks(
    np.arange(x_flat.max() + 1),
    emb_count.columns,
    rotation = 90
)
ax.set_yticks(
    np.arange(y_flat.max() + 1),
    emb_count.index
)
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("embryo_dotplot.png")
fig.savefig("embryo_dotplot.pdf")

corr_count = np.zeros((org_count.shape[1], emb_count.shape[1]), dtype = float)
for i in range(corr_count.shape[0]):
    for j in range(corr_count.shape[1]):
        corr_count[i, j] = scipy.stats.pearsonr(
            org_count.iloc[:, i], emb_count.iloc[:, j]
        ).statistic

fig = sns.clustermap(
    corr_count,
    yticklabels = org_count.columns,
    xticklabels = emb_count.columns,
    annot = True
)
fig.savefig("test.png")

##
# location
##
window = 10

seq_organoid.obs["start_bins"] = pd.cut(
    seq_organoid.obs["start"], bins=range(-1, 500, window)
)

n_hits_per_bin_organoid = seq_organoid.obs.groupby("start_bins", observed=False).size()

n_hits_per_bin_mean_organoid = (
    np.sum(
        [
            iv.mid * count
            for iv, count in zip(
                n_hits_per_bin_organoid.index, n_hits_per_bin_organoid.values
            )
        ]
    )
    / n_hits_per_bin_organoid.values.sum()
)

n_hits_per_bin_var_organoid = (
    np.sum(
        [
            count * (iv.mid - n_hits_per_bin_mean_organoid) ** 2
            for iv, count in zip(
                n_hits_per_bin_organoid.index, n_hits_per_bin_organoid.values
            )
        ]
    )
    / n_hits_per_bin_organoid.values.sum()
)

n_hits_per_bin_std_organoid = np.sqrt(n_hits_per_bin_var_organoid)

mean_organoid = n_hits_per_bin_mean_organoid
std_organoid = n_hits_per_bin_std_organoid
fig, ax = plt.subplots(figsize = (7.5, 5))
ax_bar = ax.twinx()
ax.scatter(
    seq_organoid.obs.query("start > 0")["start"],
    seq_organoid.obs.query("start > 0")["attribution"],
    s = 1, color = "black", zorder = 1
)
ax_bar.bar(
    x=[x.left for x in n_hits_per_bin_organoid.index],
    height=n_hits_per_bin_organoid.values / n_hits_per_bin_organoid.values.sum(),
    width=window,
    color=[
        "white"
        if iv.mid > mean_organoid + std_organoid * 2
        or iv.mid < mean_organoid - std_organoid * 2
        else "darkgray"
        if iv.mid > mean_organoid + std_organoid * 1
        or iv.mid < mean_organoid - std_organoid * 1
        else "dimgray"
        for iv in n_hits_per_bin_organoid.index
    ],
    edgecolor=[
        "lightgray"
        if iv.mid > mean_organoid + std_organoid * 2
        or iv.mid < mean_organoid - std_organoid * 2
        else "gray"
        if iv.mid > mean_organoid + std_organoid * 1
        or iv.mid < mean_organoid - std_organoid * 1
        else "black"
        for iv in n_hits_per_bin_organoid.index
    ],
    lw=1,
    alpha = 0.8,
)
ax_bar.text(
    0.05,
    0.05,
    s=f"μ = {int(mean_organoid) - 250}\nσ = {int(std_organoid)}",
    transform=ax_bar.transAxes,
    zorder = 2
)
ax.set_xticks(
    np.arange(0, 550, 50),
    np.arange(0, 550, 50) - 250
)
ax.set_rasterization_zorder(2)
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("Position relative to summit")
ax.set_ylabel("Attribution")
ax_bar.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig("seqlet_loc_organoid.pdf")
fig.savefig("seqlet_loc_organoid.png")


seq_embryo.obs["start_bins"] = pd.cut(
    seq_embryo.obs["start"], bins=range(-1, 500, window)
)

n_hits_per_bin_embryo = seq_embryo.obs.groupby("start_bins", observed=False).size()

n_hits_per_bin_mean_embryo = (
    np.sum(
        [
            iv.mid * count
            for iv, count in zip(
                n_hits_per_bin_embryo.index, n_hits_per_bin_embryo.values
            )
        ]
    )
    / n_hits_per_bin_embryo.values.sum()
)

n_hits_per_bin_var_embryo = (
    np.sum(
        [
            count * (iv.mid - n_hits_per_bin_mean_embryo) ** 2
            for iv, count in zip(
                n_hits_per_bin_embryo.index, n_hits_per_bin_embryo.values
            )
        ]
    )
    / n_hits_per_bin_embryo.values.sum()
)

n_hits_per_bin_std_embryo = np.sqrt(n_hits_per_bin_var_embryo)

mean_embryo = n_hits_per_bin_mean_embryo
std_embryo = n_hits_per_bin_std_embryo
fig, ax = plt.subplots(figsize = (7.5, 5))
ax_bar = ax.twinx()
ax.scatter(
    seq_embryo.obs.query("start > 0")["start"],
    seq_embryo.obs.query("start > 0")["attribution"],
    s = 1, color = "black", zorder = 1
)
ax_bar.bar(
    x=[x.left for x in n_hits_per_bin_embryo.index],
    height=n_hits_per_bin_embryo.values / n_hits_per_bin_embryo.values.sum(),
    width=window,
    color=[
        "white"
        if iv.mid > mean_embryo + std_embryo * 2
        or iv.mid < mean_embryo - std_embryo * 2
        else "darkgray"
        if iv.mid > mean_embryo + std_embryo * 1
        or iv.mid < mean_embryo - std_embryo * 1
        else "dimgray"
        for iv in n_hits_per_bin_embryo.index
    ],
    edgecolor=[
        "lightgray"
        if iv.mid > mean_embryo + std_embryo * 2
        or iv.mid < mean_embryo - std_embryo * 2
        else "gray"
        if iv.mid > mean_embryo + std_embryo * 1
        or iv.mid < mean_embryo - std_embryo * 1
        else "black"
        for iv in n_hits_per_bin_embryo.index
    ],
    lw=1,
    alpha = 0.8,
)
ax_bar.text(
    0.05,
    0.05,
    s=f"μ = {int(mean_embryo) - 250}\nσ = {int(std_embryo)}",
    transform=ax_bar.transAxes,
    zorder = 2
)
ax.set_xticks(
    np.arange(0, 550, 50),
    np.arange(0, 550, 50) - 250
)
ax.set_rasterization_zorder(2)
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("Position relative to summit")
ax.set_ylabel("Attribution")
ax_bar.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig("seqlet_loc_embryo.pdf")
fig.savefig("seqlet_loc_embryo.png")

##
# Zebrafish seqlet tSNE
##

seq_zebrafish = anndata.read_h5ad(
    "../data_prep_new/zebrafish_data/mindi/seqlet_adata_no_na.h5ad"
)

a = seq_organoid.obs.groupby("dbd_per_leiden", observed = True).size()
b = seq_embryo.obs.groupby("dbd_per_leiden", observed = True).size()
c = seq_zebrafish.obs.groupby("dbd_per_leiden", observed = True).size()

common_dbd = list(
    set(a.index) & \
    set(b.index) & \
    set(c.index)
)

print(scipy.stats.pearsonr(a.loc[common_dbd], c.loc[common_dbd]))
print(scipy.stats.pearsonr(b.loc[common_dbd], c.loc[common_dbd]))

dbd_to_color["ARID/BRIGHT"] = plt.cm.Pastel1(0)
dbd_to_color["C2H2 ZF; Homeodomain"] = plt.cm.Pastel1(1)
dbd_to_color["E2F"] = plt.cm.Pastel1(2)
dbd_to_color["SMAD"] = plt.cm.Pastel1(3)
dbd_to_color["T-box"] = plt.cm.Pastel1(4)
dbd_to_color["nan"] = "black"

seq = seq_zebrafish

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    seq.obsm["X_tsne"][:, 0], seq.obsm["X_tsne"][:, 1],
    c = [dbd_to_color[dbd] for dbd in seq.obs["dbd_per_leiden"]],
    s = 1
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "tSNE_zebrafish.png",
    dpi = 500,
    transparent=True
)

fig, ax = plt.subplots(figsize = (7, 7))
for d in dbd[np.argsort(-count)]:
    color = dbd_to_color[d]
    ax.scatter([], [], color = color, label = d)
for d in set(seq_zebrafish.obs["dbd_per_leiden"]) - set(dbd):
    color = dbd_to_color[d]
    ax.scatter([], [], color = color, label = d)
ax.legend()
ax.set_axis_off()
fig.tight_layout()
fig.savefig("dbd_legend.pdf")
fig.savefig("dbd_legend.png")

##
# cell umap
##

cell_topic = anndata.read_h5ad("../data_prep_new/zebrafish_data/300_iter.100_topics_cell_topic_adata.h5ad")

avg_loc = pd.DataFrame(cell_topic.obsm["X_umap"], index = cell_topic.obs_names) \
    .groupby(cell_topic.obs["annotation_ML_coarse"], observed=True).mean()

avg_loc = (avg_loc - avg_loc.min()) / (avg_loc.max() - avg_loc.min())

sorted_cell_types = list(avg_loc.sum(1).sort_values().index)

cell_type_to_color = {
    c: plt.cm.tab20b(i) if i < 20 else plt.cm.tab20c(i - 20)
    for i, c in enumerate(sorted_cell_types)
}

cell_to_keep = cell_topic.obs.dropna(subset = "annotation_ML_coarse").index

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    cell_topic[cell_to_keep].obsm["X_umap"][:, 0], 
    cell_topic[cell_to_keep].obsm["X_umap"][:, 1],
    c = [
        cell_type_to_color[ct] for ct in cell_topic[cell_to_keep].obs["annotation_ML_coarse"]
    ],
    s = 1
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "tSNE_zebrafish_cells.png",
    dpi = 500,
    transparent=True
)

fig, ax = plt.subplots(figsize = (10, 10))
for ct in sorted_cell_types:
    color = cell_type_to_color[ct]
    ax.scatter([], [], color = color, label = ct)
ax.legend()
ax.set_axis_off()
fig.tight_layout()
fig.savefig("ct_legend_zeb.pdf")
fig.savefig("ct_legend_zeb.png")

sc.pl.umap(
    cell_topic,
    color = cell_topic.var_names,
    ncols = 10,
    save = "_cell_topic.png"
)

zebrafish_topics = [
    73, 
    35,15, 60, 39, 11 ,25, 64, 41, 86, 90, 32, 65, 74, 54, 42, 17,          
    56, 93, 6, 71, 
    33, 30
]

zebrafish_topics = [str(t) for t in zebrafish_topics]

sc.pl.umap(
    cell_topic,
    color = [f"Topic{x}" for x in zebrafish_topics],
    ncols = 5,
    save = "_cell_topic_selected.png"
)

zeb_count = pd.crosstab(
    seq_zebrafish.obs["class"].values,
    seq_zebrafish.obs["dbd_per_leiden"].values
).loc[zebrafish_topics].T

zeb_count = zeb_count / zeb_count.sum()

dbd_order_zeb = [d for d in dbd_order if d in zeb_count.index]

zeb_count = zeb_count.loc[dbd_order_zeb]

zeb_avg_count_per_seq = seq_zebrafish.obs \
    .groupby(['region_names', 'class', 'dbd_per_leiden']).size().reset_index(name='count') \
    .query("count != 0") \
    .groupby(["class", "dbd_per_leiden"])["count"].mean() \
    .reset_index() \
    .pivot(index = "dbd_per_leiden", columns = "class")["count"] \
    .fillna(0) \
    .round() \
    .astype(int)

zeb_avg_count_per_seq = zeb_avg_count_per_seq.loc[
    dbd_order_zeb, zebrafish_topics]

cmap = matplotlib.cm.ScalarMappable(
    norm = matplotlib.colors.Normalize(vmin = 1, vmax = 5),
    cmap = matplotlib.cm.gnuplot2
)

x, y = np.meshgrid(
    np.arange(zeb_count.shape[1]),
    np.arange(zeb_count.shape[0])
)

x_flat = x.flatten()
y_flat = y.flatten()
values = zeb_count.to_numpy().flatten()
values_c = zeb_avg_count_per_seq.to_numpy().flatten()

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(
    x_flat, y_flat,
    s = values * 500,
    c = [cmap.to_rgba(x) for x in values_c],
    edgecolors="black", lw = 2
)
ax.set_xticks(
    np.arange(x_flat.max() + 1),
    zeb_count.columns,
    rotation = 90
)
ax.set_yticks(
    np.arange(y_flat.max() + 1),
    zeb_count.index
)
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("zeb_dotplot.pdf")
fig.savefig("zeb_dotplot.png")


avg_ct = cell_topic[cell_to_keep].to_df() \
    .groupby(cell_topic[cell_to_keep].obs["annotation_ML_coarse"], observed = True).mean()[
        [f"Topic{t}" for t in zebrafish_topics]
    ]

sorted_cell_types = [
    'floor_plate',
    'spinal_cord',
    'neural_posterior',
    'neural_floor_plate',
    'neurons',
    'hindbrain',
    'neural',
    'midbrain_hindbrain_boundary',
    'neural_optic',
    'neural_telencephalon',
    'neural_crest',
    'differentiating_neurons',
    'enteric_neurons',
]

avg_ct_z = (avg_ct - avg_ct.min()) / (avg_ct.max() - avg_ct.min())
fig, ax = plt.subplots(figsize = (8,4))
sns.heatmap(
    avg_ct_z.loc[sorted_cell_types],
    ax = ax,
    #vmax = 0.08,
    cmap = "viridis",
    lw = 1, linecolor = "black"
)
fig.tight_layout()
fig.savefig("avg_cell_topic.pdf")
fig.savefig("avg_cell_topic.png")


##

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
        if vmax > vmin:
            return np.clip((values - vmin) / (vmax - vmin), 0, 1)
        else:
            return values
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
    if r_vmin is None:
        r_vmin = r_values.min()
    if r_vmax is None:
        r_vmax = r_values.max()
    if g_vmin is None:
        g_vmin = g_values.min()
    if g_vmax is None:
        g_vmax = g_values.max()
    if b_vmin is None:
        b_vmin = b_values.min()
    if b_vmax is None:
        b_vmax = b_values.max()
    print(f"R: {r_vmin, r_vmax}\nG: {g_vmin, g_vmax}\nB: {b_vmin, b_vmax}")

df = cell_topic[cell_to_keep].to_df()


fig, ax = plt.subplots(figsize = (8, 8))
rgb_scatter_plot(
    x=cell_topic[cell_to_keep].obsm["X_umap"][:, 0],
    y=cell_topic[cell_to_keep].obsm["X_umap"][:, 1],
    ax=ax,
    g_cut=0,
    r_values=df["Topic73"].values,
    g_values=df["Topic71"].values,
    b_values=df["Topic30"].values,
    r_name="",
    g_name="",
    b_name="",
)
fig.tight_layout()
fig.savefig("zeb_topic_tricolor.png",dpi = 500,
    transparent=True)

##
# corr
##

common_dbd = list(set(zeb_count.index) & set(org_count.index))
zeb_org_corr = np.zeros(
    (org_count.shape[1], zeb_count.shape[1]),
    dtype = float
)
for i in tqdm(range(org_count.shape[1])):
    for j in range(zeb_count.shape[1]):
        zeb_org_corr[i, j] = scipy.stats.pearsonr(
            org_count.iloc[:, i].loc[common_dbd],
            zeb_count.iloc[:, j].loc[common_dbd]
        ).statistic

fig, ax = plt.subplots()
sns.heatmap(
    zeb_org_corr,
    xticklabels = zeb_count.columns,
    yticklabels = org_count.columns,
    ax = ax,
    cmap = "magma",
    vmin = 0, vmax = 1,
    lw = 1, linecolor = "gray"
)
fig.tight_layout()
fig.savefig("test.png")


###
# jaccard seqlets
##




seq_zebrafish_org = anndata.read_h5ad(
    "../../../../ZEBRAFISH_DEV/ZEBRAHUB/TFMINDI_ORG/seqlet_adata_no_na.h5ad"
)

common_regions = list(
    set(seq_zebrafish_org.obs["region_names"]) & \
    set(seq_organoid.obs["region_name"])
)

seq_org = seq_organoid.obs.query("region_name in @common_regions")
seq_zeb = seq_zebrafish_org.obs.rename({"region_names": "region_name"}, axis = 1).query("region_name in @common_regions")

for seq, d_out in zip(
    [seq_org, seq_zeb],
    ["organoid_zeb_bed", "zebrafish_org_bed"]
):
    if not os.path.exists(d_out):
        os.makedirs(d_out)
    for dbd in tqdm(seq["dbd_per_leiden"].unique(), desc=d_out):
        seqlets_to_bed(
            df=seq.loc[seq["dbd_per_leiden"] == dbd],
            out_f=os.path.join(d_out, f"{sanitize(dbd)}.bed")
        )


org_dbd = seq_org["dbd_per_leiden"].unique()
zeb_dbd = seq_zeb["dbd_per_leiden"].unique()

common_dbd = list(set(org_dbd) & set(zeb_dbd))

intersect_count = np.zeros(
    (len(org_dbd), len(zeb_dbd)),
    dtype=int
)

for i in tqdm(range(len(org_dbd))):
    for j in range(len(zeb_dbd)):
        a = f"organoid_zeb_bed/{sanitize(org_dbd[i])}.bed"
        b = f"zebrafish_org_bed/{sanitize(zeb_dbd[j])}.bed"
        intersect_count[i, j] = get_intersect(a, b)

union_count = np.zeros_like(intersect_count)

for i in tqdm(range(len(org_dbd))):
    for j in range(len(zeb_dbd)):
        dbd1 = org_dbd[i]
        dbd2 = zeb_dbd[j]
        union_count[i, j] = (
            sum(seq_org["dbd_per_leiden"] == dbd1) \
            + sum(seq_zeb["dbd_per_leiden"] == dbd2) \
            - intersect_count[i, j]
        )

df_jaccard= pd.DataFrame(
    np.divide(intersect_count, union_count),
    index = org_dbd,
    columns = zeb_dbd
).loc[common_dbd, common_dbd]



annot_labels = np.empty(
    (df_jaccard.shape[0], df_jaccard.shape[1]),
    dtype="<U4"
)
for i in range(df_jaccard.shape[0]):
    for j in range(df_jaccard.shape[1]):
        if df_jaccard.iloc[i, j] > 0.05:
            annot_labels[i, j] = str(np.round(df_jaccard.iloc[i, j], 2))


fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(
    df_jaccard,
    vmin = 0, vmax = 0.35,
    ax = ax,
    xticklabels=True, yticklabels=True,
    annot = annot_labels, fmt = "",
    square=True, cbar_kws = dict(label = "Jaccard"),
    linewidths=1, linecolor="white",
    cmap = "viridis"
)
fig.tight_layout()
fig.savefig("jaccard_seqlets_org_zeb.pdf")
fig.savefig("jaccard_seqlets_org_zeb.png", dpi = 500)

##

ORGANOID_GRAD_DIR="../../../../De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
EMBRYO_GRAD_DIR="../../../../De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

KEY_CONTRIB     =   "gradients_integrated"
KEY_OH          =   "oh"
KEY_REGION_N    =   "region_names"

GRAD_DIRS = {
    "organoid": ORGANOID_GRAD_DIR,
    "embryo": EMBRYO_GRAD_DIR
}

contrib         =   []
oh              =   []
class_names     =   []
region_names    =   []
model_systems   =   []

for model_system, grad_dir in GRAD_DIRS.items():
    gradient_files = [
        x for x in 
        os.listdir(grad_dir)
        if x.startswith("gradients") and x.endswith(".npz")
    ]
    for file in tqdm(gradient_files, desc=model_system):
        class_name = file.replace("gradients_", "").replace(".npz", "")
        with np.load(os.path.join(grad_dir, file)) as npz_handle:
            N = npz_handle[KEY_CONTRIB].shape[0]
            contrib.append(npz_handle[KEY_CONTRIB].squeeze())
            oh.append(npz_handle[KEY_OH])
            region_names.append(npz_handle[KEY_REGION_N])
            class_names.append(np.repeat(class_name, N))
            model_systems.append(np.repeat(model_system, N))

contrib         =   np.concatenate(contrib)
oh              =   np.concatenate(oh)
class_names     =   np.concatenate(class_names)
region_names    =   np.concatenate(region_names)
model_systems   =   np.concatenate(model_systems)

zebrafish_contr = np.load(
    "../../../../ZEBRAFISH_DEV/ZEBRAHUB/CRESTED_CLASSIFICATION/CONTR/base_202578143343_organoid_regions_selected_classes/contr_organoid_regions.npz"
)["attr"]

common_regions = list(
    set(seq_organoid.obs["region_name"]) \
    & set(seq_embryo.obs["region_name"]) \
    & set(seq_zebrafish_org.obs["region_names"])
)

seq_organoid.obs["model_class"] = seq_organoid.obs["model_class"].str.split(",")
seq_embryo.obs["model_class"] = seq_embryo.obs["model_class"].str.split(",")

seq_org = seq_organoid.obs.explode("model_class")
seq_emb = seq_embryo.obs.explode("model_class")


fp_region = "chr19:6066607-6067107"
nc_region = "chr4:13178567-13179067"
neu_region = "chr21:29454058-29454558"

seq_zebrafish_org.obs.query("region_names == @r").sort_values("start")
seq_organoid.obs.query("region_name == @r").sort_values("start")[["start", "dbd_per_leiden"]]
seq_embryo.obs.query("region_name == @r").sort_values("start")[["start", "dbd_per_leiden"]]

region = fp_region

fig, axs = plt.subplots((6, 6), nrows = 3, sharex = True)
o = oh[
    np.where(region_names == region)[0][0]
]
org_contrib = contrib[np.where(np.logical_and(
    region_names == region, 
    model_systems == "organoid"
))[0]].max(0)
_ = logomaker.Logo(
    pd.DataFrame(
        o * org_contrib,
        columns = list("ACGT")
    ),
    ax = axs[0]
)
fig.tight_layout()
fig.savefig("test.png")

for seq, d_out in zip(
    [seq_organoid.obs, seq_embryo.obs, seq_zebrafish.obs.rename({"region_names": "region_name"}, axis = 1)],
    ["organoid_bed", "embryo_bed", "zebrafish_bed"]
):
    if not os.path.exists(d_out):
        os.makedirs(d_out)
    for dbd in tqdm(seq["dbd_per_leiden"].unique(), desc=d_out):
        seqlets_to_bed(
            df=seq.loc[seq["dbd_per_leiden"] == dbd],
            out_f=os.path.join(d_out, f"{sanitize(dbd)}.bed")
        )