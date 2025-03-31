

```python

import pandas as pd
import seaborn as sns

organoid_neural_crest_region_topic = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
    index_col = 0
)
organoid_neuron_region_topic = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv",
    index_col = 0
)
organoid_progenitor_region_topic = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
    index_col = 0
)

organoid_neural_crest_region_topic.columns = [
    f"neural_crest_Topic_{topic.replace('Topic', '')}" for topic in organoid_neural_crest_region_topic.columns
]
organoid_neuron_region_topic.columns = [
    f"neuron_Topic_{topic.replace('Topic', '')}" for topic in organoid_neuron_region_topic.columns
]
organoid_progenitor_region_topic.columns = [
    f"progenitor_Topic_{topic.replace('Topic', '')}" for topic in organoid_progenitor_region_topic.columns
]

organoid_region_topic = pd.concat(
    [
        organoid_neural_crest_region_topic,
        organoid_neuron_region_topic,
        organoid_progenitor_region_topic
    ],
    axis = 1
).fillna(0)

selected_organoid_topics = [
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

organoid_selected_regions = set()
for topic in selected_organoid_topics:
    organoid_selected_regions.update(
        organoid_region_topic.sort_values(
            topic, ascending = False
        ).head(1_000).index
    )



from scenicplus.data_wrangling.cistarget_wrangling import (
  _signatures_to_iter,
  _get_cistromes,
  _merge_cistromes,
  _cistromes_to_adata
)

scplus_regions = organoid_selected_regions
paths_to_motif_enrichment_results = ["../../data_prep/data/cistarget_topics.hdf5"]

cistromes = []
for motif_enrichment_table, motif_hits in _signatures_to_iter(
    paths_to_motif_enrichment_results):
    region_set = motif_enrichment_table.Region_set[0]
    if region_set in topics_order:
      print(region_set)
      cistromes.extend(
          _get_cistromes(
              motif_enrichment_table = motif_enrichment_table, #.query("NES > 6"),
              motif_hits = motif_hits,
              scplus_regions = scplus_regions,
              direct_annotation = ["Direct_annot"],
              extended_annotation = ["Orthology_annot"]))

direct_cistromes = [cistrome for cistrome in cistromes if not cistrome.extended]
extended_cistromes = [cistrome for cistrome in cistromes if cistrome.extended]
merged_direct_cistromes = list(_merge_cistromes(direct_cistromes))
merged_extended_cistromes = list(_merge_cistromes(extended_cistromes))
adata_direct_cistromes = _cistromes_to_adata(merged_direct_cistromes)
adata_extended_cistromes = _cistromes_to_adata(merged_extended_cistromes)
adata_direct_cistromes.var["is_extended"] = False
adata_extended_cistromes.var["is_extended"] = True

human_tfs = pd.read_csv(
  "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv",
  index_col = 0
)

common_tfs = list(
    set(adata_direct_cistromes.var_names) \
    & set(human_tfs["HGNC symbol"])
)

df_motif = adata_direct_cistromes.to_df()[common_tfs]

dbd =  human_tfs.set_index("HGNC symbol")["DBD"].loc[ common_tfs ].values

df_dbd = ( df_motif.T.groupby(dbd).apply(sum) > 0 ).T

regions_order = organoid_region_topic.loc[list(organoid_selected_regions), topics_order].T.idxmax().reset_index().set_index(0).loc[topics_order]["index"].values

row_color = (
  pd.DataFrame(index = list(organoid_selected_regions)) \
  .merge(df_dbd, left_index = True, right_index = True, how = "left") \
  .fillna(False) \
  .applymap(lambda x: "#FFFFFF" if x else "#000000")
).loc[regions_order]

fig = sns.clustermap(
    organoid_region_topic.loc[organoid_selected_regions, topics_order],
    yticklabels = False,
    xticklabels = True,
    cmap = "viridis",
    robust = True,
    row_cluster = True,
    col_cluster = True,
  #colors_ratio = 0.02
)
fig.ax_row_dendrogram.set_visible(False)
fig.savefig("test_clustermap_org.png")

```


```python

from pycistarget.input_output import read_hdf5
from tangermeme.tools.tomtom import tomtom
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch
import scanpy as sc
import modiscolite
import h5py
import logomaker
import matplotlib.pyplot as plt

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

menr_organoid = read_hdf5("../../data_prep_new/organoid_data/ATAC/motif_enrichment_training_data.h5ad")
menr_embryo = read_hdf5("../../data_prep_new/embryo_data/ATAC/motif_enrichment_training_data.h5ad")

topics_to_show_organoid = []
with open("../../data_prep_new/organoid_data/ATAC/topics_hq_motif_enrichment.txt") as f:
  for l in f:
    topics_to_show_organoid.append(l.strip())

topics_to_show_embryo = []
with open("../../data_prep_new/embryo_data/ATAC/topics_hq_motif_enrichment.txt") as f:
  for l in f:
    topics_to_show_embryo.append(l.strip())

all_motif_enrichment_res = []

for topic in topics_to_show_organoid:
  m = menr_organoid[f"training_data_Topic_{topic}"].motif_enrichment
  m["data_set"] = "organoid"
  all_motif_enrichment_res.append(m)

for topic in topics_to_show_embryo:
  m = menr_embryo[f"training_data_Topic_{topic}"].motif_enrichment
  m["data_set"] = "embryo"
  all_motif_enrichment_res.append(m)

all_motif_enrichment_res = pd.concat(all_motif_enrichment_res)

motif_name_to_max_NES = (
    all_motif_enrichment_res.reset_index()
    .pivot_table(index=["data_set", "index"], columns="Region_set", values="NES")
    .fillna(0)
    .max(1)
)

def get_dataset(d):
  if "organoid" in d and "embryo" in d:
    return "both"
  elif "organoid" in d:
    return "organoid"
  elif "embryo" in d:
    return "embryo"
  else:
    raise ValueError(d)

motif_name_to_dataset = all_motif_enrichment_res.reset_index().groupby("index")["data_set"].apply(lambda x: get_dataset(list(x)))

all_motifs = []
motif_sub_names = []
motif_names = []

for topic in tqdm(topics_to_show_organoid):
    for motif_name in menr_organoid[f"training_data_Topic_{topic}"].motif_enrichment.index:
        if motif_name in motif_names:
            continue
        _motifs, _m_sub_names = load_motif(
            motif_name, "../../../../../motif_collection/cluster_buster"
        )
        all_motifs.extend(_motifs)
        motif_sub_names.extend(_m_sub_names)
        motif_names.extend(np.repeat(motif_name, len(_motifs)))

for topic in tqdm(topics_to_show_embryo):
    for motif_name in menr_embryo[f"training_data_Topic_{topic}"].motif_enrichment.index:
        if motif_name in motif_names:
            continue
        _motifs, _m_sub_names = load_motif(
            motif_name, "../../../../../motif_collection/cluster_buster"
        )
        all_motifs.extend(_motifs)
        motif_sub_names.extend(_m_sub_names)
        motif_names.extend(np.repeat(motif_name, len(_motifs)))

t_all_motifs = [torch.from_numpy(m).T for m in tqdm(all_motifs)]

pvals, scores, offsets, overlaps, strands = tomtom(t_all_motifs, t_all_motifs)

evals = pvals.numpy() * len(all_motifs)

adata_motifs = sc.AnnData(
    evals,
    obs= pd.DataFrame(
        index=motif_sub_names,
        data={
            "motif_name": motif_names,
            "max_NES_organoid": pd.DataFrame(index = list(set(motif_names))) \
              .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["organoid"]}),
                left_index = True, right_index = True, how = "left").loc[motif_names]["NES"].values,
            "max_NES_embryo": pd.DataFrame(index = list(set(motif_names))) \
              .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["embryo"]}),
                left_index = True, right_index = True, how = "left").loc[motif_names]["NES"].values,
            "data_set": motif_name_to_dataset.loc[motif_names].values
        },
    )
  )

sc.settings.figdir = "."

sc.tl.pca(adata_motifs)
sc.pp.neighbors(adata_motifs)
sc.tl.tsne(adata_motifs)

sc.tl.leiden(adata_motifs, resolution=2)

sc.pl.tsne(adata_motifs, color=["leiden", "data_set"], save="_leiden_motifs_2.pdf", legend_loc = "on data")


motifs_w_logo = adata_motifs.obs.copy().reset_index(drop = True).drop_duplicates().sort_values("leiden")

motifs_w_logo["Logo"] = [
    f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{m}.png" width="200" >'
    for m in motifs_w_logo["motif_name"]
]

motifs_w_logo["leiden"] = [f"cluster_{x}" for x in motifs_w_logo["leiden"]]

motifs_w_logo.to_html(
    "leiden_motifs_2.html", escape=False, col_space=80
)


## add tfmodisco motifs

def trim_by_ic(
  ic,
  min_v
):
  if len(np.where(np.diff((ic > min_v) * 1))[0]) == 0:
    return 0, 0
  start_index = min(np.where(np.diff((ic > min_v) * 1))[0])
  end_index = max(np.where(np.diff((ic > min_v) * 1))[0])
  return start_index, end_index + 1

def load_motif_from_modisco(filename, ic_thr, avg_ic_thr):
  with h5py.File(filename) as f:
    for pos_neg in ["pos_patterns", "neg_patterns"]:
      if pos_neg not in f.keys():
        continue
      for pattern in f[pos_neg].keys():
        ppm = f[pos_neg][pattern]["sequence"][:]
        ic = modiscolite.util.compute_per_position_ic(
          ppm = ppm,
          background = [0.27, 0.23, 0.23, 0.27], pseudocount = 1e-3)
        start, stop = trim_by_ic(ic, ic_thr)
        if stop - start <= 1:
          continue
        if ic[start: stop].mean() < avg_ic_thr:
          continue
        yield (
            filename.split("/")[-1].rsplit(".", 1)[0] + "_" + pos_neg.split("_")[0] + "_" + pattern,
            pos_neg == "pos_patterns",
            ppm[start: stop],
            ic[start: stop]
        )


organoid_dl_motif_dir = "../../../../../De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "../../../../../De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

all_motifs_dl_organoid = []
motif_names_dl_organoid = []
is_motif_pos_dl_organoid = []
ic_motifs_dl_organoid = []
for topic in tqdm(topics_to_show_organoid):
  for name, is_pos, ppm, ic in load_motif_from_modisco(
    filename=os.path.join(organoid_dl_motif_dir, f"patterns_Topic_{topic}.hdf5"),
    ic_thr=0.2, avg_ic_thr = 0.5
  ):
    all_motifs_dl_organoid.append(ppm)
    motif_names_dl_organoid.append("organoid_" + name)
    is_motif_pos_dl_organoid.append(is_pos)
    ic_motifs_dl_organoid.append(ic)

all_motifs_dl_embryo = []
motif_names_dl_embryo = []
is_motif_pos_dl_embryo = []
ic_motifs_dl_embryo = []
for topic in tqdm(topics_to_show_embryo):
  for name, is_pos, ppm, ic in load_motif_from_modisco(
    filename=os.path.join(embryo_dl_motif_dir, f"patterns_Topic_{topic}.hdf5"),
    ic_thr=0.2, avg_ic_thr = 0.5
  ):
    all_motifs_dl_embryo.append(ppm)
    motif_names_dl_embryo.append("embryo_" + name)
    is_motif_pos_dl_embryo.append(is_pos)
    ic_motifs_dl_embryo.append(ic)

motif_metadata = pd.DataFrame(
        index=motif_sub_names,
        data=dict(
            motif_name=motif_names,
            max_NES_organoid=pd.DataFrame(index = list(set(motif_names))) \
              .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["organoid"]}),
                left_index = True, right_index = True, how = "left").loc[motif_names]["NES"].values,
            max_NES_embryo=pd.DataFrame(index = list(set(motif_names))) \
              .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["embryo"]}),
                left_index = True, right_index = True, how = "left").loc[motif_names]["NES"].values,
            data_set=motif_name_to_dataset.loc[motif_names].values,
            method=np.repeat("cisTarget", len(motif_names)),
            max_ic=np.repeat(np.nan, len(motif_names)),
            avg_ic=np.repeat(np.nan, len(motif_names))
        ),
    )

motif_metadata_dl_organoid = pd.DataFrame(
  index = motif_names_dl_organoid,
  data = dict(
    motif_name=motif_names_dl_organoid,
    max_NES_organoid=np.repeat(np.nan, len(motif_names_dl_organoid)),
    max_NES_embryo=np.repeat(np.nan, len(motif_names_dl_organoid)),
    data_set=np.repeat("organoid", len(motif_names_dl_organoid)),
    method=np.repeat("deep learning", len(motif_names_dl_organoid)),
    max_ic=[ic.max() for ic in ic_motifs_dl_organoid],
    avg_ic=[ic.mean() for ic in ic_motifs_dl_organoid]
  )
)

motif_metadata_dl_embryo = pd.DataFrame(
  index = motif_names_dl_embryo,
  data = dict(
    motif_name=motif_names_dl_embryo,
    max_NES_organoid=np.repeat(np.nan, len(motif_names_dl_embryo)),
    max_NES_embryo=np.repeat(np.nan, len(motif_names_dl_embryo)),
    data_set=np.repeat("embryo", len(motif_names_dl_embryo)),
    method=np.repeat("deep learning", len(motif_names_dl_embryo)),
    max_ic=[ic.max() for ic in ic_motifs_dl_embryo],
    avg_ic=[ic.mean() for ic in ic_motifs_dl_embryo]
  )
)

motif_names_dl_ctx = [*motif_names, *motif_names_dl_organoid, *motif_names_dl_embryo]
motif_sub_names_dl_ctx = [*motif_sub_names, *motif_names_dl_organoid, *motif_names_dl_embryo]
all_motifs_dl_ctx = [*all_motifs, *all_motifs_dl_organoid, *all_motifs_dl_embryo]

motif_metadata_dl_ctx = pd.concat([motif_metadata, motif_metadata_dl_organoid, motif_metadata_dl_embryo]).loc[
  motif_sub_names_dl_ctx
]

t_all_motifs_dl_ctx = [torch.from_numpy(m).T for m in tqdm(all_motifs_dl_ctx)]

motif_metadata_dl_ctx["motif_length"] = [m.shape[1] for m in t_all_motifs_dl_ctx]

pvals, scores, offsets, overlaps, strands = tomtom(t_all_motifs_dl_ctx, t_all_motifs_dl_ctx)

evals = pvals.numpy() * len(all_motifs_dl_ctx)

adata_motifs_dl_ctx = sc.AnnData(
    evals,
    obs=motif_metadata_dl_ctx)

sc.tl.pca(adata_motifs_dl_ctx)
sc.pp.neighbors(adata_motifs_dl_ctx)
sc.tl.tsne(adata_motifs_dl_ctx)

sc.tl.leiden(adata_motifs_dl_ctx, resolution=2)

sc.pl.tsne(adata_motifs_dl_ctx, color=["leiden"], save="_leiden_motifs_dl_ctx_2.pdf", legend_loc = "on data")

sc.pl.tsne(adata_motifs_dl_ctx, color=["data_set", "method"], save="_dataset_motifs_dl_ctx.pdf")

sc.pl.tsne(adata_motifs_dl_ctx, color=["motif_length", "avg_ic", "max_ic"], save="_metrics_motifs_dl_ctx.pdf")

os.makedirs("deep_learning_motifs_png")

ic, ppm = ic_motifs_dl_embryo[0], all_motifs_dl_embryo[0]
for ic, ppm, name in tqdm(zip(
    [*ic_motifs_dl_organoid, *ic_motifs_dl_embryo],
    [*all_motifs_dl_organoid, *all_motifs_dl_embryo],
    [*motif_names_dl_organoid, *motif_names_dl_embryo]
), total = len([*motif_names_dl_organoid, *motif_names_dl_embryo])):
  fig, ax = plt.subplots(figsize = (4, 2))
  _ = logomaker.Logo(
    pd.DataFrame(
      ppm * ic[:, None],
      columns=["A", "C", "G", "T"]
    ),
    ax = ax
  )
  _ = ax.set_ylim((0, 2))
  _ = ax.set_ylabel("bits")
  _ = ax.set_yticks(
    ticks=np.arange(0, 2.2, 0.2),
    labels=[y if y == 0 or y == 1 or y == 2 else "" for y in np.arange(0, 2.2, 0.2)]
  )
  _ = ax.set_xticks(
    ticks=np.arange(0, ppm.shape[0]),
    labels=np.arange(0, ppm.shape[0]) + 1,
  )
  ax.spines[['right', 'top']].set_visible(False)
  fig.tight_layout()
  fig.savefig(f"deep_learning_motifs_png/{name}.png")
  plt.close(fig)

os.makedirs("cistarget_motifs")

import urllib.request
for motif in tqdm(adata_motifs_dl_ctx.obs.query("method == 'cisTarget'")["motif_name"].drop_duplicates()):
  urllib.request.urlretrieve(
    f"https://motifcollections.aertslab.org/v10nr_clust/logos/{motif}.png",
    f"cistarget_motifs/{motif}.png"
  )

motifs_w_logo = adata_motifs_dl_ctx.obs.copy().reset_index(drop = True).drop_duplicates().sort_values("leiden")

motifs_w_logo["Logo"] = [
    f'<img src="https://motifcollections.aertslab.org/v10nr_clust/logos/{m}.png" width="200" >'
    if method == "cisTarget"
    else
    f'<img src="file:///Users/u0138640/data_core/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_2/draft/deep_learning_motifs_png/{m}.png" width="200" >'
    for method, m in zip(motifs_w_logo["method"], motifs_w_logo["motif_name"])
]

motifs_w_logo["leiden"] = [f"cluster_{x}" for x in motifs_w_logo["leiden"]]

motifs_w_logo.set_index("Logo")[
  ["motif_name", "leiden", "method", "data_set", "max_NES_organoid", "max_NES_embryo", "max_ic", "avg_ic", "motif_length"]
].to_html(
    "leiden_motifs_dl_ctx.html", escape=False, col_space=80
)


header = [
  "logo",
  *adata_motifs_dl_ctx.obs.columns
]


import xlsxwriter

workbook = xlsxwriter.Workbook("motifs.xlsx")
worksheet = workbook.add_worksheet()
worksheet.set_column(0, 0, 15)
worksheet.write_row(0, 0, header)
for i, (motif_name, row) in enumerate(tqdm(adata_motifs_dl_ctx.obs.sort_values(
    ["leiden", "max_NES_organoid", "max_NES_embryo", "max_ic"]
  ).iterrows())):
  motif_name = row["motif_name"]
  if row["method"] == "cisTarget":
    worksheet.embed_image(i + 1, 0, f"cistarget_motifs/{motif_name}.png")
  else:
    worksheet.embed_image(i + 1, 0, f"deep_learning_motifs_png/{motif_name}.png")
  worksheet.write_row(i + 1, 1, row.fillna("na"))
  worksheet.set_row(i + 1, 50)
worksheet.autofit()
workbook.close()

```
