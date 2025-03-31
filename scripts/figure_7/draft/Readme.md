```python

from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import numpy as np
import pandas as pd
import torch
from functools import reduce
from dataclasses import dataclass
import h5py
from typing import Self
import os
import matplotlib.pyplot as plt
import pysam
from crested.utils._seq_utils import one_hot_encode_sequence

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
      region_names: list[str]
  ):
    self.contrib_scores               = p["contrib_scores"][        seqlet_idx  ]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][ seqlet_idx  ]
    self.ppm                          = p["sequence"][              seqlet_idx  ]
    self.start                        = p["start"][                 seqlet_idx  ]
    self.end                          = p["end"][                   seqlet_idx  ]
    self.is_revcomp                   = p["is_revcomp"][            seqlet_idx  ]
    region_idx                        = p["example_idx"][           seqlet_idx  ]
    self.region_name = region_names[region_idx]
    self.region_one_hot = ohs[region_idx]
    if (
        (not np.all(self.ppm == self.region_one_hot[self.start: self.end]) and not self.is_revcomp) or \
        (not np.all(self.ppm[::-1, ::-1] == self.region_one_hot[self.start: self.end]) and self.is_revcomp)
    ):
      raise ValueError(
        f"ppm does not match onehot\n" + \
        f"region_idx\t{region_idx}\n" + \
        f"start\t\t{self.start}\n" + \
        f"end\t\t{self.end}\n" + \
        f"is_revcomp\t{self.is_revcomp}\n" + \
        f"{self.ppm.argmax(1)}\n" + \
        f"{self.region_one_hot[self.start: self.end].argmax(1)}"
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
  def __init__(self, p: h5py._hl.group.Group, is_pos: bool, ohs: np.ndarray, region_names: list[str]):
    self.contrib_scores               = p["contrib_scores"][:]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][:]
    self.ppm                          = p["sequence"][:]
    self.is_pos                       = is_pos
    self.seqlets      = [Seqlet(p["seqlets"], i, ohs, region_names) for i in range(p["seqlets"]["n_seqlets"][0])]
    self.subpatterns  = [ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]
  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"
  def ic(self, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)
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
                  ModiscoPattern(f[pos_neg][pattern], pos_neg == "pos_patterns", ohs, region_names)
                )

def load_pattern_from_modisco_for_topics(
  topics:list[int],
  pattern_dir: str,
  prefix: str) -> tuple[list[ModiscoPattern], list[str]]:
  patterns = []
  pattern_names = []
  for topic in topics:
    with np.load(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz")) as gradients_data:
      ohs = gradients_data["oh"]
      region_names = gradients_data["region_names"]
    print(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz"))
    for name, pattern in load_pattern_from_modisco(
      filename=os.path.join(pattern_dir, f"patterns_Topic_{topic}.hdf5"),
      ohs=ohs,
      region_names=region_names
    ):
      patterns.append(pattern)
      pattern_names.append(prefix + name)
  return patterns, pattern_names

def get_hit_and_attribution(
    gradients_path: str,
    motifs: dict[str, np.ndarray],
    oh_key: str = "oh",
    attr_key: str = "gradients_integrated",
    region_names_key: str | None = "region_names",
) -> pd.DataFrame:
    with np.load(gradients_path) as gradients_data:
        ohs = gradients_data[oh_key]
        attr = gradients_data[attr_key]
        region_names = (
            gradients_data[region_names_key]
            if region_names_key is not None
            else np.arange(ohs.shape[0])
        )
    if not (ohs.shape[0] == attr.shape[0] == region_names.shape[0]):
        raise ValueError("Inconsistent shapes!")
    hits = pd.concat(fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2)))
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    return hits

def merge_and_max(
    left: pd.DataFrame, right: pd.DataFrame, on: list[str], max_on: str
) -> pd.DataFrame:
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1)


def get_non_overlapping_start_end_w_max_score(
    df: pd.DataFrame, max_overlap: int, score_col: str
) -> pd.DataFrame:
    df = df.sort_values("start")
    delta = np.diff(df["start"])
    delta_loc = [0, *np.where(delta > max_overlap)[0] + 1]
    groups = [
        slice(delta_loc[i], delta_loc[i + 1] if (i + 1) < len(delta_loc) else None)
        for i in range(len(delta_loc))
    ]
    return pd.DataFrame(
        [df.iloc[group].iloc[df.iloc[group][score_col].argmax()] for group in groups]
    ).reset_index(drop=True)


def get_hits_for_topics(
  organoid_topics: list[int],
  embryo_topics: list[int],
  selected_clusters: list[float],
  organoid_pattern_dir: str,
  embryo_pattern_dir: str,
  pattern_metadata_path: str,
  cluster_col: str,
  ic_trim_thr: float = 0.2
):
  # load motifs
  patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
    topics=organoid_topics,
    pattern_dir=organoid_pattern_dir,
    prefix="organoid_")
  patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
    topics=embryo_topics,
    pattern_dir=embryo_pattern_dir,
    prefix="embryo_")
  all_patterns = [*patterns_organoid, *patterns_embryo]
  all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
  pattern_metadata=pd.read_table(pattern_metadata_path, index_col = 0)
  motifs = {
    n: pattern.ppm[range(*pattern.ic_trim(ic_trim_thr))].T
    for n, pattern in zip(all_pattern_names, all_patterns)
    if n in pattern_metadata.index
  }
  hits_organoid = []
  region_order_organoid = []
  for topic in organoid_topics:
    hits=get_hit_and_attribution(
        gradients_path=os.path.join(organoid_pattern_dir, f"gradients_Topic_{topic}.npz" ),
        motifs=motifs,
    )
    hits["topic"] = topic
    hits["cluster"] = [
      pattern_metadata.loc[
        m, cluster_col
      ]
      for m in hits["motif_name"]
    ]
    hits = hits.query("cluster in @selected_clusters").reset_index(drop = True).copy()
    hits_organoid.append(hits)
    region_order_organoid.extend(hits["sequence_name"])
  hits_embryo = []
  region_order_embryo = []
  for topic in embryo_topics:
    hits=get_hit_and_attribution(
        gradients_path=os.path.join(embryo_pattern_dir, f"gradients_Topic_{topic}.npz" ),
        motifs=motifs,
    )
    hits["topic"] = topic
    hits["cluster"] = [
      pattern_metadata.loc[
        m, cluster_col
      ]
      for m in hits["motif_name"]
    ]
    hits = hits.query("cluster in @selected_clusters").reset_index(drop = True).copy()
    hits_embryo.append(hits)
    region_order_embryo.extend(hits["sequence_name"])
  hits_organoid_merged = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value", "attribution", "topic"],
      max_on = "-logp",
    ),
    hits_organoid
  )
  hits_embryo_merged = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value", "attribution", "topic"],
      max_on = "-logp",
    ),
    hits_embryo
  )
  hits_organoid_non_overlap = hits_organoid_merged \
    .groupby("sequence_name") \
    .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp")) \
    .reset_index(drop = True)
  hits_embryo_non_overlap = hits_embryo_merged \
    .groupby("sequence_name") \
    .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp")) \
    .reset_index(drop = True)
  return (
    (hits_organoid_merged, hits_organoid_non_overlap, region_order_organoid),
    (hits_embryo_merged, hits_embryo_non_overlap, region_order_embryo)
  )

@dataclass
class ModiscoClusteringResult:
  organoid_topics: list[int]
  embryo_topics: list[int]
  pattern_metadata_path: str
  cluster_col: str
  selected_patterns: list[float]


progenitor_dv_clustering_result = ModiscoClusteringResult(
  organoid_topics=[33, 38, 36, 54, 48],
  embryo_topics=[34, 38, 79, 88, 58],
  pattern_metadata_path="../../figure_3/draft/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns = [
    6, 7, 12, 15, 10, 18, 1, 19
  ]
)

progenitor_ap_clustering_result = ModiscoClusteringResult(
  organoid_topics=[],
  embryo_topics=[61, 59, 31, 62, 70, 52, 71],
  pattern_metadata_path="../../figure_4/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns=[1, 4, 8, 7, 6]
)

neural_crest_clustering_result = ModiscoClusteringResult(
  organoid_topics=[62, 60, 65, 59, 58],
  embryo_topics=[103, 105, 94, 91],
  pattern_metadata_path="../../figure_5/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2
  ]
)

neuron_clustering_result = ModiscoClusteringResult(
  organoid_topics=[6, 4, 23, 24, 13, 2],
  embryo_topics=[10, 8, 13, 24, 18, 29],
  pattern_metadata_path="../../figure_6/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0
  ]
)

cell_type_to_modisco_result = {
  "progenitor_dv": progenitor_dv_clustering_result,
  "progenitor_ap": progenitor_ap_clustering_result,
  "neural_crest": neural_crest_clustering_result,
  "neuron": neuron_clustering_result
}

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"
embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

for cell_type in cell_type_to_modisco_result:
  print(cell_type)
  (
    (hits_organoid, hits_organoid_non_overlap, region_order_organoid),
    (hits_embryo, hits_embryo_non_overlap, region_order_embryo)
  ) = get_hits_for_topics(
      organoid_pattern_dir=organoid_dl_motif_dir ,
      embryo_pattern_dir=embryo_dl_motif_dir ,
      ic_trim_thr=0.2,
      organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
      embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
      selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
      pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
      cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
    )
  cell_type_to_modisco_result[cell_type].hits_organoid = hits_organoid
  cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap = hits_organoid_non_overlap
  cell_type_to_modisco_result[cell_type].region_order_organoid = region_order_organoid
  cell_type_to_modisco_result[cell_type].hits_embryo = hits_embryo
  cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap = hits_embryo_non_overlap
  cell_type_to_modisco_result[cell_type].region_order_embryo = region_order_embryo

window = 10
thr = 0.01
for cell_type in cell_type_to_modisco_result:
  fig, axs = plt.subplots(
    figsize = (8, 4), ncols = 2, nrows = 2, sharex = True, sharey = "row",
    height_ratios = [2, 1]
  )
  transform_func = lambda X: X
  attr_z_score_organoid = cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap \
      .groupby(["cluster"])["attribution"].transform( transform_func )
  cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap["z_attribution"] = attr_z_score_organoid
  _ = axs[0, 0].scatter(
    x=cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.start,
    y=attr_z_score_organoid,
    c=cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap["-logp"],
    s=1
  )
  tmp = pd.DataFrame(
    data = {
      "start": cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.start,
      "score": attr_z_score_organoid
    }
  ).sort_values("start")
  tmp["bins"] = pd.cut(tmp["start"], bins = range(-1, 500, window))
  tmp = tmp.groupby("bins")["score"].mean()
  _ = axs[0, 0].plot(
    [iv.left for iv in tmp.index],
    tmp.values,
    color = "black"
  )
  attr_z_score_embryo = cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap \
      .groupby("cluster")["attribution"].transform( transform_func )
  cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap["z_attribution"] = attr_z_score_embryo
  _ = axs[0, 1].scatter(
    x=cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap.start,
    y=attr_z_score_embryo,
    c=cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap["-logp"],
    s=1
  )
  tmp = pd.DataFrame(
    data = {
      "start": cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap.start,
      "score": attr_z_score_embryo
    }
  ).sort_values("start")
  tmp["bins"] = pd.cut(tmp["start"], bins = range(-1, 500, window))
  tmp = tmp.groupby("bins")["score"].mean()
  _ = axs[0, 1].plot(
    [iv.left for iv in tmp.index],
    tmp.values,
    color = "black"
  )
  _ = axs[0, 0].set_title("Organoid")
  _ = axs[0, 1].set_title("Embryo")
  for ax in axs[0, :]:
    ax.grid(True)
    ax.set_axisbelow(True)
    _ = ax.set_xticks(np.arange(0, 550, 50))
    _ = ax.set_ylabel("Contribution score")
    _ = ax.set_xlabel("Start position")
    ax.axhline(thr, color = "red", ls = "dashed")
    ax.axhline(-thr, color = "red", ls = "dashed")
  tmp = pd.DataFrame(
    data = {
      "start": cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.start,
      "score": abs(attr_z_score_organoid) >= thr
    }
  ).sort_values("start")
  tmp["bins"] = pd.cut(tmp["start"], bins = range(-1, 500, window))
  tmp = tmp.groupby("bins")["score"].sum()
  tmp_mean = np.sum([iv.mid * count for iv, count in zip(tmp.index, tmp.values)]) / tmp.values.sum()
  tmp_var = np.sum([count * (iv.mid - tmp_mean)**2 for iv, count in zip(tmp.index, tmp.values)]) / tmp.values.sum()
  tmp_std = np.sqrt(tmp_var)
  ax_pdf_0 = axs[1, 0].twinx()
  ax_pdf_1 = axs[1, 1].twinx()
  axs[1, 0].bar(
    x= [x.left for x in tmp.index],
    height = tmp.values,
    width = window,
    color = [
      "white" if iv.mid > tmp_mean + tmp_std * 2 or iv.mid < tmp_mean - tmp_std * 2 else \
      "darkgray"  if iv.mid > tmp_mean + tmp_std * 1 or iv.mid < tmp_mean - tmp_std * 1 else \
      "dimgray"
      for iv in tmp.index
    ],
    edgecolor = "black",
    lw = 1,
  )
  ax_pdf_0.plot(
    [x.left for x in tmp.index],
    norm(loc = tmp_mean, scale = tmp_std).pdf([x.left for x in tmp.index]),
    color = "black"
  )
  ax_pdf_0.text(
    0.05, 0.65, s = f"μ = {int(tmp_mean)}\nσ = {int(tmp_std)}",
    transform = ax_pdf_0.transAxes
  )
  ax_pdf_0.set_ylim(0, ax_pdf_0.get_ylim()[1])
  tmp = pd.DataFrame(
    data = {
      "start": cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap.start,
      "score": abs(attr_z_score_embryo) >= thr
    }
  ).sort_values("start")
  tmp["bins"] = pd.cut(tmp["start"], bins = range(-1, 500, window))
  tmp = tmp.groupby("bins")["score"].sum()
  tmp_mean = np.sum([iv.mid * count for iv, count in zip(tmp.index, tmp.values)]) / tmp.values.sum()
  tmp_var = np.sum([count * (iv.mid - tmp_mean)**2 for iv, count in zip(tmp.index, tmp.values)]) / tmp.values.sum()
  tmp_std = np.sqrt(tmp_var)
  axs[1, 1].bar(
    x= [x.left for x in tmp.index],
    height = tmp.values,
    width = window,
    color = [
      "white" if iv.mid > tmp_mean + tmp_std * 2 or iv.mid < tmp_mean - tmp_std * 2 else \
      "darkgray"  if iv.mid > tmp_mean + tmp_std * 1 or iv.mid < tmp_mean - tmp_std * 1 else \
      "dimgray"
      for iv in tmp.index
    ],
    edgecolor = "black",
    lw = 1
  )
  ax_pdf_1.plot(
    [x.left for x in tmp.index],
    norm(loc = tmp_mean, scale = tmp_std).pdf([x.left for x in tmp.index]),
    color = "black"
  )
  ax_pdf_1.text(
    0.05, 0.65, s = f"μ = {int(tmp_mean)}\nσ = {int(tmp_std)}",
    transform = ax_pdf_1.transAxes
  )
  ax_pdf_1.set_ylim(0, ax_pdf_1.get_ylim()[1])
  for ax in [ax_pdf_0, ax_pdf_1]:
    ax.set_yticks([])
  for ax in axs[1, :]:
    ax.grid(True)
    ax.set_axisbelow(True)
    _ = ax.set_xticks(np.arange(0, 550, 50))
    _ = ax.set_ylabel("Number of hits")
    _ = ax.set_xlabel("Start position")
  fig.tight_layout()
  fig.savefig(f"plots/{cell_type}_contrib_v_start.png", dpi = 300)


def absmax(a):
  amax = a.max()
  amin = a.min()
  return np.where(-amin > amax, amin, amax)

import seaborn as sns

for cell_type in cell_type_to_modisco_result:
  print(cell_type)
  hits_organoid_non_overlap_per_seq = cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap \
    .groupby(["sequence_name", "cluster"])["z_attribution"] \
    .apply(absmax) \
    .reset_index() \
    .pivot(index = "sequence_name", columns = "cluster", values = "z_attribution") \
    .fillna(0) \
    .astype(float)
  hits_organoid_non_overlap_per_seq_count = cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap \
    .groupby(["sequence_name", "cluster"])["z_attribution"] \
    .apply(lambda x: (abs(x) >= thr).sum()) \
    .reset_index() \
    .pivot(index = "sequence_name", columns = "cluster", values = "z_attribution") \
    .fillna(0) \
    .astype(float)
  cluster_order = cell_type_to_modisco_result[cell_type].selected_patterns
  print(abs(hits_organoid_non_overlap_per_seq).mean())
  fig, axs = plt.subplots(figsize = (8, 8), ncols = 2)
  sns.heatmap(
    hits_organoid_non_overlap_per_seq.loc[
      cell_type_to_modisco_result[cell_type].region_order_organoid, 
      cluster_order],
    yticklabels = False, xticklabels = True,
    ax = axs[0],
    cmap = "bwr",
    vmin = -thr, vmax = thr
  )
  sns.heatmap(
    hits_organoid_non_overlap_per_seq_count.loc[cell_type_to_modisco_result[cell_type].region_order_organoid, cluster_order],
    yticklabels = False, xticklabels = True,
    ax = axs[1],
    cmap = "Spectral",
    robust = True
  )
  fig.tight_layout()
  fig.savefig(f"plots/hits_heatmap_{cell_type}_organoid.png")

n_hits_per_region_and_topic = []
hits_organoid_non_overlap = []
for cell_type in cell_type_to_modisco_result:
  n_hits_per_region_and_topic.append(
      cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap \
      .loc[abs(cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.z_attribution) >= thr] \
      .groupby(["sequence_name", "topic"])["cluster"] \
      .count().reset_index().pivot(index = "sequence_name", columns = ["topic"]).fillna(0)
  )
  hits_organoid_non_overlap.append(cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap \
      .loc[abs(cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.z_attribution) >= thr])

n_hits_per_region_and_topic = pd.concat(n_hits_per_region_and_topic).fillna(0)["cluster"]
hits_organoid_non_overlap = pd.concat(hits_organoid_non_overlap)

data = np.zeros((n_hits_per_region_and_topic.shape[1], int(n_hits_per_region_and_topic.max().max())))
data_2 = np.zeros_like(data)
for n in range(1, data.shape[1] + 1):
  data[:, n - 1] = (n_hits_per_region_and_topic == n).sum().values / n_hits_per_region_and_topic.sum().values
  for j, topic in enumerate(n_hits_per_region_and_topic.columns):
    region_names = n_hits_per_region_and_topic.loc[n_hits_per_region_and_topic[topic] == n].index
    data_2[j, n-1] = hits_organoid_non_overlap.query("sequence_name in @region_names")["score"].mean()


XX, YY = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

fig, ax = plt.subplots()
ax.scatter(
  x = XX.ravel(),
  y = YY.ravel(),
  s = 1.3**(data.ravel() * 100),
  c = data_2.ravel(),
  vmin = 12, vmax = 14
)
_ = ax.set_yticks(np.arange(data.shape[0]), labels = n_hits_per_region_and_topic.columns)
_ = ax.set_xticks(np.arange(data.shape[1]), labels = np.arange(data.shape[1]) + 1)
_ = ax.set_xlim(-0.5, 4.5)
_ = ax.grid()
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plots/test.pdf")

###

def get_hit_score_for_topics(
  region_names: list[str],
  genome: pysam.FastaFile,
  organoid_topics: list[int],
  embryo_topics: list[int],
  selected_clusters: list[float],
  organoid_pattern_dir: str,
  embryo_pattern_dir: str,
  pattern_metadata_path: str,
  cluster_col: str,
  ic_trim_thr: float = 0.2
):
  print("One hot encoding ... ")
  ohs = np.array(
    [
      one_hot_encode_sequence(
        genome.fetch( r.split(":")[0], *map(int, r.split(":")[1].split("-")) ),
        expand_dim = False
      )
      for r in tqdm(region_names)
    ]
  )
  print("loading data")
  patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
    topics=organoid_topics,
    pattern_dir=organoid_pattern_dir,
    prefix="organoid_")
  patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
    topics=embryo_topics,
    pattern_dir=embryo_pattern_dir,
    prefix="embryo_")
  all_patterns = [*patterns_organoid, *patterns_embryo]
  all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
  pattern_metadata=pd.read_table(pattern_metadata_path, index_col = 0)
  motifs = {
    n: pattern.ppm[range(*pattern.ic_trim(ic_trim_thr))].T
    for n, pattern in zip(all_pattern_names, all_patterns)
    if (n in pattern_metadata.index and pattern_metadata.loc[n, cluster_col] in selected_clusters)
  }
  print("Scoring hits ...")
  l_hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2), threshold = 0.5)
  for i in tqdm(range(len(l_hits)), desc = "getting max"):
    l_hits[i]["-logp"] = -np.log10(l_hits[i]["p-value"] + 1e-6)
    l_hits[i] = l_hits[i].groupby(["sequence_name", "motif_name"])[["score", "-logp"]].max().reset_index()
  hits = pd.concat(l_hits)
  hits["sequence_name"] = [region_names[x] for x in tqdm(hits["sequence_name"], desc = "sequence name")]
  hits["cluster"] = [
    pattern_metadata.loc[
      m, cluster_col
    ]
    for m in tqdm(hits["motif_name"], desc = "cluster")
  ]
  return hits


progenitor_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_progenitor_organoid = list(
  set.union(
    *[
      set(progenitor_region_topic_organoid[f"Topic{topic - 25}"]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in  cell_type_to_modisco_result["progenitor"].organoid_topics
    ]
  )
)

hg38 = pysam.FastaFile("../../../../../../resources/hg38/hg38.fa")

progenitor_hits_organoid = get_hit_score_for_topics(
  region_names = selected_regions_progenitor_organoid,
  genome = hg38,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  ic_trim_thr=0.2,
  organoid_topics=cell_type_to_modisco_result["progenitor"].organoid_topics,
  embryo_topics=cell_type_to_modisco_result["progenitor"].embryo_topics,
  selected_clusters=cell_type_to_modisco_result["progenitor"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["progenitor"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["progenitor"].cluster_col,
)

max_hits_per_seq_progenitor_organoid = progenitor_hits_organoid \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()


topics = set([f"Topic{t - 25}" for t in cell_type_to_modisco_result["progenitor"].organoid_topics])

fg = set(["Topic8"])

fg_y = np.log(
    progenitor_region_topic_organoid.loc[selected_regions_progenitor_organoid, list(fg)[0]].values + 1e-6
)
bg_y = np.log(
  progenitor_region_topic_organoid.loc[selected_regions_progenitor_organoid, list(topics - fg)].values.max(1) + 1e-6
)

fig, ax = plt.subplots()
ax.scatter(
  fg_y,
  bg_y,
  color = "black", s = 1
)
ax.set_xlabel(f"np.log10({list(fg)[0]})")
ax.set_ylabel(f"np.log10(max({', '.join(topic - fg)}))")
fig.savefig(f"plots/progenitor_organoid_r_topic_contrib_{'_'.join(fg)}_{'_'.join(topic - fg)}.pdf")

y = (fg_y - bg_y).reshape(-1, 1)
X = max_hits_per_seq_progenitor_organoid \
  .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
  .loc[selected_regions_progenitor_organoid]
feature_names = list(X.columns)
X = X.to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.33, random_state = 123
)
reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
reg.score(X_test, (y_test > 0).ravel())

fig, ax = plt.subplots()
ax.scatter(
  x = y_test,
  y = reg.predict_proba(X_test)[:, 1],
  s = 4, color = "black"
)
ax.set_xlabel("Ground truth")
ax.set_ylabel("Prediction")
fig.tight_layout()
fig.savefig(f"plots/progenitor_organoid_prediction_{'_'.join(fg)}_{'_'.join(topic - fg)}.pdf")

from sklearn.metrics import precision_recall_curve, roc_curve, auc

precision, recall, threshold = precision_recall_curve(
    (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])

fpr, tpr, thresholds = roc_curve(
    (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])

fig, axs = plt.subplots(ncols = 2, figsize = (8, 4))
_ = axs[0].plot(fpr, tpr, color = "black")
_ = axs[0].text(0.05, 0.90, s = f"AUC = {np.round(auc(fpr, tpr), 2)}", transform = axs[0].transAxes)
_ = axs[1].plot(recall, precision, color = "black")
_ = axs[1].text(0.7, 0.90, s = f"AUC = {np.round(auc(recall, precision), 2)}", transform = axs[1].transAxes)
_ = axs[0].set_xlabel("FPR")
_ = axs[0].set_ylabel("TPR")
_ = axs[1].set_xlabel("Recall")
_ = axs[1].set_ylabel("Precision")
for ax  in axs:
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.grid()
  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"plots/progenitor_organoid_PR_ROC_{'_'.join(fg)}_{'_'.join(topic - fg)}.pdf")

@dataclass
class ClassificationResult:
  model: LogisticRegression
  precision: np.ndarray
  recall: np.ndarray
  fpr: np.ndarray
  tpr: np.ndarray
  auc_pr: float
  auc_roc: float
  def __repr__(self) -> str:
    return f"Result - AUC_pr = {np.round(self.auc_pr, 2)} | AUC_roc = {np.round(self.auc_roc, 2)}"

def get_classification_results(
  fg: np.ndarray,
  bg: np.ndarray,
  X: np.ndarray,
  seed: int = 123
) -> ClassificationResult:
  y = (fg - bg).reshape(-1, 1)
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state =seed
  )
  reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
  precision, recall, threshold = precision_recall_curve(
      (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])
  fpr, tpr, thresholds = roc_curve(
      (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])
  return ClassificationResult(
    model=reg,
    precision=precision,
    recall=recall,
    fpr=fpr,
    tpr=tpr,
    auc_roc=auc(fpr, tpr),
    auc_pr=auc(recall, precision)
  )

X = max_hits_per_seq_progenitor_organoid \
  .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
  .loc[selected_regions_progenitor_organoid]
feature_names = list(X.columns)
X = X.to_numpy()

topic_to_classification_result_progenitor_organoid = {}
t_offset = 25
for topic in cell_type_to_modisco_result["progenitor"].organoid_topics:
  fg = set([f"Topic{topic - t_offset}"])
  bg = set([f"Topic{t - t_offset}" for t in cell_type_to_modisco_result["progenitor"].organoid_topics]) \
       - fg
  fg_y = np.log(
    progenitor_region_topic_organoid.loc[
      selected_regions_progenitor_organoid,
      list(fg)
    ].max(1).values + 1e-6
  )
  bg_y = np.log(
    progenitor_region_topic_organoid.loc[
      selected_regions_progenitor_organoid,
      list(bg)
    ].values.max(1) + 1e-6
  )
  topic_to_classification_result_progenitor_organoid[topic] = get_classification_results(
    fg=fg_y,
    bg=bg_y,
    X=X,
  )

fig = sns.clustermap(
  pd.DataFrame(index = feature_names, data = {t: topic_to_classification_result_progenitor_organoid[t].model.coef_.squeeze() for t in topic_to_classification_result}),
  vmin = -0.5, vmax = 0.5,
  col_cluster = False,
  cmap = "bwr"
)

hg38 = pysam.FastaFile("../../../../../../resources/hg38/hg38.fa")

#
# PROGENITOR ORGANOID
#

progenitor_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_progenitor_organoid = list(
  set.union(
    *[
      set(progenitor_region_topic_organoid[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in progenitor_region_topic_organoid.columns
    ]
  )
)

selected_regions_progenitor_organoid = [
  r for r in selected_regions_progenitor_organoid
  if r.startswith("chr")
]

progenitor_hits_organoid = get_hit_score_for_topics(
  region_names = selected_regions_progenitor_organoid,
  genome = hg38,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  ic_trim_thr=0.2,
  organoid_topics=cell_type_to_modisco_result["progenitor_dv"].organoid_topics,
  embryo_topics=cell_type_to_modisco_result["progenitor_dv"].embryo_topics,
  selected_clusters=cell_type_to_modisco_result["progenitor_dv"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["progenitor_dv"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["progenitor_dv"].cluster_col,
)

max_hits_per_seq_progenitor_organoid = progenitor_hits_organoid \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()

#
# neural_crest ORGANOID
#

neural_crest_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_neural_crest_organoid = list(
  set.union(
    *[
      set(neural_crest_region_topic_organoid[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in neural_crest_region_topic_organoid.columns
    ]
  )
)

selected_regions_neural_crest_organoid = [
  r for r in selected_regions_neural_crest_organoid
  if r.startswith("chr")
]

neural_crest_hits_organoid = get_hit_score_for_topics(
  region_names = selected_regions_neural_crest_organoid,
  genome = hg38,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  ic_trim_thr=0.2,
  organoid_topics=cell_type_to_modisco_result["neural_crest"].organoid_topics,
  embryo_topics=cell_type_to_modisco_result["neural_crest"].embryo_topics,
  selected_clusters=cell_type_to_modisco_result["neural_crest"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["neural_crest"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["neural_crest"].cluster_col,
)

max_hits_per_seq_neural_crest_organoid = neural_crest_hits_organoid \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()


#
# neuron ORGANOID
#

neuron_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_neuron_organoid = list(
  set.union(
    *[
      set(neuron_region_topic_organoid[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in neuron_region_topic_organoid.columns
    ]
  )
)

selected_regions_neuron_organoid = [
  r for r in selected_regions_neuron_organoid
  if r.startswith("chr")
]

neuron_hits_organoid = get_hit_score_for_topics(
  region_names = selected_regions_neuron_organoid,
  genome = hg38,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  ic_trim_thr=0.2,
  organoid_topics=cell_type_to_modisco_result["neuron"].organoid_topics,
  embryo_topics=cell_type_to_modisco_result["neuron"].embryo_topics,
  selected_clusters=cell_type_to_modisco_result["neuron"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["neuron"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["neuron"].cluster_col,
)

max_hits_per_seq_neuron_organoid = neuron_hits_organoid \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()


#
# PROGENITOR embryo
#

progenitor_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_progenitor_dv_embryo = list(
  set.union(
    *[
      set(progenitor_region_topic_embryo[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in progenitor_region_topic_embryo
    ]
  )
)

progenitor_dv_hits_embryo = get_hit_score_for_topics(
  region_names = selected_regions_progenitor_dv_embryo,
  genome = hg38,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  ic_trim_thr=0.2,
  embryo_topics=cell_type_to_modisco_result["progenitor_dv"].embryo_topics,
  organoid_topics=cell_type_to_modisco_result["progenitor_dv"].organoid_topics,
  selected_clusters=cell_type_to_modisco_result["progenitor_dv"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["progenitor_dv"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["progenitor_dv"].cluster_col,
)

max_hits_per_seq_progenitor_dv_embryo = progenitor_dv_hits_embryo \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()

selected_regions_progenitor_ap_embryo = list(
  set.union(
    *[
      set(progenitor_region_topic_embryo[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in progenitor_region_topic_embryo
    ]
  )
)

progenitor_ap_hits_embryo = get_hit_score_for_topics(
  region_names = selected_regions_progenitor_ap_embryo,
  genome = hg38,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  ic_trim_thr=0.2,
  embryo_topics=cell_type_to_modisco_result["progenitor_ap"].embryo_topics,
  organoid_topics=cell_type_to_modisco_result["progenitor_ap"].organoid_topics,
  selected_clusters=cell_type_to_modisco_result["progenitor_ap"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["progenitor_ap"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["progenitor_ap"].cluster_col,
)

max_hits_per_seq_progenitor_ap_embryo = progenitor_ap_hits_embryo \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()


#
# neural_crest embryo
#

neural_crest_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neural_crest_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_neural_crest_embryo = list(
  set.union(
    *[
      set(neural_crest_region_topic_embryo[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in neural_crest_region_topic_embryo.columns
    ]
  )
)

selected_regions_neural_crest_embryo = [
  r for r in selected_regions_neural_crest_embryo
  if r.startswith("chr")
]

neural_crest_hits_embryo = get_hit_score_for_topics(
  region_names = selected_regions_neural_crest_embryo,
  genome = hg38,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  ic_trim_thr=0.2,
  embryo_topics=cell_type_to_modisco_result["neural_crest"].embryo_topics,
  organoid_topics=cell_type_to_modisco_result["neural_crest"].organoid_topics,
  selected_clusters=cell_type_to_modisco_result["neural_crest"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["neural_crest"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["neural_crest"].cluster_col,
)

max_hits_per_seq_neural_crest_embryo = neural_crest_hits_embryo \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()


#
# neuron embryo
#

neuron_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neuron_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions_neuron_embryo = list(
  set.union(
    *[
      set(neuron_region_topic_embryo[topic]
      .sort_values(ascending = False)
      .head(1_000)
      .index)
      for topic in neuron_region_topic_embryo.columns
    ]
  )
)

selected_regions_neuron_embryo = [
  r for r in selected_regions_neuron_embryo
  if r.startswith("chr")
]

neuron_hits_embryo = get_hit_score_for_topics(
  region_names = selected_regions_neuron_embryo,
  genome = hg38,
  embryo_pattern_dir=embryo_dl_motif_dir ,
  organoid_pattern_dir=organoid_dl_motif_dir ,
  ic_trim_thr=0.2,
  embryo_topics=cell_type_to_modisco_result["neuron"].embryo_topics,
  organoid_topics=cell_type_to_modisco_result["neuron"].organoid_topics,
  selected_clusters=cell_type_to_modisco_result["neuron"].selected_patterns,
  pattern_metadata_path=cell_type_to_modisco_result["neuron"].pattern_metadata_path,
  cluster_col=cell_type_to_modisco_result["neuron"].cluster_col,
)

max_hits_per_seq_neuron_embryo = neuron_hits_embryo \
  .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()

"""
# I ran this before using only regions of topics of interest,
# conclusion is the same, but auPR is higher with this
# higher recall

max_hits_per_seq_progenitor_organoid.to_csv(
  "max_hits_per_seq_progenitor_organoid.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neural_crest_organoid.to_csv(
  "max_hits_per_seq_neural_crest_organoid.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neuron_organoid.to_csv(
  "max_hits_per_seq_neuron_organoid.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_progenitor_dv_embryo.to_csv(
  "max_hits_per_seq_progenitor_dv_embryo.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_progenitor_ap_embryo.to_csv(
  "max_hits_per_seq_progenitor_ap_embryo.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neural_crest_embryo.to_csv(
  "max_hits_per_seq_neural_crest_embryo.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neuron_embryo.to_csv(
  "max_hits_per_seq_neuron_embryo.tsv",
  sep = "\t", index = False
)
"""
max_hits_per_seq_progenitor_organoid.to_csv(
  "max_hits_per_seq_progenitor_organoid_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neural_crest_organoid.to_csv(
  "max_hits_per_seq_neural_crest_organoid_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neuron_organoid.to_csv(
  "max_hits_per_seq_neuron_organoid_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_progenitor_dv_embryo.to_csv(
  "max_hits_per_seq_progenitor_dv_embryo_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_progenitor_ap_embryo.to_csv(
  "max_hits_per_seq_progenitor_ap_embryo_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neural_crest_embryo.to_csv(
  "max_hits_per_seq_neural_crest_embryo_all_regions.tsv",
  sep = "\t", index = False
)

max_hits_per_seq_neuron_embryo.to_csv(
  "max_hits_per_seq_neuron_embryo_all_regions.tsv",
  sep = "\t", index = False
)



cell_type_to_hits_offset_region_topic = {
  "organoid_progenitor_dv": (
    max_hits_per_seq_progenitor_organoid,
    25,
    progenitor_region_topic_organoid,
    selected_regions_progenitor_organoid
  ),
  "organoid_neural_crest": (
    max_hits_per_seq_neural_crest_organoid,
    55,
    neural_crest_region_topic_organoid,
    selected_regions_neural_crest_organoid
  ),
  "organoid_neuron": (
    max_hits_per_seq_neuron_organoid,
    0,
    neuron_region_topic_organoid,
    selected_regions_neuron_organoid
  ),
  "embryo_progenitor_dv": (
    max_hits_per_seq_progenitor_dv_embryo,
    30,
    progenitor_region_topic_embryo,
    selected_regions_progenitor_dv_embryo
  ),
  "embryo_progenitor_ap": (
    max_hits_per_seq_progenitor_ap_embryo,
    30,
    progenitor_region_topic_embryo,
    selected_regions_progenitor_ap_embryo
  ),
  "embryo_neural_crest": (
    max_hits_per_seq_neural_crest_embryo,
    90,
    neural_crest_region_topic_embryo,
    selected_regions_neural_crest_embryo
  ),
  "embryo_neuron": (
    max_hits_per_seq_neuron_embryo,
    0,
    neuron_region_topic_embryo,
    selected_regions_neuron_embryo
  )
}

cell_type_to_classification_result = {}
for cell_type in cell_type_to_hits_offset_region_topic:
  print(cell_type)
  max_hits, t_offset, region_topic, selected_regions = cell_type_to_hits_offset_region_topic[cell_type]
  topics = cell_type_to_modisco_result[cell_type.split("_", 1)[1]].organoid_topics \
    if cell_type.split("_")[0] == "organoid" else \
    cell_type_to_modisco_result[cell_type.split("_", 1)[1]].embryo_topics
  X = max_hits  \
  .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
  .loc[selected_regions]
  feature_names = list(X.columns)
  X = X.to_numpy()
  topic_to_classification_result = {}
  for topic in topics:
    print(topic)
    if cell_type.split("_")[0] == "organoid":
      fg = set([f"Topic{topic - t_offset}"])
      bg = set(region_topic.columns) \
            - fg
    else:
      fg = set([f"Topic_{topic - t_offset}"])
      bg = set(region_topic.columns) \
            - fg
    fg_y = np.log(
      region_topic.loc[
        selected_regions,
        list(fg)
      ].max(1).values + 1e-6
    )
    bg_y = np.log(
      region_topic.loc[
        selected_regions,
        list(bg)
      ].values.max(1) + 1e-6
    )
    topic_to_classification_result[topic] = get_classification_results(
      fg=fg_y,
      bg=bg_y,
      X=X,
    )
  cell_type_to_classification_result[cell_type] = (topic_to_classification_result, feature_names)

n_motifs_acc = []
thr = 0.2
for cell_type in cell_type_to_classification_result:
  for topic in cell_type_to_classification_result[cell_type][0].keys():
    n_motifs = (cell_type_to_classification_result[cell_type][0][topic].model.coef_ > thr).sum()
    auc_pr = cell_type_to_classification_result[cell_type][0][topic].auc_pr
    auc_roc = cell_type_to_classification_result[cell_type][0][topic].auc_roc
    n_motifs_acc.append((cell_type, topic, n_motifs, auc_pr, auc_roc))

fig, ax = plt.subplots()
ax.scatter(
  [n_motifs for _, _, n_motifs, auc_pr, auc_roc in n_motifs_acc],
  [auc_roc for _, _, n_motifs, auc_pr, auc_roc in n_motifs_acc],
)
fig.tight_layout()
fig.savefig("plots/n_motifs_v_auc_roc.pdf")

color_dict = {
  "progenitor_dv": "#0066ff",
  "progenitor_ap": "#002255",
  "neuron": "#cc9900",
  "neural_crest": "#7E52A0"
}

data = sorted(n_motifs_acc, key = lambda x: x[3])
fig, axs = plt.subplots(figsize = (4, 8), ncols = 2)
axs = axs[::-1]
_ = axs[0].scatter(
  [d[2] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
_ = axs[1].scatter(
  [d[3] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
for y, d in enumerate(data):
  _ = axs[0].plot(
    [0, d[2]],
    [y, y],
    color = "black",
    zorder = 1
  )
  _ = axs[1].plot(
    [0, d[3]],
    [y, y],
    color = "black",
    zorder = 1
  )
_ = axs[1].set_xlim(0, 1)
axs[1].invert_xaxis()
axs[1].set_xticks(np.arange(0, 1.25, 0.25), labels = ["", "", "0.5", "", "1"])
axs[1].yaxis.tick_right()
_ = axs[0].set_xlim(0, 7)
_ = axs[0].set_xticks(np.arange(1, 8, 1))
_ = axs[0].set_yticks(
  np.arange(len(data)), labels = [d[0][0].upper()+"_Topic_"+str(d[1]) for d in data])
_ = axs[1].set_yticks(
  np.arange(len(data)), labels = ["" for _ in data])
_ = axs[0].set_xlabel("nr of patterns")
_ = axs[1].set_xlabel("auc PR")
for ax in axs:
  ax.grid(True)
  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plots/auc_pr_and_n_motifs.pdf")

data = sorted(n_motifs_acc, key = lambda x: x[4])
fig, axs = plt.subplots(figsize = (4, 8), ncols = 2)
axs = axs[::-1]
_ = axs[0].scatter(
  [d[2] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
_ = axs[1].scatter(
  [d[4] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
for y, d in enumerate(data):
  _ = axs[0].plot(
    [0, d[2]],
    [y, y],
    color = "black",
    zorder = 1
  )
  _ = axs[1].plot(
    [0, d[4]],
    [y, y],
    color = "black",
    zorder = 1
  )
_ = axs[1].set_xlim(0, 1)
axs[1].invert_xaxis()
axs[1].set_xticks(np.arange(0, 1.25, 0.25), labels = ["", "", "0.5", "", "1"])
axs[1].yaxis.tick_right()
_ = axs[0].set_xlim(0, 7)
_ = axs[0].set_xticks(np.arange(1, 8, 1))
_ = axs[0].set_yticks(
  np.arange(len(data)), labels = [d[0][0].upper()+"_Topic_"+str(d[1]) for d in data])
_ = axs[1].set_yticks(
  np.arange(len(data)), labels = ["" for _ in data])
_ = axs[0].set_xlabel("nr of patterns")
_ = axs[1].set_xlabel("auc ROC")
for ax in axs:
  ax.grid(True)
  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plots/auc_ROC_and_n_motifs.pdf")




```


```python

import scanpy as sc
import pandas as pd
import numpy as np

# load organoid RNA data and subset for ATAC cells
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

adata_embryo.obs["leiden_0.1"] = pd.Categorical(adata_embryo.obs["leiden_0.1"])

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

X_organoid = adata_organoid.obsm["X_pca_harmony"]

class LazyImputedAcc:
  def __init__(self, region_topic: pd.DataFrame, cell_topic: pd.DataFrame):
    self._all_regions = region_topic.index.to_list()
    self._all_cells   = cell_topic.index.to_list()
    self._region_index = {region: i for i, region in enumerate(self._all_regions)}
    self._cell_index   = {cell:   i for i, cell   in enumerate(self._all_cells)}
    self._nd_arr_region_topic = region_topic.to_numpy()
    self._nd_arr_cell_topic   = cell_topic.to_numpy().T
  def _is_empty_slice(self, s: slice):
    return s.start is None and s.step is None and s.stop is None
  def __getitem__(self, key: list[str] | str | tuple[list[str] | slice | str, list[str] | slice | str]):
    if isinstance(key, list) or isinstance(key, str):
      # imputed_acc[["region_1", "region_2"]]
      regions_to_return = key if isinstance(key, list) else [key]
      cells_to_return = self._all_cells
    elif isinstance(key, tuple):
      if len(key) > 2:
        raise ValueError("Only 2D indexis is possible!")
      if isinstance(key[0], slice) and (isinstance(key[1], list) or isinstance(key[1], str)):
        # imputed_acc[:, ["cell_1", "cell_2"]]
        if not self._is_empty_slice(key[0]):
          raise ValueError("Numeric slicing is not supported!")
        regions_to_return = self._all_regions
        cells_to_return = key[1] if isinstance(key[1], list) else [key[1]]
      elif (isinstance(key[0], list) or isinstance(key[0], str)) and isinstance(key[1], slice):
        # imputed_acc[["region_1", "region_2"], :]
        if not self._is_empty_slice(key[1]):
          raise ValueError("Numeric slicing is not supported!")
        regions_to_return = key[0] if isinstance(key[0], list) else [key[0]]
        cells_to_return = self._all_cells
      elif (isinstance(key[0], list) or isinstance(key[0], str)) and (isinstance(key[1], list) or isinstance(key[1], str)):
        regions_to_return = key[0] if isinstance(key[0], list) else [key[0]]
        cells_to_return = key[1] if isinstance(key[1], list) else [key[1]]
      else:
        raise ValueError("This type of indexing is not supported!")
    else:
      raise ValueError("This type of indexing is not supported!")
    try:
      cell_idc = [self._cell_index[c] for c in cells_to_return]
    except KeyError as c_not_found:
      raise KeyError(f"{c_not_found} not in cell_topic matrix!")
    try:
      region_idc = [self._region_index[r] for r in regions_to_return]
    except KeyError as r_not_found:
      raise KeyError(f"{r_not_found} not in region_topic matrix!")
    imp = self._nd_arr_region_topic[region_idc, :] @ self._nd_arr_cell_topic[:, cell_idc]
    return pd.DataFrame(
      data = imp,
      index = regions_to_return,
      columns = cells_to_return
    )

def rename_organoid_atac_cell(l):
  bc, sample_id = l.strip().split("-1", 1)
  sample_id = sample_id.split("___")[-1]
  return bc + "-1" + "-" + sample_id_to_num[sample_id]

progenitor_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

progenitor_cell_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/progenitor_cell_topic_contrib.tsv",
  index_col = 0
)

progenitor_cell_topic_organoid.index = [
  rename_organoid_atac_cell(c) for c in progenitor_cell_topic_organoid.index
]

neural_crest_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
  index_col = 0
)

neural_crest_cell_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neural_crest_cell_topic_contrib.tsv",
  index_col = 0
)

neural_crest_cell_topic_organoid.index = [
  rename_organoid_atac_cell(c) for c in neural_crest_cell_topic_organoid.index
]

neuron_region_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv",
  index_col = 0
)

neuron_cell_topic_organoid = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/neuron_cell_topic_contrib.tsv",
  index_col = 0
)

neuron_cell_topic_organoid.index = [
  rename_organoid_atac_cell(c) for c in neuron_cell_topic_organoid.index
]

common_regions_organoid = list(
  set(progenitor_region_topic_organoid.index) & \
  set(neural_crest_region_topic_organoid.index) & \
  set(neuron_region_topic_organoid.index)
)

organoid_dv_topics = [f"Topic{t}" for t in [8, 13, 11, 29, 23]]
organoid_neural_crest_topics = [f"Topic{t}" for t in [7, 5, 10, 4, 3]]
organoid_neuron_topics = [f"Topic{t}" for t in [6, 4, 23, 24, 13, 2]]

progenitor_imp_acc_organoid = LazyImputedAcc(
  region_topic=progenitor_region_topic_organoid.loc[
    common_regions_organoid
  ],
  cell_topic=progenitor_cell_topic_organoid.loc[
    list(set(adata_organoid.obs_names) & set(progenitor_cell_topic_organoid.index))
  ]
)
neural_crest_imp_acc_organoid = LazyImputedAcc(
  region_topic=neural_crest_region_topic_organoid.loc[
    common_regions_organoid
],
  cell_topic=neural_crest_cell_topic_organoid.loc[
    list(set(adata_organoid.obs_names) & set(neural_crest_cell_topic_organoid.index))
]
)
neuron_imp_acc_organoid = LazyImputedAcc(
  region_topic=neuron_region_topic_organoid.loc[
    common_regions_organoid
],
  cell_topic=neuron_cell_topic_organoid.loc[
    list(set(adata_organoid.obs_names) & set(neuron_cell_topic_organoid.index))
]
)

progenitor_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

progenitor_cell_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/progenitor_cell_topic_contrib.tsv",
  index_col = 0
)

progenitor_cell_topic_embryo.index = [
  c.replace("___", "-1___") for c in progenitor_cell_topic_embryo.index
]

progenitor_cell_topic_embryo.columns = [
  c.replace("Topic", "Topic_") for c in progenitor_cell_topic_embryo.columns
]



neural_crest_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neural_crest_region_topic_contrib.tsv",
  index_col = 0
)

neural_crest_cell_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neural_crest_cell_topic_contrib.tsv",
  index_col = 0
)

neural_crest_cell_topic_embryo.index = [
  c.replace("___", "-1___") for c in neural_crest_cell_topic_embryo.index
]

neural_crest_cell_topic_embryo.columns = [
  c.replace("Topic", "Topic_") for c in neural_crest_cell_topic_embryo.columns
]

neuron_region_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neuron_region_topic_contrib.tsv",
  index_col = 0
)

neuron_cell_topic_embryo = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/neuron_cell_topic_contrib.tsv",
  index_col = 0
)

neuron_cell_topic_embryo.index = [
  c.replace("___", "-1___") for c in neuron_cell_topic_embryo.index
]

neuron_cell_topic_embryo.columns = [
  c.replace("Topic", "Topic_") for c in neuron_cell_topic_embryo.columns
]


common_regions_embryo = list(
  set(progenitor_region_topic_embryo.index) & \
  set(neural_crest_region_topic_embryo.index) & \
  set(neuron_region_topic_embryo.index)
)

embryo_dv_topics = [f"Topic_{t}" for t in [4,  8, 49, 58, 28]]
embryo_ap_topics = [f"Topic_{t}" for t in [31, 29, 1, 32, 40, 22, 41]]
embryo_neural_crest_topics = [f"Topic_{t}" for t in [13, 15, 4, 1]]
embryo_neuron_topics = [f"Topic_{t}" for t in [10, 8, 13, 24, 18, 29]]

progenitor_imp_acc_embryo = LazyImputedAcc(
  region_topic=progenitor_region_topic_embryo.loc[
    common_regions_embryo
],
  cell_topic=progenitor_cell_topic_embryo.loc[
    list(set(adata_embryo.obs_names) & set(progenitor_cell_topic_embryo.index))
]
)
neural_crest_imp_acc_embryo = LazyImputedAcc(
  region_topic=neural_crest_region_topic_embryo.loc[
  common_regions_embryo
],
  cell_topic=neural_crest_cell_topic_embryo.loc[
    list(set(adata_embryo.obs_names) & set(neural_crest_cell_topic_embryo.index))
]
)
neuron_imp_acc_embryo = LazyImputedAcc(
  region_topic=neuron_region_topic_embryo.loc[
  common_regions_embryo
],
  cell_topic=neuron_cell_topic_embryo.loc[
    list(set(adata_embryo.obs_names) & set(neuron_cell_topic_embryo.index))
]
)

for topic in embryo_neural_crest_topics:
  print(topic)
  regions = neural_crest_region_topic_embryo.loc[common_regions][topic].sort_values(ascending = False).head(500).index.to_list()
  cells = np.concatenate(
      pd.concat(
        [
          imp_acc[regions] for imp_acc in
          (progenitor_imp_acc_embryo, neural_crest_imp_acc_embryo, neuron_imp_acc_embryo)
        ],axis = 1) \
      .apply(lambda region: region.nlargest(50).index.to_numpy(), axis = 1).values
    )
  print(np.unique(adata_embryo.obs.loc[cells, "COMMON_ANNOTATION"], return_counts = True))

    pd.concat(
      [
        imp_acc[regions] for imp_acc in
        (progenitor_imp_acc_embryo, neural_crest_imp_acc_embryo, neuron_imp_acc_embryo)
      ],axis = 1)

pca_organoid = pd.DataFrame(adata_organoid.obsm["X_pca_harmony"], index = adata_organoid.obs_names)
pca_embryo = pd.DataFrame(adata_embryo.obsm["X_pca"], index = adata_embryo.obs_names)


def compute_dispersion_embryo(regions: list[str], n_cells_per_region: int = 50, n_pcs: int = 20):
  top_cells_per_region = pd.concat(
    [
      imp_acc[regions] for imp_acc in
      (progenitor_imp_acc_embryo, neural_crest_imp_acc_embryo, neuron_imp_acc_embryo)
    ],axis = 1) \
  .apply(lambda region: region.nlargest(n_cells_per_region).index.to_numpy(), axis = 1).values
  quartile_coefs = np.zeros(len(regions))
  for i, cells in enumerate(top_cells_per_region):
    Q1, Q3 = np.percentile(
      np.linalg.norm(pca_embryo.loc[cells, 0:n_pcs] - pca_embryo.loc[cells, 0:n_pcs].mean(0), axis = 1),
      [25, 75]
    )
    quartile_coefs[i] = (Q3 - Q1) / (Q3 + Q1)
  return quartile_coefs

def compute_dispersion_organoid(regions: list[str], n_cells_per_region: int = 50, n_pcs: int = 20):
  top_cells_per_region = pd.concat(
    [
      imp_acc[regions] for imp_acc in
      (progenitor_imp_acc_organoid, neural_crest_imp_acc_organoid, neuron_imp_acc_organoid)
    ],axis = 1) \
  .apply(lambda region: region.nlargest(n_cells_per_region).index.to_numpy(), axis = 1).values
  quartile_coefs = np.zeros(len(regions))
  for i, cells in enumerate(top_cells_per_region):
    Q1, Q3 = np.percentile(
      np.linalg.norm(pca_organoid.loc[cells, 0:n_pcs] - pca_organoid.loc[cells, 0:n_pcs].mean(0), axis = 1),
      [25, 75]
    )
    quartile_coefs[i] = (Q3 - Q1) / (Q3 + Q1)
  return quartile_coefs

@dataclass
class ModiscoClusteringResult:
  organoid_topics: list[int]
  embryo_topics: list[int]
  pattern_metadata_path: str
  cluster_col: str
  selected_patterns: list[float]


progenitor_dv_clustering_result = ModiscoClusteringResult(
  organoid_topics=[33, 38, 36, 54, 48],
  embryo_topics=[34, 38, 79, 88, 58],
  pattern_metadata_path="../../figure_3/draft/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns = [
    6, 7, 12, 15, 10, 18, 1, 19
  ]
)

progenitor_ap_clustering_result = ModiscoClusteringResult(
  organoid_topics=[],
  embryo_topics=[61, 59, 31, 62, 70, 52, 71],
  pattern_metadata_path="../../figure_4/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns=[1, 4, 8, 7, 6]
)

neural_crest_clustering_result = ModiscoClusteringResult(
  organoid_topics=[62, 60, 65, 59, 58],
  embryo_topics=[103, 105, 94, 91],
  pattern_metadata_path="../../figure_5/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2
  ]
)

neuron_clustering_result = ModiscoClusteringResult(
  organoid_topics=[6, 4, 23, 24, 13, 2],
  embryo_topics=[10, 8, 13, 24, 18, 29],
  pattern_metadata_path="../../figure_6/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0
  ]
)

cell_type_to_modisco_result = {
  "progenitor_dv": progenitor_dv_clustering_result,
  "progenitor_ap": progenitor_ap_clustering_result,
  "neural_crest": neural_crest_clustering_result,
  "neuron": neuron_clustering_result
}

max_hits_per_seq_progenitor_organoid = pd.read_table(
  "max_hits_per_seq_progenitor_organoid.tsv",
)
max_hits_per_seq_neural_crest_organoid = pd.read_table(
  "max_hits_per_seq_neural_crest_organoid.tsv",
)
max_hits_per_seq_neuron_organoid = pd.read_table(
  "max_hits_per_seq_neuron_organoid.tsv",
)
max_hits_per_seq_progenitor_dv_embryo = pd.read_table(
  "max_hits_per_seq_progenitor_dv_embryo.tsv",
)
max_hits_per_seq_progenitor_ap_embryo = pd.read_table(
  "max_hits_per_seq_progenitor_ap_embryo.tsv",
)
max_hits_per_seq_neural_crest_embryo = pd.read_table(
  "max_hits_per_seq_neural_crest_embryo.tsv",
)
max_hits_per_seq_neuron_embryo = pd.read_table(
  "max_hits_per_seq_neuron_embryo.tsv",
)

cell_type_to_hits_offset_region_topic = {
  "organoid_progenitor_dv": (
    max_hits_per_seq_progenitor_organoid,
    25,
    progenitor_region_topic_organoid,
    list(set(max_hits_per_seq_progenitor_organoid.sequence_name)),
  ),
  "organoid_neural_crest": (
    max_hits_per_seq_neural_crest_organoid,
    55,
    neural_crest_region_topic_organoid,
    list(set(max_hits_per_seq_neural_crest_organoid.sequence_name)),
  ),
  "organoid_neuron": (
    max_hits_per_seq_neuron_organoid,
    0,
    neuron_region_topic_organoid,
    list(set(max_hits_per_seq_neuron_organoid.sequence_name)),
  ),
  "embryo_progenitor_dv": (
    max_hits_per_seq_progenitor_dv_embryo,
    30,
    progenitor_region_topic_embryo,
    list(set(max_hits_per_seq_progenitor_dv_embryo.sequence_name)),
  ),
  "embryo_progenitor_ap": (
    max_hits_per_seq_progenitor_ap_embryo,
    30,
    progenitor_region_topic_embryo,
  list(set(max_hits_per_seq_progenitor_ap_embryo.sequence_name)), 
  ),
  "embryo_neural_crest": (
    max_hits_per_seq_neural_crest_embryo,
    90,
    neural_crest_region_topic_embryo,
    list(set(max_hits_per_seq_neural_crest_embryo.sequence_name)),
  ),
  "embryo_neuron": (
    max_hits_per_seq_neuron_embryo,
    0,
    neuron_region_topic_embryo,
    list(set(max_hits_per_seq_neuron_embryo.sequence_name)),
  )
}

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc

@dataclass
class ClassificationResult:
  model: LogisticRegression
  precision: np.ndarray
  recall: np.ndarray
  fpr: np.ndarray
  tpr: np.ndarray
  auc_pr: float
  auc_roc: float
  def __repr__(self) -> str:
    return f"Result - AUC_pr = {np.round(self.auc_pr, 2)} | AUC_roc = {np.round(self.auc_roc, 2)}"

def get_classification_results(
  fg: np.ndarray,
  bg: np.ndarray,
  X: np.ndarray,
  seed: int = 123
) -> ClassificationResult:
  y = (fg - bg).reshape(-1, 1)
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state =seed
  )
  reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
  precision, recall, threshold = precision_recall_curve(
      (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])
  fpr, tpr, thresholds = roc_curve(
      (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])
  return ClassificationResult(
    model=reg,
    precision=precision,
    recall=recall,
    fpr=fpr,
    tpr=tpr,
    auc_roc=auc(fpr, tpr),
    auc_pr=auc(recall, precision)
  )


cell_type_to_classification_result = {}
for cell_type in cell_type_to_hits_offset_region_topic:
  print(cell_type)
  max_hits, t_offset, region_topic, selected_regions = cell_type_to_hits_offset_region_topic[cell_type]
  topics = cell_type_to_modisco_result[cell_type.split("_", 1)[1]].organoid_topics \
    if cell_type.split("_")[0] == "organoid" else \
    cell_type_to_modisco_result[cell_type.split("_", 1)[1]].embryo_topics
  X = max_hits  \
  .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
  .loc[selected_regions]
  feature_names = list(X.columns)
  X = X.to_numpy()
  topic_to_classification_result = {}
  for topic in topics:
    print(topic)
    if cell_type.split("_")[0] == "organoid":
      fg = set([f"Topic{topic - t_offset}"])
      bg = set(region_topic.columns) \
            - fg
    else:
      fg = set([f"Topic_{topic - t_offset}"])
      bg = set(region_topic.columns) \
            - fg
    fg_y = np.log(
      region_topic.loc[
        selected_regions,
        list(fg)
      ].max(1).values + 1e-6
    )
    bg_y = np.log(
      region_topic.loc[
        selected_regions,
        list(bg)
      ].values.max(1) + 1e-6
    )
    topic_to_classification_result[topic] = get_classification_results(
      fg=fg_y,
      bg=bg_y,
      X=X,
    )
  cell_type_to_classification_result[cell_type] = (topic_to_classification_result, feature_names)

cell_type_to_dispersion = {}
for cell_type in cell_type_to_hits_offset_region_topic:
  print(cell_type)
  _, t_offset, region_topic, _ = cell_type_to_hits_offset_region_topic[cell_type]
  topics = cell_type_to_modisco_result[cell_type.split("_", 1)[1]].organoid_topics \
    if cell_type.split("_")[0] == "organoid" else \
    cell_type_to_modisco_result[cell_type.split("_", 1)[1]].embryo_topics
  topic_to_disp = {}
  for topic_n in topics:
    if cell_type.split("_")[0] == "organoid":
      topic = f"Topic{topic_n - t_offset}"
    else:
      topic = f"Topic_{topic_n - t_offset}"
    print(topic)
    regions = region_topic.loc[common_regions][topic].sort_values(ascending = False).head(500).index.to_list()
    if cell_type.split("_")[0] == "organoid":
      topic_to_disp[topic] = compute_dispersion_organoid(regions, 100)
    else:
      topic_to_disp[topic] = compute_dispersion_embryo(regions, 100)
  cell_type_to_dispersion[cell_type] = topic_to_disp

for cell_type in cell_type_to_dispersion:
  print(cell_type)
  for topic in cell_type_to_dispersion[cell_type]:
    print(f"\t{topic}: {cell_type_to_dispersion[cell_type][topic].mean()}")


n_motifs_acc = []
thr = 0
for cell_type in cell_type_to_classification_result:
  for topic in cell_type_to_classification_result[cell_type][0].keys():
    n_motifs = (cell_type_to_classification_result[cell_type][0][topic].model.coef_ > thr).sum()
    auc_pr = cell_type_to_classification_result[cell_type][0][topic].auc_pr
    auc_roc = cell_type_to_classification_result[cell_type][0][topic].auc_roc
    n_motifs_acc.append((cell_type, topic, n_motifs, auc_pr, auc_roc))

color_dict = {
  "progenitor_dv": "#0066ff",
  "progenitor_ap": "#002255",
  "neuron": "#cc9900",
  "neural_crest": "#7E52A0"
}

data = sorted(n_motifs_acc, key = lambda x: x[3])
fig, axs = plt.subplots(figsize = (4, 8), ncols = 2)
axs = axs[::-1]
_ = axs[0].scatter(
  [d[2] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
_ = axs[1].scatter(
  [d[3] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
for y, d in enumerate(data):
  _ = axs[0].plot(
    [0, d[2]],
    [y, y],
    color = "black",
    zorder = 1
  )
  _ = axs[1].plot(
    [0, d[3]],
    [y, y],
    color = "black",
    zorder = 1
  )
_ = axs[1].set_xlim(0, 1)
axs[1].invert_xaxis()
axs[1].set_xticks(np.arange(0, 1.25, 0.25), labels = ["", "", "0.5", "", "1"])
axs[1].yaxis.tick_right()
_ = axs[0].set_xlim(0, 7)
_ = axs[0].set_xticks(np.arange(1, 8, 1))
_ = axs[0].set_yticks(
  np.arange(len(data)), labels = [d[0][0].upper()+"_Topic_"+str(d[1]) for d in data])
_ = axs[1].set_yticks(
  np.arange(len(data)), labels = ["" for _ in data])
_ = axs[0].set_xlabel("nr of patterns")
_ = axs[1].set_xlabel("auc PR")
for ax in axs:
  ax.grid(True)
  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plots/auc_pr_and_n_motifs.pdf")

data = sorted(n_motifs_acc, key = lambda x: x[4])
fig, axs = plt.subplots(figsize = (4, 8), ncols = 2)
axs = axs[::-1]
_ = axs[0].scatter(
  [d[2] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
_ = axs[1].scatter(
  [d[4] for d in data],
  np.arange(len(data)),
  color = [color_dict[d[0].split("_", 1)[1]] for d in data],
  lw = 1,
  edgecolor = "black",
  zorder = 2
)
for y, d in enumerate(data):
  _ = axs[0].plot(
    [0, d[2]],
    [y, y],
    color = "black",
    zorder = 1
  )
  _ = axs[1].plot(
    [0, d[4]],
    [y, y],
    color = "black",
    zorder = 1
  )
_ = axs[1].set_xlim(0, 1)
axs[1].invert_xaxis()
axs[1].set_xticks(np.arange(0, 1.25, 0.25), labels = ["", "", "0.5", "", "1"])
axs[1].yaxis.tick_right()
_ = axs[0].set_xlim(0, 7)
_ = axs[0].set_xticks(np.arange(1, 8, 1))
_ = axs[0].set_yticks(
  np.arange(len(data)), labels = [d[0][0].upper()+"_Topic_"+str(d[1]) for d in data])
_ = axs[1].set_yticks(
  np.arange(len(data)), labels = ["" for _ in data])
_ = axs[0].set_xlabel("nr of patterns")
_ = axs[1].set_xlabel("auc ROC")
for ax in axs:
  ax.grid(True)
  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plots/auc_ROC_and_n_motifs.pdf")


for cell_type in cell_type_to_classification_result:
  pd.DataFrame(
    data = {topic: cell_type_to_classification_result[cell_type][0][topic].model.coef_.squeeze()
      for topic in cell_type_to_classification_result[cell_type][0]},
    index = cell_type_to_classification_result[cell_type][1]
  ).to_csv(f"{cell_type}_pattern_coef.tsv", sep = "\t", index = True)

```


```python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import crested
import random
from dataclasses import dataclass
import h5py
from typing import Self
import numpy as np
import pandas as pd

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
      region_names: list[str]
  ):
    self.contrib_scores               = p["contrib_scores"][        seqlet_idx  ]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][ seqlet_idx  ]
    self.ppm                          = p["sequence"][              seqlet_idx  ]
    self.start                        = p["start"][                 seqlet_idx  ]
    self.end                          = p["end"][                   seqlet_idx  ]
    self.is_revcomp                   = p["is_revcomp"][            seqlet_idx  ]
    region_idx                        = p["example_idx"][           seqlet_idx  ]
    self.region_name = region_names[region_idx]
    self.region_one_hot = ohs[region_idx]
    if (
        (not np.all(self.ppm == self.region_one_hot[self.start: self.end]) and not self.is_revcomp) or \
        (not np.all(self.ppm[::-1, ::-1] == self.region_one_hot[self.start: self.end]) and self.is_revcomp)
    ):
      raise ValueError(
        f"ppm does not match onehot\n" + \
        f"region_idx\t{region_idx}\n" + \
        f"start\t\t{self.start}\n" + \
        f"end\t\t{self.end}\n" + \
        f"is_revcomp\t{self.is_revcomp}\n" + \
        f"{self.ppm.argmax(1)}\n" + \
        f"{self.region_one_hot[self.start: self.end].argmax(1)}"
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
  def __init__(self, p: h5py._hl.group.Group, is_pos: bool, ohs: np.ndarray, region_names: list[str]):
    self.contrib_scores               = p["contrib_scores"][:]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][:]
    self.ppm                          = p["sequence"][:]
    self.is_pos                       = is_pos
    self.seqlets      = [Seqlet(p["seqlets"], i, ohs, region_names) for i in range(p["seqlets"]["n_seqlets"][0])]
    self.subpatterns  = [ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]
  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"
  def ic(self, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)
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
                  ModiscoPattern(f[pos_neg][pattern], pos_neg == "pos_patterns", ohs, region_names)
                )

def load_pattern_from_modisco_for_topics(
  topics:list[int],
  pattern_dir: str,
  prefix: str) -> tuple[list[ModiscoPattern], list[str]]:
  patterns = []
  pattern_names = []
  for topic in topics:
    with np.load(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz")) as gradients_data:
      ohs = gradients_data["oh"]
      region_names = gradients_data["region_names"]
    print(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz"))
    for name, pattern in load_pattern_from_modisco(
      filename=os.path.join(pattern_dir, f"patterns_Topic_{topic}.hdf5"),
      ohs=ohs,
      region_names=region_names
    ):
      patterns.append(pattern)
      pattern_names.append(prefix + name)
  return patterns, pattern_names


genome = crested.Genome(
  "../../../../../../resources/hg38/hg38.fa",
  "../../../../../../resources/hg38/hg38.chrom.sizes"
)

consensus_peaks_embryo = []
with open("../../data_prep_new/embryo_data/ATAC/embry_atac_region_names.txt", "rt") as f:
  for l in f:
    consensus_peaks_embryo.append(l.strip())

n_to_sample = 10_000

random.seed(123)
acgt_distribution = crested.utils._utils.calculate_nucleotide_distribution(
  input=random.sample(consensus_peaks_embryo, n_to_sample),
  genome=genome,
  per_position=True
)

# load human model 
path_model = "../../data_prep_new/organoid_data/MODELS/"
model_organoid = tf.keras.models.model_from_json(
    open(
        os.path.join(path_model, "model.json")
    ).read(),
    custom_objects = {'Functional':tf.keras.models.Model}
)
model_organoid.load_weights(
    os.path.join(path_model, "model_epoch_23.hdf5")
)

path_model = "../../data_prep_new/embryo_data/MODELS/"
model_embryo = tf.keras.models.model_from_json(
    open(
        os.path.join(path_model, "model.json")
    ).read(),
    custom_objects = {'Functional':tf.keras.models.Model}
)
model_embryo.load_weights(
    os.path.join(path_model, "model_epoch_36.hdf5")
)





@dataclass
class ModiscoClusteringResult:
  organoid_topics: list[int]
  embryo_topics: list[int]
  pattern_metadata_path: str
  cluster_col: str
  selected_patterns: list[float]


progenitor_dv_clustering_result = ModiscoClusteringResult(
  organoid_topics=[33, 38, 36, 54, 48],
  embryo_topics=[34, 38, 79, 88, 58],
  pattern_metadata_path="../../figure_3/draft/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns = [
    6, 7, 12, 15, 10, 18, 1, 19
  ]
)
progenitor_ap_clustering_result = ModiscoClusteringResult(
  organoid_topics=[],
  embryo_topics=[61, 59, 31, 62, 70, 52, 71],
  pattern_metadata_path="../../figure_4/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns=[1, 4, 8, 7, 6]
)
neural_crest_clustering_result = ModiscoClusteringResult(
  organoid_topics=[62, 60, 65, 59, 58],
  embryo_topics=[103, 105, 94, 91],
  pattern_metadata_path="../../figure_5/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2
  ]
)
neuron_clustering_result = ModiscoClusteringResult(
  organoid_topics=[6, 4, 23, 24, 13, 2],
  embryo_topics=[10, 8, 13, 24, 18, 29],
  pattern_metadata_path="../../figure_6/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0
  ]
)
cell_type_to_modisco_result = {
  "progenitor_dv": progenitor_dv_clustering_result,
  "progenitor_ap": progenitor_ap_clustering_result,
  "neural_crest": neural_crest_clustering_result,
  "neuron": neuron_clustering_result
}

def ic(ppm, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
  return (ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)

def ic_trim(ic, min_v: float) -> tuple[int, int]:
  if all(ic > min_v):
    return 0, len(ic)
  delta = np.where(np.diff((ic > min_v) * 1))[0]
  if ic[-1] > min_v:
    delta = np.array([*delta, len(ic) - 1])
  if ic[0] > min_v:
    delta = np.array([0, *delta])
  if len(delta) == 0:
    return 0, 0
  start_index = min(delta)
  end_index = max(delta)
  return start_index, end_index + 1

import logomaker
import matplotlib.pyplot as plt


def get_consensus(
  organoid_topics,
  embryo_topics,
  organoid_pattern_di,
  embryo_pattern_dir,
  pattern_metadata_path,
  cluster_col,
  selected_patterns
):
# load motifs
  patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
    topics=organoid_topics,
    pattern_dir=organoid_pattern_dir,
    prefix="organoid_")
  patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
    topics=embryo_topics,
    pattern_dir=embryo_pattern_dir,
    prefix="embryo_")
  all_patterns = [*patterns_organoid, *patterns_embryo]
  all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
  pattern_metadata=pd.read_table(pattern_metadata_path, index_col = 0)
  if "ic_start" not in pattern_metadata.columns:
    pattern_metadata["ic_start"] = 0
  if "ic_stop" not in pattern_metadata.columns:
    pattern_metadata["ic_stop"] = 30
  for cluster_id in selected_patterns:
    cluster_patterns = pattern_metadata.loc[pattern_metadata[cluster_col] == cluster_id].index.to_list()
    P = []
    for pattern_name in cluster_patterns:
      if pattern_name not in all_pattern_names:
        continue
      pattern = all_patterns[all_pattern_names.index(pattern_name)]
      ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
        pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
      ]
      pattern_ic = [s.ppm[ic_start: ic_end] for s in pattern.seqlets]
      if is_rc_to_root:
        pattern_ic = [s[::-1, ::-1] for s in pattern_ic]
      if offset_to_root > 0:
        pattern_ic = [np.concatenate(
          [
            np.zeros((offset_to_root, 4)),
            s
          ]
        )
        for s in pattern_ic
      ]
      elif offset_to_root < 0:
        pattern_ic = [s[abs(offset_to_root):, :] for s in pattern_ic]
      P.extend(pattern_ic)
    max_len = max([p.shape[0] for p in P])
    P = np.array([np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P])
    P += 1e-6
    P = (P.sum(0).T / P.sum(0).sum(1)).T
    consensus = "".join(["ACGT"[n] for n in P[range(*ic_trim(ic(P), 0.2))].argmax(1)])
    fig, axs = plt.subplots(figsize = (8, 2), ncols = 2)
    _ = logomaker.Logo(
      pd.DataFrame(
        (P * ic(P)[:, None])[range(*ic_trim(ic(P), 0.2))],
        columns = ["A", "C", "G", "T"]
      ),
      ax = axs[0]
    )
    _ = logomaker.Logo(
      pd.DataFrame(
        (P * ic(P)[:, None]),
        columns = ["A", "C", "G", "T"]
      ),
      ax = axs[1]
    )
    fig.tight_layout()
    fig.savefig(f"plots/{cell_type}_{cluster_id}_{consensus}_avg.pdf")
    plt.close(fig)
    yield (cluster_id, consensus)

cell_type_to_pattern_consensus = {}
for cell_type in cell_type_to_modisco_result:
  organoid_topics = cell_type_to_modisco_result[cell_type].organoid_topics
  embryo_topics = cell_type_to_modisco_result[cell_type].embryo_topics
  organoid_pattern_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"
  embryo_pattern_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"
  pattern_metadata_path = cell_type_to_modisco_result[cell_type].pattern_metadata_path
  cluster_col = cell_type_to_modisco_result[cell_type].cluster_col
  selected_patterns = cell_type_to_modisco_result[cell_type].selected_patterns
  cell_type_to_pattern_consensus[cell_type] = {}
  for cluster, consensus in get_consensus(
    organoid_topics,
    embryo_topics,
    organoid_pattern_dir,
    embryo_pattern_dir,
    pattern_metadata_path,
    cluster_col,
    selected_patterns
  ):
    cell_type_to_pattern_consensus[cell_type][cluster] = consensus

cell_type_organoid_to_coef = {
  cell_type: pd.read_table(f"organoid_{cell_type}_pattern_coef.tsv", index_col = 0)
  for cell_type in cell_type_to_modisco_result
  if os.path.exists(f"organoid_{cell_type}_pattern_coef.tsv")
}

cell_type_embryo_to_coef = {
  cell_type: pd.read_table(f"embryo_{cell_type}_pattern_coef.tsv", index_col = 0)
  for cell_type in cell_type_to_modisco_result
  if os.path.exists(f"embryo_{cell_type}_pattern_coef.tsv")
}

experimental_design_organoid = {}
for cell_type in cell_type_organoid_to_coef:
  for topic in cell_type_organoid_to_coef[cell_type].columns:
    pattern_names = cell_type_organoid_to_coef[cell_type].index[cell_type_organoid_to_coef[cell_type][topic] > 0]
    if len(pattern_names) > 2:
      patterns = {str(n): cell_type_to_pattern_consensus[cell_type][n] for n in pattern_names}
    elif len(pattern_names) == 2:
      patterns = {
        str(n) + f"_{i}": cell_type_to_pattern_consensus[cell_type][n]
        for i, n in zip([1, 1, 2, 2,], [*pattern_names, *pattern_names])
      }
    elif len(pattern_names) == 1:
      patterns = {
        str(n) + f"_{i}": cell_type_to_pattern_consensus[cell_type][n]
        for i, n in zip([1, 2, 3, 4], [*pattern_names, *pattern_names, *pattern_names, *pattern_names])
      }
    else:
      print("NO PATTERNS??")
    experimental_design_organoid[f"{cell_type}_{topic}"] = {
      "patterns": patterns,
      "target": int(topic) - 1
    }


experimental_design_embryo = {}
for cell_type in cell_type_embryo_to_coef:
  for topic in cell_type_embryo_to_coef[cell_type].columns:
    pattern_names = cell_type_embryo_to_coef[cell_type].index[cell_type_embryo_to_coef[cell_type][topic] > 0]
    if len(pattern_names) > 2:
      patterns = {str(n): cell_type_to_pattern_consensus[cell_type][n] for n in pattern_names}
    elif len(pattern_names) == 2:
      patterns = {
        str(n) + f"_{i}": cell_type_to_pattern_consensus[cell_type][n]
        for i, n in zip([1, 1, 2, 2,], [*pattern_names, *pattern_names])
      }
    elif len(pattern_names) == 1:
      patterns = {
        str(n) + f"_{i}": cell_type_to_pattern_consensus[cell_type][n]
        for i, n in zip([1, 2, 3, 4], [*pattern_names, *pattern_names, *pattern_names, *pattern_names])
      }
    else:
      print("NO PATTERNS??")
    experimental_design_embryo[f"{cell_type}_{topic}"] = {
      "patterns": patterns,
      "target": int(topic) - 1
    }

from tqdm import tqdm
import pickle

n_seq = 200
results_per_experiment_organoid = {}
for experiment in tqdm(experimental_design_organoid):
  print(experiment)
  print(experimental_design_organoid[experiment])
  results_per_experiment_organoid[experiment] = crested.tl.enhancer_design_motif_insertion(
      model=model_organoid,
      acgt_distribution=acgt_distribution,
      n_sequences=n_seq,
      target_len=500,
      return_intermediate=True,
      **experimental_design_organoid[experiment]
  )

for experiment in experimental_design_organoid:
  mn =  np.mean(
      [
        results_per_experiment_organoid[experiment][0][i]["predictions"][-1][experimental_design_organoid[experiment]["target"]]
        for i in range(n_seq)
    ]
  )
  print(f"{experiment}\t{mn}")


pickle.dump(
  results_per_experiment_organoid,
  open("motif_embedding_organoid.pkl", "wb")
)

results_per_experiment_embryo = {}
for experiment in tqdm(experimental_design_embryo):
  print(experiment)
  print(experimental_design_embryo[experiment])
  results_per_experiment_embryo[experiment] = crested.tl.enhancer_design_motif_insertion(
      model=model_embryo,
      acgt_distribution=acgt_distribution,
      n_sequences=n_seq,
      target_len=500,
      return_intermediate=True,
      **experimental_design_embryo[experiment]
  )

for experiment in experimental_design_embryo:
  mn =  np.mean(
      [
        results_per_experiment_embryo[experiment][0][i]["predictions"][-1][experimental_design_embryo[experiment]["target"]]
        for i in range(n_seq)
    ]
  )
  print(f"{experiment}\t{mn}")

pickle.dump(
  results_per_experiment_embryo,
  open("motif_embedding_embryo.pkl", "wb")
)

```


```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

results_per_experiment_organoid = pickle.load(
  open("motif_embedding_organoid.pkl", "rb")
)

results_per_experiment_embryo = pickle.load(
  open("motif_embedding_embryo.pkl", "rb")
)

def scale(X):
  return (X - X.min()) / (X.max() - X.min())

def get_pred_and_l2_for_cell_type(result, cell_type,):
  topic = int(cell_type.split("_")[-1]) - 1
  # create array with all zeros, except for topic which is one
  target = np.zeros(result[cell_type][0][0]["predictions"][-1].shape[0])
  target[topic] = 1
  l2s = np.array(
    [
      np.linalg.norm(target - scale(result[cell_type][0][seq]["predictions"][-1]))
      for seq in range(len(result[cell_type][0]))
    ]
  )
  pred = np.array(
    [
      result[cell_type][0][seq]["predictions"][-1][topic]
      for seq in range(len(result[cell_type][0]))
    ]
  )
  return pred, l2s


cell_type_to_pred_l2_organoid = {
  cell_type: get_pred_and_l2_for_cell_type(results_per_experiment_organoid, cell_type)
  for cell_type in results_per_experiment_organoid
}

cell_type_to_pred_l2_embryo = {
  cell_type: get_pred_and_l2_for_cell_type(results_per_experiment_embryo, cell_type)
  for cell_type in results_per_experiment_embryo
}

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_organoid[ct][0] for ct in cell_type_to_pred_l2_organoid],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_organoid])
fig.tight_layout()
fig.savefig("plots/pred_organoid_design.pdf")

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_embryo[ct][0] for ct in cell_type_to_pred_l2_embryo],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_embryo])
fig.tight_layout()
fig.savefig("plots/pred_embryo_design.pdf")

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_organoid[ct][1] for ct in cell_type_to_pred_l2_organoid],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_organoid])
fig.tight_layout()
fig.savefig("plots/l2_organoid_design.pdf")

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_embryo[ct][1] for ct in cell_type_to_pred_l2_embryo],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_embryo])
fig.tight_layout()
fig.savefig("plots/l2_embryo_design.pdf")

```


```python

from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import numpy as np
import pandas as pd
import torch
from functools import reduce
from dataclasses import dataclass
import h5py
from typing import Self
import os
import matplotlib.pyplot as plt
import pysam
from crested.utils._seq_utils import one_hot_encode_sequence

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
      region_names: list[str]
  ):
    self.contrib_scores               = p["contrib_scores"][        seqlet_idx  ]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][ seqlet_idx  ]
    self.ppm                          = p["sequence"][              seqlet_idx  ]
    self.start                        = p["start"][                 seqlet_idx  ]
    self.end                          = p["end"][                   seqlet_idx  ]
    self.is_revcomp                   = p["is_revcomp"][            seqlet_idx  ]
    region_idx                        = p["example_idx"][           seqlet_idx  ]
    self.region_name = region_names[region_idx]
    self.region_one_hot = ohs[region_idx]
    if (
        (not np.all(self.ppm == self.region_one_hot[self.start: self.end]) and not self.is_revcomp) or \
        (not np.all(self.ppm[::-1, ::-1] == self.region_one_hot[self.start: self.end]) and self.is_revcomp)
    ):
      raise ValueError(
        f"ppm does not match onehot\n" + \
        f"region_idx\t{region_idx}\n" + \
        f"start\t\t{self.start}\n" + \
        f"end\t\t{self.end}\n" + \
        f"is_revcomp\t{self.is_revcomp}\n" + \
        f"{self.ppm.argmax(1)}\n" + \
        f"{self.region_one_hot[self.start: self.end].argmax(1)}"
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
  def __init__(self, p: h5py._hl.group.Group, is_pos: bool, ohs: np.ndarray, region_names: list[str]):
    self.contrib_scores               = p["contrib_scores"][:]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][:]
    self.ppm                          = p["sequence"][:]
    self.is_pos                       = is_pos
    self.seqlets      = [Seqlet(p["seqlets"], i, ohs, region_names) for i in range(p["seqlets"]["n_seqlets"][0])]
    self.subpatterns  = [ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]
  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"
  def ic(self, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)
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
                  ModiscoPattern(f[pos_neg][pattern], pos_neg == "pos_patterns", ohs, region_names)
                )

def load_pattern_from_modisco_for_topics(
  topics:list[int],
  pattern_dir: str,
  prefix: str) -> tuple[list[ModiscoPattern], list[str]]:
  patterns = []
  pattern_names = []
  for topic in topics:
    with np.load(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz")) as gradients_data:
      ohs = gradients_data["oh"]
      region_names = gradients_data["region_names"]
    print(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz"))
    for name, pattern in load_pattern_from_modisco(
      filename=os.path.join(pattern_dir, f"patterns_Topic_{topic}.hdf5"),
      ohs=ohs,
      region_names=region_names
    ):
      patterns.append(pattern)
      pattern_names.append(prefix + name)
  return patterns, pattern_names

def get_hit_and_attribution(
    gradients_path: str,
    motifs: dict[str, np.ndarray],
    oh_key: str = "oh",
    attr_key: str = "gradients_integrated",
    region_names_key: str | None = "region_names",
) -> pd.DataFrame:
    with np.load(gradients_path) as gradients_data:
        ohs = gradients_data[oh_key]
        attr = gradients_data[attr_key]
        region_names = (
            gradients_data[region_names_key]
            if region_names_key is not None
            else np.arange(ohs.shape[0])
        )
    if not (ohs.shape[0] == attr.shape[0] == region_names.shape[0]):
        raise ValueError("Inconsistent shapes!")
    hits = pd.concat(fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2)))
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    return hits

def merge_and_max(
    left: pd.DataFrame, right: pd.DataFrame, on: list[str], max_on: str
) -> pd.DataFrame:
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1)

def get_non_overlapping_start_end_w_max_score(
    df: pd.DataFrame, max_overlap: int, score_col: str
) -> pd.DataFrame:
    df = df.sort_values("start")
    delta = np.diff(df["start"])
    delta_loc = [0, *np.where(delta > max_overlap)[0] + 1]
    groups = [
        slice(delta_loc[i], delta_loc[i + 1] if (i + 1) < len(delta_loc) else None)
        for i in range(len(delta_loc))
    ]
    return pd.DataFrame(
        [df.iloc[group].iloc[df.iloc[group][score_col].argmax()] for group in groups]
    ).reset_index(drop=True)


def get_hits_for_topics(
  organoid_topics: list[int],
  embryo_topics: list[int],
  selected_clusters: list[float],
  organoid_pattern_dir: str,
  embryo_pattern_dir: str,
  pattern_metadata_path: str,
  cluster_col: str,
  ic_trim_thr: float = 0.2
):
  # load motifs
  patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
    topics=organoid_topics,
    pattern_dir=organoid_pattern_dir,
    prefix="organoid_")
  patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
    topics=embryo_topics,
    pattern_dir=embryo_pattern_dir,
    prefix="embryo_")
  all_patterns = [*patterns_organoid, *patterns_embryo]
  all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
  pattern_metadata=pd.read_table(pattern_metadata_path, index_col = 0)
  motifs = {
    n: pattern.ppm[range(*pattern.ic_trim(ic_trim_thr))].T
    for n, pattern in zip(all_pattern_names, all_patterns)
    if n in pattern_metadata.index
  }
  hits_organoid = []
  region_order_organoid = []
  for topic in organoid_topics:
    hits=get_hit_and_attribution(
        gradients_path=os.path.join(organoid_pattern_dir, f"gradients_Topic_{topic}.npz" ),
        motifs=motifs,
    )
    hits["topic"] = topic
    hits["cluster"] = [
      pattern_metadata.loc[
        m, cluster_col
      ]
      for m in hits["motif_name"]
    ]
    hits = hits.query("cluster in @selected_clusters").reset_index(drop = True).copy()
    hits_organoid.append(hits)
    region_order_organoid.extend(hits["sequence_name"])
  hits_embryo = []
  region_order_embryo = []
  for topic in embryo_topics:
    hits=get_hit_and_attribution(
        gradients_path=os.path.join(embryo_pattern_dir, f"gradients_Topic_{topic}.npz" ),
        motifs=motifs,
    )
    hits["topic"] = topic
    hits["cluster"] = [
      pattern_metadata.loc[
        m, cluster_col
      ]
      for m in hits["motif_name"]
    ]
    hits = hits.query("cluster in @selected_clusters").reset_index(drop = True).copy()
    hits_embryo.append(hits)
    region_order_embryo.extend(hits["sequence_name"])
  if len(organoid_topics) > 0:
    hits_organoid_merged = reduce(
    lambda left, right:
      merge_and_max(
        left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
        right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
        on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value", "attribution", "topic"],
        max_on = "-logp",
      ),
      hits_organoid
    )
  else:
    hits_organoid_merged = None
  if len(embryo_topics) > 0:
    hits_embryo_merged = reduce(
    lambda left, right:
      merge_and_max(
        left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
        right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value", "-logp", "topic"]],
        on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value", "attribution", "topic"],
        max_on = "-logp",
      ),
      hits_embryo
    )
  else:
    hits_embryo_merged = None
  hits_organoid_non_overlap = hits_organoid_merged \
    .groupby("sequence_name") \
    .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp")) \
    .reset_index(drop = True) if len(organoid_topics) > 0 else None
  hits_embryo_non_overlap = hits_embryo_merged \
    .groupby("sequence_name") \
    .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp")) \
    .reset_index(drop = True) if len(embryo_topics) > 0 else None
  return (
    (hits_organoid_merged, hits_organoid_non_overlap, region_order_organoid),
    (hits_embryo_merged, hits_embryo_non_overlap, region_order_embryo)
  )

@dataclass
class ModiscoClusteringResult:
  organoid_topics: list[int]
  embryo_topics: list[int]
  pattern_metadata_path: str
  cluster_col: str
  selected_patterns: list[float]


progenitor_dv_clustering_result = ModiscoClusteringResult(
  organoid_topics=[33, 38, 36, 54, 48],
  embryo_topics=[34, 38, 79, 88, 58],
  pattern_metadata_path="../../figure_3/draft/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns = [
    6, 7, 12, 15, 10, 18, 1, 19
  ]
)

progenitor_ap_clustering_result = ModiscoClusteringResult(
  organoid_topics=[],
  embryo_topics=[61, 59, 31, 62, 70, 52, 71],
  pattern_metadata_path="../../figure_4/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns=[1, 4, 8, 7, 6]
)

neural_crest_clustering_result = ModiscoClusteringResult(
  organoid_topics=[62, 60, 65, 59, 58],
  embryo_topics=[103, 105, 94, 91],
  pattern_metadata_path="../../figure_5/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2
  ]
)

neuron_clustering_result = ModiscoClusteringResult(
  organoid_topics=[6, 4, 23, 24, 13, 2],
  embryo_topics=[10, 8, 13, 24, 18, 29],
  pattern_metadata_path="../../figure_6/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0
  ]
)

cell_type_to_modisco_result = {
  "progenitor_dv": progenitor_dv_clustering_result,
  "progenitor_ap": progenitor_ap_clustering_result,
  "neural_crest": neural_crest_clustering_result,
  "neuron": neuron_clustering_result
}

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"
embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

for cell_type in cell_type_to_modisco_result:
  print(cell_type)
  (
    (hits_organoid, hits_organoid_non_overlap, region_order_organoid),
    (hits_embryo, hits_embryo_non_overlap, region_order_embryo)
  ) = get_hits_for_topics(
      organoid_pattern_dir=organoid_dl_motif_dir ,
      embryo_pattern_dir=embryo_dl_motif_dir ,
      ic_trim_thr=0.2,
      organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
      embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
      selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
      pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
      cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
    )
  cell_type_to_modisco_result[cell_type].hits_organoid = hits_organoid
  cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap = hits_organoid_non_overlap
  cell_type_to_modisco_result[cell_type].region_order_organoid = region_order_organoid
  cell_type_to_modisco_result[cell_type].hits_embryo = hits_embryo
  cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap = hits_embryo_non_overlap
  cell_type_to_modisco_result[cell_type].region_order_embryo = region_order_embryo

def get_chrom_start_stop_topic_hit(r):
  for _ , h in r.iterrows():
    chrom, start, _ = h["sequence_name"].replace("-", ":").split(":")
    start = int(start)
    #end = int(end)
    start += h["start"]
    #end = start + h["end"]
    yield chrom, int(start), int(start), h["topic"]


import pyBigWig

# ttesting

bw_per_topic = {
  int(f.split(".")[0].replace("progenitor_Topic", "")) + 25: pyBigWig.open(
    f"../../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/{f}"
  )
  for f in os.listdir("../../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/")
  if f.startswith("progenitor_Topic")
}

topic_to_region_name_to_grad = {}
for f in os.listdir(organoid_dl_motif_dir):
  if not f.startswith("gradients_Topic"):
    continue
  print(f)
  with np.load(os.path.join(organoid_dl_motif_dir, f)) as gradients_data:
      oh = gradients_data["oh"]
      attr = gradients_data["gradients_integrated"]
      region_names = gradients_data["region_names"]
  topic_to_region_name_to_grad[int(f.split(".")[0].replace("gradients_Topic_", ""))] = dict(
    zip(region_names, attr.squeeze() * oh)
  )

D = cell_type_to_modisco_result["progenitor_dv"].hits_organoid_non_overlap.loc[
  cell_type_to_modisco_result["progenitor_dv"].hits_organoid_non_overlap.groupby("sequence_name")["attribution"].idxmax()
]
#D = D.query("cluster == 6")
S = 500
bin_size = 5
x = get_chrom_start_stop_topic_hit(
  D
)
test_cov = np.zeros(
  (
      len(D),
      S
  ),
  dtype = float
)
test_attr = np.full((len(D), S), np.nan)
for i, (chrom, start, end, topic) in tqdm(enumerate(x), total = test_cov.shape[0]):
  test_cov[i] = np.array(
        bw_per_topic[topic].values(
          chrom,
          start - S // 2,
          end + S // 2,
      ),
    dtype = float
  )
  mid = S // 2
  seq_name = D.iloc[i]["sequence_name"]
  start = mid - int(D.iloc[i]["start"])
  end = start + 500
  l_start = max(0, start)
  l_end = min(S, end)
  x_start = max(0, -start)
  x_end = x_start + (l_end - l_start)
  test_attr[i][l_start: l_end] = topic_to_region_name_to_grad[topic][seq_name][x_start:x_end, :].sum(1)

to_plot = np.nan_to_num(np.nanmean(np.lib.stride_tricks.sliding_window_view(test_cov, bin_size, axis = 1), axis = 2))
to_plot = (to_plot.T / to_plot.sum(1)).T

to_plot_attr = np.nan_to_num(np.nanmean(np.lib.stride_tricks.sliding_window_view(test_attr, bin_size, axis = 1), axis = 2))
to_plot_attr = (to_plot_attr.T / to_plot_attr.sum(1)).T

fig, ax = plt.subplots()
ax.plot(
  np.arange(to_plot.shape[1]),
  scale(to_plot.mean(0))
)
ax.plot(
  np.arange(to_plot.shape[1]),
  scale(to_plot_attr.mean(0))
)
fig.savefig("plots/test.pdf")

fig, ax = plt.subplots(figsize = (4, 10))
sns.heatmap(
  to_plot[D.reset_index(drop = True).sort_values(["cluster", "attribution"]).index][:, 200: 300],
  ax = ax,
  cmap = "gist_ncar",
  robust = True,
  vmin = 0.001, vmax = 0.004
)
#ax.set_xlim(150, 350)
fig.tight_layout()
fig.savefig("plots/test_hm.png", dpi = 500)


###

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

bw_per_topic_organoid = {
  topic_name_to_model_index_organoid(f.replace("Topic", "Topic_").split(".")[0]) + 1: pyBigWig.open(
    f"../../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/{f}"
  )
  for f in os.listdir("../../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/")
  if not f.startswith("all_Topic")
}

topic_to_region_name_to_grad_organoid = {}
for f in os.listdir(organoid_dl_motif_dir):
  if not f.startswith("gradients_Topic"):
    continue
  print(f)
  with np.load(os.path.join(organoid_dl_motif_dir, f)) as gradients_data:
      oh = gradients_data["oh"]
      attr = gradients_data["gradients_integrated"]
      region_names = gradients_data["region_names"]
  topic_to_region_name_to_grad_organoid[int(f.split(".")[0].replace("gradients_Topic_", ""))] = dict(
    zip(region_names, attr.squeeze() * oh)
  )

# total size
S = 500
bin_size = 5

hit_per_seq_organoid = []
for cell_type in cell_type_to_modisco_result:
  if cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap is None:
    continue
  tmp = cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.copy()
  tmp["cluster"] = cell_type + "_" + tmp["cluster"].astype(str)
  hit_per_seq_organoid.append(tmp)

hit_per_seq_organoid = pd.concat(hit_per_seq_organoid).reset_index(drop = True)

max_hit_per_seq_organoid = hit_per_seq_organoid.loc[
  hit_per_seq_organoid.groupby(["sequence_name", "topic"])["attribution"].idxmax()
]

min_hit_per_seq_organoid = hit_per_seq_organoid.loc[
  hit_per_seq_organoid.groupby(["sequence_name", "topic"])["attribution"].idxmin()
]

for pattern in max_hit_per_seq_organoid["cluster"].unique():
  D = max_hit_per_seq_organoid.query("cluster == @pattern")
  x = get_chrom_start_stop_topic_hit(
    D
  )
  test_cov = np.zeros(
    (
        len(D),
        S
    ),
    dtype = float
  )
  test_attr = np.full((len(D), S), np.nan)
  for i, (chrom, start, end, topic) in tqdm(enumerate(x), total = test_cov.shape[0]):
    test_cov[i] = np.array(
          bw_per_topic_organoid[topic].values(
            chrom,
            start - S // 2,
            end + S // 2,
        ),
      dtype = float
    )
    mid = S // 2
    seq_name = D.iloc[i]["sequence_name"]
    start = mid - int(D.iloc[i]["start"])
    end = start + 500
    l_start = max(0, start)
    l_end = min(S, end)
    x_start = max(0, -start)
    x_end = x_start + (l_end - l_start)
    test_attr[i][l_start: l_end] = topic_to_region_name_to_grad_organoid[topic][seq_name][x_start:x_end, :].sum(1)
  to_plot = np.nanmean(np.lib.stride_tricks.sliding_window_view(test_cov, bin_size, axis = 1), axis = 2)
  #to_plot = (to_plot.T / to_plot.sum(1)).T
  to_plot_attr = np.nan_to_num(np.nanmean(np.lib.stride_tricks.sliding_window_view(test_attr, bin_size, axis = 1), axis = 2))
  #to_plot_attr = (to_plot_attr.T / to_plot_attr.sum(1)).T
  fig, axs = plt.subplots(figsize = (4, 10), height_ratios = (2, 10), nrows = 2, sharex = True)
  axs[0].plot(
    np.arange(to_plot.shape[1]),
    scale(np.nanmean(to_plot, axis = 0)),
    color = "black",
    label = "cut sites",
    zorder = 2
  )
  axs[0].plot(
    np.arange(to_plot.shape[1]),
    scale(to_plot_attr.mean(0)),
    color = "red",
    label = "attribution",
    zorder = 1
  )
  axs[0].legend(loc = "upper left")
  sns.heatmap(
    np.nan_to_num(to_plot[np.argsort(np.nan_to_num(to_plot).mean(1))[::-1]]),
    ax = axs[1],
    cmap = "Spectral_r",
    robust = True,
    cbar = False,
    yticklabels = False
  )
  _ = axs[1].set_xticks(
      np.arange(0, S, 10),
  )
  axs[0].grid(True)
  axs[0].set_axisbelow(True)
  fig.tight_layout()
  fig.savefig(f"plots/footprint_{pattern}_organoid.png", dpi = 500)
  plt.close(fig)
  print(f"{pattern} done!")

for pattern in max_hit_per_seq_organoid["cluster"].unique():
  #for pattern in ["progenitor_dv_6", "neural_crest_13.2"]:
  fig, axs = plt.subplots(
    figsize = (4 * len(max_hit_per_seq_organoid["topic"].unique()), 4),
    ncols = len(max_hit_per_seq_organoid["topic"].unique()),
    sharex = True,
    sharey = True
  )
  for i, topic in enumerate(max_hit_per_seq_organoid["topic"].unique()):
    ax = axs[i]
    D_max = max_hit_per_seq_organoid.query("cluster == @pattern")
    x_max = get_chrom_start_stop_topic_hit(
      D_max
    )
    cov_max = np.zeros(
      (
          len(D_max),
          S
      ),
      dtype = float
    )
    D_min = min_hit_per_seq_organoid.query("cluster == @pattern")
    x_min = get_chrom_start_stop_topic_hit(
      D_min
    )
    cov_min = np.zeros(
      (
          len(D_min),
          S
      ),
      dtype = float
    )
    for i, (chrom, start, end, _) in tqdm(enumerate(x_max), total = cov_max.shape[0]):
      cov_max[i] = np.array(
            bw_per_topic_organoid[topic].values(
              chrom,
              start - S // 2,
              end + S // 2,
          ),
        dtype = float
      )
    for i, (chrom, start, end, _) in tqdm(enumerate(x_min), total = cov_min.shape[0]):
      cov_min[i] = np.array(
            bw_per_topic_organoid[topic].values(
              chrom,
              start - S // 2,
              end + S // 2,
          ),
        dtype = float
      )
    to_plot_max = np.nanmean(np.lib.stride_tricks.sliding_window_view(cov_max, bin_size, axis = 1), axis = 2)
    to_plot_min = np.nanmean(np.lib.stride_tricks.sliding_window_view(cov_min, bin_size, axis = 1), axis = 2)
    ax.plot(
      np.arange(to_plot_max.shape[1]),
      np.nan_to_num(to_plot_max).mean(0),
      color = "black",
      label = "max attr",
      zorder = 3
    )
    ax.plot(
      np.arange(to_plot_min.shape[1]),
      np.nan_to_num(to_plot_min).mean(0),
      color = "gray",
      label = "min attr",
      zorder = 2,
      ls = "dashed"
    )
    ax.grid(True)
    ax.set_axisbelow(True)
    _ = ax.set_xticks(
        np.arange(0, S, 10),
    )
    _ = ax.legend(loc = "upper left")
    _ = ax.set_title(topic)
  fig.tight_layout()
  fig.savefig(f"plots/footprint_{pattern}_organoid_all_topic.png", dpi = 500)
  plt.close(fig)
  print(f"{pattern} done")
 

 patterns_to_show = [
  "neuron_6.0",
  "neuron_3.1",
  "neuron_2.2",
  "neuron_2.1",
  "neuron_8.0",
  "neuron_5.2",
  "neuron_7.5",
  "progenitor_dv_18",
  "progenitor_dv_12",
  "progenitor_dv_1",
  "progenitor_dv_6",
  "progenitor_dv_10",
  "neural_crest_13.1",
  "neural_crest_9.1",
  "neural_crest_3.0",
  "neural_crest_2.2",
  "neural_crest_14.0",
  "neural_crest_11.1",
  "neural_crest_2.1",
  "neural_crest_13.2"
]

data = {}
for pattern in tqdm(patterns_to_show):
  data[pattern] = {}
  for i, topic in enumerate(max_hit_per_seq_organoid["topic"].unique()):
    ax = axs[i]
    D_max = max_hit_per_seq_organoid.query("cluster == @pattern")
    x_max = get_chrom_start_stop_topic_hit(
      D_max
    )
    cov_max = np.zeros(
      (
          len(D_max),
          S
      ),
      dtype = float
    )
    D_min = min_hit_per_seq_organoid.query("cluster == @pattern")
    x_min = get_chrom_start_stop_topic_hit(
      D_min
    )
    cov_min = np.zeros(
      (
          len(D_min),
          S
      ),
      dtype = float
    )
    for i, (chrom, start, end, _) in enumerate(x_max):
      cov_max[i] = np.array(
            bw_per_topic_organoid[topic].values(
              chrom,
              start - S // 2,
              end + S // 2,
          ),
        dtype = float
      )
    for i, (chrom, start, end, _) in enumerate(x_min):
      cov_min[i] = np.array(
            bw_per_topic_organoid[topic].values(
              chrom,
              start - S // 2,
              end + S // 2,
          ),
        dtype = float
      )
    to_plot_max = np.nanmean(np.lib.stride_tricks.sliding_window_view(cov_max, bin_size, axis = 1), axis = 2)
    to_plot_min = np.nanmean(np.lib.stride_tricks.sliding_window_view(cov_min, bin_size, axis = 1), axis = 2)
    data[pattern][topic] = (to_plot_max, to_plot_min)
 
topic_order = sorted(max_hit_per_seq_organoid["topic"].unique())


fig, axs = plt.subplots(
  figsize = (4 * len(topic_order), 2 * len(patterns_to_show)),
  ncols = len(topic_order),
  nrows = len(patterns_to_show),
  sharex = True,
  sharey = "col"
)
for i, pattern in tqdm(enumerate(patterns_to_show), total = len(patterns_to_show)):
  for j, topic in enumerate(topic_order):
    axs[i, j].plot(
      np.arange(data[pattern][topic][0].shape[1]),
      np.nan_to_num(data[pattern][topic][0]).mean(0),
      color = "black",
    )
    axs[i, j].plot(
      np.arange(data[pattern][topic][1].shape[1]),
      np.nan_to_num(data[pattern][topic][1]).mean(0),
      color = "gray",
    )
    if j == 0:
      axs[i, j].set_ylabel(pattern)
    if i == 0:
      axs[i, j].set_title(topic)
fig.tight_layout()
fig.savefig("plots/footpring_organoid_selected_patterns.pdf")

```


```python

import random
from dataclasses import dataclass
import h5py
from typing import Self
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt


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
      region_names: list[str]
  ):
    self.contrib_scores               = p["contrib_scores"][        seqlet_idx  ]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][ seqlet_idx  ]
    self.ppm                          = p["sequence"][              seqlet_idx  ]
    self.start                        = p["start"][                 seqlet_idx  ]
    self.end                          = p["end"][                   seqlet_idx  ]
    self.is_revcomp                   = p["is_revcomp"][            seqlet_idx  ]
    region_idx                        = p["example_idx"][           seqlet_idx  ]
    self.region_name = region_names[region_idx]
    self.region_one_hot = ohs[region_idx]
    if (
        (not np.all(self.ppm == self.region_one_hot[self.start: self.end]) and not self.is_revcomp) or \
        (not np.all(self.ppm[::-1, ::-1] == self.region_one_hot[self.start: self.end]) and self.is_revcomp)
    ):
      raise ValueError(
        f"ppm does not match onehot\n" + \
        f"region_idx\t{region_idx}\n" + \
        f"start\t\t{self.start}\n" + \
        f"end\t\t{self.end}\n" + \
        f"is_revcomp\t{self.is_revcomp}\n" + \
        f"{self.ppm.argmax(1)}\n" + \
        f"{self.region_one_hot[self.start: self.end].argmax(1)}"
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
  def __init__(self, p: h5py._hl.group.Group, is_pos: bool, ohs: np.ndarray, region_names: list[str]):
    self.contrib_scores               = p["contrib_scores"][:]
    self.hypothetical_contrib_scores  = p["hypothetical_contribs"][:]
    self.ppm                          = p["sequence"][:]
    self.is_pos                       = is_pos
    self.seqlets      = [Seqlet(p["seqlets"], i, ohs, region_names) for i in range(p["seqlets"]["n_seqlets"][0])]
    self.subpatterns  = [ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]
  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"
  def ic(self, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)
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
                  ModiscoPattern(f[pos_neg][pattern], pos_neg == "pos_patterns", ohs, region_names)
                )

def load_pattern_from_modisco_for_topics(
  topics:list[int],
  pattern_dir: str,
  prefix: str) -> tuple[list[ModiscoPattern], list[str]]:
  patterns = []
  pattern_names = []
  for topic in topics:
    with np.load(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz")) as gradients_data:
      ohs = gradients_data["oh"]
      region_names = gradients_data["region_names"]
    print(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz"))
    for name, pattern in load_pattern_from_modisco(
      filename=os.path.join(pattern_dir, f"patterns_Topic_{topic}.hdf5"),
      ohs=ohs,
      region_names=region_names
    ):
      patterns.append(pattern)
      pattern_names.append(prefix + name)
  return patterns, pattern_names


@dataclass
class ModiscoClusteringResult:
  organoid_topics: list[int]
  embryo_topics: list[int]
  pattern_metadata_path: str
  cluster_col: str
  selected_patterns: list[float]


progenitor_dv_clustering_result = ModiscoClusteringResult(
  organoid_topics=[33, 38, 36, 54, 48],
  embryo_topics=[34, 38, 79, 88, 58],
  pattern_metadata_path="../../figure_3/draft/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns = [
    6, 7, 12, 15, 10, 18, 1, 19
  ]
)
progenitor_ap_clustering_result = ModiscoClusteringResult(
  organoid_topics=[],
  embryo_topics=[61, 59, 31, 62, 70, 52, 71],
  pattern_metadata_path="../../figure_4/motif_metadata.tsv",
  cluster_col="hier_cluster",
  selected_patterns=[1, 4, 8, 7, 6]
)
neural_crest_clustering_result = ModiscoClusteringResult(
  organoid_topics=[62, 60, 65, 59, 58],
  embryo_topics=[103, 105, 94, 91],
  pattern_metadata_path="../../figure_5/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2
  ]
)
neuron_clustering_result = ModiscoClusteringResult(
  organoid_topics=[6, 4, 23, 24, 13, 2],
  embryo_topics=[10, 8, 13, 24, 18, 29],
  pattern_metadata_path="../../figure_6/draft/pattern_metadata.tsv",
  cluster_col="cluster_sub_cluster",
  selected_patterns = [
    1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0
  ]
)
cell_type_to_modisco_result = {
  "progenitor_dv": progenitor_dv_clustering_result,
  "progenitor_ap": progenitor_ap_clustering_result,
  "neural_crest": neural_crest_clustering_result,
  "neuron": neuron_clustering_result
}

def calc_ic(ppm, bg = np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
  return (ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)

def ic_trim(ic, min_v: float) -> tuple[int, int]:
  if all(ic > min_v):
    return 0, len(ic)
  delta = np.where(np.diff((ic > min_v) * 1))[0]
  if ic[-1] > min_v:
    delta = np.array([*delta, len(ic) - 1])
  if ic[0] > min_v:
    delta = np.array([0, *delta])
  if len(delta) == 0:
    return 0, 0
  start_index = min(delta)
  end_index = max(delta)
  return start_index, end_index + 1

def get_consensus(
  organoid_topics,
  embryo_topics,
  organoid_pattern_di,
  embryo_pattern_dir,
  pattern_metadata_path,
  cluster_col,
  selected_patterns
):
  # load motifs
  patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
    topics=organoid_topics,
    pattern_dir=organoid_pattern_dir,
    prefix="organoid_")
  patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
    topics=embryo_topics,
    pattern_dir=embryo_pattern_dir,
    prefix="embryo_")
  all_patterns = [*patterns_organoid, *patterns_embryo]
  all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
  pattern_metadata=pd.read_table(pattern_metadata_path, index_col = 0)
  if "ic_start" not in pattern_metadata.columns:
    pattern_metadata["ic_start"] = 0
  if "ic_stop" not in pattern_metadata.columns:
    pattern_metadata["ic_stop"] = 30
  for cluster_id in selected_patterns:
    cluster_patterns = pattern_metadata.loc[pattern_metadata[cluster_col] == cluster_id].index.to_list()
    P = []
    for pattern_name in cluster_patterns:
      if pattern_name not in all_pattern_names:
        continue
      pattern = all_patterns[all_pattern_names.index(pattern_name)]
      ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
        pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
      ]
      pattern_ic = [s.ppm[ic_start: ic_end] for s in pattern.seqlets]
      if is_rc_to_root:
        pattern_ic = [s[::-1, ::-1] for s in pattern_ic]
      if offset_to_root > 0:
        pattern_ic = [np.concatenate(
          [
            np.zeros((offset_to_root, 4)),
            s
          ]
        )
        for s in pattern_ic
      ]
      elif offset_to_root < 0:
        pattern_ic = [s[abs(offset_to_root):, :] for s in pattern_ic]
      P.extend(pattern_ic)
    max_len = max([p.shape[0] for p in P])
    P = np.array([np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P])
    P += 1e-6
    P = (P.sum(0).T / P.sum(0).sum(1)).T
    consensus = "".join(["ACGT"[n] for n in P[range(*ic_trim(calc_ic(P), 0.2))].argmax(1)])
    fig, axs = plt.subplots(figsize = (8, 2), ncols = 2)
    _ = logomaker.Logo(
      pd.DataFrame(
        (P * calc_ic(P)[:, None])[range(*ic_trim(calc_ic(P), 0.2))],
        columns = ["A", "C", "G", "T"]
      ),
      ax = axs[0]
    )
    _ = logomaker.Logo(
      pd.DataFrame(
        (P * calc_ic(P)[:, None]),
        columns = ["A", "C", "G", "T"]
      ),
      ax = axs[1]
    )
    fig.tight_layout()
    fig.savefig(f"plots/{cell_type}_{cluster_id}_{consensus}_avg.pdf")
    plt.close(fig)
    yield (cluster_id, P[range(*ic_trim(calc_ic(P), 0.2))])

cell_type_to_pattern_consensus = {}
for cell_type in cell_type_to_modisco_result:
  organoid_topics = cell_type_to_modisco_result[cell_type].organoid_topics
  embryo_topics = cell_type_to_modisco_result[cell_type].embryo_topics
  organoid_pattern_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"
  embryo_pattern_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"
  pattern_metadata_path = cell_type_to_modisco_result[cell_type].pattern_metadata_path
  cluster_col = cell_type_to_modisco_result[cell_type].cluster_col
  selected_patterns = cell_type_to_modisco_result[cell_type].selected_patterns
  cell_type_to_pattern_consensus[cell_type] = {}
  for cluster, consensus in get_consensus(
    organoid_topics,
    embryo_topics,
    organoid_pattern_dir,
    embryo_pattern_dir,
    pattern_metadata_path,
    cluster_col,
    selected_patterns
  ):
    cell_type_to_pattern_consensus[cell_type][cluster] = consensus

cell_type_organoid_to_coef = {
  cell_type: pd.read_table(f"organoid_{cell_type}_pattern_coef.tsv", index_col = 0)
  for cell_type in cell_type_to_modisco_result
  if os.path.exists(f"organoid_{cell_type}_pattern_coef.tsv")
}

cell_type_embryo_to_coef = {
  cell_type: pd.read_table(f"embryo_{cell_type}_pattern_coef.tsv", index_col = 0)
  for cell_type in cell_type_to_modisco_result
  if os.path.exists(f"embryo_{cell_type}_pattern_coef.tsv")
}

cell_type_to_pattern_consensus_filtered = {}
cell_type_to_pattern_consensus_index = {}
index_to_cell_type_pattern = {}
all_patterns = []
i = 0
for cell_type in cell_type_to_pattern_consensus:
  cell_type_to_pattern_consensus_filtered[cell_type] = {}
  cell_type_to_pattern_consensus_index[cell_type] = {}
  for pattern in cell_type_to_pattern_consensus[cell_type]:
    if (
        (
          cell_type in cell_type_organoid_to_coef and \
          cell_type_organoid_to_coef[cell_type].loc[pattern].max() > 0
      ) or \
        (
          cell_type in cell_type_embryo_to_coef and \
          cell_type_embryo_to_coef[cell_type].loc[pattern].max() > 0
      )
    ):
      cell_type_to_pattern_consensus_filtered[cell_type][pattern] = cell_type_to_pattern_consensus[cell_type][pattern]
      all_patterns.append(cell_type_to_pattern_consensus[cell_type][pattern])
      cell_type_to_pattern_consensus_index[cell_type][pattern] = i
      index_to_cell_type_pattern[i] = (cell_type, pattern)
      i += 1

from tangermeme.tools.tomtom import tomtom
import torch
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import logomaker

pvals, scores, offsets, overlaps, strands = tomtom(
    [torch.from_numpy(p.T) for p in all_patterns], [torch.from_numpy(p.T) for p in all_patterns]
)

evals = pvals.numpy() * len(all_patterns)

dat = 1 - np.corrcoef(evals)

row_linkage = hierarchy.linkage(
    distance.pdist(dat), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(dat.T), method='average')

clusters = hierarchy.fcluster(row_linkage, t = 16, criterion = "maxclust")
fig = sns.clustermap(
  dat,
  vmin = 0, vmax = 0.4,
  cmap = "rainbow_r",
  row_colors = [plt.cm.tab20(x) for x in clusters],
  col_colors = [plt.cm.tab20(x) for x in clusters]
)
lut = {cl: plt.cm.tab20(cl) for cl in np.unique(clusters)}
handles = [Patch(facecolor=lut[name]) for name in lut]
plt.legend(handles, lut, title='cluster',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right',
           ncols = 2)
fig.savefig("plots/motif_distance_matrix.png")

for cluster in np.unique(clusters):
  print(cluster)
  patterns_of_cluster = np.where(clusters == cluster)[0]
  n_patterns = len(patterns_of_cluster)
  fig, axs = plt.subplots(figsize = (4, 2 * n_patterns), nrows = n_patterns)
  if n_patterns == 1:
    axs = [axs]
  for pattern_idx, ax in zip(patterns_of_cluster, axs):
    pattern = all_patterns[pattern_idx]
    name = " ".join([str(x) for x in index_to_cell_type_pattern[pattern_idx]])
    _ = logomaker.Logo(
      pd.DataFrame(
          (pattern * calc_ic(pattern)[:, None]),
          columns = ["A", "C", "G", "T"]
      ),
      ax = ax
    )
    _ = ax.set_ylim(0, 2)
    _ = ax.set_title(name)
  fig.tight_layout()
  fig.savefig(f"plots/patterns_cluster_{cluster}.pdf")
  plt.close(fig)

pattern_metadata = pd.DataFrame(
  index = [" ".join(map(str, index_to_cell_type_pattern[i])) for i in range(len(all_patterns))],
  data = {
    "cluster": clusters
  }
)

to_sub_cluster = {
  5: 2,
  4: 2,
  11: 2,
  13: 2,
  14: 2
}
pattern_metadata["sub_cluster"] = 0
for cluster, n_clusters in to_sub_cluster.items():
  patterns_of_cluster = np.where(clusters == cluster)[0]
  ppm_cluster = [
    pattern for (pattern, in_cluster) in zip(all_patterns, clusters == cluster)
    if in_cluster
  ]
  pvals_cluster, _, _, _, _ = tomtom(
    [torch.from_numpy(p.T) for p in ppm_cluster], [torch.from_numpy(p.T) for p in ppm_cluster]
  )
  evals = pvals_cluster.numpy() * len(ppm_cluster)
  dat = 1 - np.corrcoef(evals)
  row_linkage = hierarchy.linkage(
    distance.pdist(dat), method='average')
  col_linkage = hierarchy.linkage(
    distance.pdist(dat.T), method='average')
  subclusters = hierarchy.fcluster(row_linkage, t = n_clusters, criterion = "maxclust")
  fig = sns.clustermap(
    dat,
    vmin = 0, vmax = 0.4,
    cmap = "rainbow_r",
    row_colors = [plt.cm.tab20(x) for x in subclusters],
    col_colors = [plt.cm.tab20(x) for x in subclusters]
  )
  lut = {cl: plt.cm.tab20(cl) for cl in np.unique(subclusters)}
  handles = [Patch(facecolor=lut[name]) for name in lut]
  plt.legend(handles, lut, title='cluster',
            bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
  fig.savefig(f"plots/subclusters_{cluster}_distance_matrix.png")
  for subcluster in np.unique(subclusters):
    print(f"{cluster}.{subcluster}")
    patterns_of_subcluster = np.where(subclusters == subcluster)[0]
    n_patterns = len(patterns_of_subcluster)
    fig, axs = plt.subplots(figsize = (4, 2 * n_patterns), nrows = n_patterns)
    if n_patterns == 1:
      axs = [axs]
    for pattern_idx, ax in zip(patterns_of_subcluster, axs):
      pattern = all_patterns[patterns_of_cluster[pattern_idx]]
      name = " ".join(map(str, index_to_cell_type_pattern[patterns_of_cluster[pattern_idx]]))
      pattern_metadata.loc[name, "sub_cluster"] = int(subcluster)
      _ = logomaker.Logo(
        pd.DataFrame(
            (pattern * calc_ic(pattern)[:, None]),
            columns = ["A", "C", "G", "T"]
        ),
        ax = ax
      )
      _ = ax.set_ylim(0, 2)
      _ = ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"plots/patterns_cluster_{cluster}.{subcluster}.pdf")
    plt.close(fig)

pattern_metadata["cluster_sub_cluster"] = pattern_metadata["cluster"].astype(str) \
  + "." + pattern_metadata["sub_cluster"].astype(str)
pattern_metadata["is_root_motif"] = False
pattern_metadata["is_rc_to_root"] = False
pattern_metadata["offset_to_root"] = 0

all_patterns_names = [" ".join(map(str, index_to_cell_type_pattern[i])) for i in index_to_cell_type_pattern]

cluster_to_avg_pattern = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
  cluster_idc = np.where(pattern_metadata["cluster_sub_cluster"] == cluster)[0]
  n_clusters = len(cluster_idc)
  print(f"cluster: {cluster}")
  _, _, o, _, s = tomtom(
      [torch.from_numpy(all_patterns[m].T) for m in cluster_idc],
      [torch.from_numpy(all_patterns[m].T) for m in cluster_idc]
  )
  fig, axs = plt.subplots(nrows = n_clusters, figsize = (4, 2 * n_clusters), sharex = True)
  pwms_aligned = []
  if n_clusters == 1:
    axs = [axs]
  for i, m in enumerate(cluster_idc):
    print(i)
    rel = np.argmax([np.mean(calc_ic(all_patterns[m])) for m in cluster_idc])
    pattern_metadata.loc[all_patterns_names[cluster_idc[rel]], "is_root_motif"] = True
    is_rc = s[i, rel] == 1
    offset = int(o[i, rel].numpy())
    pwm = all_patterns[m]
    pwm_rel = all_patterns[cluster_idc[rel]]
    ic = calc_ic(all_patterns[m])[:, None]
    if is_rc:
      pwm = pwm[::-1, ::-1]
      ic = ic[::-1, ::-1]
      offset = pwm_rel.shape[0] - pwm.shape[0] - offset
    pattern_metadata.loc[all_patterns_names[m], "is_rc_to_root"] = bool(is_rc)
    pattern_metadata.loc[all_patterns_names[m], "offset_to_root"] = offset
    if offset > 0:
      pwm = np.concatenate([np.zeros((offset, 4)), pwm])
      ic = np.concatenate([np.zeros((offset, 1)), ic])
    elif offset < 0:
      pwm = pwm[abs(offset):, :]
      ic = ic[abs(offset):, :]
    pwms_aligned.append(pwm)
    _ = logomaker.Logo(
          pd.DataFrame(pwm * ic,
                      columns=["A", "C", "G", "T"]
                      ),
      ax=axs[i]
    )
    _ = axs[i].set_title(f"{offset}, {is_rc}")
  fig.tight_layout()
  fig.savefig(f"plots/cluster_{cluster}_patterns_aligned.pdf")
  max_len = max([x.shape[0] for x in pwms_aligned])
  pwm_avg = np.array(
    [
      np.concatenate([p, np.zeros((max_len - p.shape[0], 4))])
      for p in pwms_aligned
    ]
  ).mean(0)
  ic = calc_ic(pwm_avg)
  fig, ax = plt.subplots(figsize = (4,2))
  _ = logomaker.Logo(
    pd.DataFrame(pwm_avg * ic[:, None], columns = ["A", "C", "G", "T"]),
    ax = ax
  )
  fig.tight_layout()
  fig.savefig(f"plots/cluster_{cluster}_avg_patterns_aligned.pdf")
  cluster_to_avg_pattern[cluster] = pwm_avg

pattern_metadata.to_csv("pattern_metadata.tsv", sep = "\t", header = True, index = True)

pattern_code = pd.DataFrame(
  index = pattern_metadata["cluster_sub_cluster"].unique(),
  columns = [
    *["O_" + topic for cell_type in cell_type_organoid_to_coef for topic in cell_type_organoid_to_coef[cell_type]],
    *["E_" + topic for cell_type in cell_type_embryo_to_coef for topic in cell_type_embryo_to_coef[cell_type]]
  ]
).fillna(0)

for cell_type in cell_type_organoid_to_coef:
  for topic in cell_type_organoid_to_coef[cell_type].columns:
    patterns = cell_type_organoid_to_coef[cell_type].loc[ cell_type_organoid_to_coef[cell_type][topic] > 0].index.to_list()
    if len(patterns) == 2:
      patterns = [*patterns, *patterns]
    elif len(patterns) == 1:
      patterns = [*patterns, *patterns, *patterns, *patterns]
    for pattern in patterns:
      pattern_code.loc[
          pattern_metadata.loc[
            all_patterns_names[cell_type_to_pattern_consensus_index[cell_type][pattern]], "cluster_sub_cluster"
          ],
        "O_" + topic
    ] += 1

for cell_type in cell_type_embryo_to_coef:
  for topic in cell_type_embryo_to_coef[cell_type].columns:
    patterns = cell_type_embryo_to_coef[cell_type].loc[ cell_type_embryo_to_coef[cell_type][topic] > 0].index.to_list()
    if len(patterns) == 2:
      patterns = [*patterns, *patterns]
    elif len(patterns) == 1:
      patterns = [*patterns, *patterns, *patterns, *patterns]
    for pattern in patterns:
      pattern_code.loc[
          pattern_metadata.loc[
            all_patterns_names[cell_type_to_pattern_consensus_index[cell_type][pattern]], "cluster_sub_cluster"
          ],
        "E_" + topic
    ] += 1

topic_order = [
    # DV
    "O_33",
    "E_34",
    "O_38",
    "E_38",
    "O_36",
    "E_79",
    "O_54",
    "E_88",
    "O_48",
    "E_58",
    # AP
    "E_61",
    "E_59",
    "E_31",
    "E_62",
    "E_70",
    "E_52",
    "E_71",
    # NC
    "O_62",
    "E_103",
    "O_60",
    "E_105",
    "O_65",
    "E_94",
    "O_59",
    "O_58",
    "E_91",
    # NEU
    "O_6",
    "E_10",
    "O_4",
    "E_8",
    "O_23",
    "E_13",
    "O_24",
    "E_24",
    "O_13",
    "E_18",
    "O_2", 
    "E_29",
]

sorted_patterns = pattern_code.T.idxmax().sort_values(key = lambda X: [topic_order.index(x) for x in X]).index[::-1]

import pickle

results_per_experiment_organoid = pickle.load(
  open("motif_embedding_organoid.pkl", "rb")
)

results_per_experiment_embryo = pickle.load(
  open("motif_embedding_embryo.pkl", "rb")
)

def scale(X):
  return (X - X.min()) / (X.max() - X.min())

def get_pred_and_l2_for_cell_type(result, cell_type,):
  topic = int(cell_type.split("_")[-1]) - 1
  # create array with all zeros, except for topic which is one
  target = np.zeros(result[cell_type][0][0]["predictions"][-1].shape[0])
  target[topic] = 1
  l2s = np.array(
    [
      np.linalg.norm(target - scale(result[cell_type][0][seq]["predictions"][-1]))
      for seq in range(len(result[cell_type][0]))
    ]
  )
  pred = np.array(
    [
      result[cell_type][0][seq]["predictions"][-1][topic]
      for seq in range(len(result[cell_type][0]))
    ]
  )
  return pred, l2s

cell_type_to_pred_l2_organoid = {
  "O_" + cell_type.split("_")[-1]: get_pred_and_l2_for_cell_type(results_per_experiment_organoid, cell_type)
  for cell_type in results_per_experiment_organoid
}

cell_type_to_pred_l2_embryo = {
  "E_" + cell_type.split("_")[-1]: get_pred_and_l2_for_cell_type(results_per_experiment_embryo, cell_type)
  for cell_type in results_per_experiment_embryo
}

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_organoid[ct][0] for ct in cell_type_to_pred_l2_organoid],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_organoid])
fig.tight_layout()
fig.savefig("plots/pred_organoid_design.pdf")

fig, ax = plt.subplots(figsize = (4, 8))
_ = ax.boxplot(
  [cell_type_to_pred_l2_embryo[ct][0] for ct in cell_type_to_pred_l2_embryo],
  vert = False
)
_ = ax.set_yticklabels([ct for ct in cell_type_to_pred_l2_embryo])
fig.tight_layout()
fig.savefig("plots/pred_embryo_design.pdf")

color_dict = {
  "progenitor_dv": "#0066ff",
  "progenitor_ap": "#002255",
  "neuron": "#cc9900",
  "neural_crest": "#7E52A0"
}

topic_to_ct = {
    # DV
    "O_33": "progenitor_dv",
    "E_34": "progenitor_dv",
    "O_38": "progenitor_dv",
    "E_38": "progenitor_dv",
    "O_36": "progenitor_dv",
    "E_79": "progenitor_dv",
    "O_54": "progenitor_dv",
    "E_88": "progenitor_dv",
    "O_48": "progenitor_dv",
    "E_58": "progenitor_dv",
    # AP
    "E_61": "progenitor_ap",
    "E_59": "progenitor_ap",
    "E_31": "progenitor_ap",
    "E_62": "progenitor_ap",
    "E_70": "progenitor_ap",
    "E_52": "progenitor_ap",
    "E_71": "progenitor_ap",
    # NC
    "O_62": "neural_crest",
    "E_103": "neural_crest",
    "O_60": "neural_crest",
    "E_105": "neural_crest",
    "O_65": "neural_crest",
    "E_94": "neural_crest",
    "O_59": "neural_crest",
    "O_58": "neural_crest",
    "E_91": "neural_crest",
    # NEU
    "O_6": "neuron",
    "E_10": "neuron",
    "O_4": "neuron",
    "E_8": "neuron",
    "O_23": "neuron",
    "E_13": "neuron",
    "O_24": "neuron",
    "E_24": "neuron",
    "O_13": "neuron",
    "E_18": "neuron",
    "O_2": "neuron",
    "E_29": "neuron"
}


from matplotlib.lines import Line2D

fig = plt.figure(figsize = (10, 12))
gs = fig.add_gridspec(len(sorted_patterns) * 2 + 6, 15)
for i, pattern_name in enumerate(sorted_patterns[::-1]):
  ax = fig.add_subplot(gs[i * 2: i * 2 + 2, 0:2])
  pattern = cluster_to_avg_pattern[pattern_name]
  _ = logomaker.Logo(
    pd.DataFrame(
      pattern * calc_ic(pattern)[:, None],
      columns = ["A", "C", "G", "T"],
    ),
    ax = ax
  )
  ax.set_axis_off()
ax = fig.add_subplot(gs[0: len(sorted_patterns) * 2, 2:15])
XX, YY = np.meshgrid(
  np.arange(pattern_code.shape[1]),
  np.arange(pattern_code.shape[0])
)
for j in range(pattern_code.shape[1]):
  idx = np.where(pattern_code.loc[sorted_patterns, topic_order[j]].values)[0]
  _ = ax.scatter(
    XX[idx, j], YY[idx, j],
    s = pattern_code.loc[sorted_patterns, topic_order[j]].values[idx] * 20,
    color = color_dict[topic_to_ct[topic_order[j]]],
    zorder = 3,
    edgecolor = "black",
    lw = 1
  )
  _ = ax.plot(
    [
      XX[idx[YY[idx, j].argmin()], j],
      XX[idx[YY[idx, j].argmax()], j]
    ],
    [
      YY[idx, j].min(),
      YY[idx, j].max()
    ],
    color = color_dict[topic_to_ct[topic_order[j]]],
    zorder =2
  )
ax.legend(
  handles = [
    Line2D([0], [0], color = color_dict[ct], markerfacecolor = "o", label = ct, markersize = 10)
    for ct in color_dict
  ],
  loc = "lower left"
)
_ = ax.set_xlim(-0.5, XX.max() + 0.5)
_ = ax.set_xticks(
  np.arange(pattern_code.shape[1]),
  labels = ["" for _ in range(pattern_code.shape[1])],
  rotation = 90
)
_ = ax.set_yticks(
  np.arange(pattern_code.shape[0]),
  labels = sorted_patterns
)
ax.grid(True)
ax.set_axisbelow(True)
ax = fig.add_subplot(gs[len(sorted_patterns) * 2 + 1:, 2:15])
_ = ax.boxplot(
  [{**cell_type_to_pred_l2_organoid, **cell_type_to_pred_l2_embryo}[t][0] for t in topic_order]
)
_ = ax.set_xticks(
  np.arange(pattern_code.shape[1]) + 1,
  labels = topic_order,
  rotation = 90
)
ax.grid(True)
ax.set_axisbelow(True)
ax.set_ylabel("pred. score")
fig.tight_layout()
fig.savefig("plots/pattern_code.pdf")


organoid_topics = [int(x.split("_")[1]) for x in  topic_order if x.startswith("O_")]

confusion_organoid = np.array(
  [
    np.mean(
      [results_per_experiment_organoid[cell_type][0][seq]["predictions"][-1][[t - 1 for t in organoid_topics]] for seq in range(200)],
      axis = 0
    )
    for cell_type in results_per_experiment_organoid
  ]
)

df_confusion_organoid = pd.DataFrame(
  confusion_organoid,
  index = [int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_organoid],
  columns = organoid_topics
).loc[organoid_topics]

fig, ax = plt.subplots()
sns.heatmap(
  df_confusion_organoid,
  xticklabels = True,
  yticklabels = True,
  robust = True,
  ax = ax,
  vmin = 0, vmax = 1
)
ax.set_ylabel("Designed for ...")
ax.set_xlabel("Predicted to be ...")
fig.tight_layout()
fig.savefig("plots/confusion_organoid.pdf")


embryo_topics = [int(x.split("_")[1]) for x in  topic_order if x.startswith("E_")]

confusion_embryo = np.array(
  [
    np.mean(
      [results_per_experiment_embryo[cell_type][0][seq]["predictions"][-1][[t - 1 for t in embryo_topics]] for seq in range(200)],
      axis = 0
    )
    for cell_type in results_per_experiment_embryo
  ]
)

df_confusion_embryo = pd.DataFrame(
  confusion_embryo,
  index = [int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_embryo],
  columns = embryo_topics
).loc[embryo_topics]

fig, ax = plt.subplots()
sns.heatmap(
  df_confusion_embryo,
  xticklabels = True,
  yticklabels = True,
  robust = True,
  ax = ax,
  vmin = 0, vmax = 1
)
ax.set_ylabel("Designed for ...")
ax.set_xlabel("Predicted to be ...")
fig.tight_layout()
fig.savefig("plots/confusion_embryo.pdf")

import tensorflow as tf
from crested.utils._seq_utils import one_hot_encode_sequence

zebrafish_model = tf.keras.models.load_model(
  "../../../../../sun_et_al_zebrafish_emb_scatac/crested/danRer_development_500bp_rescaled/2024-08-30_10:46/checkpoints/30.keras"
)

cell_type_to_zebrafish_organoid = {
  cell_type: zebrafish_model.predict(
      np.array(
        [
          one_hot_encode_sequence(s, expand_dim = False)
          for s in results_per_experiment_organoid[cell_type][1]
        ],
      dtype = int)
  )
  for cell_type in results_per_experiment_organoid
}

cell_type_to_zebrafish_embryo = {
  cell_type: zebrafish_model.predict(
      np.array(
        [
          one_hot_encode_sequence(s, expand_dim = False)
          for s in results_per_experiment_embryo[cell_type][1]
        ],
      dtype = int)
  )
  for cell_type in results_per_experiment_embryo
}

import scanpy as sc
zebrafish_adata = sc.read_h5ad(
  "../../../../CREsted_2025/figure_5/data_prep/data/zebrafish_anndata.h5ad"
)

stage = "hpf72:"

zebrafish_organoid = np.array(
  [
    np.mean(
      cell_type_to_zebrafish_organoid[cell_type][:, [i for i, x in enumerate( zebrafish_adata.obs_names) if x.startswith(stage)]],
      axis = 0
    )
    for cell_type in results_per_experiment_organoid
  ]
)

df_zebrafish_organoid = pd.DataFrame(
  zebrafish_organoid,
  index = [int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_organoid],
  columns = zebrafish_adata.obs_names[[i for i, x in enumerate( zebrafish_adata.obs_names) if x.startswith(stage)]]
).loc[organoid_topics]

zebrafish_embryo = np.array(
  [
    np.mean(
      cell_type_to_zebrafish_embryo[cell_type][:, [i for i, x in enumerate( zebrafish_adata.obs_names) if x.startswith(stage)]],
      axis = 0
    )
    for cell_type in results_per_experiment_embryo
  ]
)

df_zebrafish_embryo = pd.DataFrame(
  zebrafish_embryo,
  index = [int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_embryo],
  columns = zebrafish_adata.obs_names[[i for i, x in enumerate( zebrafish_adata.obs_names) if x.startswith(stage)]]
).loc[embryo_topics]


sorted_ct = df_zebrafish_organoid.idxmax().sort_values(key = lambda X: [organoid_topics.index(x) for x in X]).index
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(
  df_zebrafish_organoid[sorted_ct].T,
  ax = ax,
  robust = True,
  xticklabels = True,
  yticklabels = True,
  vmin = 3, vmax = 8,
  cbar_kws = {"shrink": 0.1}
)
fig.tight_layout()
fig.savefig("plots/organoids_zebrafish.pdf")


sorted_ct = df_zebrafish_embryo.idxmax().sort_values(key = lambda X: [embryo_topics.index(x) for x in X]).index
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(
  df_zebrafish_embryo[sorted_ct].T,
  ax = ax,
  robust = True,
  xticklabels = True,
  yticklabels = True,
  vmin = 3, vmax = 8,
  square = True,
  cbar_kws = {"shrink": 0.1}
)
fig.tight_layout()
fig.savefig("plots/embryos_zebrafish.pdf")





```


