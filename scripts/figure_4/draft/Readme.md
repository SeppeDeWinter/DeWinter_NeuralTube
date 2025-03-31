```python

import numpy as np
from dataclasses import dataclass
from typing import Self
import h5py
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import torch
from tangermeme.tools.tomtom import tomtom
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib.patches import Patch
from tangermeme.tools.fimo import fimo
import modiscolite
import tensorflow as tf

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

def get_value_seqlets(seqlets: list[Seqlet], v: np.ndarray):
  if v.shape[0] != len(seqlets):
    raise ValueError(f"{v.shape[0]} != {len(seqlets)}")
  for i, seqlet in enumerate(seqlets):
    if seqlet.is_revcomp:
      yield v[i, seqlet.start: seqlet.end, :][::-1, ::-1]
    else:
      yield v[i, seqlet.start: seqlet.end, :]

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


ap_topics = np.array(
  [31, 29, 1, 32, 40, 22, 41]
) + 30

dv_topics = np.array(
  [
    4,
    44,
    57,
    8,
    49,
    21,
    58,
    28
  ]
) + 30

embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(ap_topics):
    ohs = np.load(os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["oh"]
    region_names = np.load(os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            embryo_dl_motif_dir, f"patterns_Topic_{topic}.hdf5",
        ),
        ohs = ohs,
        region_names = region_names
    ):
        pattern_names_dl_embryo.append("embryo_" + name)
        patterns_dl_embryo.append(pattern)

pattern_topic = np.array([x.split("_")[3] for x in pattern_names_dl_embryo])

for topic in ap_topics:
  print(topic)
  patterns_of_topic = np.where(pattern_topic == str(topic))[0]
  n_patterns = len(patterns_of_topic)
  fig, axs = plt.subplots(figsize = (4, 2 * n_patterns), nrows = n_patterns)
  for pattern_idx, ax in zip(patterns_of_topic, axs.ravel()):
    pattern = patterns_dl_embryo[pattern_idx]
    _ = logomaker.Logo(
      pd.DataFrame(
          (pattern.ppm * pattern.ic()[:, None])[range(*pattern.ic_trim(0.2))],
          columns = ["A", "C", "G", "T"]
      ),
      ax = ax
    )
    avg_ic = np.round(pattern.ic()[range(*pattern.ic_trim(0.2))].mean(), 3)
    _ = ax.set_ylim(0, 2)
    _ = ax.set_title( ("pos pattern " if pattern.is_pos else "neg pattern ") + str(avg_ic))
  fig.tight_layout()
  fig.savefig(f"plots/patterns_per_topic_{topic}.pdf")
  plt.close(fig)

patterns_to_cluster = []
patterns_to_cluster_names = []
avg_ic_thr = 0.6

for pattern, pattern_name in zip(patterns_dl_embryo, pattern_names_dl_embryo):
  if not pattern.is_pos:
    continue
  avg_ic = pattern.ic()[range(*pattern.ic_trim(0.2))].mean()
  if avg_ic > avg_ic_thr:
    patterns_to_cluster.append(pattern)
    patterns_to_cluster_names.append(pattern_name)

ppm_patterns_to_cluster = [
  torch.from_numpy(pattern.ppm[range(*pattern.ic_trim(0.2))]).T
  for pattern in patterns_to_cluster
]

pvals, scores, offsets, overlaps, strands = tomtom(
    ppm_patterns_to_cluster, ppm_patterns_to_cluster
)

evals = pvals.numpy() * len(patterns_to_cluster)

dat = 1 - np.corrcoef(evals)

row_linkage = hierarchy.linkage(
    distance.pdist(dat), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(dat.T), method='average')

clusters = hierarchy.fcluster(row_linkage, t = 10, criterion = "maxclust")
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
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
fig.savefig("plots/motif_distance_matrix.png")

for cluster in np.unique(clusters):
  print(cluster)
  patterns_of_cluster = np.where(clusters == cluster)[0]
  n_patterns = len(patterns_of_cluster)
  fig, axs = plt.subplots(figsize = (4, 2 * n_patterns), nrows = n_patterns)
  if n_patterns == 1:
    axs = [axs]
  for pattern_idx, ax in zip(patterns_of_cluster, axs):
    pattern = patterns_to_cluster[pattern_idx]
    _ = logomaker.Logo(
      pd.DataFrame(
          (pattern.ppm * pattern.ic()[:, None])[range(*pattern.ic_trim(0.2))],
          columns = ["A", "C", "G", "T"]
      ),
      ax = ax
    )
    _ = ax.set_ylim(0, 2)
  fig.tight_layout()
  fig.savefig(f"plots/patterns_cluster_{cluster}.pdf")
  plt.close(fig)

region_topic = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv",
  index_col = 0
)

selected_regions = set()
top_n = 1_000
for topic in [*ap_topics, *dv_topics]:
  print(topic)
  topic_name = model_index_to_topic_name_embryo(topic - 1).replace("progenitor_", "")
  selected_regions = selected_regions | set(region_topic[topic_name].sort_values(ascending = False).head(top_n).index)

ap_dv_topics_names = [
  model_index_to_topic_name_embryo(topic - 1).replace("progenitor_", "")
  for topic in [*ap_topics, *dv_topics]
]

dat = region_topic.loc[list(selected_regions), ap_dv_topics_names].corr()

fig = sns.clustermap(
  dat,
  annot = np.round(dat, 2).mask(dat < 0.15).fillna(""),
  fmt = "",
  cmap = "Spectral_r",
  vmin = 0, vmax = 0.6,
  row_cluster = False,
  col_cluster = False,
  linewidths = 1,
  linecolor = "black",
  col_colors = [*np.repeat("blue", len(ap_topics)), *np.repeat("red", len(dv_topics))]
)
fig.savefig("plots/clustermap_ap_dv_topics.png")

!wget https://hugheslab.ccbr.utoronto.ca/supplementary-data/homeodomains1/pwm_all_102107.txt

pwms = {}
name = None
pwm = None
with open("pwm_all_102107.txt", "rt") as f:
  for line in f:
    line = line.strip()
    if len(line) == 0:
      continue
    if not "\t" in line:
      if name is not None:
        pwms[name] = np.array(pwm).T
      name = line
      pwm = []
    else:
      pwm.append(list(map(float, line.split()[1:])))

from matplotlib.backends.backend_pdf import PdfPages

bg = np.array([0.27, 0.23, 0.23, 0.27])
eps=1e-3
with PdfPages("plots/pwms_homeboxs.pdf") as pdf:
  for tf in pwms:
    fig, ax = plt.subplots(figsize = (4, 2))
    print(tf)
    ppm = pwms[tf]
    ic = (ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)
    _ = logomaker.Logo(
      pd.DataFrame(
        ppm * ic[:, None],
        columns = ["A", "C", "G", "T"]
      ),
      ax = ax
    )
    _ = ax.set_title(tf)
    pdf.savefig(fig)
    plt.close(fig)

#

selected_regions = list(
    set(region_topic["Topic_1"].sort_values(ascending = False).head(3_000).index) \
  | set(region_topic["Topic_31"].sort_values(ascending = False).head(3_000).index)
)

import pysam
hg38 = pysam.FastaFile("/data/projects/c20/sdewin/resources/hg38/hg38.fa")

from crested.utils._seq_utils import one_hot_encode_sequence

ohs = np.array(
  [
    one_hot_encode_sequence(
      hg38.fetch( r.split(":")[0], *map(int, r.split(":")[1].split("-")) ),
      expand_dim = False
    )
    for r in selected_regions
  ]
)

motifs = {
  **{
    f"cluster_4_pattern_{pattern_idx}": patterns_to_cluster[pattern_idx].ppm[
      range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))
    ].T
    for pattern_idx in np.where(clusters == 4)[0]
  },
  **{
    f"cluster_8_pattern_{pattern_idx}": patterns_to_cluster[pattern_idx].ppm[
      range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))
    ].T
    for pattern_idx in np.where(clusters == 8)[0]
  }
}

hits = pd.concat(
  fimo(
    motifs=motifs,
    sequences=ohs.swapaxes(1,2),
    threshold = 0.5
  )
)

hits["cluster"] = [n.split("_")[1] for n in hits["motif_name"]]
hits["-log(p-value)"] = -np.log(hits["p-value"] + 1e-6)

max_hits_per_seq = hits.groupby(["sequence_name", "cluster"])[["score", "-log(p-value)"]].max().reset_index()

hits["sequence_name"] = [selected_regions[x] for x in hits["sequence_name"]]

fig, ax = plt.subplots()
ax.scatter(
  np.log(region_topic.loc[selected_regions, "Topic_1"].values + 1e-6),
  np.log(region_topic.loc[selected_regions, "Topic_31"].values + 1e-6),
  color = "black", s = 1
)
fig.savefig("plots/topic_1_v_topic_31.pdf")

y = (
      np.log(region_topic.loc[selected_regions, "Topic_1"].values + 1e-6) \
    - np.log(region_topic.loc[selected_regions, "Topic_31"].values + 1e-6)
  ).reshape(-1, 1)

cluster_4_score = max_hits_per_seq.set_index("cluster") \
  .loc["4"] \
  .reset_index(drop = True) \
  .set_index("sequence_name") \
  .iloc[np.arange(len(y))] \
  ["-log(p-value)"].values
cluster_8_score = max_hits_per_seq.set_index("cluster") \
  .loc["8"] \
  .reset_index(drop = True) \
  .set_index("sequence_name") \
  .iloc[np.arange(len(y))] \
  ["-log(p-value)"].values

X = np.array([cluster_4_score, cluster_8_score]).T

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression, LogisticRegression

reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
reg.score(X_test, (y_test > 0).ravel())

fig, ax = plt.subplots()
ax.scatter(
  x = y_test,
  y = reg.predict_proba(X_test)[:, 1],
  s = 4, color = "black"
)
fig.tight_layout()
fig.savefig("plots/prediction_topic_1_v_topic_31.pdf")

from sklearn.metrics import precision_recall_curve, roc_curve

precision, recall, threshold = precision_recall_curve(
    (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])

fpr, tpr, thresholds = roc_curve(
    (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])

fig, axs = plt.subplots(ncols = 2, figsize = (8, 4))
_ = axs[0].plot(fpr, tpr, color = "black")
_ = axs[1].plot(recall, precision, color = "black")
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
fig.savefig("plots/pr_roc_prediction_topic_1_v_31.pdf")


###



motif_metadata = pd.DataFrame(
  index = patterns_to_cluster_names
)

motif_metadata["hier_cluster"] = clusters

motif_metadata["ic_start"] = [pattern.ic_trim(0.2)[0] for pattern in patterns_to_cluster]
motif_metadata["ic_stop"] = [pattern.ic_trim(0.2)[1] for pattern in patterns_to_cluster]
motif_metadata["is_root_motif"] = False
motif_metadata["is_rc_to_root"] = False
motif_metadata["offset_to_root"] = 0

for cluster in set(clusters):
  cluster_idc = np.where(clusters == cluster)[0]
  print(f"cluster: {cluster}")
  _, _, o, _, s = tomtom(
      [torch.from_numpy(patterns_to_cluster[m].ppm).T for m in cluster_idc],
      [torch.from_numpy(patterns_to_cluster[m].ppm).T for m in cluster_idc]
  )
  fig, axs = plt.subplots(nrows = sum(clusters == cluster), figsize = (4, 2 * sum(clusters == cluster)), sharex = True)
  pwms_aligned = []
  if sum(clusters == cluster) == 1:
    axs = [axs]
  for i, m in enumerate(cluster_idc):
    print(i)
    rel = np.argmax([np.mean(patterns_to_cluster[m].ic()) for m in cluster_idc])
    motif_metadata.loc[patterns_to_cluster_names[cluster_idc[rel]], "is_root_motif"] = True
    is_rc = s[i, rel] == 1
    offset = int(o[i, rel].numpy())
    pwm = patterns_to_cluster[m].ppm
    ic = patterns_to_cluster[m].ic()[:, None]
    pwm_rel = patterns_to_cluster[cluster_idc[rel]].ppm
    if is_rc:
      pwm = pwm[::-1, ::-1]
      ic = ic[::-1, ::-1]
      offset = pwm_rel.shape[0] - pwm.shape[0] - offset
    motif_metadata.loc[patterns_to_cluster_names[m], "is_rc_to_root"] = bool(is_rc)
    motif_metadata.loc[patterns_to_cluster_names[m], "offset_to_root"] = offset
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
  pwm_std = np.array(
    [
      np.concatenate([p, np.zeros((max_len - p.shape[0], 4))])
      for p in pwms_aligned
    ]
  ).std(0)
  ic = modiscolite.util.compute_per_position_ic(
        ppm=pwm_avg, background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
  )
  fig, ax = plt.subplots(figsize = (4,2))
  _ = logomaker.Logo(
    pd.DataFrame(pwm_avg * ic[:, None], columns = ["A", "C", "G", "T"]),
    ax = ax
  )
  fig.tight_layout()
  fig.savefig(f"plots/cluster_{cluster}_avg_patterns_aligned.pdf")

def get_non_overlapping_start_end_w_max_score(df, max_overlap, score_col):
  df = df.sort_values("start")
  delta = np.diff(df["start"])
  delta_loc = [0, *np.where(delta > max_overlap)[0] + 1]
  groups = [
    slice(delta_loc[i], delta_loc[i + 1] if (i + 1) < len(delta_loc) else None)
    for i in range(len(delta_loc))
  ]
  return pd.DataFrame(
    [
        df.iloc[group].iloc[df.iloc[group][score_col].argmax()]
        for group in groups
    ]
  ).reset_index(drop = True)

COMPLEMENT = {
  "A": "T",
  "T": "A",
  "C": "G",
  "G": "C",
  "a": "t",
  "t": "a",
  "c": "g",
  "g": "c",
  "N": "N",
  "n": "n"
}

letter_to_color = {
  "A": "#008000",
  "C": "#0000ff",
  "G": "#ffa600",
  "T": "#ff0000"
}

letter_to_val = {
  c: v
  for c, v in zip(list("ACGT"), np.linspace(0, 1, 4))
}

import matplotlib
nuc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nuc", list(letter_to_color.values()), 4)

def reverse_complement(s):
  return "".join(map(COMPLEMENT.get, s[::-1]))

def get_sequence_hit(hit, alignment_info, genome, target_len):
  chrom, start_offset, _ = hit.sequence_name.replace(":", "-").split("-")
  start_offset = int(start_offset)
  hit_start = hit.start
  hit_end = hit.end
  hit_strand = hit.strand
  is_rc_hit = hit_strand == "-"
  is_rc_to_root = alignment_info.is_rc_to_root
  offset_to_root = alignment_info.offset_to_root
  is_root = alignment_info.is_root_motif
  if is_rc_to_root and not is_root:
    offset_to_root -= 1
  if not is_rc_to_root and not is_root:
    offset_to_root += 1
  if is_rc_to_root ^ is_rc_hit:
    # align end position
    _start = start_offset + hit_start
    _end   = start_offset + hit_end - offset_to_root
    to_pad = target_len - (_end - _start)
    # add padding to the start
    _start -= to_pad
    _end += target_len
    seq = genome.fetch(
      chrom,
      _start,
      _end
    )
    seq = reverse_complement(seq)
  else:
    # align start
    _start = start_offset + hit_start + offset_to_root
    _end   = start_offset + hit_end
    to_pad = target_len - (_end - _start)
    # add padding to end
    _end += to_pad
    _start -= target_len
    seq = genome.fetch(
      chrom,
      _start,
      _end
    )
  return seq

selected_regions = list(
    set(region_topic["Topic_1"].sort_values(ascending = False).head(3_000).index) \
  | set(region_topic["Topic_31"].sort_values(ascending = False).head(3_000).index)
)

hg38 = pysam.FastaFile("/data/projects/c20/sdewin/resources/hg38/hg38.fa")

ohs = np.array(
  [
    one_hot_encode_sequence(
      hg38.fetch( r.split(":")[0], *map(int, r.split(":")[1].split("-")) ),
      expand_dim = False
    )
    for r in selected_regions
  ]
)

motifs = {
  **{
      patterns_to_cluster_names[pattern_idx]: patterns_to_cluster[pattern_idx].ppm[
      range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))
    ].T
    for pattern_idx in np.where(clusters == 4)[0]
  },
  **{
      patterns_to_cluster_names[pattern_idx]: patterns_to_cluster[pattern_idx].ppm[
      range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))
    ].T
    for pattern_idx in np.where(clusters == 8)[0]
  }
}

hits = pd.concat(
  fimo(
    motifs=motifs,
    sequences=ohs.swapaxes(1,2),
    threshold = 0.0001
  )
)

hits["cluster"] = [motif_metadata.loc[n, "hier_cluster"] for n in hits["motif_name"]]
hits["-log(p-value)"] = -np.log(hits["p-value"] + 1e-6)
hits["sequence_name"] = [selected_regions[x] for x in hits["sequence_name"]]

hits_non_overlap = hits \
  .groupby(["sequence_name", "cluster"]) \
  .apply(lambda hit_region_cluster: get_non_overlapping_start_end_w_max_score(hit_region_cluster, 10, "-log(p-value)")) \
  .reset_index(drop = True)

MX = 40
for cluster in [4, 8]:
  print(cluster)
  pattern_seq = []
  for _, hit in hits_non_overlap.query("cluster == @cluster").sort_values("sequence_name").iterrows():
    s = get_sequence_hit(
      hit=hit,
      alignment_info=motif_metadata.loc[hit.motif_name],
      genome=hg38,
      target_len=20
    )
    pattern_seq.append(s)
  ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
  pc = 1e-3
  ppm += pc
  ppm = (ppm.T / ppm.sum(1)).T
  ic = modiscolite.util.compute_per_position_ic(
    ppm=ppm.to_numpy(), background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
  )
  fig, axs = plt.subplots(nrows = 2, figsize = (4, 8), height_ratios = [1, 10], sharex = False)
  _ = logomaker.Logo((ppm * ic[:, None]), ax = axs[0])
  D = np.array([[letter_to_val[nuc.upper()] for nuc in seq] for seq in pattern_seq])
  sns.heatmap(
    D,
    cmap = nuc_cmap,
    ax = axs[1],
    cbar = False,
    yticklabels = False
  )
  axs[1].set_ylabel("motif instances")
  axs[0].set_ylabel("bits")
  axs[1].set_xlabel("position")
  axs[0].set_ylim((0, 2))
  axs[0].set_xticks(np.arange(D.shape[1]), labels = [])
  axs[1].set_xticks(
    np.arange(D.shape[1]) - 0.5,
    labels = np.arange(D.shape[1])
  )
  axs[0].set_xlim(-0.5, MX + 0.5)
  axs[1].set_xlim(0, MX + 1)
  _ = axs[0].set_title(f"cluster {cluster} instances")
  fig.tight_layout()
  fig.savefig(f"plots/cluster_{cluster}_aligned_hits.png")

def replace_hit(
  hit, alignment_info, genome, replace_to,
  verbose = False, add_extra_to_start = False
):
  chrom, region_start, region_end = hit.sequence_name.replace(":", "-").split("-")
  region_start = int(region_start)
  region_end = int(region_end)
  hit_start = hit.start
  hit_end = hit.end
  hit_strand = hit.strand
  is_rc_hit = hit_strand == "-"
  is_rc_to_root = alignment_info.is_rc_to_root
  offset_to_root = alignment_info.offset_to_root
  is_root = alignment_info.is_root_motif
  if is_rc_to_root and not is_root:
    offset_to_root -= 1
  if not is_rc_to_root and not is_root:
    offset_to_root += 1
  if is_rc_to_root ^ is_rc_hit:
    # align end position
    hit_start = region_start + hit_start
    hit_end   = region_start + hit_end - offset_to_root
  else:
    # align start
    hit_start = region_start + hit_start + offset_to_root
    hit_end   = region_start + hit_end
  orig_pattern = genome.fetch(
    chrom,
    hit_start,
    hit_end
  )
  delta = len(orig_pattern) - len(replace_to)
  if delta < 0:
    raise ValueError("replace_to is shorter than orig_pattern")
  if is_rc_hit:
    if not add_extra_to_start:
      replace_to += "".join([COMPLEMENT.get(n, "") for n in orig_pattern[0:delta]]).lower()
    else:
      replace_to = "".join([COMPLEMENT.get(n, "") for n in orig_pattern[::-1][0:delta]]).lower() + replace_to
    replace_to = reverse_complement(replace_to)
  else:
    if not add_extra_to_start:
      replace_to += orig_pattern[::-1][0:delta].lower()
    else:
      replace_to = orig_pattern[0:delta].lower() + replace_to
  hit_start_relative = hit_start - region_start
  hit_end_relative = hit_end - region_start
  if verbose:
    print(
      f">{chrom}:{region_start}-{region_end}\t" \
        +f"[{hit_start_relative}, {hit_end_relative}]\t{is_rc_hit}\n\t{orig_pattern}\n\t{replace_to}"
    )
  return (
    f"{chrom}:{region_start}-{region_end}",
    [hit_start_relative, hit_end_relative],
    replace_to
  )

modifications_per_sequence_4 = {}
for _, hit in hits_non_overlap \
  .query("cluster == 4") \
  .sort_values("sequence_name") \
  .iterrows():
  region_name, start_end, replace_to = replace_hit(
    hit,
    motif_metadata.loc[hit.motif_name],
    hg38,
    "AGGGATTAG",
    verbose = True
  )
  if region_name not in modifications_per_sequence_4:
    modifications_per_sequence_4[region_name] = []
  modifications_per_sequence_4[region_name].append((start_end, replace_to))

modifications_per_sequence_8 = {}
for _, hit in hits_non_overlap \
  .query("cluster == 8") \
  .sort_values("sequence_name") \
  .iterrows():
  region_name, start_end, replace_to = replace_hit(
    hit,
    motif_metadata.loc[hit.motif_name],
    hg38,
    "CTAATTAG",
    verbose = True,
    add_extra_to_start = True
  )
  if region_name not in modifications_per_sequence_8:
    modifications_per_sequence_8[region_name] = []
  modifications_per_sequence_8[region_name].append((start_end, replace_to))

cluster_4_seq_orig = []
cluster_4_seq_modi = []
for region in modifications_per_sequence_4:
  print(region)
  chrom, start, end = region.replace(":", "-").split("-")
  start = int(start)
  end = int(end)
  orig_seq = hg38.fetch(
    chrom, start, end
  )
  modi_seq = list(orig_seq)
  for (start, end), new_tfbs in modifications_per_sequence_4[region]:
    print(f"Replacing:\n\t{orig_seq[start: end]}")
    if end - start != len(new_tfbs):
      raise ValueError("Start end is not length of new tfbs")
    for nuc in range(len(new_tfbs)):
      modi_seq[nuc + start] = new_tfbs[nuc]
    tmp_modi_seq = "".join(modi_seq)
    print(f"\t{tmp_modi_seq[start:end]}")
  cluster_4_seq_orig.append(orig_seq)
  cluster_4_seq_modi.append("".join(modi_seq))

cluster_8_seq_orig = []
cluster_8_seq_modi = []
for region in modifications_per_sequence_8:
  print(region)
  chrom, start, end = region.replace(":", "-").split("-")
  start = int(start)
  end = int(end)
  orig_seq = hg38.fetch(
    chrom, start, end
  )
  modi_seq = list(orig_seq)
  for (start, end), new_tfbs in modifications_per_sequence_8[region]:
    print(f"Replacing:\n\t{orig_seq[start: end]}")
    if end - start != len(new_tfbs):
      raise ValueError("Start end is not length of new tfbs")
    for nuc in range(len(new_tfbs)):
      modi_seq[nuc + start] = new_tfbs[nuc]
    tmp_modi_seq = "".join(modi_seq)
    print(f"\t{tmp_modi_seq[start:end]}")
  cluster_8_seq_orig.append(orig_seq)
  cluster_8_seq_modi.append("".join(modi_seq))

import numpy as np
import tensorflow as tf
import os

# load human model 
path_to_human_model = "../../data_prep_new/embryo_data/MODELS/"

model = tf.keras.models.model_from_json(
    open(
        os.path.join(path_to_human_model, "model.json")
    ).read(),
    custom_objects = {'Functional':tf.keras.models.Model}
)

model.load_weights(
    os.path.join(path_to_human_model, "model_epoch_36.hdf5")
)

prediction_score_cluster_4_orig = model.predict(
  np.array([one_hot_encode_sequence(s, expand_dim = False) for s in cluster_4_seq_orig]),
  verbose = True
)

prediction_score_cluster_4_modi = model.predict(
  np.array([one_hot_encode_sequence(s, expand_dim = False) for s in cluster_4_seq_modi]),
  verbose = True
)

prediction_score_cluster_8_orig = model.predict(
  np.array([one_hot_encode_sequence(s, expand_dim = False) for s in cluster_8_seq_orig]),
  verbose = True
)

prediction_score_cluster_8_modi = model.predict(
  np.array([one_hot_encode_sequence(s, expand_dim = False) for s in cluster_8_seq_modi]),
  verbose = True
)

data_4 = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
          [
            prediction_score_cluster_4_orig[:, ap_topics - 1],
            np.repeat("wt", prediction_score_cluster_4_orig.shape[0])[:, None]
          ],
          axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "condition"
      ],
      index = modifications_per_sequence_4.keys()
    ),
  pd.DataFrame(
      np.concatenate(
          [
            prediction_score_cluster_4_modi[:, ap_topics - 1],
            np.repeat("alt", prediction_score_cluster_4_modi.shape[0])[:, None]
          ],
          axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "condition"
      ],
      index = modifications_per_sequence_4.keys()
    )
  ]
) \
  .melt(id_vars = "condition", ignore_index = False) \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )

data_8 = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
          [
            prediction_score_cluster_8_orig[:, ap_topics - 1],
            np.repeat("wt", prediction_score_cluster_8_orig.shape[0])[:, None]
          ],
          axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "condition"
      ],
      index = modifications_per_sequence_8
    ),
  pd.DataFrame(
      np.concatenate(
          [
            prediction_score_cluster_8_modi[:, ap_topics - 1],
            np.repeat("alt", prediction_score_cluster_8_modi.shape[0])[:, None]
          ],
          axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "condition"
      ],
      index = modifications_per_sequence_8
    )
  ]
) \
  .melt(id_vars = "condition", ignore_index = False) \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )

data_4["prediction score"] = data_4["prediction score"].astype(float)
data_8["prediction score"] = data_8["prediction score"].astype(float)

fig, axs = plt.subplots(ncols = 2, figsize = (8, 4))
for ax, data in zip(axs, [data_4, data_8]):
  regions = data.loc[data["prediction score"] >= 0.4].index.drop_duplicates()
  sns.boxplot(
    data=data.loc[regions],
    x = "condition",
    y = "prediction score",
    hue = "Topic",
    ax = ax,
    palette = "Spectral",
    legend = False
  )
  ax.grid()
axs[0].set_title("CTAATTAG -> AGGGATTAG")
axs[1].set_title("AGGGATTAG -> CTAATTAG")
fig.savefig("plots/prediction_score_cluster_4_8_sub.pdf")

```

```python

import tensorflow as tf
import os
import crested
import random
import numpy as np
from dataclasses import dataclass
import h5py
from typing import Self

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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

genome = crested.Genome(
    "/data/projects/c20/sdewin/resources/hg38/hg38.fa",
    "/data/projects/c20/sdewin/resources/hg38/hg38.chrom.sizes"
)

consensus_peaks_embryo = []
with open("../../data_prep_new/embryo_data/ATAC/embry_atac_region_names.txt", "rt") as f:
  for l in f:
    consensus_peaks_embryo.append(l.strip())

n_to_sample = 10_000

random.seed(123)
acgt_distribution = crested.utils.calculate_nucleotide_distribution(
  input=random.sample(consensus_peaks_embryo, n_to_sample),
  genome=genome,
  per_position=True
)

# load human model 
path_to_human_model = "../../data_prep_new/embryo_data/MODELS/"
model = tf.keras.models.model_from_json(
    open(
        os.path.join(path_to_human_model, "model.json")
    ).read(),
    custom_objects = {'Functional':tf.keras.models.Model}
)
model.load_weights(
    os.path.join(path_to_human_model, "model_epoch_36.hdf5")
)

ap_topics = np.array(
  [31, 29, 1, 32, 40, 22, 41]
) + 30

experimental_design = {
  "CTAATTAG_Topic_31_5": dict(
    patterns = {
      "HOX_1": "CTAATTAG"
    },
    insertions_per_pattern = {
      "HOX_1": 5
    },
    target = ap_topics[0] - 1
  ),
  "CTAATTAG_Topic_1_5": dict(
    patterns = {
      "HOX_1": "CTAATTAG"
    },
    insertions_per_pattern = {
      "HOX_1": 5
    },
    target = ap_topics[2] - 1
  ),
  "GGATTAG_Topic_31_5": dict(
    patterns = {
      "HOX_2": "GGATTAG"
    },
    insertions_per_pattern = {
      "HOX_2": 5
    },
    target = ap_topics[0] - 1
  ),
  "GGATTAG_Topic_1_5": dict(
    patterns = {
      "HOX_2": "GGATTAG"
    },
    insertions_per_pattern = {
      "HOX_2": 5
    },
    target = ap_topics[2] - 1
  ),
  "SOX_CTAATTAG_Topic_31_5": dict(
    patterns = {
      "SOX": "ACAA",
      "HOX_1": "CTAATTAG"
    },
    insertions_per_pattern = {
      "SOX": 1,
      "HOX_1": 5
    },
    target = ap_topics[0] - 1
  ),
  "SOX_CTAATTAG_Topic_1_5": dict(
    patterns = {
      "SOX": "ACAA",
      "HOX_1": "CTAATTAG"
    },
    insertions_per_pattern = {
      "SOX": 1,
      "HOX_1": 5
    },
    target = ap_topics[2] - 1
  ),
  "SOX_GGATTAG_Topic_31_5": dict(
    patterns = {
      "SOX": "ACAA",
      "HOX_2": "GGATTAG"
    },
    insertions_per_pattern = {
      "SOX": 1,
      "HOX_2": 5
    },
    target = ap_topics[0] - 1
  ),
  "SOX_GGATTAG_Topic_1_5": dict(
    patterns = {
      "SOX": "ACAA",
      "HOX_2": "GGATTAG"
    },
    insertions_per_pattern = {
      "SOX": 1,
      "HOX_2": 5
    },
    target = ap_topics[2] - 1
  ),
}

from tqdm import tqdm

n_seq = 200

results_per_experiment = {}
for experiment in tqdm(experimental_design):
  print(experiment)
  print(experimental_design[experiment])
  results_per_experiment[experiment] = crested.tl.enhancer_design_motif_insertion(
      model=model,
      acgt_distribution=acgt_distribution,
      n_sequences=n_seq,
      target_len=500,
      return_intermediate=True,
      **experimental_design[experiment]
  )

import pandas as pd

data = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
            [
              np.array(
                [results_per_experiment[experiment][0][i]["predictions"][iter][ap_topics - 1] for i in range(n_seq)]
              ),
              np.repeat(experiment, n_seq)[:, None],
              np.repeat(iter, n_seq)[:, None]
            ],
            axis = 1
          ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "condition",
          "iter"
      ],
    )
    for iter in range(len(results_per_experiment[experiment][0][0]["changes"]) - 1) for experiment in experimental_design
  ]
).melt(id_vars = ["condition", "iter"]) \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )


data["iter"] = data["iter"].astype(int)
data["prediction score"] = data["prediction score"].astype(float)

import matplotlib.pyplot as plt
import seaborn as sns

for experiment in experimental_design:
  print(experiment)
  fig, ax = plt.subplots(figsize = (8, 4))
  sns.boxplot(
    data=data.query("condition == @experiment"),
    x = "iter",
    y = "prediction score",
    hue = "Topic",
    ax = ax,
    palette = "Spectral",
    legend = False
  )
  fig.savefig(f"plots/motif_embedding_{experiment}.pdf")
  plt.close(fig)

import pickle
with open("motif_embedding.pkl", "wb") as f:
  pickle.dump(results_per_experiment, f)


from crested.tl._explainer_tf import Explainer
from crested.utils._seq_utils import one_hot_encode_sequence


for class_index in [ap_topics[0] - 1, ap_topics[2] - 1]:
  explainer = Explainer(model = model, class_index = class_index)
  for experiment in results_per_experiment:
    print(f"{experiment}_class_{class_index}")
    initial_ohs = np.concatenate(
      [
        one_hot_encode_sequence(results_per_experiment[experiment][0][i]["initial_sequence"])
        for i in range(n_seq)
      ]
    )
    designed_ohs = np.concatenate(
      [
        one_hot_encode_sequence(results_per_experiment[experiment][0][i]["designed_sequence"])
        for i in range(n_seq)
      ]
    )
    gradients_initial = explainer.integrated_grad(X = initial_ohs)
    gradients_designed = explainer.integrated_grad(X = designed_ohs)
    np.savez_compressed(
      f"motif_embedding_gradients/oh_gradient_initial_{experiment}_class_{class_index}",
      oh = initial_ohs,
      gradient = gradients_initial
    )
    np.savez_compressed(
      f"motif_embedding_gradients/oh_gradient_designed_{experiment}_class_{class_index}",
      oh = designed_ohs,
      gradient = gradients_designed
    )

from custom_modisco import workflow
from joblib import Parallel, delayed
import custom_modisco
import numpy as np
import modiscolite
import os
from tqdm import tqdm

params = dict(
    sliding_window_size=15,
    flank_size=5,
    min_metacluster_size=100,
    weak_threshold_for_counting_sign=0.8,
    target_seqlet_fdr=0.2,
    min_passing_windows_frac=0.03,
    max_passing_windows_frac=0.2,
    n_leiden_runs=50,
    n_leiden_iterations=-1,
    min_overlap_while_sliding=0.7,
    nearest_neighbors_to_compute=500,
    affmat_correlation_threshold=0.15,
    tsne_perplexity=10.0,
    frac_support_to_trim_to=0.2,
    min_num_to_trim_to=30,
    trim_to_window_size=20,
    initial_flank_to_add=5,
    prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
    prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
    subcluster_perplexity=50,
    merging_max_seqlets_subsample=300,
    final_min_cluster_size=20,
    min_ic_in_window=0.6,
    min_ic_windowsize=6,
    ppm_pseudocount=0.001,
    number_of_seqlets_to_sample = 1000)


gradient_files = [
  x for x in os.listdir("motif_embedding_gradients") if x.startswith("oh_gradient_") and x.endswith(".npz")]

def get_modisco_patterns_for_topic_and_save(
  path_to_gradients_result: str,
  out_file_result: str,
  params
):
  one_hot = np.load(path_to_gradients_result)["oh"]
  gradients_integrated = np.load(path_to_gradients_result)["gradient"].squeeze()
  task = path_to_gradients_result.split("/")[-1].replace("npz", "").replace("oh_gradient_", "")
  pos_patterns, neg_patterns = workflow.get_patterns_for_task(
    task = task,
    one_hot = one_hot,
    task_to_hyp_scores = {task: gradients_integrated},
    **params
  )
  modiscolite.io.save_hdf5(
    filename=out_file_result, pos_patterns=pos_patterns, neg_patterns=neg_patterns,
    window_size = params["sliding_window_size"]
  )

Parallel(n_jobs = 10)(
  delayed(get_modisco_patterns_for_topic_and_save)(
    path_to_gradients_result = os.path.join("motif_embedding_gradients", gf),
    out_file_result = os.path.join("motif_embedding_patterns", gf.replace("oh_gradient", "patterns").replace("npz", "hdf5")),
    params = params
  )
  for gf in tqdm(gradient_files)
)


patterns = []
pattern_names = []
for experiment in tqdm(experimental_design):
  for class_index in [ap_topics[0] - 1, ap_topics[2] - 1]:
    ohs_initial = np.load(
      f"motif_embedding_gradients/oh_gradient_initial_{experiment}_class_{class_index}.npz"
    )["oh"]
    ohs_designed = np.load(
      f"motif_embedding_gradients/oh_gradient_designed_{experiment}_class_{class_index}.npz"
    )["oh"]
    region_names_initial = [f"{experiment}_initial_{i}" for i in range(n_seq)]
    region_names_designed = [f"{experiment}_designed_{i}" for i in range(n_seq)]
    for name, pattern in load_pattern_from_modisco(
        filename = f"motif_embedding_patterns/patterns_initial_{experiment}_class_{class_index}.hdf5",
        ohs = ohs_initial,
        region_names = region_names_initial
    ):
        pattern_names.append(name)
        patterns.append(pattern)
    for name, pattern in load_pattern_from_modisco(
        filename = f"motif_embedding_patterns/patterns_designed_{experiment}_class_{class_index}.hdf5",
        ohs = ohs_designed,
        region_names = region_names_designed
    ):
        pattern_names.append(name)
        patterns.append(pattern)

from matplotlib.backends.backend_pdf import PdfPages
import logomaker

for experiment in tqdm(experimental_design):
  for class_index in [ap_topics[0] - 1, ap_topics[2] - 1]:
    for d in ["initial", "designed"]:
      prefix = f"patterns_{d}_{experiment}_class_{class_index}"
      print(prefix)
      with PdfPages(f"plots/{prefix}.pdf") as pdf:
        for pattern, pattern_name in zip(patterns, pattern_names):
          if not pattern_name.startswith(prefix):
            continue
          fig, ax = plt.subplots(figsize = (4, 2))
          _ = logomaker.Logo(
            pd.DataFrame(
              (pattern.ppm * pattern.ic()[:, None]) * (2 * pattern.is_pos - 1),
              columns = ["A", "C", "G", "T"]
            ),
            ax = ax
          )
          if pattern.is_pos:
            _ = ax.set_ylim(0, 2)
          else:
            _ = ax.set_ylim(-2, 0)
          _ = ax.set_ylabel("Bits")
          fig.tight_layout()
          pdf.savefig(fig)
          plt.close(fig)

def pairwise_distance(l: int):
  dist = np.zeros((len(l), len(l)))
  for i, a in enumerate(l):
    for j, b in enumerate(l):
      if i == j:
        dist[i, j] = np.inf
      else:
        dist[i, j] = abs(a - b)
  return dist

fig, axs = plt.subplots(ncols = 2, nrows = 4, figsize = (8, 16), sharex = True, sharey = True)
for n_sites in range(2, 6):
  ctaattag_distances = np.array([
    pairwise_distance(
      [x[0] for x in results_per_experiment["CTAATTAG_Topic_31_5"][0][i]["changes"][1:n_sites + 1]]
    ).min() - len("CTAATTAG")
    for i in range(n_seq)
  ])
  ggattag_distances = np.array([
    pairwise_distance(
      [x[0] for x in results_per_experiment["GGATTAG_Topic_1_5"][0][i]["changes"][1:n_sites + 1]]
    ).min() - len("GGATTAG")
    for i in range(n_seq)
  ])
  for i, dist in enumerate([ctaattag_distances, ggattag_distances]):
    ax = axs[n_sites - 2, i]
    for d in range(100):
      ax.bar(
        x = d,
        height = sum(dist == d),
        width = 1,
        color = "black"
      )
      ax.set_ylabel("Frequency")
      ax.set_xlabel(f"min distance ({n_sites} instances).")
      ax.grid(True)
      ax.set_axisbelow(True)
axs[0, 0].set_title("CTAATTAG")
axs[0, 1].set_title("GGATTAG")
fig.tight_layout()
fig.savefig("plots/motif_embedding_distances.pdf")



random.seed(123)
motif_to_put = "CTAATTAG"
ctaag_rand_prediction_scores = np.empty((n_seq, 6, model.output.shape[1]))
for i in tqdm(range(n_seq)):
  seqs = np.empty((6, 500, 4))
  seqs[0] = one_hot_encode_sequence(
    results_per_experiment["CTAATTAG_Topic_31_5"][0][i]["initial_sequence"],
    expand_dim = False
  )
  locations = np.zeros(5) + np.inf
  for j in range(1, 6):
    if j == 1:
      loc = random.randint(200, 300)
    else:
      loc = random.randint(200, 300)
      while abs((locations - loc)).min() <= len(motif_to_put):
        loc = random.randint(200, 300)
    locations[j - 1] = loc
    seqs[j] = seqs[j - 1].copy()
    seqs[j][loc:loc + len(motif_to_put)] = one_hot_encode_sequence(motif_to_put, expand_dim = False)
  ctaag_rand_prediction_scores[i] = model.predict(seqs, verbose = 0)

random.seed(123)
motif_to_put = "GGATTAG"
ggattag_rand_prediction_scores = np.empty((n_seq, 6, model.output.shape[1]))
for i in tqdm(range(n_seq)):
  seqs = np.empty((6, 500, 4))
  seqs[0] = one_hot_encode_sequence(
    results_per_experiment["GGATTAG_Topic_1_5"][0][i]["initial_sequence"],
    expand_dim = False
  )
  locations = np.zeros(5) + np.inf
  for j in range(1, 6):
    if j == 1:
      loc = random.randint(200, 300)
    else:
      loc = random.randint(200, 300)
      while abs((locations - loc)).min() <= len(motif_to_put):
        loc = random.randint(200, 300)
    locations[j - 1] = loc
    seqs[j] = seqs[j - 1].copy()
    seqs[j][loc:loc + len(motif_to_put)] = one_hot_encode_sequence(motif_to_put, expand_dim = False)
  ggattag_rand_prediction_scores[i] = model.predict(seqs, verbose = 0)


data_ctaag = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
        [
          ctaag_rand_prediction_scores[:, i, ap_topics - 1],
          np.repeat(i, n_seq)[:, None]
        ],
        axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "iter"
      ],
    )
    for i in range(6)
  ]
).melt(id_vars = "iter") \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )

data_ggattag = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
        [
          ggattag_rand_prediction_scores[:, i, ap_topics - 1],
          np.repeat(i, n_seq)[:, None]
        ],
        axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "iter"
      ],
    )
    for i in range(6)
  ]
).melt(id_vars = "iter") \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )



fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(
  data=data_ctaag,
  x = "iter",
  y = "prediction score",
  hue = "Topic",
  ax = ax,
  palette = "Spectral",
  legend = False
)
fig.savefig(f"plots/motif_embedding_random_CTAATTAG.pdf")
  
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(
  data=data_ggattag,
  x = "iter",
  y = "prediction score",
  hue = "Topic",
  ax = ax,
  palette = "Spectral",
  legend = False
)
fig.savefig(f"plots/motif_embedding_random_GGATTAG.pdf")
 

random.seed(123)
motif_to_put = "CTAATTAGGGCTAATTAG"
ctaag_rand_prediction_scores = np.empty((n_seq, 3, model.output.shape[1]))
for i in tqdm(range(n_seq)):
  seqs = np.empty((3, 500, 4))
  seqs[0] = one_hot_encode_sequence(
    results_per_experiment["CTAATTAG_Topic_31_5"][0][i]["initial_sequence"],
    expand_dim = False
  )
  locations = np.zeros(2) + np.inf
  for j in range(1, 3):
    if j == 1:
      loc = random.randint(200, 300)
    else:
      loc = random.randint(200, 300)
      while abs((locations - loc)).min() <= len(motif_to_put):
        loc = random.randint(200, 300)
    locations[j - 1] = loc
    seqs[j] = seqs[j - 1].copy()
    seqs[j][loc:loc + len(motif_to_put)] = one_hot_encode_sequence(motif_to_put, expand_dim = False)
  ctaag_rand_prediction_scores[i] = model.predict(seqs, verbose = 0)

data_ctaag = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
        [
          ctaag_rand_prediction_scores[:, i, ap_topics - 1],
          np.repeat(i, n_seq)[:, None]
        ],
        axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "iter"
      ],
    )
    for i in range(3)
  ]
).melt(id_vars = "iter") \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(
  data=data_ctaag,
  x = "iter",
  y = "prediction score",
  hue = "Topic",
  ax = ax,
  palette = "Spectral",
  legend = False
)
fig.savefig(f"plots/motif_embedding_random_{motif_to_put}.pdf")


random.seed(123)
motif_to_put = "GGATTAGCGGGGGATTAG"
ggattag_rand_prediction_scores = np.empty((n_seq, 3, model.output.shape[1]))
for i in tqdm(range(n_seq)):
  seqs = np.empty((3, 500, 4))
  seqs[0] = one_hot_encode_sequence(
    results_per_experiment["GGATTAG_Topic_1_5"][0][i]["initial_sequence"],
    expand_dim = False
  )
  locations = np.zeros(2) + np.inf
  for j in range(1, 3):
    if j == 1:
      loc = random.randint(200, 300)
    else:
      loc = random.randint(200, 300)
      while abs((locations - loc)).min() <= len(motif_to_put):
        loc = random.randint(200, 300)
    locations[j - 1] = loc
    seqs[j] = seqs[j - 1].copy()
    seqs[j][loc:loc + len(motif_to_put)] = one_hot_encode_sequence(motif_to_put, expand_dim = False)
  ggattag_rand_prediction_scores[i] = model.predict(seqs, verbose = 0)

data_ggattag = pd.concat(
  [
    pd.DataFrame(
      np.concatenate(
        [
          ggattag_rand_prediction_scores[:, i, ap_topics - 1],
          np.repeat(i, n_seq)[:, None]
        ],
        axis = 1
      ),
      columns = [
          *[model_index_to_topic_name_embryo(x - 1).replace("progenitor_",  "") for x in ap_topics],
          "iter"
      ],
    )
    for i in range(3)
  ]
).melt(id_vars = "iter") \
  .rename(
    {
      "variable": "Topic",
      "value": "prediction score"
    },
    axis = 1
  )

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(
  data=data_ggattag,
  x = "iter",
  y = "prediction score",
  hue = "Topic",
  ax = ax,
  palette = "Spectral",
  legend = False
)
fig.savefig(f"plots/motif_embedding_random_{motif_to_put}.pdf")
 



```
