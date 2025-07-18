```python

from dataclasses import dataclass
import h5py
from typing import Self
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch
from tangermeme.tools.tomtom import tomtom
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import logomaker
import modiscolite

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

organoid_neural_crest_topics = [
  1, 3, 4, 5,
  7, 9, 10
]

embryo_neural_crest_topics = [
  1, 4, 7, 8, 9,
  10, 11, 12, 13,
  15, 19, 20, 21,
  22, 29,
]

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_neural_crest_topics):
    topic = topic + 55
    ohs = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["oh"]
    region_names = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            organoid_dl_motif_dir, f"patterns_Topic_{topic}.hdf5",
        ),
        ohs = ohs,
        region_names = region_names
    ):
        pattern_names_dl_organoid.append("organoid_" + name)
        patterns_dl_organoid.append(pattern)

embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_neural_crest_topics):
    topic = topic + 90
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

for pattern, name in zip(patterns_dl_embryo, pattern_names_dl_embryo):
  if not (f"Topic_{12 + 90}_" in name or f"Topic_{13 + 90}_" in name):
    continue
  fig, ax = plt.subplots()
  _ = logomaker.Logo(
      pd.DataFrame(
          (pattern.ppm * pattern.ic()[:, None])[range(*pattern.ic_trim(0.2))],
          columns = ["A", "C", "G", "T"]
      ),
      ax = ax
    )
  _ = ax.set_ylim(0, 2)
  _ = ax.set_title(name)
  print(name)
  fig.savefig(f"plots/{name}.pdf")
  plt.close(fig)


patterns_to_cluster = []
patterns_to_cluster_names = []
avg_ic_thr = 0.6
for pattern, pattern_name in zip(
  [*patterns_dl_organoid, *patterns_dl_embryo],
  [*pattern_names_dl_organoid, *pattern_names_dl_embryo]
):
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

clusters = hierarchy.fcluster(row_linkage, t = 15, criterion = "maxclust")
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
    name = patterns_to_cluster_names[pattern_idx]
    _ = logomaker.Logo(
      pd.DataFrame(
          (pattern.ppm * pattern.ic()[:, None])[range(*pattern.ic_trim(0.2))],
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
  index = patterns_to_cluster_names,
  data = {
    "cluster": clusters
  }
)

to_sub_cluster = {
  1: 2,
  2: 3,
  4: 2,
  9: 3,
  10: 3,
  11: 2,
  13: 2,
}
cluster_to_subclusters = {}
pattern_metadata["sub_cluster"] = 0
for cluster, n_clusters in to_sub_cluster.items():
  patterns_of_cluster = np.where(clusters == cluster)[0]
  ppm_cluster = [
    pattern for (pattern, in_cluster) in zip(ppm_patterns_to_cluster, clusters == cluster)
    if in_cluster
  ]
  pvals_cluster, _, _, _, _ = tomtom(
    ppm_cluster, ppm_cluster
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
      pattern = patterns_to_cluster[patterns_of_cluster[pattern_idx]]
      name = patterns_to_cluster_names[patterns_of_cluster[pattern_idx]]
      pattern_metadata.loc[name, "sub_cluster"] = int(subcluster)
      _ = logomaker.Logo(
        pd.DataFrame(
            (pattern.ppm * pattern.ic()[:, None])[range(*pattern.ic_trim(0.2))],
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

for cluster in set(pattern_metadata["cluster_sub_cluster"]):
  cluster_idc = np.where(pattern_metadata["cluster_sub_cluster"] == cluster)[0]
  n_clusters = len(cluster_idc)
  print(f"cluster: {cluster}")
  _, _, o, _, s = tomtom(
      [torch.from_numpy(patterns_to_cluster[m].ppm.T) for m in cluster_idc],
      [torch.from_numpy(patterns_to_cluster[m].ppm.T) for m in cluster_idc]
  )
  fig, axs = plt.subplots(nrows = n_clusters, figsize = (4, 2 * n_clusters), sharex = True)
  pwms_aligned = []
  if n_clusters == 1:
    axs = [axs]
  for i, m in enumerate(cluster_idc):
    print(i)
    rel = np.argmax([np.mean(patterns_to_cluster[m].ic()) for m in cluster_idc])
    pattern_metadata.loc[patterns_to_cluster_names[cluster_idc[rel]], "is_root_motif"] = True
    is_rc = s[i, rel] == 1
    offset = int(o[i, rel].numpy())
    pwm = patterns_to_cluster[m].ppm
    pwm_rel = patterns_to_cluster[cluster_idc[rel]].ppm
    ic = patterns_to_cluster[m].ic()[:, None]
    if is_rc:
      pwm = pwm[::-1, ::-1]
      ic = ic[::-1, ::-1]
      offset = pwm_rel.shape[0] - pwm.shape[0] - offset
    pattern_metadata.loc[patterns_to_cluster_names[m], "is_rc_to_root"] = bool(is_rc)
    pattern_metadata.loc[patterns_to_cluster_names[m], "offset_to_root"] = offset
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


pattern_metadata.to_csv("pattern_metadata.tsv", sep = "\t", header = True, index = True)

```

```python

import os
os.environ["CUDA_VISIBALE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import os
import crested
from tqdm import tqdm
from crested.utils._seq_utils import one_hot_encode_sequence
from crested.tl._explainer_tf import Explainer
import h5py
import pandas as pd
from dataclasses import dataclass
import h5py
from typing import Self
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import logomaker
import modiscolite

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

organoid_neural_crest_topics = [
  1, 3, 4, 5,
  7, 9, 10
]

embryo_neural_crest_topics = [
  1, 4, 7, 8, 9,
  10, 11, 12, 13,
  15, 19, 20, 21,
  22, 29,
]

path_to_organoid_model = "../../data_prep_new/organoid_data/MODELS/"

organoid_model = tf.keras.models.model_from_json(
  open(os.path.join(path_to_organoid_model, "model.json")).read(),
  custom_objects = {"Functional": tf.keras.models.Model}
)

organoid_model.load_weights(os.path.join(path_to_organoid_model, "model_epoch_23.hdf5"))

##

path_to_embryo_model = "../../data_prep_new/embryo_data/MODELS/"

embryo_model = tf.keras.models.model_from_json(
  open(os.path.join(path_to_embryo_model, "model.json")).read(),
  custom_objects = {"Functional": tf.keras.models.Model}
)

embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_neural_crest_topics):
    topic = topic + 55
    ohs = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["oh"]
    region_names = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            organoid_dl_motif_dir, f"patterns_Topic_{topic}.hdf5",
        ),
        ohs = ohs,
        region_names = region_names
    ):
        pattern_names_dl_organoid.append("organoid_" + name)
        patterns_dl_organoid.append(pattern)

embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_neural_crest_topics):
    topic = topic + 90
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

all_patterns = [*patterns_dl_organoid, *patterns_dl_embryo]
all_pattern_names = [*pattern_names_dl_organoid, *pattern_names_dl_embryo]

pattern_metadata = pd.read_table("pattern_metadata.tsv", index_col = 0)

pattern_to_topic_to_grad_organoid = {}
for pattern_name in tqdm(pattern_metadata.index):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad_organoid[pattern_name] = {}
  for topic in tqdm(organoid_neural_crest_topics, leave = False):
    topic = topic + 55
    class_idx = topic - 1
    explainer = Explainer(model = organoid_model, class_index = int(class_idx))
    #gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
    gradients_integrated = explainer.saliency_maps(X = oh_sequences)
    pattern_grads = list(get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze()))
    pattern_to_topic_to_grad_organoid[pattern_name][topic] = (pattern_grads, pattern_ohs)

pattern_to_topic_to_grad_embryo = {}
for pattern_name in tqdm(pattern_metadata.index):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad_embryo[pattern_name] = {}
  for topic in tqdm(embryo_neural_crest_topics, leave = False):
    topic = topic + 90
    class_idx = topic - 1
    explainer = Explainer(model = embryo_model, class_index = int(class_idx))
    #gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
    gradients_integrated = explainer.saliency_maps(X = oh_sequences)
    pattern_grads = list(get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze()))
    pattern_to_topic_to_grad_embryo[pattern_name][topic] = (pattern_grads, pattern_ohs)

def allign_patterns_of_cluster_for_topic(
    pattern_to_topic_to_grad: dict[str, dict[int, np.ndarray]],
    pattern_metadata: pd.DataFrame,
    cluster_id: int,
    topic:int
  ):
  P = []
  O = []
  cluster_patterns = pattern_metadata.query("cluster_sub_cluster == @cluster_id").index.to_list()
  for pattern_name in cluster_patterns:
    pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
    ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
      pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
    ]
    pattern_grads_ic = [p[ic_start: ic_end] for p in pattern_grads]
    pattern_ohs_ic = [o[ic_start: ic_end] for o in pattern_ohs]
    if is_rc_to_root:
      pattern_grads_ic = [p[::-1, ::-1] for p in pattern_grads_ic]
      pattern_ohs_ic = [o[::-1, ::-1] for o in pattern_ohs_ic]
    if offset_to_root > 0:
      pattern_grads_ic = [
        np.concatenate(
          [
            np.zeros((offset_to_root, 4)),
            p_ic
          ]
        )
        for p_ic in pattern_grads_ic
      ]
      pattern_ohs_ic = [
        np.concatenate(
          [
            np.zeros((offset_to_root, 4)),
            o_ic
          ]
        )
        for o_ic in pattern_ohs_ic
      ]
    elif offset_to_root < 0:
      pattern_grads_ic = [
        p[abs(offset_to_root):, :]
        for p in pattern_grads_ic
      ]
      pattern_ohs_ic = [
        o[abs(offset_to_root):, :]
        for o in pattern_ohs_ic
      ]
    P.extend(pattern_grads_ic)
    O.extend(pattern_ohs_ic)
  max_len = max([p.shape[0] for p in P])
  P = [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P]
  O = [np.concatenate([o, np.zeros((max_len - o.shape[0], 4))]) for o in O]
  return np.array(P), np.array(O)


pattern_metadata["ic_start"] = 0
#[
#  all_patterns[all_pattern_names.index(x)].ic_trim(0.2)[0] for x in  pattern_metadata.index
#]
pattern_metadata["ic_stop"] = 30
#[
#  all_patterns[all_pattern_names.index(x)].ic_trim(0.2)[1] for x in  pattern_metadata.index
#]

cluster_to_topic_to_avg_pattern_organoid = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
  cluster_to_topic_to_avg_pattern_organoid[cluster] = {}
  for topic in organoid_neural_crest_topics:
    topic = topic + 55
    P, O = allign_patterns_of_cluster_for_topic(
        pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid,
        pattern_metadata=pattern_metadata,
        cluster_id=cluster,
        topic=topic
      )
    cluster_to_topic_to_avg_pattern_organoid[cluster][topic] = (
      P * O
    ).mean(0)

cluster_to_topic_to_avg_pattern_embryo = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
  cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
  for topic in embryo_neural_crest_topics:
    topic = topic + 90
    P, O = allign_patterns_of_cluster_for_topic(
        pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
        pattern_metadata=pattern_metadata,
        cluster_id=cluster,
        topic=topic
      )
    cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (
      P * O
    ).mean(0)

organoid_neural_crest_topics = [
  62, 56, 60, 65, 59, 58, 64
]

def K(idx):
  return [organoid_neural_crest_topics.index(x) for x in idx]

pattern_order = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid).applymap(np.mean).idxmax().reset_index().set_index(0).sort_index(key = K)["index"].to_list()

n_clusters = len(cluster_to_topic_to_avg_pattern_organoid)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_neural_crest_topics),
  figsize = (len(organoid_neural_crest_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(pattern_order)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(organoid_neural_crest_topics):
    if i == 0:
      axs[i, j].set_title(f"Topic {topic}")
    pwm = cluster_to_topic_to_avg_pattern_organoid[cluster][topic]
    _ = logomaker.Logo(
        pd.DataFrame(
          pwm,
          columns=["A", "C", "G", "T"]
        ),
        ax = axs[i, j]
      )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("plots/code_table_organoid.pdf")

selected_patterns_organoid = [
      1.1,
      1.2,
      3.0,
      5.0,
      13.1,
      9.2,
      4.2,
      14.0,
      11.1,
      9.1,
#     12.0,
      10.2,
      2.2,
      2.1,
      13.2,
      10.1,
      6.0,
      8.0
]

n_clusters = len(selected_patterns_organoid)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_neural_crest_topics),
  figsize = (len(organoid_neural_crest_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_patterns_organoid)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(organoid_neural_crest_topics):
    if i == 0:
      axs[i, j].set_title(f"Topic {topic}")
    pwm = cluster_to_topic_to_avg_pattern_organoid[cluster][topic]
    _ = logomaker.Logo(
        pd.DataFrame(
          pwm,
          columns=["A", "C", "G", "T"]
        ),
        ax = axs[i, j]
      )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("plots/code_table_organoid.pdf")

def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

data = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid).map(absmax).loc[
      organoid_neural_crest_topics,
      selected_patterns_organoid
    ].T

fig, ax = plt.subplots()
sns.heatmap(
  (data.T / data.sum(1).abs()).T,
  cmap = "bwr",
  vmax = 0.7,
  vmin = -0.7,
  ax = ax
)
fig.tight_layout()
fig.savefig("plots/heatmap_code_table_organoid.pdf")


n_clusters = len(cluster_to_topic_to_avg_pattern_embryo)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(embryo_neural_crest_topics),
  figsize = (len(embryo_neural_crest_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(cluster_to_topic_to_avg_pattern_embryo)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(embryo_neural_crest_topics):
    topic = topic + 90
    if i == 0:
      axs[i, j].set_title(f"Topic {topic}")
    pwm = cluster_to_topic_to_avg_pattern_embryo[cluster][topic]
    _ = logomaker.Logo(
        pd.DataFrame(
          pwm,
          columns=["A", "C", "G", "T"]
        ),
        ax = axs[i, j]
      )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("plots/code_table_embryo.pdf")

data = pd.DataFrame(cluster_to_topic_to_avg_pattern_embryo).map(absmax).loc[
      :,
      selected_patterns_organoid
    ].T

def K(idx):
  return [selected_patterns_organoid.index(x) for x in idx]

embryo_topic_order = data.idxmax().sort_values(key = K).index.to_list()

fig, ax = plt.subplots()
sns.heatmap(
  (data.T / data.sum(1).abs()).T[embryo_topic_order],
  cmap = "bwr",
  vmax = 0.7,
  vmin = -0.7,
  ax = ax
)
fig.tight_layout()
fig.savefig("plots/heatmap_code_table_embryo.pdf")

selected_patterns = [
  3.0,
  13.1,
  9.2,
  14.0,
  11.1,
  9.1,
  10.2,
  2.2,
  2.1,
  13.2
]

organoid_neural_crest_topics = [
  62, 60, 65, 59, 58
]

n_clusters = len(selected_patterns)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_neural_crest_topics),
  figsize = (len(organoid_neural_crest_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_patterns)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(organoid_neural_crest_topics):
    if i == 0:
      axs[i, j].set_title(f"Topic {topic}")
    pwm = cluster_to_topic_to_avg_pattern_organoid[cluster][topic]
    _ = logomaker.Logo(
        pd.DataFrame(
          pwm,
          columns=["A", "C", "G", "T"]
        ),
        ax = axs[i, j]
      )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("plots/code_table_organoid_selected.pdf")

embryo_neural_crest_topics = [
  103, 102, 105, 94, 91, 109,
]

n_clusters = len(selected_patterns)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(embryo_neural_crest_topics),
  figsize = (len(embryo_neural_crest_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_patterns)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(embryo_neural_crest_topics):
    if i == 0:
      axs[i, j].set_title(f"Topic {topic}")
    pwm = cluster_to_topic_to_avg_pattern_embryo[cluster][topic]
    _ = logomaker.Logo(
        pd.DataFrame(
          pwm,
          columns=["A", "C", "G", "T"]
        ),
        ax = axs[i, j]
      )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("plots/code_table_embryo_selected.pdf")

```


```python

import os
import numpy as np
import tensorflow as tf
import crested
from tqdm import tqdm
from crested.utils._seq_utils import one_hot_encode_sequence
import h5py
import pandas as pd
from dataclasses import dataclass
import h5py
from typing import Self
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import logomaker
import modiscolite
from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import torch

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

organoid_neural_crest_topics = [
  1, 3, 4, 5,
  7, 9, 10
]

embryo_neural_crest_topics = [
  1, 4, 7, 8, 9,
  10, 11, 12, 13,
  15, 19, 20, 21,
  22, 29,
]

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_neural_crest_topics):
    topic = topic + 55
    ohs = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["oh"]
    region_names = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            organoid_dl_motif_dir, f"patterns_Topic_{topic}.hdf5",
        ),
        ohs = ohs,
        region_names = region_names
    ):
        pattern_names_dl_organoid.append("organoid_" + name)
        patterns_dl_organoid.append(pattern)

embryo_dl_motif_dir = "../../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_neural_crest_topics):
    topic = topic + 90
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

all_patterns = [*patterns_dl_organoid, *patterns_dl_embryo]
all_pattern_names = [*pattern_names_dl_organoid, *pattern_names_dl_embryo]

pattern_metadata = pd.read_table("pattern_metadata.tsv", index_col = 0)

def merge_and_max(left, right, on, max_on, l):
  global a
  if a:
    print(" "*(l - 1) + "|", end = "\r", flush=True)
    a = False
  print("x", end = "", flush=True)
  x = pd.merge(
    left, right,
    on=on,
    how = "outer"
  )
  x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
  return x.drop([f"{max_on}_x", f"{max_on}_y"], axis = 1).copy()

motifs = {
  n: pattern.ppm[range(*pattern.ic_trim(0.2))].T
  for n, pattern in zip(all_pattern_names, all_patterns)
  if n in pattern_metadata.index
}

def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def merge_and_absmax(left, right, on, max_on, l):
  global a
  if a:
    print(" "*(l - 1) + "|", end = "\r", flush=True)
    a = False
  print("x", end = "", flush=True)
  x = pd.merge(
    left, right,
    on=on,
    how = "outer"
  )
  x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).T.apply(absmax)
  return x.drop([f"{max_on}_x", f"{max_on}_y"], axis = 1).copy()

from functools import reduce

all_hits_organoid_subset = []
for topic in [62, 60, 65, 59, 58]:
  f = f"gradients_Topic_{topic}.npz"
  print(f)
  ohs = np.load(os.path.join(organoid_dl_motif_dir, f))["oh"]
  attr = np.load(os.path.join(organoid_dl_motif_dir, f))["gradients_integrated"]
  region_names = np.load(os.path.join(organoid_dl_motif_dir, f))["region_names"]
  hits = fimo(
    motifs=motifs,
    sequences=ohs.swapaxes(1,2)
  )
  hits = pd.concat(hits)
  hits["attribution"] = extract_signal(
    hits[["sequence_name", "start", "end"]],
    torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
    verbose = True
  ).sum(dim = 1)
  hits["cluster"] = [pattern_metadata.loc[
      m, "cluster_sub_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
  all_hits_organoid_subset.append(hits)

from functools import reduce
a = True
hist_merged_organoid_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "p-value", "-logp"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "p-value", "-logp"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "p-value", "attribution"],
      max_on = "-logp",
      l = len(all_hits_organoid_subset)
    ),
  all_hits_organoid_subset
)

hist_merged_organoid_subset_per_seq_and_cluster_max = hist_merged_organoid_subset \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .apply(absmax) \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0) \
  .astype(float)

hist_merged_organoid_subset_per_seq_and_cluster_max_scaled = hist_merged_organoid_subset_per_seq_and_cluster_max / hist_merged_organoid_subset_per_seq_and_cluster_max.sum()

region_order_organoid_subset = []
for x in tqdm(all_hits_organoid_subset):
  for r in x["sequence_name"]:
    if r not in region_order_organoid_subset:
      region_order_organoid_subset.append(r)

pattern_order_manual = [
  3.0,
  13.1,
  9.2,
  14.0,
  11.1,
  9.1,
  10.2,
  2.2,
  2.1,
  13.2
]

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hist_merged_organoid_subset_per_seq_and_cluster_max_scaled.loc[
    region_order_organoid_subset, pattern_order_manual].astype(float),
  yticklabels = False, xticklabels = True,
  ax = ax,
  cmap = "bwr",
  vmin = -0.0008, vmax = 0.0008
)
fig.tight_layout()
fig.savefig("plots/hits_merged_organoid_per_seq_and_cluster_max.png")

data = (
    hist_merged_organoid_subset_per_seq_and_cluster_max_scaled.loc[
      region_order_organoid_subset, pattern_order_manual
    ].abs() > 0.0008
) * 1
cooc = data.T @ data
for cluster in pattern_order_manual:
  cooc.loc[cluster, cluster] = 0

cooc_perc = (cooc / cooc.sum()).T

import matplotlib
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(pattern_order_manual), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)

cluster_to_color = {
  c: mapper.to_rgba(i)
  for i, c in enumerate(pattern_order_manual)
}

G_organoid = nx.DiGraph()

binom_stats = np.zeros((len(pattern_order_manual), len(pattern_order_manual)))
binom_pvals = np.zeros_like(binom_stats)

fig, ax = plt.subplots()
p = 1 / (len(pattern_order_manual) - 1)
for y, cluster in enumerate(pattern_order_manual):
  left = 0.0
  _sorted_vals = cooc_perc.loc[cluster]
  n = sum(cooc.loc[cluster])
  for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
    k = cooc.loc[cluster, _cluster]
    t = binomtest(k, n, p, alternative = "greater")
    stat, pval = t.statistic, t.pvalue
    binom_stats[y, pattern_order_manual.index(_cluster)] = stat
    binom_pvals[y, pattern_order_manual.index(_cluster)]  = pval
    _ = ax.barh(
      y = y,
      left = left,
      width = _width,
      label = _cluster,
      color = cluster_to_color[_cluster],
      lw = 1, edgecolor = "black"
    )
    if pval < (0.01 / 100) and (_cluster != cluster):
      _ = ax.text(
        x = left + _width / 2, y = y,
        s = f"p=1e{np.round(np.log(pval),2)}", va= "center", ha = "center",
        weight = "bold"
      )
      G_organoid.add_edge(cluster, _cluster, weight = _width)
    left += _width
  if y == 0:
    ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
_ = ax.set_yticks(
  np.arange(len(pattern_order_manual)),
  labels = pattern_order_manual
)
ax.grid(color = "black")
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(10) / 10)
fig.tight_layout()
fig.savefig("plots/perc_other_organoid.png")

pos = nx.forceatlas2_layout(G_organoid, seed = 12)
fig, ax = plt.subplots()
nx.draw(
  G=G_organoid,
  pos=pos,
  ax=ax,
  node_color=[cluster_to_color[n] for n in G_organoid.nodes],
  with_labels=True,
  width=[10**d["weight"] for (_, _, d) in G_organoid.edges(data = True)]
)
fig.tight_layout()
fig.savefig("plots/organoid_graph.pdf")

##


all_hits_embryo_subset = []
for topic in [103, 102, 105, 94, 91, 109]:
  f = f"gradients_Topic_{topic}.npz"
  print(f)
  ohs = np.load(os.path.join(embryo_dl_motif_dir, f))["oh"]
  attr = np.load(os.path.join(embryo_dl_motif_dir, f))["gradients_integrated"]
  region_names = np.load(os.path.join(embryo_dl_motif_dir, f))["region_names"]
  hits = fimo(
    motifs=motifs,
    sequences=ohs.swapaxes(1,2)
  )
  hits = pd.concat(hits)
  hits["attribution"] = extract_signal(
    hits[["sequence_name", "start", "end"]],
    torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
    verbose = True
  ).sum(dim = 1)
  hits["cluster"] = [pattern_metadata.loc[
      m, "cluster_sub_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
  all_hits_embryo_subset.append(hits)

from functools import reduce
a = True
hist_merged_embryo_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "p-value", "-logp"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "p-value", "-logp"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "p-value", "attribution"],
      max_on = "-logp",
      l = len(all_hits_embryo_subset)
    ),
  all_hits_embryo_subset
)

hist_merged_embryo_subset_per_seq_and_cluster_max = hist_merged_embryo_subset.loc[
    hist_merged_embryo_subset.groupby(['sequence_name', 'cluster'])['-logp'].idxmax()
].pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0) \
  .astype(float)

hist_merged_embryo_subset_per_seq_and_cluster_max_scaled = hist_merged_embryo_subset_per_seq_and_cluster_max / hist_merged_embryo_subset_per_seq_and_cluster_max.sum()

region_order_embryo_subset = []
for x in tqdm(all_hits_embryo_subset):
  for r in x["sequence_name"]:
    if r not in region_order_embryo_subset:
      region_order_embryo_subset.append(r)

pattern_order_manual = [
  3.0,
  13.1,
  9.2,
  14.0,
  11.1,
  9.1,
  10.2,
  2.2,
  2.1,
  13.2
]

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hist_merged_embryo_subset_per_seq_and_cluster_max_scaled.loc[
    region_order_embryo_subset, pattern_order_manual].astype(float),
  yticklabels = False, xticklabels = True,
  ax = ax,
  cmap = "bwr",
  vmin = -0.001, vmax = 0.001
)
fig.tight_layout()
fig.savefig("plots/hits_merged_embryo_per_seq_and_cluster_max.png")

data = (
    hist_merged_embryo_subset_per_seq_and_cluster_max_scaled.loc[
      region_order_embryo_subset, pattern_order_manual
    ].abs() > 0.001
) * 1
cooc = data.T @ data
for cluster in pattern_order_manual:
  cooc.loc[cluster, cluster] = 0

cooc_perc = (cooc / cooc.sum()).T

import matplotlib
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(pattern_order_manual), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)

cluster_to_color = {
  c: mapper.to_rgba(i)
  for i, c in enumerate(pattern_order_manual)
}

G_embryo = nx.DiGraph()

binom_stats = np.zeros((len(pattern_order_manual), len(pattern_order_manual)))
binom_pvals = np.zeros_like(binom_stats)

fig, ax = plt.subplots()
p = 1 / (len(pattern_order_manual) - 1)
for y, cluster in enumerate(pattern_order_manual):
  left = 0.0
  _sorted_vals = cooc_perc.loc[cluster]
  n = sum(cooc.loc[cluster])
  for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
    k = cooc.loc[cluster, _cluster]
    t = binomtest(k, n, p, alternative = "greater")
    stat, pval = t.statistic, t.pvalue
    binom_stats[y, pattern_order_manual.index(_cluster)] = stat
    binom_pvals[y, pattern_order_manual.index(_cluster)]  = pval
    _ = ax.barh(
      y = y,
      left = left,
      width = _width,
      label = _cluster,
      color = cluster_to_color[_cluster],
      lw = 1, edgecolor = "black"
    )
    if pval < (0.01 / 100) and (_cluster != cluster):
      _ = ax.text(
        x = left + _width / 2, y = y,
        s = f"p=1e{np.round(np.log(pval),2)}", va= "center", ha = "center",
        weight = "bold"
      )
      G_embryo.add_edge(cluster, _cluster, weight = _width)
    left += _width
  if y == 0:
    ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
_ = ax.set_yticks(
  np.arange(len(pattern_order_manual)),
  labels = pattern_order_manual
)
ax.grid(color = "black")
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(10) / 10)
fig.tight_layout()
fig.savefig("plots/perc_other_embryo.png")

pos = nx.forceatlas2_layout(G_embryo, seed = 12)
fig, ax = plt.subplots()
nx.draw(
  G=G_embryo,
  pos=pos,
  ax=ax,
  node_color=[cluster_to_color[n] for n in G_embryo.nodes],
  with_labels=True,
  width=[10**d["weight"] for (_, _, d) in G_embryo.edges(data = True)]
)
fig.tight_layout()
fig.savefig("plots/embryo_graph.pdf")


import pysam
hg38 = pysam.FastaFile("/data/projects/c20/sdewin/resources/hg38/hg38.fa")

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

nuc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("nuc", list(letter_to_color.values()), 4)

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


hits_organoid_non_overlap = hist_merged_organoid_subset \
  .groupby("sequence_name") \
  .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp")) \
  .reset_index(drop = True)

def dna_ansi(nuc):
  u_nuc = nuc.upper()
  if u_nuc == "A":
    return f"\033[0;32m{nuc}"
  if u_nuc == "C":
    return f"\033[0;34m{nuc}"
  if u_nuc == "G":
    return f"\033[1;33m{nuc}"
  if u_nuc == "T":
    return f"\033[0;36m{nuc}"

pattern_order = [
  "organoid_patterns_Topic_58_neg_pattern_1",
  "embryo_patterns_Topic_91_neg_pattern_3",
  "embryo_patterns_Topic_97_neg_pattern_1",
  "embryo_patterns_Topic_101_neg_pattern_0",
  "embryo_patterns_Topic_109_neg_pattern_2",
  "organoid_patterns_Topic_60_pos_pattern_0",
  "organoid_patterns_Topic_65_pos_pattern_2",
  "embryo_patterns_Topic_105_pos_pattern_0",
  "organoid_patterns_Topic_62_neg_pattern_1",
  #
  #"organoid_patterns_Topic_59_pos_pattern_0",
  "embryo_patterns_Topic_112_pos_pattern_0",
  "embryo_patterns_Topic_94_pos_pattern_1",
  "embryo_patterns_Topic_94_pos_pattern_0",
  "embryo_patterns_Topic_94_pos_pattern_2"
]
import pyBigWig
bw_topic_60 = pyBigWig.open(
  f"../../data_prep_new/organoid_data/ATAC/bw_per_topic/{model_index_to_topic_name_organoid(59).replace('Topic_', 'Topic')}.fragments.tsv.bigWig"
)
bw_topic_59 = pyBigWig.open(
  f"../../data_prep_new/organoid_data/ATAC/bw_per_topic/{model_index_to_topic_name_organoid(58).replace('Topic_', 'Topic')}.fragments.tsv.bigWig"
)
pattern_seq = []
cov = []
target_size = 25
patterns = []
for _, hit in tqdm(
    hits_organoid_non_overlap.query("cluster == 2.2").set_index("motif_name").loc[pattern_order].reset_index().iterrows(),
    total = len(hits_organoid_non_overlap.query("cluster == 2.2"))
):
  chrom, start, end = hit.sequence_name.replace(":", "-").split("-")
  seq = hg38.fetch(chrom, int(start), int(end))
  topic_60_cov = bw_topic_60.stats(chrom, int(start), int(end))
  topic_59_cov = bw_topic_59.stats(chrom, int(start), int(end))
  h_start, h_end = int(hit.start), int(hit.end)
  if hit.strand == "-":
    seq = reverse_complement(seq)
    h_start = 500 - h_start
    h_end = 500 - h_end
  meta = pattern_metadata.loc[hit.motif_name]
  if meta.is_rc_to_root:
    seq = reverse_complement(seq)
    h_start = 500 - h_start
    h_end = 500 - h_end
  if not meta.is_rc_to_root:
    to_add_start = (all_patterns[all_pattern_names.index(hit.motif_name)].ic_trim(0.2)[0] + meta.offset_to_root)
  else:
    to_add_start = (30 - all_patterns[all_pattern_names.index(hit.motif_name)].ic_trim(0.2)[1] + meta.offset_to_root)
  if to_add_start > min(h_start, h_end):
    continue
  total_size = max(h_start, h_end) - (min(h_start, h_end) - to_add_start)
  to_add_end = target_size - total_size
  if max(h_start, h_end) + to_add_end >= 500:
    continue
  s = "".join(list(seq)[min(h_start, h_end) - to_add_start: max(h_start, h_end) + to_add_end])
  if len(s) != target_size:
    raise ValueError()
  #s = "".join([dna_ansi(n) for n in s]) + "\033[0m"
  #print(f"{str(h_start):<5}{str(h_end):<5}{str(to_add_start):<5}{s:<3}")
  pattern_seq.append(s)
  cov.append([topic_60_cov, topic_59_cov])
  patterns.append(hit.motif_name)
#
window = 500
D = np.array([[letter_to_val[nuc.upper()] for nuc in seq] for seq in pattern_seq])
fig, axs = plt.subplots(figsize = (6, 8), ncols = 2, width_ratios = [4, 1], sharey = True)
sns.heatmap(
    D,
    cmap = nuc_cmap,
    ax = axs[0],
    cbar = False,
    yticklabels = False
  )
_ = axs[0].set_xticks(np.arange(D.shape[1]) + 0.6, labels = [x if x % 2 else "" for x in np.arange(D.shape[1]) + 1])
a = np.lib.stride_tricks.sliding_window_view(
    np.nan_to_num(np.array(cov).astype(float))[:, 0, :].squeeze(), window).mean(axis = -1)
b = np.lib.stride_tricks.sliding_window_view(
    np.nan_to_num(np.array(cov).astype(float))[:, 1, :].squeeze(), window).mean(axis = -1)
axs[1].plot(
  (a - a.min()) / (a.max() - a.min()),
  np.arange(D.shape[0] - window + 1),
  color = "#D64933",
  label = "Topic 60 acc."
)
axs[1].plot(
  (b - b.min()) / (b.max() - b.min()),
  np.arange(D.shape[0] - window + 1),
  color = "#0C7C59",
  label = "Topic 59 acc."
)
_ = axs[1].set_xlim(0, 1)
_ = axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0], labels = [0, "", "", "", 1])
_ = axs[1].set_yticks(np.arange(0, D.shape[0], window), labels = [])
_ = axs[1].legend()
axs[1].grid(True)
fig.tight_layout()
fig.savefig("plots/SOX_hits.png")

data = pd.DataFrame(data = dict(pattern = patterns, topic_60 = np.nan_to_num(np.array(cov).astype(float))[:, 0, :].squeeze(), topic_59 = np.nan_to_num(np.array(cov).astype(float))[:, 1, :].squeeze()))
data["topic_60"] = data["topic_60"] / data["topic_60"].max()
data["topic_59"] = data["topic_59"] / data["topic_59"].max()
data["ratio"] = ((data["topic_59"] + 1) / (data["topic_60"] + 1))

data = data.melt(id_vars=["pattern"]).rename({"variable": "Topic", "value": "Accessiblity"}, axis = 1)

dimers = [
  "embryo_patterns_Topic_112_pos_pattern_0",
  "embryo_patterns_Topic_94_pos_pattern_1",
  "embryo_patterns_Topic_94_pos_pattern_0",
  "embryo_patterns_Topic_94_pos_pattern_2"
]

fig, ax = plt.subplots()
sns.boxplot(
  data.query("Topic == 'ratio'"),
  y = "pattern",
  x = "Accessiblity",
  ax = ax
)
ax.axvline(1, color = "black")
fig.tight_layout()
fig.savefig("plots/bxplot_sox_hits.pdf")






```
