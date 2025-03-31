

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

organoid_neuron_topics = [
  1,
  2,
  3,
  4,
  6,
  8,
  10,
  11,
  12,
  13,
  15,
  16,
  18,
  19,
  23,
  24,
  25
]

embryo_neuron_topics = [
  1,
  3,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  15,
  17,
  18,
  19,
  22,
  24,
  26,
  27,
  29,
  30
]

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_neuron_topics):
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
for topic in tqdm(embryo_neuron_topics):
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

clusters = hierarchy.fcluster(row_linkage, t = 8, criterion = "maxclust")
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
  1: 3,
  2: 2,
  3: 3,
  5: 3,
  7: 5,
}
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


organoid_neuron_topics = [
  1,
  2,
  3,
  4,
  6,
  8,
  10,
  11,
  12,
  13,
  15,
  16,
  18,
  19,
  23,
  24,
  25
]

embryo_neuron_topics = [
  1,
  3,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  15,
  17,
  18,
  19,
  22,
  24,
  26,
  27,
  29,
  30
]

organoid_dl_motif_dir = "../../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_neuron_topics):
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
for topic in tqdm(embryo_neuron_topics):
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

all_patterns = [*patterns_dl_organoid, *patterns_dl_embryo]
all_pattern_names = [*pattern_names_dl_organoid, *pattern_names_dl_embryo]

pattern_metadata = pd.read_table("pattern_metadata.tsv", index_col = 0)

selected_clusters = [
  1.1,
  1.3,
  2.1,
  2.2,
  3.1,
  4.0,
  5.1,
  5.2,
  5.3,
  6.0,
  7.3,
  7.5,
  8.0
]

pattern_to_topic_to_grad_organoid = {}
for pattern_name in tqdm(pattern_metadata.query("cluster_sub_cluster in @selected_clusters").index):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad_organoid[pattern_name] = {}
  for topic in tqdm(organoid_neuron_topics, leave = False):
    class_idx = topic - 1
    explainer = Explainer(model = organoid_model, class_index = int(class_idx))
    #gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
    gradients_integrated = explainer.saliency_maps(X = oh_sequences)
    pattern_grads = list(get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze()))
    pattern_to_topic_to_grad_organoid[pattern_name][topic] = (pattern_grads, pattern_ohs)

pattern_to_topic_to_grad_embryo = {}
for pattern_name in tqdm(pattern_metadata.query("cluster_sub_cluster in @selected_clusters").index):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad_embryo[pattern_name] = {}
  for topic in tqdm(embryo_neuron_topics, leave = False):
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
for cluster in set(selected_clusters):
  cluster_to_topic_to_avg_pattern_organoid[cluster] = {}
  for topic in organoid_neuron_topics:
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
for cluster in set(selected_clusters):
  cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
  for topic in embryo_neuron_topics:
    P, O = allign_patterns_of_cluster_for_topic(
        pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
        pattern_metadata=pattern_metadata,
        cluster_id=cluster,
        topic=topic
      )
    cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (
      P * O
    ).mean(0)

organoid_neuron_topics_selected = [
  6, 4, 23, 24, 13, 2
]

def K(idx):
  return [organoid_neuron_topics_selected.index(x) for x in idx]

pattern_order = pd.DataFrame(cluster_to_topic_to_avg_pattern_organoid).loc[organoid_neuron_topics_selected].applymap(np.mean).idxmax().reset_index().set_index(0).sort_index(key = K)["index"].to_list()

n_clusters = len(cluster_to_topic_to_avg_pattern_organoid)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_neuron_topics_selected),
  figsize = (len(organoid_neuron_topics_selected) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(pattern_order)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(organoid_neuron_topics_selected):
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


embryo_neuron_topics_selected = [
  10, 8, 13, 24, 18, 29
]

n_clusters = len(cluster_to_topic_to_avg_pattern_embryo)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(embryo_neuron_topics_selected),
  figsize = (len(embryo_neuron_topics_selected) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(pattern_order)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(embryo_neuron_topics_selected):
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


```

