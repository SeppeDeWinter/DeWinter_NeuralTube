

```python

import numpy as np

organoid_dv_topics = np.array(
  [
    8,
    16,
    13,
    9,
    11,
    19,
    25,
    1,
    29,
    23,
    3
  ]
) + 25

embryo_dv_topics = np.array(
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

import h5py
import modiscolite
from tqdm import tqdm
import os

def trim_by_ic(ic, min_v):
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
                    ppm=ppm, background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
                )
                start, stop = trim_by_ic(ic, ic_thr)
                if stop - start <= 1:
                    continue
                if ic[start:stop].mean() < avg_ic_thr:
                    continue
                yield (
                    filename.split("/")[-1].rsplit(".", 1)[0]
                    + "_"
                    + pos_neg.split("_")[0]
                    + "_"
                    + pattern,
                    pos_neg == "pos_patterns",
                    ppm[start:stop],
                    ic[start:stop],
                    (start, stop)
                )

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


organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

all_motifs_dl_organoid = []
motif_names_dl_organoid = []
is_motif_pos_dl_organoid = []
ic_motifs_dl_organoid = []
ic_start_stop_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
    for name, is_pos, ppm, ic, ic_start_stop in load_motif_from_modisco(
        filename=os.path.join(
            organoid_dl_motif_dir, f"patterns_Topic_{topic}.hdf5"
        ),
        ic_thr=0.2,
        avg_ic_thr=0.5,
    ):
        all_motifs_dl_organoid.append(ppm)
        motif_names_dl_organoid.append("organoid_" + name)
        is_motif_pos_dl_organoid.append(is_pos)
        ic_motifs_dl_organoid.append(ic)
        ic_start_stop_dl_organoid.append(ic_start_stop)

all_motifs_dl_embryo = []
motif_names_dl_embryo = []
is_motif_pos_dl_embryo = []
ic_motifs_dl_embryo = []
ic_start_stop_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
    for name, is_pos, ppm, ic, ic_start_stop in load_motif_from_modisco(
        filename=os.path.join(embryo_dl_motif_dir, f"patterns_Topic_{topic}.hdf5"),
        ic_thr=0.2,
        avg_ic_thr=0.5,
    ):
        all_motifs_dl_embryo.append(ppm)
        motif_names_dl_embryo.append("embryo_" + name)
        is_motif_pos_dl_embryo.append(is_pos)
        ic_motifs_dl_embryo.append(ic)
        ic_start_stop_dl_embryo.append(ic_start_stop)

from tangermeme.tools.tomtom import tomtom
import scanpy as sc
import torch
import pandas as pd

motif_metadata = pd.DataFrame(
  index = [*motif_names_dl_organoid, *motif_names_dl_embryo],
  data = dict(
    is_pos = [*is_motif_pos_dl_organoid, *is_motif_pos_dl_embryo],
    mean_ic = [np.mean(x) for x in [*ic_motifs_dl_organoid, *ic_motifs_dl_embryo]],
  )
)

motif_metadata["topic"] = [
  x.rsplit("_", 3)[0] for x in motif_metadata.index
]

all_motifs = [*all_motifs_dl_organoid, *all_motifs_dl_embryo]

t_all_motifs = [torch.from_numpy(m).T for m in tqdm(all_motifs)]

all_motifs_collection = []
cb_dir = "/data/projects/c20/sdewin/PhD/motif_collection/cluster_buster"
for f in tqdm(os.listdir(cb_dir), total = len(os.listdir(cb_dir))):
  m, _ = load_motif(f.replace(".cb", ""), cb_dir)
  all_motifs_collection.extend(m)

t_all_motifs_collection = [torch.from_numpy(m).T for m in tqdm(all_motifs_collection)]

pvals, scores, offsets, overlaps, strands = tomtom(
    t_all_motifs, t_all_motifs_collection
)

evals = pvals.numpy() * len(all_motifs)
adata_motifs = sc.AnnData(evals, obs=motif_metadata)

sc.settings.figdir = os.getcwd()

sc.tl.pca(adata_motifs)

sc.pp.neighbors(adata_motifs, use_rep = "X")
sc.tl.tsne(adata_motifs, use_rep = "X")
sc.tl.leiden(adata_motifs, resolution=3)

sc.pl.pca(
    adata_motifs,
    color=["leiden"],
    save="_leiden_motifs.pdf",
)

sc.pl.tsne(
    adata_motifs,
    color=["leiden", "topic"],
    save="_leiden_motifs.pdf",
)

motifs_w_logo = adata_motifs.obs.copy()

motifs_w_logo["Logo"] = [
    f'<img src="file:///Users/u0138640/data_core/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_2/draft/deep_learning_motifs_png/{m}.png" width="200" >'
    for m in motifs_w_logo.index
]

motifs_w_logo["leiden"] = [f"cluster_{x}" for x in motifs_w_logo["leiden"]]

motifs_w_logo.reset_index().set_index("Logo").sort_values("leiden").to_html(
    "leiden_motifs.html", escape=False, col_space=80
)


import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.cluster import hierarchy

dat = 1 - adata_motifs.to_df().T.corr().to_numpy()

row_linkage = hierarchy.linkage(
    distance.pdist(dat), method='average')

col_linkage = hierarchy.linkage(
    distance.pdist(dat.T), method='average')

clusters = hierarchy.fcluster(row_linkage, t = 20, criterion = "maxclust")

fig = sns.clustermap(
  1 - adata_motifs.to_df().T.corr().to_numpy(),
  vmin = 0, vmax = 0.6,
  cmap = "rainbow_r",
  row_colors = [plt.cm.tab20(x) for x in clusters],
  col_colors = [plt.cm.tab20(x) for x in clusters]
)
fig.savefig("motif_distance_matrix.png")

motifs_w_logo = adata_motifs.obs.copy()

motifs_w_logo["hier_cluster"] = clusters

motifs_w_logo["Logo"] = [
    f'<img src="file:///Users/u0138640/data_core/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/figure_2/draft/deep_learning_motifs_png/{m}.png" width="200" >'
    for m in motifs_w_logo.index
]

motifs_w_logo.reset_index().set_index("Logo").sort_values("hier_cluster").to_html(
    "hier_clust_motifs.html", escape=False, col_space=80
)


import logomaker

motif_metadata = adata_motifs.obs.copy()
motif_metadata["hier_cluster"] = clusters

ic_all_motifs = [*ic_motifs_dl_organoid, *ic_motifs_dl_embryo]
ic_start_stop_all_motif = [*ic_start_stop_dl_organoid, *ic_start_stop_dl_embryo]
all_motif_names = [*motif_names_dl_organoid, *motif_names_dl_embryo]

motif_metadata["ic_start"] = [x[0] for x in ic_start_stop_all_motif]
motif_metadata["ic_stop"] = [x[1] for x in ic_start_stop_all_motif]
motif_metadata["is_root_motif"] = False
motif_metadata["is_rc_to_root"] = False
motif_metadata["offset_to_root"] = 0

for cluster in set(clusters):
  cluster_idc = np.where(clusters == cluster)[0]
  print(f"cluster: {cluster}")
  _, _, o, _, s = tomtom(
      [t_all_motifs[m] for m in cluster_idc],
      [t_all_motifs[m] for m in cluster_idc]
  )
  fig, axs = plt.subplots(nrows = sum(clusters == cluster), figsize = (4, 2 * sum(clusters == cluster)), sharex = True)
  pwms_aligned = []
  if sum(clusters == cluster) == 1:
    axs = [axs]
  for i, m in enumerate(cluster_idc):
    print(i)
    rel = np.argmax([np.mean(ic_all_motifs[m]) for m in cluster_idc])
    motif_metadata.loc[all_motif_names[cluster_idc[rel]], "is_root_motif"] = True
    is_rc = s[i, rel] == 1
    offset = int(o[i, rel].numpy())
    pwm = t_all_motifs[m].T.numpy()
    ic = ic_all_motifs[m][:, None]
    if is_rc:
      pwm = pwm[::-1, ::-1]
      ic = ic[::-1, ::-1]
      offset = t_all_motifs[cluster_idc[rel]].shape[1] - pwm.shape[0] - offset
    motif_metadata.loc[all_motif_names[m], "is_rc_to_root"] = bool(is_rc)
    motif_metadata.loc[all_motif_names[m], "offset_to_root"] = offset
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
  fig.savefig(f"cluster_{cluster}_patterns_aligned.pdf")
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
  fig.savefig(f"cluster_{cluster}_avg_patterns_aligned.pdf")

#manual for cluster 10 and 11

motif_metadata.loc[motif_metadata["hier_cluster"] == 10, "is_root_motif"] = False
motif_metadata.loc[motif_metadata["hier_cluster"] == 10, "is_rc_to_root"] = False
motif_metadata.loc[motif_metadata["hier_cluster"] == 10, "offset_to_root"] = 0

motif_metadata.loc[motif_metadata["hier_cluster"] == 11, "is_root_motif"] = False
motif_metadata.loc[motif_metadata["hier_cluster"] == 11, "is_rc_to_root"] = False
motif_metadata.loc[motif_metadata["hier_cluster"] == 11, "offset_to_root"] = 0

cluster = 10
cluster_idc = np.where(clusters == cluster)[0]
print(f"cluster: {cluster}")
_, _, o, _, s = tomtom(
    [t_all_motifs[m] for m in cluster_idc],
    [t_all_motifs[m] for m in cluster_idc]
)
fig, axs = plt.subplots(nrows = sum(clusters == cluster), figsize = (4, 2 * sum(clusters == cluster)), sharex = True)
pwms_aligned = []
if sum(clusters == cluster) == 1:
  axs = [axs]
for i, m in enumerate(cluster_idc):
  print(i)
  rel = 1 #np.argmax([np.mean(ic_all_motifs[m]) for m in cluster_idc])
  motif_metadata.loc[all_motif_names[cluster_idc[rel]], "is_root_motif"] = True
  is_rc = s[i, rel] == 1
  offset = int(o[i, rel].numpy())
  pwm = t_all_motifs[m].T.numpy()
  ic = ic_all_motifs[m][:, None]
  if is_rc:
    pwm = pwm[::-1, ::-1]
    ic = ic[::-1, ::-1]
    offset = t_all_motifs[cluster_idc[rel]].shape[1] - pwm.shape[0] - offset
  motif_metadata.loc[all_motif_names[m], "is_rc_to_root"] = bool(is_rc)
  motif_metadata.loc[all_motif_names[m], "offset_to_root"] = offset
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
fig.savefig(f"cluster_{cluster}_patterns_aligned.pdf")

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
fig.savefig(f"cluster_{cluster}_avg_patterns_aligned.pdf")

cluster = 11
cluster_idc = np.where(clusters == cluster)[0]
print(f"cluster: {cluster}")
_, _, o, _, s = tomtom(
    [t_all_motifs[m] for m in cluster_idc],
    [t_all_motifs[m] for m in cluster_idc]
)
fig, axs = plt.subplots(nrows = sum(clusters == cluster), figsize = (4, 2 * sum(clusters == cluster)), sharex = True)
pwms_aligned = []
if sum(clusters == cluster) == 1:
  axs = [axs]
for i, m in enumerate(cluster_idc):
  print(i)
  rel = 0 #np.argmax([np.mean(ic_all_motifs[m]) for m in cluster_idc])
  motif_metadata.loc[all_motif_names[cluster_idc[rel]], "is_root_motif"] = True
  is_rc = s[i, rel] == 1
  offset = int(o[i, rel].numpy())
  pwm = t_all_motifs[m].T.numpy()
  ic = ic_all_motifs[m][:, None]
  if is_rc:
    pwm = pwm[::-1, ::-1]
    ic = ic[::-1, ::-1]
    offset = t_all_motifs[cluster_idc[rel]].shape[1] - pwm.shape[0] - offset
  motif_metadata.loc[all_motif_names[m], "is_rc_to_root"] = bool(is_rc)
  motif_metadata.loc[all_motif_names[m], "offset_to_root"] = offset
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
fig.savefig(f"cluster_{cluster}_patterns_aligned.pdf")

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
fig.savefig(f"cluster_{cluster}_avg_patterns_aligned.pdf")

motif_metadata.to_csv("motif_metadata.tsv", sep = "\t", header = True, index = True)


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

path_to_organoid_model = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/global_model/model_20230216"

organoid_model = tf.keras.models.model_from_json(
  open(os.path.join(path_to_organoid_model, "model.json")).read(),
  custom_objects = {"Functional": tf.keras.models.Model}
)

organoid_model.load_weights(os.path.join(path_to_organoid_model, "model_epoch_23.hdf5"))

##

path_to_embryo_model = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/model_20241213"

embryo_model = tf.keras.models.model_from_json(
  open(os.path.join(path_to_embryo_model, "model.json")).read(),
  custom_objects = {"Functional": tf.keras.models.Model}
)

embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))

organoid_dv_topics = np.array(
  [
    8,
    16,
    13,
    9,
    11,
    19,
    25,
    1,
    29,
    23,
    3
  ]
) + 25

embryo_dv_topics = np.array(
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

from dataclasses import dataclass
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

organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
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

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
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

pattern_metadata = pd.read_table("motif_metadata.tsv", index_col = 0)

topic_6_patterns = pattern_metadata.query("hier_cluster == 6").index.to_list()

pattern = all_patterns[all_pattern_names.index(topic_6_patterns[0])]
oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]).astype(np.int8)

topic_to_grad = {}
for topic in tqdm(organoid_dv_topics):
  class_idx = topic - 1
  explainer = Explainer(model = organoid_model, class_index = int(class_idx))
  gradients_integrated = explainer.integrated_grad(X = oh_sequences)
  topic_to_grad[topic] = gradients_integrated

import logomaker
import matplotlib.pyplot as plt

fig, axs = plt.subplots(
  figsize = (4 * len(topic_to_grad.keys()), 2),
  ncols = len(topic_to_grad.keys()),
  sharey = True
)
for topic, ax in zip(organoid_dv_topics, axs):
  print(topic)
  pattern_grads = list(get_value_seqlets(pattern.seqlets, topic_to_grad[topic].squeeze()))
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  _ = logomaker.Logo(
    pd.DataFrame(
      (np.array(pattern_grads) * np.array(pattern_ohs)).mean(0).astype(float),
      columns=["A", "C", "G", "T"]
    ),
    ax = ax
  )
  ax.set_ylim((0, 0.05))
fig.tight_layout()
fig.savefig("test.png")


pattern_to_topic_to_grad = {}
for pattern_name in tqdm(topic_6_patterns):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad[pattern_name] = {}
  for topic in tqdm(organoid_dv_topics, leave = False):
    class_idx = topic - 1
    explainer = Explainer(model = organoid_model, class_index = int(class_idx))
    #gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
    gradients_integrated = explainer.saliency_maps(X = oh_sequences)
    pattern_grads = list(get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze()))
    pattern_to_topic_to_grad[pattern_name][topic] = (pattern_grads, pattern_ohs)


fig, axs = plt.subplots(
  figsize = (len(organoid_dv_topics) * 4, 2 * len(topic_6_patterns)),
  nrows = len(topic_6_patterns), ncols = len(organoid_dv_topics),
  sharex=True
)

for i, pattern_name in enumerate(tqdm(topic_6_patterns)):
  for j, topic in enumerate(organoid_dv_topics):
    pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
    pwm = (np.array(pattern_grads) * np.array(pattern_ohs)).mean(0).astype(float)
    ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
      pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
    ]
    pwm_ic = pwm[ic_start: ic_end]
    if is_rc_to_root:
      pwm_ic = pwm_ic[::-1, ::-1] 
    if offset_to_root > 0:
      pwm_ic = np.concatenate([pwm[ic_start - offset_to_root: ic_start], pwm_ic])
    elif offset_to_root < 0:
      pwm_ic = pwm_ic[abs(offset_to_root):, :] 
    _ = logomaker.Logo(
      pd.DataFrame(
        pwm_ic,
        columns=["A", "C", "G", "T"]
      ),
      ax = axs[i, j]
    )
    _ = axs[i, j].set_ylim(0, 0.15)
fig.tight_layout()
fig.savefig("test.pdf")

topic_to_patterns = {}
for topic in organoid_dv_topics:
  P = []
  O = []
  for pattern_name in topic_6_patterns:
    pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
    ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
      pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
    ]
    pattern_grads_ic = [p[ic_start: ic_end] for p in pattern_grads]
    pattern_ohs_ic = [o[ic_start: ic_end] for o in pattern_ohs]
    if is_rc_to_root:
      pattern_grads_ic = [p[::-1, ::-1] for p in pattern_grads_ic]
      pattern_ohs_ic = [o[::-1, ::-1] for o in pattern_ohs_ic]
      pwm_ic = pwm_ic[::-1, ::-1] 
    if offset_to_root > 0:
      if is_rc_to_root:
        pattern_grads_ic = [
          np.concatenate(
            [
              p[ic_end: ic_end + offset_to_root][::-1, ::-1],
              p_ic
            ]
          )
          for p, p_ic in zip(pattern_grads, pattern_grads_ic)
        ]
        pattern_ohs_ic = [
          np.concatenate(
            [
              o[ic_end: ic_end + offset_to_root][::-1, ::-1],
              o_ic
            ]
          )
          for o, o_ic in zip(pattern_ohs, pattern_ohs_ic)
        ]
      else:
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
  topic_to_patterns[topic] = (
    np.array(P), np.array(O)
  )

fig, axs = plt.subplots(figsize = (4 * len(organoid_dv_topics), 2), ncols = len(organoid_dv_topics))
for i, topic in enumerate(organoid_dv_topics):
  pwm = (topic_to_patterns[topic][0] * topic_to_patterns[topic][1]).mean(0)
  _ = logomaker.Logo(
      pd.DataFrame(
        pwm,
        columns=["A", "C", "G", "T"]
      ),
      ax = axs[i]
    )
  _ = axs[i].set_ylim(0, 0.15)
fig.tight_layout()
fig.savefig("test.pdf")

pattern_to_topic_to_grad_organoid = {}
for pattern_name in tqdm(pattern_metadata.index):
  pattern = all_patterns[all_pattern_names.index(pattern_name)]
  oh_sequences = np.array([x.region_one_hot for x in pattern.seqlets]) #.astype(np.int8)
  pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
  pattern_to_topic_to_grad_organoid[pattern_name] = {}
  for topic in tqdm(organoid_dv_topics, leave = False):
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
  for topic in tqdm(embryo_dv_topics, leave = False):
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
  cluster_patterns = pattern_metadata.query("hier_cluster == @cluster_id").index.to_list()
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

cluster_to_topic_to_avg_pattern_organoid = {}
for cluster in set(pattern_metadata["hier_cluster"]):
  cluster_to_topic_to_avg_pattern_organoid[cluster] = {}
  for topic in organoid_dv_topics:
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
for cluster in set(pattern_metadata["hier_cluster"]):
  cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
  for topic in embryo_dv_topics:
    P, O = allign_patterns_of_cluster_for_topic(
        pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
        pattern_metadata=pattern_metadata,
        cluster_id=cluster,
        topic=topic
      )
    cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (
      P * O
    ).mean(0)


n_clusters = len(set(pattern_metadata["hier_cluster"]))
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_dv_topics),
  figsize = (len(organoid_dv_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(set(pattern_metadata["hier_cluster"]))):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(organoid_dv_topics):
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
fig.savefig("code_table_organoid.pdf")

n_clusters = len(set(pattern_metadata["hier_cluster"]))
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(embryo_dv_topics),
  figsize = (len(embryo_dv_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(set(pattern_metadata["hier_cluster"]))):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(embryo_dv_topics):
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
fig.savefig("code_table_embryo.pdf")



selected_clusters = [
  6, 7, 12, 15, 10, 18, 2, 17, 1, 19
]
selected_topics = [
  33, 41, 38, 36, 26, 54, 48, 28
]

n_clusters = len(selected_clusters)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(selected_topics),
  figsize = (len(selected_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_clusters)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(selected_topics):
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
fig.savefig("code_table_organoid_selected.pdf")

selected_clusters = [
  6, 7, 12, 15, 10, 18, 2, 17, 1, 19
]
selected_topics = embryo_dv_topics

n_clusters = len(selected_clusters)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(selected_topics),
  figsize = (len(selected_topics) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_clusters)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, topic in enumerate(selected_topics):
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
fig.savefig("code_table_embryo_selected.pdf")

organoid_embryo = [
  (33, 34),
  (None, 74),
  (41, None),
  (None, 87),
  (38, 38),
  (36, 79),
  (54, 88),
  (48, 58),
  (28, None)
]

n_clusters = len(selected_clusters)
fig, axs = plt.subplots(
  nrows = n_clusters,
  ncols = len(organoid_embryo),
  figsize = (len(organoid_embryo) * 4, n_clusters * 2)
)
for i, cluster in enumerate(tqdm(selected_clusters)):
  YMIN = np.inf
  YMAX = -np.inf
  for j, (topic_org, topic_embr) in enumerate(organoid_embryo):
    if i == 0:
      axs[i, j].set_title(f"Topic {topic_org} {topic_embr}")
    if topic_org is not None:
      pwm = cluster_to_topic_to_avg_pattern_organoid[cluster][topic_org]
      _ = logomaker.Logo(
          pd.DataFrame(
            pwm,
            columns=["A", "C", "G", "T"]
          ),
          ax = axs[i, j],
          alpha = 0.5,
          color_scheme = "classic"
        )
    if topic_embr is not None:
      pwm = cluster_to_topic_to_avg_pattern_embryo[cluster][topic_embr]
      _ = logomaker.Logo(
          pd.DataFrame(
            pwm,
            columns=["A", "C", "G", "T"]
          ),
          ax = axs[i, j],
          alpha = 0.5,
          color_scheme = "grays"
        )
    ymn, ymx = axs[i, j].get_ylim()
    YMIN = min(ymn, YMIN)
    YMAX = max(ymx, YMAX)
  _ = axs[i, 0].set_ylabel(f"cluster_{cluster}")
  for ax in axs[i, :]:
    _ = ax.set_ylim(YMIN, YMAX)
fig.tight_layout()
fig.savefig("code_table_embryo_organoid_selected.pdf")


```

```python

from dataclasses import dataclass
from typing import Self
from tangermeme.tools.fimo import fimo
import numpy as np
import h5py
from tqdm import tqdm
import os
import pandas as pd
from tangermeme.utils import extract_signal
import torch
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib
from scipy.stats import binomtest
import networkx as nx

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
    self.subpatterns  = [
      ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]

  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"

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

organoid_dv_topics = np.array(
  [
    8,
    16,
    13,
    9,
    11,
    19,
    25,
    1,
    29,
    23,
    3
  ]
) + 25

embryo_dv_topics = np.array(
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

organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
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

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
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

pattern_metadata = pd.read_table("motif_metadata.tsv", index_col = 0)

## test --> jump to mark c
ohs = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_33.npz"))["oh"]
attr = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_33.npz"))["gradients_integrated"]
motifs = {
  n: pattern.ppm[pattern_metadata.loc[n].ic_start: pattern_metadata.loc[n].ic_stop].T
  for n, pattern in zip(all_pattern_names, all_patterns)
  if n in pattern_metadata.index
}

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
    m, "hier_cluster"
  ]
  for m in hits["motif_name"]
]

fig, ax = plt.subplots()
_ = ax.hist(hits.query("cluster == 12")["attribution"], bins = 500, label = 12, color = "black")
ax.axvline(0.035, color = "red")
fig.savefig("test.png")


fig, ax = plt.subplots()
sorted_idx = np.argsort(hits.query("cluster == 6")["p-value"])[::-1]
_ = ax.scatter(
  x = hits.query("cluster == 6")["start"],
  y = hits.query("cluster == 6")["attribution"],
  s = 3,
  c = hits.query("cluster == 6")["p-value"]
)
ax.set_xlabel("start position of motif instance")
ax.set_ylabel("FOX attribution score")
fig.savefig("attr_v_start.png")

sequences_two = []
for seq in set(hits["sequence_name"]):
  if 6 in hits.query("sequence_name == @seq")["cluster"].to_list() and \
    12 in hits.query("sequence_name == @seq")["cluster"].to_list():
    sequences_two.append(seq)

distances = []
attributions = []
for seq in sequences_two:
  t = np.array(hits.query("sequence_name == @seq & cluster == 6")[["start", "attribution"]])
  a = t[:, 0]
  attributions.extend(t[:, 1])
  b = np.array(hits.query("sequence_name == @seq & cluster == 12")["start"])
  d = np.zeros((len(a), len(b)))
  for i in range(len(a)):
    for j in range(len(b)):
      d[i, j] = abs(a[i] - b[j])
  distances.extend(d.min(1))

fig, ax = plt.subplots()
ax.scatter(
  x = distances,
  y = attributions,
  s = 2
)
ax.set_xlabel("abs distances between FOX and RFX")
ax.set_ylabel("FOX attribution score")
fig.savefig("attr_v_rfx.png")

distances = []
attributions = []
for seq in sequences_two:
  t = np.array(hits.query("sequence_name == @seq & cluster == 12")[["start", "attribution"]])
  a = t[:, 0]
  attributions.extend(t[:, 1])
  b = np.array(hits.query("sequence_name == @seq & cluster == 6")["start"])
  d = np.zeros((len(a), len(b)))
  for i in range(len(a)):
    for j in range(len(b)):
      d[i, j] = abs(a[i] - b[j])
  distances.extend(d.min(1))

fig, ax = plt.subplots()
ax.scatter(
  x = distances,
  y = attributions,
  s = 2
)
ax.set_xlabel("abs distances between FOX and RFX")
ax.set_ylabel("RFX attribution score")
fig.savefig("attr_v_fox.png")



tmp = hits.groupby(["sequence_name", "cluster"])["attribution"].sum().reset_index().pivot(index = "sequence_name", columns = "cluster", values = "attribution").fillna(0)

tmp = tmp[[6, 12]]
spearmanr(tmp[6], tmp[12])


###


all_hits = []
for topic in organoid_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits.append(hits)

def merge_and_max(left, right, on, max_on):
  global a
  global all_hits
  if a:
    print(" "*(len(all_hits) - 1) + "|", end = "\r", flush=True)
    a = False
  print("x", end = "", flush=True)
  x = pd.merge(
    left, right,
    on=on,
    how = "outer"
  )
  x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
  return x.drop([f"{max_on}_x", f"{max_on}_y"], axis = 1).copy()

a = True
hits_merged = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution"
    ),
  all_hits
)
# USE MAX to deal with overlapping hits
tmp = hits_merged.groupby(["sequence_name", "cluster"])["attribution"].sum().reset_index().pivot(index = "sequence_name", columns = "cluster", values = "attribution").fillna(0)

region_order = []
for x in tqdm(all_hits):
  for r in x["sequence_name"]:
    if r not in region_order:
      region_order.append(r)

import matplotlib.pyplot as plt

pattern_order = tmp.columns.values[np.argsort([region_order.index(x) for x in tmp.idxmax().values])]

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  ((tmp - tmp.min()) / (tmp.max() - tmp.min())).loc[region_order, pattern_order],
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("test2.png")

## end test

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
  n: pattern.ppm[pattern_metadata.loc[n].ic_start: pattern_metadata.loc[n].ic_stop].T
  for n, pattern in zip(all_pattern_names, all_patterns)
  if n in pattern_metadata.index
}

all_hits_organoid = []
for topic in organoid_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_organoid.append(hits)

subset_hits_organoid = []
for topic in [
  33, 41, 38, 36, 26, 54, 48, 28]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_organoid.append(hits)

a = True
hits_merged_organoid = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(all_hits_organoid)
    ),
  all_hits_organoid
)

a = True
hits_merged_organoid_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(subset_hits_organoid)
    ),
  subset_hits_organoid
)

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum = hits_merged_organoid \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum_subset = hits_merged_organoid_subset \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

region_order_organoid = []
for x in tqdm(all_hits_organoid):
  for r in x["sequence_name"]:
    if r not in region_order_organoid:
      region_order_organoid.append(r)

region_order_organoid_subset = []
for x in tqdm(subset_hits_organoid):
  for r in x["sequence_name"]:
    if r not in region_order_organoid_subset:
      region_order_organoid_subset.append(r)

pattern_order_organoid = hits_merged_organoid_per_seq_and_cluster_sum \
  .columns.values[
    np.argsort(
    [
      region_order_organoid.index(x)
      for x in hits_merged_organoid_per_seq_and_cluster_sum.idxmax().values
    ]
  )
]

hits_merged_organoid_per_seq_and_cluster_sum_scaled = \
  ( hits_merged_organoid_per_seq_and_cluster_sum - hits_merged_organoid_per_seq_and_cluster_sum.min(0) ) / \
  (hits_merged_organoid_per_seq_and_cluster_sum.max(0) - hits_merged_organoid_per_seq_and_cluster_sum.min(0)  )


hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled = \
  ( hits_merged_organoid_per_seq_and_cluster_sum_subset - hits_merged_organoid_per_seq_and_cluster_sum_subset.min(0) ) / \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset.max(0) - hits_merged_organoid_per_seq_and_cluster_sum_subset.min(0)  )


fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_organoid_per_seq_and_cluster_sum_scaled.loc[region_order_organoid, pattern_order_organoid],
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_organoid_per_seq_and_cluster_max.png")

pattern_order_manual = [
  12, 6, 7, 15, 10, 18, 2, 17, 1, 19
]

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[region_order_organoid_subset, pattern_order_manual],
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_organoid_per_seq_and_cluster_max_selected_patterns.png")

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[region_order_organoid_subset, pattern_order_manual] > 0.2,
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_organoid_per_seq_and_cluster_max_selected_patterns_bin.png")

data = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[region_order_organoid_subset, pattern_order_manual] > 0.2
) * 1
cooc = data.T @ data
for cluster in pattern_order_manual:
  cooc.loc[cluster, cluster] = 0

cooc_perc = (cooc / cooc.sum()).T

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
fig.savefig("perc_other_organoid.png")

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
fig.savefig("organoid_graph.pdf")

##

all_hits_embryo = []
for topic in embryo_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_embryo.append(hits)

subset_hits_embryo = []
for topic in [
  34, 74, 87, 38, 79, 51, 88, 58
]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_embryo.append(hits)

a = True
hits_merged_embryo = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(all_hits_embryo)
    ),
  all_hits_embryo
)

a = True
hits_merged_embryo_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(subset_hits_embryo)
    ),
  subset_hits_embryo
)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum = hits_merged_embryo \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum_subset = hits_merged_embryo_subset \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

region_order_embryo = []
for x in tqdm(all_hits_embryo):
  for r in x["sequence_name"]:
    if r not in region_order_embryo:
      region_order_embryo.append(r)

region_order_embryo_subset = []
for x in tqdm(subset_hits_embryo):
  for r in x["sequence_name"]:
    if r not in region_order_embryo_subset:
      region_order_embryo_subset.append(r)

pattern_order_embryo = hits_merged_embryo_per_seq_and_cluster_sum \
  .columns.values[
    np.argsort(
    [
      region_order_embryo.index(x)
      for x in hits_merged_embryo_per_seq_and_cluster_sum.idxmax().values
    ]
  )
]

hits_merged_embryo_per_seq_and_cluster_sum_scaled = \
  ( hits_merged_embryo_per_seq_and_cluster_sum - hits_merged_embryo_per_seq_and_cluster_sum.min(0) ) / \
  (hits_merged_embryo_per_seq_and_cluster_sum.max(0) - hits_merged_embryo_per_seq_and_cluster_sum.min(0)  )

hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled = \
  ( hits_merged_embryo_per_seq_and_cluster_sum_subset - hits_merged_embryo_per_seq_and_cluster_sum_subset.min(0) ) / \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset.max(0) - hits_merged_embryo_per_seq_and_cluster_sum_subset.min(0)  )

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_embryo_per_seq_and_cluster_sum_scaled.loc[region_order_embryo, pattern_order_embryo],
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_embryo_per_seq_and_cluster_max.png")

pattern_order_manual = [
  12, 6, 7, 15, 10, 18, 2, 17, 1, 19
]

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[region_order_embryo_subset, pattern_order_manual],
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_embryo_per_seq_and_cluster_max_selected_patterns.png")

fig, ax = plt.subplots(figsize = (4, 8))
sns.heatmap(
  hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[region_order_embryo_subset, pattern_order_manual] > 0.1,
  yticklabels = False, xticklabels = True,
  robust = True, ax = ax)
fig.tight_layout()
fig.savefig("hits_merged_embryo_per_seq_and_cluster_max_selected_patterns_bin.png")

data = (
    hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[region_order_embryo_subset, pattern_order_manual] > 0.1
) * 1
cooc = data.T @ data
for cluster in pattern_order_manual:
  cooc.loc[cluster, cluster] = 0

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
        s = f"p=1e{np.round(np.log(pval),2)}",
        va= "center", ha = "center", weight = "bold")
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
fig.savefig("perc_other_embryo.png")


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
fig.savefig("embryo_graph.pdf")


attr = hits_merged_organoid.query("cluster == 6")["attribution"].values
fig, ax = plt.subplots()
ax.scatter(
  x = hits_merged_organoid.query("cluster == 6")["start"].values,
  y = (attr - attr.min()) / (attr.max() - attr.min()),
  s = 1,
  color = "black"
)
ax.axhline(0.1, color = "red")
fig.tight_layout()
fig.savefig("ps_v_cluster_6_attr.png")


topic = 33
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
    m, "hier_cluster"
  ]
    for m in hits["motif_name"]
  ]

from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(torch.from_numpy((ohs * attr.squeeze()).sum(2)))

seqlets[["example_idx", "start", "end"]].to_csv("seqlets.bed", sep = "\t", header = False, index = False)

hits[["sequence_name", "start", "end"]].to_csv("hits.bed", sep = "\t", header = False, index = False)

hits_w_seqlet = pd.read_table("hits_w_seqlet.bed", header = None)
hits_w_seqlet.columns = ["sequence_name", "start", "end"]
hits_w_seqlet["is_seqlet"] = True

hits = pd.merge(left = hits, right = hits_w_seqlet, on = ["sequence_name", "start", "end"], how = "outer").fillna(False)

attr = hits.query("cluster == 6")["attribution"].values
start = hits.query("cluster == 6")["start"].values
is_seqlet = hits.query("cluster == 6")["is_seqlet"].values

fig, axs = plt.subplots(ncols = 2, figsize = (8, 4), sharex = True, sharey = True)
axs[0].scatter(
  x = start[~is_seqlet],
  y = ((attr - attr.min()) / (attr.max() - attr.min()))[~is_seqlet],
  s = 1,
  color = "black"
)
axs[0].set_title("not seqlet")
axs[1].scatter(
  x = start[is_seqlet],
  y = ((attr - attr.min()) / (attr.max() - attr.min()))[is_seqlet],
  s = 1,
  color = "black"
)
axs[1].set_title("seqlet")
fig.tight_layout()
fig.savefig("ps_v_cluster_6_attr.png")

```

```bash

wget -O HEPG2_FOXA2_ENCFF466FCB.bed.gz https://www.encodeproject.org/files/ENCFF466FCB/@@download/ENCFF466FCB.bed.gz
wget -O A549_FOXA2_ENCFF686MSH.bed.gz https://www.encodeproject.org/files/ENCFF686MSH/@@download/ENCFF686MSH.bed.gz

```

```python

from dataclasses import dataclass
from typing import Self
from tangermeme.tools.fimo import fimo
import numpy as np
import h5py
from tqdm import tqdm
import os
import pandas as pd
from tangermeme.utils import extract_signal
import torch
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib
from scipy.stats import binomtest
import networkx as nx

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
    self.subpatterns  = [
      ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]

  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"

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

organoid_dv_topics = np.array(
  [
    8,
    16,
    13,
    9,
    11,
    19,
    25,
    1,
    29,
    23,
    3
  ]
) + 25

embryo_dv_topics = np.array(
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

organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
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

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
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

pattern_metadata = pd.read_table("motif_metadata.tsv", index_col = 0)

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
  n: pattern.ppm[pattern_metadata.loc[n].ic_start: pattern_metadata.loc[n].ic_stop].T
  for n, pattern in zip(all_pattern_names, all_patterns)
  if n in pattern_metadata.index
}

all_hits_organoid = []
for topic in organoid_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_organoid.append(hits)

subset_hits_organoid = []
for topic in [
  33, 41, 38, 36, 26, 54, 48, 28]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_organoid.append(hits)

a = True
hits_merged_organoid = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(all_hits_organoid)
    ),
  all_hits_organoid
)

a = True
hits_merged_organoid_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(subset_hits_organoid)
    ),
  subset_hits_organoid
)

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum = hits_merged_organoid \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum_subset = hits_merged_organoid_subset \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_organoid_per_seq_and_cluster_sum_scaled = \
  ( hits_merged_organoid_per_seq_and_cluster_sum - hits_merged_organoid_per_seq_and_cluster_sum.min(0) ) / \
  (hits_merged_organoid_per_seq_and_cluster_sum.max(0) - hits_merged_organoid_per_seq_and_cluster_sum.min(0)  )


hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled = \
  ( hits_merged_organoid_per_seq_and_cluster_sum_subset - hits_merged_organoid_per_seq_and_cluster_sum_subset.min(0) ) / \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset.max(0) - hits_merged_organoid_per_seq_and_cluster_sum_subset.min(0)  )

all_hits_embryo = []
for topic in embryo_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_embryo.append(hits)

subset_hits_embryo = []
for topic in [
  34, 74, 87, 38, 79, 51, 88, 58
]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_embryo.append(hits)

a = True
hits_merged_embryo = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(all_hits_embryo)
    ),
  all_hits_embryo
)

a = True
hits_merged_embryo_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand"],
      max_on = "attribution",
      l = len(subset_hits_embryo)
    ),
  subset_hits_embryo
)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum = hits_merged_embryo \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum_subset = hits_merged_embryo_subset \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_embryo_per_seq_and_cluster_sum_scaled = \
  ( hits_merged_embryo_per_seq_and_cluster_sum - hits_merged_embryo_per_seq_and_cluster_sum.min(0) ) / \
  (hits_merged_embryo_per_seq_and_cluster_sum.max(0) - hits_merged_embryo_per_seq_and_cluster_sum.min(0)  )

hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled = \
  ( hits_merged_embryo_per_seq_and_cluster_sum_subset - hits_merged_embryo_per_seq_and_cluster_sum_subset.min(0) ) / \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset.max(0) - hits_merged_embryo_per_seq_and_cluster_sum_subset.min(0)  )

with open("FOX_hits_organoid.bed", "wt") as f:
  for region in hits_merged_organoid_per_seq_and_cluster_sum_scaled. \
    index[hits_merged_organoid_per_seq_and_cluster_sum_scaled[6] > 0.02]:
    _ = f.write(region.replace(":", "\t").replace("-", "\t") + "\n")

with open("FOX_hits_organoid_subset.bed", "wt") as f:
  for region in hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled. \
    index[hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled[6] > 0.02]:
    _ = f.write(region.replace(":", "\t").replace("-", "\t") + "\n")

with open("FOX_hits_embryo.bed", "wt") as f:
  for region in hits_merged_embryo_per_seq_and_cluster_sum_scaled. \
    index[hits_merged_embryo_per_seq_and_cluster_sum_scaled[6] > 0.01]:
    _ = f.write(region.replace(":", "\t").replace("-", "\t") + "\n")

with open("FOX_hits_embryo_subset.bed", "wt") as f:
  for region in hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled. \
    index[hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled[6] > 0.01]:
    _ = f.write(region.replace(":", "\t").replace("-", "\t") + "\n")


```

```bash

module load BEDTools/2.31.0-GCC-12.3.0

bedtools intersect -b HEPG2_FOXA2_ENCFF466FCB.bed.gz -a FOX_hits_organoid.bed -F 0.4 -wa \
  > FOX_hits_organoid_w_HEPG2.bed

bedtools intersect -b HEPG2_FOXA2_ENCFF466FCB.bed.gz -a FOX_hits_organoid_subset.bed -F 0.4 -wa \
  > FOX_hits_organoid_subset_w_HEPG2.bed

bedtools intersect -b A549_FOXA2_ENCFF686MSH.bed.gz -a FOX_hits_organoid.bed -F 0.4 -wa \
  > FOX_hits_organoid_w_A549.bed

bedtools intersect -b A549_FOXA2_ENCFF686MSH.bed.gz -a FOX_hits_organoid_subset.bed -F 0.4 -wa \
  > FOX_hits_organoid_subset_w_A549.bed

bedtools intersect -a HEPG2_FOXA2_ENCFF466FCB.bed.gz -b FOX_hits_organoid.bed -F 0.4 -wa \
  > HEPG2_w_FOX_hits_organoid.bed

bedtools intersect -a HEPG2_FOXA2_ENCFF466FCB.bed.gz -b FOX_hits_organoid_subset.bed -F 0.4 -wa \
  > HEPG2_w_FOX_hits_organoid_subset.bed

bedtools intersect -a A549_FOXA2_ENCFF686MSH.bed.gz -b FOX_hits_organoid.bed -F 0.4 -wa \
  > A549_w_FOX_hits_organoid.bed

bedtools intersect -a A549_FOXA2_ENCFF686MSH.bed.gz -b FOX_hits_organoid_subset.bed -F 0.4 -wa \
  > A549_w_FOX_hits_organoid_subset.bed

##

bedtools intersect -b HEPG2_FOXA2_ENCFF466FCB.bed.gz -a FOX_hits_embryo.bed -F 0.4 -wa \
  > FOX_hits_embryo_w_HEPG2.bed

bedtools intersect -b HEPG2_FOXA2_ENCFF466FCB.bed.gz -a FOX_hits_embryo_subset.bed -F 0.4 -wa \
  > FOX_hits_embryo_subset_w_HEPG2.bed

bedtools intersect -b A549_FOXA2_ENCFF686MSH.bed.gz -a FOX_hits_embryo.bed -F 0.4 -wa \
  > FOX_hits_embryo_w_A549.bed

bedtools intersect -b A549_FOXA2_ENCFF686MSH.bed.gz -a FOX_hits_embryo_subset.bed -F 0.4 -wa \
  > FOX_hits_embryo_subset_w_A549.bed

bedtools intersect -a HEPG2_FOXA2_ENCFF466FCB.bed.gz -b FOX_hits_embryo.bed -F 0.4 -wa \
  > HEPG2_w_FOX_hits_embryo.bed

bedtools intersect -a HEPG2_FOXA2_ENCFF466FCB.bed.gz -b FOX_hits_embryo_subset.bed -F 0.4 -wa \
  > HEPG2_w_FOX_hits_embryo_subset.bed

bedtools intersect -a A549_FOXA2_ENCFF686MSH.bed.gz -b FOX_hits_embryo.bed -F 0.4 -wa \
  > A549_w_FOX_hits_embryo.bed

bedtools intersect -a A549_FOXA2_ENCFF686MSH.bed.gz -b FOX_hits_embryo_subset.bed -F 0.4 -wa \
  > A549_w_FOX_hits_embryo_subset.bed



```


```python

from dataclasses import dataclass
from typing import Self
from tangermeme.tools.fimo import fimo
import numpy as np
import h5py
from tqdm import tqdm
import os
import pandas as pd
from tangermeme.utils import extract_signal
import torch
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import matplotlib
from scipy.stats import binomtest
import networkx as nx

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
    self.subpatterns  = [
      ModiscoPattern(p[sub], is_pos, ohs, region_names) for sub in p.keys() if sub.startswith("subpattern_")]

  def __repr__(self):
    return f"ModiscoPattern with {len(self.seqlets)} seqlets"

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

organoid_dv_topics = np.array(
  [
    8,
    16,
    13,
    9,
    11,
    19,
    25,
    1,
    29,
    23,
    3
  ]
) + 25

embryo_dv_topics = np.array(
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

organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
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

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
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

pattern_metadata = pd.read_table("motif_metadata.tsv", index_col = 0)

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
  n: pattern.ppm[pattern_metadata.loc[n].ic_start: pattern_metadata.loc[n].ic_stop].T
  for n, pattern in zip(all_pattern_names, all_patterns)
  if n in pattern_metadata.index
}

all_hits_organoid = []
for topic in organoid_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_organoid.append(hits)

subset_hits_organoid = []
for topic in [
  33, 41, 38, 36, 26, 54, 48, 28]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_organoid.append(hits)

a = True
hits_merged_organoid = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value"],
      max_on = "attribution",
      l = len(all_hits_organoid)
    ),
  all_hits_organoid
)

a = True
hits_merged_organoid_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value"],
      max_on = "attribution",
      l = len(subset_hits_organoid)
    ),
  subset_hits_organoid
)

all_hits_embryo = []
for topic in embryo_dv_topics:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  all_hits_embryo.append(hits)

subset_hits_embryo = []
for topic in [
  34, 74, 87, 38, 79, 51, 88, 58
]:
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
      m, "hier_cluster"
    ]
      for m in hits["motif_name"]
    ]
  hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
  subset_hits_embryo.append(hits)

a = True
hits_merged_embryo = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value"],
      max_on = "attribution",
      l = len(all_hits_embryo)
    ),
  all_hits_embryo
)

a = True
hits_merged_embryo_subset = reduce(
  lambda left, right:
    merge_and_max(
      left[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      right[["motif_name", "cluster", "sequence_name", "start", "end", "strand", "attribution", "score", "p-value"]],
      on = ["motif_name", "cluster", "sequence_name", "start", "end", "strand", "score", "p-value"],
      max_on = "attribution",
      l = len(subset_hits_embryo)
    ),
  subset_hits_embryo
)

region_names_organoid_w_HEPG2 = []
with open("FOX_hits_organoid_w_HEPG2.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_organoid_w_HEPG2.append(
      f"{chrom}:{start}-{end}"
    )

region_names_organoid_w_A549 = []
with open("FOX_hits_organoid_w_A549.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_organoid_w_A549.append(
      f"{chrom}:{start}-{end}"
    )

region_names_organoid_subset_w_HEPG2 = []
with open("FOX_hits_organoid_subset_w_HEPG2.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_organoid_subset_w_HEPG2.append(
      f"{chrom}:{start}-{end}"
    )

region_names_organoid_subset_w_A549 = []
with open("FOX_hits_organoid_subset_w_A549.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_organoid_subset_w_A549.append(
      f"{chrom}:{start}-{end}"
    )

region_names_embryo_w_HEPG2 = []
with open("FOX_hits_embryo_w_HEPG2.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_embryo_w_HEPG2.append(
      f"{chrom}:{start}-{end}"
    )

region_names_embryo_w_A549 = []
with open("FOX_hits_embryo_w_A549.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_embryo_w_A549.append(
      f"{chrom}:{start}-{end}"
    )

region_names_embryo_subset_w_HEPG2 = []
with open("FOX_hits_embryo_subset_w_HEPG2.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_embryo_subset_w_HEPG2.append(
      f"{chrom}:{start}-{end}"
    )

region_names_embryo_subset_w_A549 = []
with open("FOX_hits_embryo_subset_w_A549.bed") as f:
  for l in f:
    chrom, start, end = l.strip().split()
    region_names_embryo_subset_w_A549.append(
      f"{chrom}:{start}-{end}"
    )

hits_merged_organoid["overlaps_w_HEPG2"] = False
hits_merged_organoid["overlaps_w_A549"] = False
hits_merged_organoid_subset["overlaps_w_HEPG2"] = False
hits_merged_organoid_subset["overlaps_w_A549"] = False

hits_merged_embryo["overlaps_w_HEPG2"] = False
hits_merged_embryo["overlaps_w_A549"] = False
hits_merged_embryo_subset["overlaps_w_HEPG2"] = False
hits_merged_embryo_subset["overlaps_w_A549"] = False

hits_merged_organoid.loc[
  [r in region_names_organoid_w_HEPG2 for r in hits_merged_organoid["sequence_name"]],
  "overlaps_w_HEPG2"
] = True
hits_merged_organoid.loc[
  [r in region_names_organoid_w_A549 for r in hits_merged_organoid["sequence_name"]],
  "overlaps_w_A549"
] = True
hits_merged_organoid_subset.loc[
  [r in region_names_organoid_subset_w_HEPG2 for r in hits_merged_organoid_subset["sequence_name"]],
  "overlaps_w_HEPG2"
] = True
hits_merged_organoid_subset.loc[
  [r in region_names_organoid_subset_w_A549 for r in hits_merged_organoid_subset["sequence_name"]],
  "overlaps_w_A549"
] = True

hits_merged_embryo.loc[
  [r in region_names_embryo_w_HEPG2 for r in hits_merged_embryo["sequence_name"]],
  "overlaps_w_HEPG2"
] = True
hits_merged_embryo.loc[
  [r in region_names_embryo_w_A549 for r in hits_merged_embryo["sequence_name"]],
  "overlaps_w_A549"
] = True
hits_merged_embryo_subset.loc[
  [r in region_names_embryo_subset_w_HEPG2 for r in hits_merged_embryo_subset["sequence_name"]],
  "overlaps_w_HEPG2"
] = True
hits_merged_embryo_subset.loc[
  [r in region_names_embryo_subset_w_A549 for r in hits_merged_embryo_subset["sequence_name"]],
  "overlaps_w_A549"
] = True

def get_number_of_non_overlapping_sites(df: pd.DataFrame, max_overlap: int, broadcast = False):
  n = sum(np.diff(df["start"].sort_values()) > max_overlap) + 1
  if not broadcast:
    return n
  else:
    return [n for _ in range(len(df))]

hits_organoid_number_sites_specific = hits_merged_organoid \
  .query("cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
  .groupby("sequence_name") \
  .apply(lambda x: get_number_of_non_overlapping_sites(x, 10), include_groups = False)

hits_organoid_number_sites_general = hits_merged_organoid \
  .query("cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)") \
  .groupby("sequence_name") \
  .apply(lambda x: get_number_of_non_overlapping_sites(x, 10), include_groups = False)


from scipy.stats import mannwhitneyu


pval = mannwhitneyu(hits_organoid_number_sites_general, hits_organoid_number_sites_specific).pvalue

fig, axs = plt.subplots(nrows = 2, sharex = True, sharey = True)
_ = axs[0].hist(
  hits_organoid_number_sites_general,
  bins = np.arange( max(max(hits_organoid_number_sites_general), max(hits_organoid_number_sites_specific)) ) + 1,
  density = True,
  color = "gray",
  edgecolor = "black",
  lw = 1,
)
_ = axs[1].hist(
  hits_organoid_number_sites_specific,
  bins = np.arange( max(max(hits_organoid_number_sites_general), max(hits_organoid_number_sites_specific)) ) + 1,
  density = True,
  color = "gray",
  edgecolor = "black",
  lw = 1,
)
_ = axs[0].set_title(f"General - pval = {round(pval * 10**round(-np.log10(pval)))}e{round(np.log10(pval))}")
_ = axs[1].set_title("Specific")
_ = axs[1].set_xlabel("Number of sites")
_ = axs[0].set_ylabel("Frequency")
_ = axs[1].set_ylabel("Frequency")
for ax in axs:
  _ = ax.set_xticks(np.arange( max(max(hits_organoid_number_sites_general), max(hits_organoid_number_sites_specific)) ) + 1)
  ax.grid(color = "gray")
  ax.set_axisbelow(True)
fig.savefig("nr_FOX_hits_organoid_spec_v_general.png")

hits_embryo_number_sites_specific = hits_merged_embryo \
.query("cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
.groupby("sequence_name") \
.apply(lambda x: get_number_of_non_overlapping_sites(x, 10), include_groups = False)

hits_embryo_number_sites_general = hits_merged_embryo \
.query("cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)") \
.groupby("sequence_name") \
.apply(lambda x: get_number_of_non_overlapping_sites(x, 10), include_groups = False)

pval = mannwhitneyu(hits_embryo_number_sites_general, hits_embryo_number_sites_specific).pvalue

fig, axs = plt.subplots(nrows = 2, sharex = True, sharey = True)
_ = axs[0].hist(
hits_embryo_number_sites_general,
bins = np.arange( max(max(hits_embryo_number_sites_general), max(hits_embryo_number_sites_specific)) ) + 1,
density = True,
color = "gray",
edgecolor = "black",
lw = 1,
)
_ = axs[1].hist(
hits_embryo_number_sites_specific,
bins = np.arange( max(max(hits_embryo_number_sites_general), max(hits_embryo_number_sites_specific)) ) + 1,
density = True,
color = "gray",
edgecolor = "black",
lw = 1,
)
_ = axs[0].set_title(f"General - pval = {round(pval * 10**round(-np.log10(pval)))}e{round(np.log10(pval))}")
_ = axs[1].set_title("Specific")
_ = axs[1].set_xlabel("Number of sites")
_ = axs[0].set_ylabel("Frequency")
_ = axs[1].set_ylabel("Frequency")
for ax in axs:
  _ = ax.set_xticks(np.arange( max(max(hits_embryo_number_sites_general), max(hits_embryo_number_sites_specific)) ) + 1)
  ax.grid(color = "gray")
  ax.set_axisbelow(True)
fig.savefig("nr_FOX_hits_embryo_spec_v_general.png")

def get_non_overlapping_start_end_w_max_attr(df, max_overlap):
  df = df.sort_values("start")
  delta = np.diff(df["start"])
  delta_loc = [0, *np.where(delta > max_overlap)[0] + 1]
  groups = [
    slice(delta_loc[i], delta_loc[i + 1] if (i + 1) < len(delta_loc) else None)
    for i in range(len(delta_loc))
  ]
  return pd.DataFrame(
    [
        df.iloc[group].iloc[df.iloc[group].attribution.argmax()]
        for group in groups
    ]
  ).reset_index(drop = True)

hits_organoid_non_overlap_specific = hits_merged_organoid \
  .query("cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
  .groupby("sequence_name") \
  .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10), include_groups = False) \
  .reset_index() \
  .drop("level_1", axis = 1)

hits_organoid_non_overlap_general = hits_merged_organoid \
  .query("cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)") \
  .groupby("sequence_name") \
  .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10), include_groups = False) \
  .reset_index() \
  .drop("level_1", axis = 1)

specific_n_sites_organoid = hits_organoid_non_overlap_specific.groupby("sequence_name").apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True), include_groups = False).explode().values

general_n_sites_organoid = hits_organoid_non_overlap_general.groupby("sequence_name").apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True), include_groups = False).explode().values


fig, axs = plt.subplots(ncols = 2, sharey = True, sharex = True, figsize = (8, 4))
bplot = axs[0].boxplot(
  [
    -np.log10(hits_organoid_non_overlap_specific.loc[specific_n_sites_organoid == (n + 1), "p-value"] + 0.000001)
    for n in range(max(specific_n_sites_organoid)) if sum(specific_n_sites_organoid == (n + 1)) > 100
  ],
  tick_labels = [n + 1 for n in range(max(specific_n_sites_organoid)) if sum(specific_n_sites_organoid == (n + 1)) > 100],
  patch_artist=True
)
for patch in bplot['boxes']:
    patch.set_facecolor("orange")
bplot = axs[1].boxplot(
  [
    -np.log10(hits_organoid_non_overlap_general.loc[general_n_sites_organoid == (n + 1), "p-value"] + 0.000001)
    for n in range(max(general_n_sites_organoid)) if sum(general_n_sites_organoid == (n + 1)) > 100
  ],
  tick_labels = [n + 1 for n in range(max(general_n_sites_organoid)) if sum(general_n_sites_organoid == (n + 1)) > 100], 
  patch_artist=True
)
for patch in bplot['boxes']:
    patch.set_facecolor("orange")
_ = axs[0].set_title("specific")
_ = axs[1].set_title("general")
_ = axs[0].set_ylabel("$-log10(pval)$")
for ax in axs:
  ax.set_ylim((3.5, 6.5))
  ax.grid(color = "gray")
  ax.set_axisbelow(True)
  _ = ax.set_xlabel("n. sites")
fig.savefig("bplot_n_sites_v_motif_score_organoid.png")

hits_embryo_non_overlap_specific = hits_merged_embryo \
  .query("cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
  .groupby("sequence_name") \
  .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10), include_groups = False) \
  .reset_index() \
  .drop("level_1", axis = 1)

hits_embryo_non_overlap_general = hits_merged_embryo \
  .query("cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)") \
  .groupby("sequence_name") \
  .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10), include_groups = False) \
  .reset_index() \
  .drop("level_1", axis = 1)

specific_n_sites_embryo = hits_embryo_non_overlap_specific.groupby("sequence_name").apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True), include_groups = False).explode().values

general_n_sites_embryo = hits_embryo_non_overlap_general.groupby("sequence_name").apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True), include_groups = False).explode().values

fig, axs = plt.subplots(ncols = 2, sharey = True, sharex = True, figsize = (8, 4))
bplot = axs[0].boxplot(
  [
    -np.log10(hits_embryo_non_overlap_specific.loc[specific_n_sites_embryo == (n + 1), "p-value"] + 0.000001)
    for n in range(max(specific_n_sites_embryo)) if sum(specific_n_sites_embryo == (n + 1)) > 100
  ],
  tick_labels = [n + 1 for n in range(max(specific_n_sites_embryo)) if sum(specific_n_sites_embryo == (n + 1)) > 100],
  patch_artist=True
)
for patch in bplot['boxes']:
    patch.set_facecolor("orange")
bplot = axs[1].boxplot(
  [
    -np.log10(hits_embryo_non_overlap_general.loc[general_n_sites_embryo == (n + 1), "p-value"] + 0.000001)
    for n in range(max(general_n_sites_embryo)) if sum(general_n_sites_embryo == (n + 1)) > 100
  ],
  tick_labels = [n + 1 for n in range(max(general_n_sites_embryo)) if sum(general_n_sites_embryo == (n + 1)) > 100], 
  patch_artist=True
)
for patch in bplot['boxes']:
    patch.set_facecolor("orange")
_ = axs[0].set_title("specific")
_ = axs[1].set_title("general")
_ = axs[0].set_ylabel("$-log10(pval)$")
for ax in axs:
  ax.set_ylim((3.5, 6.5))
  ax.grid(color = "gray")
  ax.set_axisbelow(True)
  _ = ax.set_xlabel("n. sites")
fig.savefig("bplot_n_sites_v_motif_score_embryo.png")

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

def get_sequence_hit(hit, allignment_info, genome, target_len):
  chrom, start_offset, _ = hit.sequence_name.replace(":", "-").split("-")
  start_offset = int(start_offset)
  hit_start = hit.start
  hit_end = hit.end
  hit_strand = hit.strand
  is_rc_hit = hit_strand == "-"
  is_rc_to_root = allignment_info.is_rc_to_root
  offset_to_root = allignment_info.offset_to_root
  if is_rc_to_root ^ is_rc_hit:
    # align end position
    _start = start_offset + hit_start
    _end   = start_offset + hit_end + offset_to_root
    to_pad = target_len - (_end - _start)
    # add padding to the start
    _start -= to_pad
    seq = genome.fetch(
      chrom,
      _start,
      _end
    )
    seq = reverse_complement(seq)
  else:
    # align start
    _start = start_offset + hit_start - offset_to_root
    _end   = start_offset + hit_end
    to_pad = target_len - (_end - _start)
    # add padding to end
    _end += to_pad
    seq = genome.fetch(
      chrom,
      _start,
      _end
    )
  return seq

import logomaker
import modiscolite

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

MX = 13
for n_sites in range(1, 7):
  print(n_sites)
  pattern_seq = []
  for _, hit in hits_organoid_non_overlap_general.loc[
    general_n_sites_organoid == n_sites].sort_values("sequence_name").iterrows():
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30
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
  _ = axs[0].set_title(f"Organoid general regions\n{n_sites} instance(s) per region.")
  fig.tight_layout()
  fig.savefig(f"organoid_general_{n_sites}_site.png")
  fig.savefig(f"organoid_general_{n_sites}_site.pdf")

MX = 13
for n_sites in range(1, 7):
  print(n_sites)
  pattern_seq = []
  for _, hit in hits_organoid_non_overlap_specific.loc[
    specific_n_sites_organoid == n_sites].sort_values("sequence_name").iterrows():
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30
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
  _ = axs[0].set_title(f"Organoid specific regions\n{n_sites} instance(s) per region.")
  fig.tight_layout()
  fig.savefig(f"organoid_specific_{n_sites}_site.png")
  fig.savefig(f"organoid_specific_{n_sites}_site.pdf")

n_sites = 1
pattern_seq = []
for _, hit in hits_organoid_non_overlap_general.loc[
  general_n_sites_organoid == n_sites].sort_values("sequence_name").iterrows():
  s = get_sequence_hit(
      hit=hit,
      allignment_info=pattern_metadata.loc[hit.motif_name],
      genome=hg38,
      target_len=30
    )
  pattern_seq.append(s)
ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
#pc = 1e-3
#ppm += pc
#ppm = (ppm.T / ppm.sum(1)).T
ppm_general = ppm.copy()

n_sites = 1
pattern_seq = []
for _, hit in hits_organoid_non_overlap_specific.loc[
  specific_n_sites_organoid == n_sites].sort_values("sequence_name").iterrows():
  s = get_sequence_hit(
      hit=hit,
      allignment_info=pattern_metadata.loc[hit.motif_name],
      genome=hg38,
      target_len=30
    )
  pattern_seq.append(s)
ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
#pc = 1e-3
#ppm += pc
#ppm = (ppm.T / ppm.sum(1)).T
ppm_specific = ppm.copy()

fig, ax = plt.subplots()
_ = logomaker.Logo(
  ppm_general[0:10] - ppm_specific[0:10],
  ax = ax
)
fig.savefig("test.png")


import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import fisher_exact
import warnings

def compare_ppm_counts(ppm1, ppm2):
    results = []
    for pos in ppm1.index:
        # Get counts for this position
        counts1 = ppm1.loc[pos]
        counts2 = ppm2.loc[pos]
        # Create contingency table
        contingency = np.vstack([counts1, counts2])
        # Calculate row totals (sample sizes)
        n1 = counts1.sum()
        n2 = counts2.sum()
        # Perform chi-square test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, p_chi2 = stats.chi2_contingency(contingency)[:2]
        # Calculate effect sizes
        # Cramer's V
        n_total = n1 + n2
        min_dim = min(2, 4) - 1  # min(rows-1, cols-1)
        cramer_v = np.sqrt(chi2 / (n_total * min_dim))
        # Calculate proportions and their differences
        props1 = counts1 / n1
        props2 = counts2 / n2
        max_diff = np.max(np.abs(props1 - props2))
        # Calculate standard errors for proportion differences
        max_diff_se = np.max([
            np.sqrt((p1 * (1-p1))/n1 + (p2 * (1-p2))/n2)
            for p1, p2 in zip(props1, props2)
        ])
        # Store results
        results.append({
            'position': pos,
            'chi2_statistic': chi2,
            'p_value_chi2': p_chi2,
            'cramers_v': cramer_v,
            'max_prop_diff': max_diff,
            'max_prop_diff_se': max_diff_se,
            'n1': n1,
            'n2': n2
        })
    results_df = pd.DataFrame(results).set_index('position')
    # Add significance levels
    results_df['significance'] = pd.cut(
        results_df['p_value_chi2'],
        bins=[0, 0.001, 0.01, 0.05, 1],
        labels=['***', '**', '*', 'ns']
    )
    return results_df.round(4)

ppm1 = ppm_specific[0:10]
ppm2 = ppm_general[0:10]
res = compare_ppm_counts(ppm1 + 1, ppm2 + 1)


fig, ax = plt.subplots()
_ = logomaker.Logo(
  -((ppm1 + pc).T / (ppm1 + pc).sum(1)).T + ((ppm2 + pc).T / (ppm2 + pc).sum(1)).T,
  ax = ax
)
ax.scatter(
  x = np.where(res["p_value_chi2"] < 0.01)[0],
  y = np.repeat(0.2, sum(res["p_value_chi2"] < 0.01)),
  marker = "*",
  color = "black"
)
ax.set_xticks(np.arange(10), labels = np.arange(10) + 1)
ax.set_ylabel("$ppm_{general} - ppm_{specific}$")
ax.set_xlabel("Position")
ax.set_ylim((-0.25, 0.25))
fig.savefig("delta_general_specific_organoid.png")


MX = 13
for n_sites in range(1, 7):
  print(n_sites)
  pattern_seq = []
  for _, hit in hits_embryo_non_overlap_general.loc[
    general_n_sites_embryo == n_sites].sort_values("sequence_name").iterrows():
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30
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
  _ = axs[0].set_title(f"embryo general regions\n{n_sites} instance(s) per region.")
  fig.tight_layout()
  fig.savefig(f"embryo_general_{n_sites}_site.png")
  fig.savefig(f"embryo_general_{n_sites}_site.pdf")

MX = 13
for n_sites in range(1, 7):
  print(n_sites)
  pattern_seq = []
  for _, hit in hits_embryo_non_overlap_specific.loc[
    specific_n_sites_embryo == n_sites].sort_values("sequence_name").iterrows():
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30
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
  _ = axs[0].set_title(f"embryo specific regions\n{n_sites} instance(s) per region.")
  fig.tight_layout()
  fig.savefig(f"embryo_specific_{n_sites}_site.png")
  fig.savefig(f"embryo_specific_{n_sites}_site.pdf")

n_sites = 1
pattern_seq = []
for _, hit in hits_embryo_non_overlap_general.loc[
  general_n_sites_embryo == n_sites].sort_values("sequence_name").iterrows():
  s = get_sequence_hit(
      hit=hit,
      allignment_info=pattern_metadata.loc[hit.motif_name],
      genome=hg38,
      target_len=30
    )
  pattern_seq.append(s)
ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
#pc = 1e-3
#ppm += pc
#ppm = (ppm.T / ppm.sum(1)).T
ppm_general = ppm.copy()

n_sites = 1
pattern_seq = []
for _, hit in hits_embryo_non_overlap_specific.loc[
  specific_n_sites_embryo == n_sites].sort_values("sequence_name").iterrows():
  s = get_sequence_hit(
      hit=hit,
      allignment_info=pattern_metadata.loc[hit.motif_name],
      genome=hg38,
      target_len=30
    )
  pattern_seq.append(s)
ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
#pc = 1e-3
#ppm += pc
#ppm = (ppm.T / ppm.sum(1)).T
ppm_specific = ppm.copy()


ppm1 = ppm_specific[0:10]
ppm2 = ppm_general[0:10]
res = compare_ppm_counts(ppm1 + 1, ppm2 + 1)


fig, ax = plt.subplots()
_ = logomaker.Logo(
  -((ppm1 + pc).T / (ppm1 + pc).sum(1)).T + ((ppm2 + pc).T / (ppm2 + pc).sum(1)).T,
  ax = ax
)
ax.scatter(
  x = np.where(res["p_value_chi2"] < 0.01)[0],
  y = np.repeat(0.23, sum(res["p_value_chi2"] < 0.01)),
  marker = "*",
  color = "black"
)
ax.set_ylabel("$ppm_{general} - ppm_{specific}$")
ax.set_xlabel("Position")
ax.set_ylim((-0.25, 0.25))
fig.savefig("delta_general_specific_embryo.png")



####

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum_subset_specific = hits_merged_organoid_subset \
  .query("overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_organoid_per_seq_and_cluster_sum_subset_general = hits_merged_organoid_subset \
  .query("overlaps_w_HEPG2 == True | overlaps_w_A549 == True") \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_organoid_per_seq_and_cluster_sum_subset_specific_scaled = \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset_specific - hits_merged_organoid_per_seq_and_cluster_sum_subset_specific.min(0)) / \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset_specific.max(0) - hits_merged_organoid_per_seq_and_cluster_sum_subset_specific.min(0))

hits_merged_organoid_per_seq_and_cluster_sum_subset_general_scaled = \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset_general - hits_merged_organoid_per_seq_and_cluster_sum_subset_general.min(0)) / \
  (hits_merged_organoid_per_seq_and_cluster_sum_subset_general.max(0) - hits_merged_organoid_per_seq_and_cluster_sum_subset_general.min(0))

pattern_order_manual = [
  12, 6, 7, 15, 10, 18, 2, 17, 1, 19
]

data = (
  hits_merged_organoid_per_seq_and_cluster_sum_subset_specific_scaled[pattern_order_manual] > 0.2
) * 1

cooc_specific = data.T @ data
for cluster in pattern_order_manual:
  cooc_specific.loc[cluster, cluster] = 0

cooc_perc_specific = (cooc_specific / cooc_specific.sum()).T

data = (
  hits_merged_organoid_per_seq_and_cluster_sum_subset_general_scaled[pattern_order_manual] > 0.2
) * 1

cooc_general = data.T @ data
for cluster in pattern_order_manual:
  cooc_general.loc[cluster, cluster] = 0

cooc_perc_general = (cooc_general / cooc_general.sum()).T

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(pattern_order_manual), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)
cluster_to_color = {
  c: mapper.to_rgba(i)
  for i, c in enumerate(pattern_order_manual)
}

fig, ax = plt.subplots(figsize = (8, 4))
p = 1 / (len(pattern_order_manual) - 1)
cluster = 6
y = 0
cooc_perc = cooc_perc_specific
cooc = cooc_specific
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
y = 1
cooc_perc = cooc_perc_general
cooc = cooc_general
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
_ = ax.set_yticks(
  np.arange(2),
  labels = ["specific", "general"]
)
ax.grid(color = "black")
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(10) / 10)
fig.tight_layout()
fig.savefig("perc_other_organoid_general_specific.png")


fig, ax = plt.subplots(figsize = (8, 4))
p = 1 / (len(pattern_order_manual) - 1)
cluster = 6
y = 0
cooc_perc = cooc_specific
cooc = cooc_specific
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
y = 1
cooc_perc = cooc_general
cooc = cooc_general
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
_ = ax.set_yticks(
  np.arange(2),
  labels = ["specific", "general"]
)
ax.grid(color = "black")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("abs_other_organoid_general_specific.png")


####

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_embryo_per_seq_and_cluster_sum_subset_specific = hits_merged_embryo_subset \
  .query("overlaps_w_HEPG2 == False & overlaps_w_A549 == False") \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_embryo_per_seq_and_cluster_sum_subset_general = hits_merged_embryo_subset \
  .query("overlaps_w_HEPG2 == True | overlaps_w_A549 == True") \
  .groupby(["sequence_name", "cluster"])["attribution"] \
  .max() \
  .reset_index() \
  .pivot(index = "sequence_name", columns = "cluster", values = "attribution") \
  .fillna(0)

hits_merged_embryo_per_seq_and_cluster_sum_subset_specific_scaled = \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset_specific - hits_merged_embryo_per_seq_and_cluster_sum_subset_specific.min(0)) / \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset_specific.max(0) - hits_merged_embryo_per_seq_and_cluster_sum_subset_specific.min(0))

hits_merged_embryo_per_seq_and_cluster_sum_subset_general_scaled = \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset_general - hits_merged_embryo_per_seq_and_cluster_sum_subset_general.min(0)) / \
  (hits_merged_embryo_per_seq_and_cluster_sum_subset_general.max(0) - hits_merged_embryo_per_seq_and_cluster_sum_subset_general.min(0))

pattern_order_manual = [
  12, 6, 7, 15, 10, 18, 2, 17, 1, 19
]

data = (
  hits_merged_embryo_per_seq_and_cluster_sum_subset_specific_scaled[pattern_order_manual] > 0.1
) * 1

cooc_specific = data.T @ data
for cluster in pattern_order_manual:
  cooc_specific.loc[cluster, cluster] = 0

cooc_perc_specific = (cooc_specific / cooc_specific.sum()).T

data = (
  hits_merged_embryo_per_seq_and_cluster_sum_subset_general_scaled[pattern_order_manual] > 0.1
) * 1

cooc_general = data.T @ data
for cluster in pattern_order_manual:
  cooc_general.loc[cluster, cluster] = 0

cooc_perc_general = (cooc_general / cooc_general.sum()).T

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(pattern_order_manual), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)
cluster_to_color = {
  c: mapper.to_rgba(i)
  for i, c in enumerate(pattern_order_manual)
}

fig, ax = plt.subplots(figsize = (8, 4))
p = 1 / (len(pattern_order_manual) - 1)
cluster = 6
y = 0
cooc_perc = cooc_perc_specific
cooc = cooc_specific
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
y = 1
cooc_perc = cooc_perc_general
cooc = cooc_general
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
_ = ax.set_yticks(
  np.arange(2),
  labels = ["specific", "general"]
)
ax.grid(color = "black")
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(10) / 10)
fig.tight_layout()
fig.savefig("perc_other_embryo_general_specific.png")


fig, ax = plt.subplots(figsize = (8, 4))
p = 1 / (len(pattern_order_manual) - 1)
cluster = 6
y = 0
cooc_perc = cooc_specific
cooc = cooc_specific
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05), ncols = len(pattern_order_manual) // 2)
y = 1
cooc_perc = cooc_general
cooc = cooc_general
left = 0.0
_sorted_vals = cooc_perc.loc[cluster]
n = sum(cooc.loc[cluster])
for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
  k = cooc.loc[cluster, _cluster]
  t = binomtest(k, n, p, alternative = "greater")
  stat, pval = t.statistic, t.pvalue
  _ = ax.barh(
    y = y,
    left = left,
    width = _width,
    label = _cluster,
    color = cluster_to_color[_cluster],
    lw = 1, edgecolor = "black"
  )
  if pval < 1e-6 and (_cluster != cluster):
    _ = ax.text(
      x = left + _width / 2, y = y,
      s = "*", va= "center", ha = "center",
      weight = "bold"
    )
  left += _width
_ = ax.set_yticks(
  np.arange(2),
  labels = ["specific", "general"]
)
ax.grid(color = "black")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("abs_other_embryo_general_specific.png")

```

```python

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd

###############################################################################################################
#                                             LOAD DATA                                                       #
###############################################################################################################

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

sc.settings.figdir = "."

sc.pl.umap(
    adata_embryo_progenitor,
    color=["FOXA2", "NKX2-2", "PAX6"],
    s=50,
    layer="log_cpm",
    vmax=1.5,
    save="_dv_embr.pdf",
    components=["2, 3"],
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


## plot topic contr

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


organoid_progenitor_cell_topic = pd.read_table(
  "../../data_prep_new/organoid_data/ATAC/progenitor_cell_topic_contrib.tsv",
  index_col = 0
)

organoid_progenitor_cell_topic.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in organoid_progenitor_cell_topic
]

def atac_to_rna(l):
  bc, sample_id = l.strip().split("-1", 1)
  sample_id = sample_id.split("___")[-1]
  return bc + "-1" + "-" + sample_id_to_num[sample_id]

organoid_progenitor_cell_topic.index = [
  atac_to_rna(x) for x in organoid_progenitor_cell_topic.index
]

embryo_progenitor_cell_topic = pd.read_table(
  "../../data_prep_new/embryo_data/ATAC/progenitor_cell_topic_contrib.tsv",
  index_col = 0
)

embryo_progenitor_cell_topic.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in embryo_progenitor_cell_topic
]

embryo_progenitor_cell_topic.index = [
  x.split("___")[0] + "-1" + "___" + x.split("___")[1] for x in embryo_progenitor_cell_topic.index]

import matplotlib
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



fig, ax = plt.subplots()
organoid_cells_both = list(set(organoid_progenitor_cell_topic.index) & set(adata_organoid_progenitor.obs_names))
r_values = organoid_progenitor_cell_topic.loc[organoid_cells_both, model_index_to_topic_name_organoid(33 - 1)].values
g_values = organoid_progenitor_cell_topic.loc[organoid_cells_both, model_index_to_topic_name_organoid(38 - 1)].values
b_values = organoid_progenitor_cell_topic.loc[organoid_cells_both, model_index_to_topic_name_organoid(36 - 1)].values
rgb_scatter_plot(
    x=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    #r_vmin=0,
    #r_vmax=1.5,
    g_vmin=0,
    g_vmax=0.1,
    #b_vmin=0,
    #b_vmax=1.5,
)
fig.savefig("rgb_topics_33_38_36_organoid.png")

fig, ax = plt.subplots()
organoid_cells_both = list(set(organoid_progenitor_cell_topic.index) & set(adata_organoid_progenitor.obs_names))
r_values = organoid_progenitor_cell_topic.loc[organoid_cells_both, model_index_to_topic_name_organoid(54 - 1)].values
g_values = organoid_progenitor_cell_topic.loc[organoid_cells_both, model_index_to_topic_name_organoid(48 - 1)].values
b_values = np.zeros_like(g_values)
rgb_scatter_plot(
    x=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    r_vmin=0,
    r_vmax=0.1,
  #g_vmin=0,
  #g_vmax=0.1,
    #b_vmin=0,
    #b_vmax=1.5,
)
fig.savefig("rgb_topics_54_48_organoid.png")

fig, ax = plt.subplots()
embryo_cells_both = list(set(embryo_progenitor_cell_topic.index) & set(adata_embryo_progenitor.obs_names))
r_values = embryo_progenitor_cell_topic.loc[embryo_cells_both, model_index_to_topic_name_embryo(34 - 1)].values
g_values = embryo_progenitor_cell_topic.loc[embryo_cells_both, model_index_to_topic_name_embryo(38 - 1)].values
b_values = embryo_progenitor_cell_topic.loc[embryo_cells_both, model_index_to_topic_name_embryo(79 - 1)].values
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 2],
    ax=ax,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    #r_vmin=0,
    #r_vmax=1.5,
    g_vmin=0,
    g_vmax=0.2,
    #b_vmin=0,
    #b_vmax=1.5,
)
fig.savefig("rgb_topics_33_38_36_embryo.png")

fig, ax = plt.subplots()
embryo_cells_both = list(set(embryo_progenitor_cell_topic.index) & set(adata_embryo_progenitor.obs_names))
r_values = embryo_progenitor_cell_topic.loc[embryo_cells_both, model_index_to_topic_name_embryo(88 - 1)].values
g_values = embryo_progenitor_cell_topic.loc[embryo_cells_both, model_index_to_topic_name_embryo(58 - 1)].values
b_values = np.zeros_like(g_values)
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 2],
    ax=ax,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
  #r_vmin=0,
  # r_vmax=0.1,
  #g_vmin=0,
  #g_vmax=0.1,
    #b_vmin=0,
    #b_vmax=1.5,
)
fig.savefig("rgb_topics_54_48_embryo.png")


cell_topic_bin_organoid = pd.read_table(
    "../../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)

from pycisTopic.topic_binarization import binarize_topics
cells, scores, thresholds = binarize_topics(embryo_progenitor_cell_topic.to_numpy(), embryo_progenitor_cell_topic.index, "li")

cell_topic_bin_embryo = dict(
  cell_barcode = [],
  topic_name = [],
  group = [],
  topic_prob = []
)
for topic_idx in range(len(cells)):
  cell_topic_bin_embryo["cell_barcode"].extend( cells[topic_idx] )
  cell_topic_bin_embryo["topic_name"].extend( np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx])) )
  cell_topic_bin_embryo["group"].extend( np.repeat("progenitor", len(cells[topic_idx])) )
  cell_topic_bin_embryo["topic_prob"].extend( scores[topic_idx] )

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

##

exp_organoid = adata_organoid_progenitor.to_df(layer="log_cpm")

organoid_progenitor_topics_to_show = [
  33,
  38,
  36,
  54,
  48
]

SHH_expr_org_per_topic = {}
for topic in organoid_progenitor_topics_to_show:
  cell_names = list(
    set(
      [
        atac_to_rna(x)
        for x in cell_topic_bin_organoid.set_index("topic_name").loc[
          model_index_to_topic_name_organoid(topic - 1).replace("progenitor", "").replace("_", ""), "cell_barcode"
      ].values
      ]
    ) & set(adata_organoid_progenitor.obs_names)
  )
  SHH_expr_org_per_topic[topic] = np.exp(exp_organoid.loc[cell_names, "SHH"].values)

fig, ax = plt.subplots()
_ = ax.boxplot(
  SHH_expr_org_per_topic.values(),
  labels = SHH_expr_org_per_topic.keys()
)
fig.savefig("organoid_shh_bxplot.pdf")

exp_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")

embryo_progenitor_topics_to_show = [
  34,
  38,
  79,
  88,
  58
]

SHH_expr_org_per_topic = {}
for topic in embryo_progenitor_topics_to_show:
  cell_names = list(
    set(
      cell_topic_bin_embryo.set_index("topic_name").loc[
          model_index_to_topic_name_embryo(topic - 1).replace("progenitor", "").replace("_", ""), "cell_barcode"
      ].values
    ) & set(adata_embryo_progenitor.obs_names)
  )
  SHH_expr_org_per_topic[topic] = np.exp(exp_embryo.loc[cell_names, "SHH"].values)

fig, ax = plt.subplots()
_ = ax.boxplot(
  SHH_expr_org_per_topic.values(),
  labels = SHH_expr_org_per_topic.keys()
)
fig.savefig("embryo_shh_bxplot.pdf")


import pyBigWig
import os
import gzip

transcript_id = "NM_000193"

transcript = {
  "exon": [],
  "3UTR": [],
  "5UTR": [],
  "transcript": []
}
with gzip.open("/home/VIB.LOCAL/seppe.dewinter/sdewin/resources/hg38/hg38.refGene.gtf.gz") as f:
  for line in f:
    line = line.decode().strip()
    if f'transcript_id "{transcript_id}"' not in line:
      continue
    _, _, t, start, stop = line.split()[0:5]
    if t in transcript:
      transcript[t].append((int(start), int(stop)))

transcript_feature_to_size = {
  "exon": 1,
  "3UTR": .5,
  "5UTR": .5
}

bw_dir_organoid = "../../data_prep_new/organoid_data/ATAC/bw_per_topic"

topic_organoid_to_bw = {
  topic: os.path.join(
    bw_dir_organoid,
    model_index_to_topic_name_organoid(topic - 1)[::-1].replace("_", "", 1)[::-1] + ".fragments.tsv.bigWig"
  )
  for topic in organoid_progenitor_topics_to_show
}

SFPE1_hg38_coord = (155824153, 155824653)
SFPE2_hg38_coord = (155804373, 155804873)

locus = ("chr7", 155_799_664, 155_827_483)
nbp_per_bin = 1
nbins = (locus[2] - locus[1]) // nbp_per_bin

def None_to_0(x):
  if x is None:
    return 0
  else:
    return x

topic_organoid_to_locus_val = {}
for topic in topic_organoid_to_bw:
  print(topic)
  with pyBigWig.open(topic_organoid_to_bw[topic]) as bw:
    topic_organoid_to_locus_val[topic] = np.array(list(map(None_to_0, bw.stats(*locus, nBins = nbins))))


fig, axs = plt.subplots(nrows = len(topic_organoid_to_locus_val), figsize = (20, 10), sharex = True, sharey = True)
for ax, topic in zip(axs.ravel(), topic_organoid_to_locus_val):
  ax.fill_between(
    np.arange(locus[1], locus[2]),
    np.zeros(locus[2] - locus[1]),
    topic_organoid_to_locus_val[topic]
  )
ax.plot(
  [SFPE1_hg38_coord[0], SFPE1_hg38_coord[1]],
  [-1, -1],
  color = "black",
  lw = 2
)
ax.plot(
  [SFPE2_hg38_coord[0], SFPE2_hg38_coord[1]],
  [-1, -1],
  color = "black",
  lw = 2
)
for k in transcript:
  if k == "transcript":
    for start, stop in transcript[k]:
      ax.plot(
        [start, stop],
        [-2, -2],
        color = "black",
        lw = 2
      )
  else:
    size = transcript_feature_to_size[k]
    for start, stop in transcript[k]:
      p = matplotlib.patches.Rectangle(
        (start, -2 - size / 2), stop - start, size,
        facecolor = "black",
        lw = 0, fill = True
      )
      ax.add_patch(p)
fig.tight_layout()
fig.savefig("organoid_shh_locus.pdf")


bw_dir_embryo = "../../data_prep_new/embryo_data/ATAC/bw_per_topic"

topic_embryo_to_bw = {
  topic: os.path.join(
    bw_dir_embryo,
    model_index_to_topic_name_embryo(topic - 1)[::-1].replace("_", "", 1)[::-1] + ".fragments.tsv.bigWig"
  )
  for topic in embryo_progenitor_topics_to_show
}

SFPE1_hg38_coord = (155824153, 155824653)
SFPE2_hg38_coord = (155804373, 155804873)

locus = ("chr7", 155_799_664, 155_827_483)
nbp_per_bin = 1
nbins = (locus[2] - locus[1]) // nbp_per_bin

def None_to_0(x):
  if x is None:
    return 0
  else:
    return x

topic_embryo_to_locus_val = {}
for topic in topic_embryo_to_bw:
  print(topic)
  with pyBigWig.open(topic_embryo_to_bw[topic]) as bw:
    topic_embryo_to_locus_val[topic] = np.array(list(map(None_to_0, bw.stats(*locus, nBins = nbins))))


fig, axs = plt.subplots(nrows = len(topic_embryo_to_locus_val), figsize = (20, 10), sharex = True, sharey = True)
for ax, topic in zip(axs.ravel(), topic_embryo_to_locus_val):
  ax.fill_between(
    np.arange(locus[1], locus[2]),
    np.zeros(locus[2] - locus[1]),
    topic_embryo_to_locus_val[topic]
  )
ax.plot(
  [SFPE1_hg38_coord[0], SFPE1_hg38_coord[1]],
  [-1, -1],
  color = "black",
  lw = 2
)
ax.plot(
  [SFPE2_hg38_coord[0], SFPE2_hg38_coord[1]],
  [-1, -1],
  color = "black",
  lw = 2
)
for k in transcript:
  if k == "transcript":
    for start, stop in transcript[k]:
      ax.plot(
        [start, stop],
        [-2, -2],
        color = "black",
        lw = 2
      )
  else:
    size = transcript_feature_to_size[k]
    for start, stop in transcript[k]:
      p = matplotlib.patches.Rectangle(
        (start, -2 - size / 2), stop - start, size,
        facecolor = "black",
        lw = 0, fill = True
      )
      ax.add_patch(p)
fig.tight_layout()
fig.savefig("embryo_shh_locus.pdf")



```





