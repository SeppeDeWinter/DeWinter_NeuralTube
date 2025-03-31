import os

os.environ["CUDA_VISIBALE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import matplotlib
import scanpy as sc
import pandas as pd
import numpy as np
from dataclasses import dataclass
import h5py
from typing import Self
import tensorflow as tf
from tqdm import tqdm
from crested.tl._explainer_tf import Explainer
from crested.utils._seq_utils import one_hot_encode_sequence
import pickle
import logomaker
from functools import reduce
from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import torch
import seaborn as sns
from pycisTopic.topic_binarization import binarize_topics
import networkx as nx
from scipy.stats import binomtest
import json
import requests
import time

###############################################################################################################
#                                                                                                             #
#                                                   FUNCTIONS                                                 #
#                                                                                                             #
###############################################################################################################


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


def region_to_chrom_start_end(r):
    chrom, start, end = r.replace(":", "-").split("-")
    return chrom, int(start), int(end)


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
        region_names: list[str],
    ):
        self.contrib_scores = p["contrib_scores"][seqlet_idx]
        self.hypothetical_contrib_scores = p["hypothetical_contribs"][seqlet_idx]
        self.ppm = p["sequence"][seqlet_idx]
        self.start = p["start"][seqlet_idx]
        self.end = p["end"][seqlet_idx]
        self.is_revcomp = p["is_revcomp"][seqlet_idx]
        region_idx = p["example_idx"][seqlet_idx]
        self.region_name = region_names[region_idx]
        self.region_one_hot = ohs[region_idx]
        if (
            not np.all(self.ppm == self.region_one_hot[self.start : self.end])
            and not self.is_revcomp
        ) or (
            not np.all(
                self.ppm[::-1, ::-1] == self.region_one_hot[self.start : self.end]
            )
            and self.is_revcomp
        ):
            raise ValueError(
                f"ppm does not match onehot\n"
                + f"region_idx\t{region_idx}\n"
                + f"start\t\t{self.start}\n"
                + f"end\t\t{self.end}\n"
                + f"is_revcomp\t{self.is_revcomp}\n"
                + f"{self.ppm.argmax(1)}\n"
                + f"{self.region_one_hot[self.start: self.end].argmax(1)}"
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
    def __init__(
        self,
        p: h5py._hl.group.Group,
        is_pos: bool,
        ohs: np.ndarray,
        region_names: list[str],
    ):
        self.contrib_scores = p["contrib_scores"][:]
        self.hypothetical_contrib_scores = p["hypothetical_contribs"][:]
        self.ppm = p["sequence"][:]
        self.is_pos = is_pos
        self.seqlets = [
            Seqlet(p["seqlets"], i, ohs, region_names)
            for i in range(p["seqlets"]["n_seqlets"][0])
        ]
        self.subpatterns = [
            ModiscoPattern(p[sub], is_pos, ohs, region_names)
            for sub in p.keys()
            if sub.startswith("subpattern_")
        ]
    def __repr__(self):
        return f"ModiscoPattern with {len(self.seqlets)} seqlets"
    def ic(self, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
        return (
            self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)
        ).sum(1)
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
                    ModiscoPattern(
                        f[pos_neg][pattern],
                        pos_neg == "pos_patterns",
                        ohs,
                        region_names,
                    ),
                )


def get_value_seqlets(seqlets: list[Seqlet], v: np.ndarray):
    if v.shape[0] != len(seqlets):
        raise ValueError(f"{v.shape[0]} != {len(seqlets)}")
    for i, seqlet in enumerate(seqlets):
        if seqlet.is_revcomp:
            yield v[i, seqlet.start : seqlet.end, :][::-1, ::-1]
        else:
            yield v[i, seqlet.start : seqlet.end, :]


def allign_patterns_of_cluster_for_topic(
    pattern_to_topic_to_grad: dict[str, dict[int, np.ndarray]],
    pattern_metadata: pd.DataFrame,
    cluster_id: int,
    topic: int,
):
    P = []
    O = []
    cluster_patterns = pattern_metadata.query(
        "cluster_sub_cluster == @cluster_id"
    ).index.to_list()
    for pattern_name in cluster_patterns:
        pattern_grads, pattern_ohs = pattern_to_topic_to_grad[pattern_name][topic]
        ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
            pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
        ]
        pattern_grads_ic = [p[ic_start:ic_end] for p in pattern_grads]
        pattern_ohs_ic = [o[ic_start:ic_end] for o in pattern_ohs]
        if is_rc_to_root:
            pattern_grads_ic = [p[::-1, ::-1] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[::-1, ::-1] for o in pattern_ohs_ic]
        if offset_to_root > 0:
            pattern_grads_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), p_ic])
                for p_ic in pattern_grads_ic
            ]
            pattern_ohs_ic = [
                np.concatenate([np.zeros((offset_to_root, 4)), o_ic])
                for o_ic in pattern_ohs_ic
            ]
        elif offset_to_root < 0:
            pattern_grads_ic = [p[abs(offset_to_root) :, :] for p in pattern_grads_ic]
            pattern_ohs_ic = [o[abs(offset_to_root) :, :] for o in pattern_ohs_ic]
        P.extend(pattern_grads_ic)
        O.extend(pattern_ohs_ic)
    max_len = max([p.shape[0] for p in P])
    P = [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P]
    O = [np.concatenate([o, np.zeros((max_len - o.shape[0], 4))]) for o in O]
    return np.array(P), np.array(O)


def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def merge_and_absmax(left, right, on, max_on, l):
    global a
    if a:
        print(" " * (l - 1) + "|", end="\r", flush=True)
        a = False
    print("x", end="", flush=True)
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).T.apply(absmax)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1).copy()


def merge_and_max(left, right, on, max_on, l):
    global a
    if a:
        print(" " * (l - 1) + "|", end="\r", flush=True)
        a = False
    print("x", end="", flush=True)
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1).copy()


# load organoid RNA data and subset for ATAC cells
adata_organoid = sc.read_h5ad("../figure_1/adata_organoid.h5ad")

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

adata_organoid_neuron = sc.read_h5ad("../figure_1/adata_organoid_neuron.h5ad")

adata_embryo_neuron = sc.read_h5ad("../figure_1/adata_embryo_neuron.h5ad")

# load cell topic

organoid_neuron_cell_topic = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neuron_cell_topic_contrib.tsv",
    index_col=0,
)

organoid_neuron_cell_topic.columns = [
    f"neuron_Topic_{c.replace('Topic', '')}" for c in organoid_neuron_cell_topic
]


def rename_organoid_atac_cell(l):
    bc, sample_id = l.strip().split("-1", 1)
    sample_id = sample_id.split("___")[-1]
    return bc + "-1" + "-" + sample_id_to_num[sample_id]


organoid_neuron_cell_topic.index = [
    rename_organoid_atac_cell(x) for x in organoid_neuron_cell_topic.index
]

embryo_neuron_cell_topic = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/neuron_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_neuron_cell_topic.columns = [
    f"neuron_Topic_{c.replace('Topic', '')}" for c in embryo_neuron_cell_topic
]

embryo_neuron_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_neuron_cell_topic.index
]

# load and score patterns

neuron_topics_organoid = [6, 4, 23, 24, 13, 2]

neuron_topics_embryo = [10, 8, 13, 24, 18, 29]

all_neuron_topics_organoid = [
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
    25,
]

all_neuron_topics_embryo = [
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
    30,
]


path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"

organoid_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_organoid_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)

organoid_model.load_weights(os.path.join(path_to_organoid_model, "model_epoch_23.hdf5"))

##

path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"

embryo_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_embryo_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)

embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))

organoid_dl_motif_dir = "../data_prep_new/organoid_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(all_neuron_topics_organoid):
    ohs = np.load(os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz"))[
        "oh"
    ]
    region_names = np.load(
        os.path.join(organoid_dl_motif_dir, f"gradients_Topic_{topic}.npz")
    )["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            organoid_dl_motif_dir,
            f"patterns_Topic_{topic}.hdf5",
        ),
        ohs=ohs,
        region_names=region_names,
    ):
        pattern_names_dl_organoid.append("organoid_" + name)
        patterns_dl_organoid.append(pattern)

embryo_dl_motif_dir = "../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(all_neuron_topics_embryo):
    ohs = np.load(os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz"))[
        "oh"
    ]
    region_names = np.load(
        os.path.join(embryo_dl_motif_dir, f"gradients_Topic_{topic}.npz")
    )["region_names"]
    for name, pattern in load_pattern_from_modisco(
        filename=os.path.join(
            embryo_dl_motif_dir,
            f"patterns_Topic_{topic}.hdf5",
        ),
        ohs=ohs,
        region_names=region_names,
    ):
        pattern_names_dl_embryo.append("embryo_" + name)
        patterns_dl_embryo.append(pattern)

all_patterns = [*patterns_dl_organoid, *patterns_dl_embryo]
all_pattern_names = [*pattern_names_dl_organoid, *pattern_names_dl_embryo]

pattern_metadata = pd.read_table("draft/pattern_metadata.tsv", index_col=0)

if not os.path.exists("pattern_to_topic_to_grad_organoid.pkl"):
    pattern_to_topic_to_grad_organoid = {}
    for pattern_name in tqdm(pattern_metadata.index):
        pattern = all_patterns[all_pattern_names.index(pattern_name)]
        oh_sequences = np.array(
            [x.region_one_hot for x in pattern.seqlets]
        )  # .astype(np.int8)
        pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
        pattern_to_topic_to_grad_organoid[pattern_name] = {}
        for topic in tqdm(all_neuron_topics_organoid, leave=False):
            class_idx = topic - 1
            explainer = Explainer(model=organoid_model, class_index=int(class_idx))
            # gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
            gradients_integrated = explainer.saliency_maps(X=oh_sequences)
            pattern_grads = list(
                get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze())
            )
            pattern_to_topic_to_grad_organoid[pattern_name][topic] = (
                pattern_grads,
                pattern_ohs,
            )
    pickle.dump(
        pattern_to_topic_to_grad_organoid,
        open("pattern_to_topic_to_grad_organoid.pkl", "wb"),
    )
else:
    pattern_to_topic_to_grad_organoid = pickle.load(
        open("pattern_to_topic_to_grad_organoid.pkl", "rb")
    )

if not os.path.exists("pattern_to_topic_to_grad_embryo.pkl"):
    pattern_to_topic_to_grad_embryo = {}
    for pattern_name in tqdm(pattern_metadata.index):
        pattern = all_patterns[all_pattern_names.index(pattern_name)]
        oh_sequences = np.array(
            [x.region_one_hot for x in pattern.seqlets]
        )  # .astype(np.int8)
        pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
        pattern_to_topic_to_grad_embryo[pattern_name] = {}
        for topic in tqdm(all_neuron_topics_embryo, leave=False):
            class_idx = topic - 1
            explainer = Explainer(model=embryo_model, class_index=int(class_idx))
            # gradients_integrated = explainer.integrated_grad(X = oh_sequences) change to this for real fig
            gradients_integrated = explainer.saliency_maps(X=oh_sequences)
            pattern_grads = list(
                get_value_seqlets(pattern.seqlets, gradients_integrated.squeeze())
            )
            pattern_to_topic_to_grad_embryo[pattern_name][topic] = (
                pattern_grads,
                pattern_ohs,
            )
    pickle.dump(
        pattern_to_topic_to_grad_embryo,
        open("pattern_to_topic_to_grad_embryo.pkl", "wb"),
    )
else:
    pattern_to_topic_to_grad_embryo = pickle.load(
        open("pattern_to_topic_to_grad_embryo.pkl", "rb")
    )

pattern_metadata["ic_start"] = 0
pattern_metadata["ic_stop"] = 30


def ic(ppm, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)


def ic_trim(ic, min_v: float) -> tuple[int, int]:
    delta = np.where(np.diff((ic > min_v) * 1))[0]
    if len(delta) == 0:
        return 0, 0
    start_index = min(delta)
    end_index = max(delta)
    return start_index, end_index + 1


cluster_to_topic_to_avg_pattern_organoid = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_organoid[cluster] = {}
    for topic in all_neuron_topics_organoid:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.8)
        cluster_to_topic_to_avg_pattern_organoid[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_embryo = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
    for topic in all_neuron_topics_embryo:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.8)
        cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

selected_clusters = [1.1, 2.1, 8.0, 2.2, 4.0, 5.2, 6.0, 7.3, 3.1, 7.5, 5.1]

cluster_to_name = {
    1.1: "TEAD1",
    2.1: "FOX(P1|P2|P4|J3|A2|O1)",
    2.2: "SOX2",
    3.1: "ONECUT(1|2|3)",
    4.0: "(NR2F6)|(NR1D2)",
    5.1: "ZEB(1|2)",
    5.2: "(ASCL1)|(NEUROD(1|4))",
    6.0: "EBF(1|2|3)",
    7.3: "(NKX2-2)|(ISL1)",
    7.5: "GATA2",
    8.0: "RFX(3|4)",
}

organoid_embryo = [(6, 10), (4, 8), (23, 13), (24, 24), (13, 18), (2, 29)]

motifs = {
    n: pattern.ppm[range(*pattern.ic_trim(0.2))].T
    for n, pattern in zip(all_pattern_names, all_patterns)
    if n in pattern_metadata.index
}

all_hits_organoid_subset = []
for topic in neuron_topics_organoid:
    f = f"gradients_Topic_{topic}.npz"
    print(f)
    ohs = np.load(os.path.join(organoid_dl_motif_dir, f))["oh"]
    attr = np.load(os.path.join(organoid_dl_motif_dir, f))["gradients_integrated"]
    region_names = np.load(os.path.join(organoid_dl_motif_dir, f))["region_names"]
    hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2))
    hits = pd.concat(hits)
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["cluster"] = [
        pattern_metadata.loc[m, "cluster_sub_cluster"] for m in hits["motif_name"]
    ]
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    all_hits_organoid_subset.append(hits)

a = True
hits_merged_organoid_subset = reduce(
    lambda left, right: merge_and_max(
        left[
            [
                "motif_name",
                "cluster",
                "sequence_name",
                "start",
                "end",
                "strand",
                "attribution",
                "p-value",
                "-logp",
            ]
        ],
        right[
            [
                "motif_name",
                "cluster",
                "sequence_name",
                "start",
                "end",
                "strand",
                "attribution",
                "p-value",
                "-logp",
            ]
        ],
        on=[
            "motif_name",
            "cluster",
            "sequence_name",
            "start",
            "end",
            "strand",
            "p-value",
            "attribution",
        ],
        max_on="-logp",
        l=len(all_hits_organoid_subset),
    ),
    all_hits_organoid_subset,
)

hits_merged_organoid_subset_per_seq_and_cluster_max = (
    hits_merged_organoid_subset.groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
    .astype(float)
)

hits_merged_organoid_subset_per_seq_and_cluster_max_scaled = (
    hits_merged_organoid_subset_per_seq_and_cluster_max
    / hits_merged_organoid_subset_per_seq_and_cluster_max.sum()
)


region_order_organoid_subset = []
for x in tqdm(all_hits_organoid_subset):
    for r in x["sequence_name"]:
        if r not in region_order_organoid_subset:
            region_order_organoid_subset.append(r)

hits_organoid_bin = (
    hits_merged_organoid_subset_per_seq_and_cluster_max_scaled.loc[
        region_order_organoid_subset, selected_clusters
    ].abs()
    > 0.0008
)
hits_organoid_bin.columns = [cluster_to_name[x] for x in hits_organoid_bin.columns]


def jaccard(s_a: set[str], s_b: set[str]):
    return len(s_a & s_b) / len(s_a | s_b)


jaccard_organoid = pd.DataFrame(
    index=hits_organoid_bin.columns, columns=hits_organoid_bin.columns
).fillna(0)

for tf_1 in jaccard_organoid.columns:
    for tf_2 in jaccard_organoid.index:
        jaccard_organoid.loc[tf_1, tf_2] = jaccard(
            set(hits_organoid_bin.loc[hits_organoid_bin[tf_1]].index),
            set(hits_organoid_bin.loc[hits_organoid_bin[tf_2]].index),
        )


cell_topic_bin_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)

cell_topic_bin_organoid.cell_barcode = [
    rename_organoid_atac_cell(x) for x in cell_topic_bin_organoid.cell_barcode
]

exp_organoid = adata_organoid_neuron.to_df(layer="log_cpm")

exp_per_topic_organoid = pd.DataFrame(
    index=[
        model_index_to_topic_name_organoid(t - 1)
        .replace("neuron_", "")
        .replace("_", "")
        for t in neuron_topics_organoid
    ],
    columns=exp_organoid.columns,
)

for topic in tqdm(exp_per_topic_organoid.index):
    cells = list(
        set(exp_organoid.index)
        & set(
            cell_topic_bin_organoid.query(
                "group == 'neuron' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_organoid.loc[topic] = exp_organoid.loc[cells].mean()

cells, scores, thresholds = binarize_topics(
    embryo_neuron_cell_topic.to_numpy(),
    embryo_neuron_cell_topic.index,
    "li",
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(np.repeat("neuron", len(cells[topic_idx])))
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_embryo = adata_embryo_neuron.to_df(layer="log_cpm")

exp_per_topic_embryo = pd.DataFrame(
    index=[
        model_index_to_topic_name_embryo(t - 1).replace("neuron_", "").replace("_", "")
        for t in neuron_topics_embryo
    ],
    columns=exp_embryo.columns,
)

for topic in tqdm(exp_per_topic_embryo.index):
    cells = list(
        set(exp_embryo.index)
        & set(
            cell_topic_bin_embryo.query(
                "group == 'neuron' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_embryo.loc[topic] = exp_embryo.loc[cells].mean()


import re

cluster_to_tf = cluster_to_name

tf_expr_matrix_per_topic_organoid = (
    pd.DataFrame(exp_per_topic_organoid)
)

tf_expr_matrix_per_topic_organoid = tf_expr_matrix_per_topic_organoid[
    [c for c in tf_expr_matrix_per_topic_organoid.columns if any([re.fullmatch(p, c) for p in cluster_to_tf.values() ])]
]

tf_expr_matrix_per_topic_organoid.index = [
    topic_name_to_model_index_organoid("neuron_Topic_" + t.replace("Topic", "")) + 1
    for t in tf_expr_matrix_per_topic_organoid.index
]

tf_expr_matrix_per_topic_organoid = tf_expr_matrix_per_topic_organoid[
    tf_expr_matrix_per_topic_organoid.idxmax().sort_values(
        key = lambda X: [[oe[0] for oe in organoid_embryo].index(x) for x in X]).index
]

tf_expr_matrix_per_topic_embryo = (
    pd.DataFrame(exp_per_topic_embryo)
)

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    [c for c in tf_expr_matrix_per_topic_embryo.columns if any([re.fullmatch(p, c) for p in cluster_to_tf.values() ])]
]

tf_expr_matrix_per_topic_embryo.index = [
    topic_name_to_model_index_embryo("neuron_Topic_" + t.replace("Topic", "")) + 1
    for t in tf_expr_matrix_per_topic_embryo.index
]

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    tf_expr_matrix_per_topic_organoid.columns
]


N_PIXELS_PER_GRID = 50

plt.style.use("../paper.mplstyle")

fig = plt.figure()
width, height = fig.get_size_inches()
n_w_pixels = fig.get_dpi() * width
n_h_pixels = fig.get_dpi() * height
ncols = int((n_w_pixels) // N_PIXELS_PER_GRID)
nrows = int((n_h_pixels) // N_PIXELS_PER_GRID)
gs = fig.add_gridspec(
    nrows, ncols, wspace=0.05, hspace=0.1, left=0.05, right=0.97, bottom=0.05, top=0.95
)
ax_organoid_umap_1 = fig.add_subplot(gs[0:5, 0:5])
ax_organoid_umap_2 = fig.add_subplot(gs[5:10, 0:5])
ax_embryo_umap_1 = fig.add_subplot(gs[11:16, 0:5])
ax_embryo_umap_2 = fig.add_subplot(gs[16:21, 0:5])
organoid_cells_both = list(
    set(organoid_neuron_cell_topic.index) & set(adata_organoid_neuron.obs_names)
)
rgb_scatter_plot(
    x=adata_organoid_neuron[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_neuron[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_organoid_umap_1,
    g_cut=0,
    r_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[0] - 1),
    ].values,
    g_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[1] - 1),
    ].values,
    b_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[2] - 1),
    ].values,
    r_name=model_index_to_topic_name_organoid(neuron_topics_organoid[0] - 1).replace(
        "neuron_", ""
    ),
    g_name=model_index_to_topic_name_organoid(neuron_topics_organoid[1] - 1).replace(
        "neuron_", ""
    ),
    b_name=model_index_to_topic_name_organoid(neuron_topics_organoid[2] - 1).replace(
        "neuron_", ""
    ),
    r_vmin=0,
    r_vmax=0.15,
    g_vmin=0,
    g_vmax=0.1,
)
rgb_scatter_plot(
    x=adata_organoid_neuron[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_neuron[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_organoid_umap_2,
    g_cut=0,
    r_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[3] - 1),
    ].values,
    g_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[4] - 1),
    ].values,
    b_values=organoid_neuron_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neuron_topics_organoid[5] - 1),
    ].values,
    r_name=model_index_to_topic_name_organoid(neuron_topics_organoid[3] - 1).replace(
        "neuron_", ""
    ),
    g_name=model_index_to_topic_name_organoid(neuron_topics_organoid[4] - 1).replace(
        "neuron_", ""
    ),
    b_name=model_index_to_topic_name_organoid(neuron_topics_organoid[4] - 1).replace(
        "neuron_", ""
    ),
)
embryo_cells_both = list(
    set(embryo_neuron_cell_topic.index) & set(adata_embryo_neuron.obs_names)
)
rgb_scatter_plot(
    x=adata_embryo_neuron[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_neuron[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_embryo_umap_1,
    g_cut=0,
    r_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[0] - 1),
    ].values,
    g_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[1] - 1),
    ].values,
    b_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[2] - 1),
    ].values,
    r_name=model_index_to_topic_name_embryo(neuron_topics_embryo[0] - 1).replace(
        "neuron_", ""
    ),
    g_name=model_index_to_topic_name_embryo(neuron_topics_embryo[1] - 1).replace(
        "neuron_", ""
    ),
    b_name=model_index_to_topic_name_embryo(neuron_topics_embryo[2] - 1).replace(
        "neuron_", ""
    ),
    r_vmin=0,
    r_vmax=0.1,
)
rgb_scatter_plot(
    x=adata_embryo_neuron[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_neuron[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_embryo_umap_2,
    g_cut=0,
    r_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[3] - 1),
    ].values,
    g_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[4] - 1),
    ].values,
    b_values=embryo_neuron_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neuron_topics_embryo[5] - 1),
    ].values,
    r_name=model_index_to_topic_name_embryo(neuron_topics_embryo[3] - 1).replace(
        "neuron_", ""
    ),
    g_name=model_index_to_topic_name_embryo(neuron_topics_embryo[4] - 1).replace(
        "neuron_", ""
    ),
    b_name=model_index_to_topic_name_embryo(neuron_topics_embryo[5] - 1).replace(
        "neuron_", ""
    ),
    g_vmin=0,
    g_vmax=0.05,
)
x_current = 7
y_current = 0
pattern_height = 2
pattern_width = 3
for i, cluster in enumerate(tqdm(selected_clusters)):
    YMIN = np.inf
    YMAX = -np.inf
    axs = []
    for j, (topic_org, topic_embr) in enumerate(organoid_embryo):
        ax = fig.add_subplot(
            gs[
                y_current + i * pattern_height : y_current
                + i * pattern_height
                + pattern_height,
                x_current + j * pattern_width : x_current
                + j * pattern_width
                + pattern_width,
            ]
        )
        if i == 0:
            ax.set_title(f"Topic {topic_org} {topic_embr}")
        if topic_org is not None:
            pwm = cluster_to_topic_to_avg_pattern_organoid[cluster][topic_org]
            if cluster == 2.2:
                pwm = pwm[0:8, :]
            if cluster == 3.1:
                pwm = pwm[0:10, :]
            _ = logomaker.Logo(
                pd.DataFrame(pwm, columns=["A", "C", "G", "T"]),
                ax=ax,
                # alpha=0.5,
                # color_scheme="red",
            )
            ymn, ymx = ax.get_ylim()
            YMIN = min(ymn, YMIN)
            YMAX = max(ymx, YMAX)
        if (topic_embr is not None) and True:
            pwm = cluster_to_topic_to_avg_pattern_embryo[cluster][topic_embr]
            if cluster == 2.2:
                pwm = pwm[0:8, :]
            if cluster == 3.1:
                pwm = pwm[0:10]
            _ = logomaker.Logo(
                pd.DataFrame(-pwm, columns=["A", "C", "G", "T"]),
                ax=ax,
                edgecolor="black",
                edgewidth=0.4,
                # alpha=0.5,
                # color_scheme="blue",
            )
            ymn, ymx = ax.get_ylim()
            YMIN = min(ymn, YMIN)
            YMAX = max(ymx, YMAX)
        if j == 0:
            _ = ax.set_ylabel(f"cluster_{cluster}")
        axs.append(ax)
    for ax, (organoid_topic, embryo_topic) in zip(axs, organoid_embryo):
        _ = ax.set_ylim(YMIN, YMAX)
        _ = ax.set_axis_off()
ax_hit_heatmap = fig.add_subplot(gs[0:21, 25:30])
sns.heatmap(
    hits_merged_organoid_subset_per_seq_and_cluster_max_scaled.loc[
        region_order_organoid_subset, selected_clusters
    ].astype(float),
    yticklabels=False,
    xticklabels=[cluster_to_name[x] for x in selected_clusters],
    ax=ax_hit_heatmap,
    cmap="bwr",
    vmin=-0.0008,
    vmax=0.0008,
    cbar_kws=dict(shrink=0.5, format=lambda x, _: "{:.0e}".format(x)),
    cbar=False,
)
ax_hit_heatmap.set_ylabel("")
ax_organoid_expr_heatmap = fig.add_subplot(gs[0:12, 33:36])
ax_embryo_expr_heatmap = fig.add_subplot(gs[0:12, 36:39])
sns.heatmap(
    (
        (tf_expr_matrix_per_topic_organoid - tf_expr_matrix_per_topic_organoid.min())
        / (
            tf_expr_matrix_per_topic_organoid.max()
            - tf_expr_matrix_per_topic_organoid.min()
        )
    )
    .astype(float)
    .T,
    cmap="Greys",
    ax=ax_organoid_expr_heatmap,
    xticklabels=True,
    yticklabels=True,
    cbar=False,
    vmin=0,
    vmax=1,
    lw=0.5,
    linecolor="black",
)
sns.heatmap(
    (
        (tf_expr_matrix_per_topic_embryo - tf_expr_matrix_per_topic_embryo.min())
        / (
            tf_expr_matrix_per_topic_embryo.max()
            - tf_expr_matrix_per_topic_embryo.min()
        )
    )
    .astype(float)
    .T,
    cmap="Greys",
    ax=ax_embryo_expr_heatmap,
    yticklabels=False,
    xticklabels=True,
    cbar=False,
    vmin=0,
    vmax=1,
    lw=0.5,
    linecolor="black",
)
# ax_network = fig.add_subplot(gs[12:20, 31:38])
ax = fig.add_subplot(gs[13:21, 33:39])
sns.heatmap(
    jaccard_organoid,
    cmap="viridis",
    vmin=0,
    vmax=0.2,
    ax=ax,
    linecolor="black",
    lw=0.5,
    cbar=False,
    yticklabels=True,
    xticklabels=True,
)
fig.tight_layout()
fig.savefig("Figure_6.png", transparent=False)
fig.savefig("Figure_6.pdf")
