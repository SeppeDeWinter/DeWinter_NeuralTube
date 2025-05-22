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
from bs4 import BeautifulSoup
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


def fetch_rsid_batched(rsids: list[str], batch_size: int, tries: int) -> list[str]:
    for i in range(len(rsids) // batch_size + 1):
        left = i * batch_size
        right = min(left + batch_size, len(rsids))
        got_result = False
        nth_try = 0
        while not got_result and nth_try < tries:
            response = requests.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&id={','.join(snps[left:right])}&report=DocSet"
            )
            if response.ok:
                got_result = True
                time.sleep(0.1)
                break
            else:
                nth_try += 1
                time.sleep(0.25)
        if got_result:
            data = response.content.decode().splitlines()
            for d in data:
                yield d
        else:
            raise ValueError(f"Did not get result after {tries} tries")


# load organoid RNA data and subset for ATAC cells
adata_organoid = sc.read_h5ad("../figure_1/adata_organoid.h5ad")

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

adata_organoid_neural_crest = sc.read_h5ad("../figure_1/adata_organoid_neural_crest.h5ad")

adata_embryo_neural_crest = sc.read_h5ad("../figure_1/adata_embryo_neural_crest.h5ad")


# load cell topic

organoid_neural_crest_cell_topic = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neural_crest_cell_topic_contrib.tsv",
    index_col=0,
)

organoid_neural_crest_cell_topic.columns = [
    f"neural_crest_Topic_{c.replace('Topic', '')}"
    for c in organoid_neural_crest_cell_topic
]


def rename_organoid_atac_cell(l):
    bc, sample_id = l.strip().split("-1", 1)
    sample_id = sample_id.split("___")[-1]
    return bc + "-1" + "-" + sample_id_to_num[sample_id]


organoid_neural_crest_cell_topic.index = [
    rename_organoid_atac_cell(x) for x in organoid_neural_crest_cell_topic.index
]

embryo_neural_crest_cell_topic = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/neural_crest_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_neural_crest_cell_topic.columns = [
    f"neural_crest_Topic_{c.replace('Topic', '')}"
    for c in embryo_neural_crest_cell_topic
]

embryo_neural_crest_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_neural_crest_cell_topic.index
]

# load and score patterns

neural_crest_topics_organoid = [
    62,
    60,
    65,
    59,
    58,
]

neural_crest_topics_embryo = [103, 105, 94, 91]

all_neural_crest_topics_organoid = (np.array([1, 3, 4, 5, 7, 9, 10]) + 55).tolist()

all_neural_crest_topics_embryo = (
    np.array(
        [
            1,
            4,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
            19,
            20,
            21,
            22,
            29,
        ]
    )
    + 90
).tolist()


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
for topic in tqdm(all_neural_crest_topics_organoid):
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
for topic in tqdm(all_neural_crest_topics_embryo):
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
        for topic in tqdm(all_neural_crest_topics_organoid, leave=False):
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
        for topic in tqdm(all_neural_crest_topics_embryo, leave=False):
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
    for topic in all_neural_crest_topics_organoid:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_organoid[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

cluster_to_topic_to_avg_pattern_embryo = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
    for topic in all_neural_crest_topics_embryo:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        ic_start, ic_end = ic_trim(ic((O.sum(0).T / O.sum(0).sum(1)).T), 0.2)
        cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (P * O).mean(0)[
            ic_start:ic_end
        ]

selected_clusters = [3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2]

cluster_to_name = {
    3.0: "TEAD(3|4)",
    13.1: "GRHL(1|2)",
    9.2: "ZEB(1|2)",
    14.0: "RFX(3|4)",
    11.1: "NR2(C|F)2",
    9.1: "TFAP2(A|B|C)",
    10.2: "ZIC(1|2|4|5)",
    2.2: "SOX(5|10)",
    2.1: "FOX(P1|P2|C1|C2|D1)",
    13.2: "(TWIST1)|(ALX(1|4))|(MSX1)|(PRRX1)",
}

organoid_embryo = [(62, 103), (60, 105), (65, 94), (59, None), (58, 91)]

from scipy import stats

corrs_patterns = []
for cluster in selected_clusters:
    corrs_patterns.append(
        stats.pearsonr(
            [
                cluster_to_topic_to_avg_pattern_organoid[cluster][o].sum() for o, e in organoid_embryo
                if not (o is None or e is None)
            ],
            [
                cluster_to_topic_to_avg_pattern_embryo[cluster][e].sum() for o, e in organoid_embryo
                if not (o is None or e is None)
            ]
        ).statistic
    )
print(np.mean(corrs_patterns))



motifs = {
    n: pattern.ppm[range(*pattern.ic_trim(0.2))].T
    for n, pattern in zip(all_pattern_names, all_patterns)
    if n in pattern_metadata.index
}

all_hits_organoid_subset = []
for topic in neural_crest_topics_organoid:
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

exp_organoid = adata_organoid_neural_crest.to_df(layer="log_cpm")

exp_per_topic_organoid = pd.DataFrame(
    index=[
        model_index_to_topic_name_organoid(t - 1)
        .replace("neural_crest_", "")
        .replace("_", "")
        for t in neural_crest_topics_organoid
    ],
    columns=exp_organoid.columns,
)

for topic in tqdm(exp_per_topic_organoid.index):
    cells = list(
        set(exp_organoid.index)
        & set(
            cell_topic_bin_organoid.query(
                "group == 'neural_crest' & topic_name == @topic"
            ).cell_barcode
        )
    )
    exp_per_topic_organoid.loc[topic] = exp_organoid.loc[cells].mean()

cells, scores, thresholds = binarize_topics(
    embryo_neural_crest_cell_topic.to_numpy(),
    embryo_neural_crest_cell_topic.index,
    "li",
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(
        np.repeat("neural_crest", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_embryo = adata_embryo_neural_crest.to_df(layer="log_cpm")

exp_per_topic_embryo = pd.DataFrame(
    index=[
        model_index_to_topic_name_embryo(t - 1)
        .replace("neural_crest_", "")
        .replace("_", "")
        for t in neural_crest_topics_embryo
    ],
    columns=exp_embryo.columns,
)

for topic in tqdm(exp_per_topic_embryo.index):
    cells = list(
        set(exp_embryo.index)
        & set(
            cell_topic_bin_embryo.query(
                "group == 'neural_crest' & topic_name == @topic"
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
    topic_name_to_model_index_organoid("neural_crest_Topic_" + t.replace("Topic", "")) + 1
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
    topic_name_to_model_index_embryo("neural_crest_Topic_" + t.replace("Topic", "")) + 1
    for t in tf_expr_matrix_per_topic_embryo.index
]

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    tf_expr_matrix_per_topic_organoid.columns
]


data = (
    hits_merged_organoid_subset_per_seq_and_cluster_max_scaled.loc[
        region_order_organoid_subset, selected_clusters
    ].abs()
    > 0.0008
) * 1
cooc = data.T @ data
for cluster in selected_clusters:
    cooc.loc[cluster, cluster] = 0

cooc_perc = (cooc / cooc.sum()).T

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(selected_clusters), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)

cluster_to_color = {
    cluster_to_name[c]: mapper.to_rgba(i) for i, c in enumerate(selected_clusters)
}

G_organoid = nx.DiGraph()

p = 1 / (len(selected_clusters) - 1)
for y, cluster in enumerate(selected_clusters):
    left = 0.0
    _sorted_vals = cooc_perc.loc[cluster]
    n = sum(cooc.loc[cluster])
    for _cluster, _width in zip(_sorted_vals.index, _sorted_vals.values):
        k = cooc.loc[cluster, _cluster]
        t = binomtest(k, n, p, alternative="greater")
        stat, pval = t.statistic, t.pvalue
        if pval < (0.01 / 100) and (_cluster != cluster):
            G_organoid.add_edge(
                cluster_to_name[cluster], cluster_to_name[_cluster], weight=_width
            )

pos = nx.forceatlas2_layout(G_organoid, seed=12)


@dataclass
class SequenceRefAlt:
    rsid: str
    ref: str
    alts: list[str]


seq_ref_alts = []
for x in json.load(
    open("../data_prep_new/facial_gwas/white/ref_alt_all_sign_snps.json")
):
    seq_ref_alts.append(SequenceRefAlt(rsid=x["rsid"], ref=x["ref"], alts=x["alts"]))


@dataclass
class PredictionRefAlt:
    rsid: str
    ref: np.ndarray
    alts: np.ndarray


refs = np.concatenate(
    [one_hot_encode_sequence(variant.ref) for variant in seq_ref_alts]
)

alts = np.concatenate(
    [
        np.concatenate([one_hot_encode_sequence(a) for a in variant.alts])
        for variant in seq_ref_alts
    ]
)

pred_ref_organoid = organoid_model.predict(refs)
pred_alt_organoid = organoid_model.predict(alts)

pred_ref_alts_organoid = []
alt_id = 0
for ref_id, variant in tqdm(enumerate(seq_ref_alts), total=len(seq_ref_alts)):
    n_alts = len(variant.alts)
    pred_ref_alts_organoid.append(
        PredictionRefAlt(
            rsid=variant.rsid,
            ref=pred_ref_organoid[ref_id : ref_id + 1, :],
            alts=pred_alt_organoid[alt_id : alt_id + n_alts, :],
        )
    )
    alt_id = alt_id + n_alts

pred_ref_embryo = embryo_model.predict(refs)
pred_alt_embryo = embryo_model.predict(alts)

pred_ref_alts_embryo = []
alt_id = 0
for ref_id, variant in tqdm(enumerate(seq_ref_alts), total=len(seq_ref_alts)):
    n_alts = len(variant.alts)
    pred_ref_alts_embryo.append(
        PredictionRefAlt(
            rsid=variant.rsid,
            ref=pred_ref_embryo[ref_id : ref_id + 1, :],
            alts=pred_alt_embryo[alt_id : alt_id + n_alts, :],
        )
    )
    alt_id = alt_id + n_alts


@dataclass
class DeltaPredRefAlt:
    rsid: str
    deltas: np.ndarray


delta_ref_alt_organoid = []
for variant in pred_ref_alts_organoid:
    delta = np.zeros_like(variant.alts)
    for i in range(variant.alts.shape[0]):
        delta[i] = variant.alts[i] - variant.ref[0]
    delta_ref_alt_organoid.append(DeltaPredRefAlt(rsid=variant.rsid, deltas=delta))

delta_ref_alt_embryo = []
for variant in pred_ref_alts_embryo:
    delta = np.zeros_like(variant.alts)
    for i in range(variant.alts.shape[0]):
        delta[i] = variant.alts[i] - variant.ref[0]
    delta_ref_alt_embryo.append(DeltaPredRefAlt(rsid=variant.rsid, deltas=delta))


chrom_alias = (
    pd.read_table(
        "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/chromAlias.txt.gz",
        header=None,
    )
    .set_index(0)[1]
    .to_dict()
)

with open("../data_prep_new/facial_gwas/white/all_sign_snps.rs.uniq.txt") as f:
    snps = f.read().splitlines()

data = []
for d in tqdm(fetch_rsid_batched(rsids=snps, batch_size=100, tries=5), total=len(snps)):
    data.append(d)


def decode_data(d: str):
    soup = BeautifulSoup(d)
    return ("rs" + soup.snp_id.string, soup.spdi.string)


rsid_to_spdi = dict(
    decode_data(d) for d in tqdm(data, total=len(data)) if "error" not in d
)
missing_snps = set(snps) - set(rsid_to_spdi.keys())
# these are indeed missing from db snp, in total 631


@dataclass
class Variant:
    rsid: str
    chrom: str
    pos: int
    ref: str
    alts: list[str]


def decode_spdi(spdi: str, rsid) -> Variant:
    variants_spdi = spdi.split(",")
    chrom, pos, ref, _ = variants_spdi[0].split(":")
    alts = []
    for s in variants_spdi:
        c, p, r, alt = s.split(":")
        if c != chrom or p != pos or r != ref:
            raise ValueError("Multiple chroms, pos, ref")
        alts.append(alt)
    return Variant(rsid=rsid, chrom=chrom, pos=int(pos), ref=ref, alts=alts)


variants = []
nones = 0
for rsid in rsid_to_spdi:
    if rsid_to_spdi[rsid] is not None:
        variants.append(decode_spdi(rsid_to_spdi[rsid], rsid))
    else:
        nones += 1

# one None

variants_ucsc = []
for variant in variants:
    variants_ucsc.append(
        Variant(
            rsid=variant.rsid,
            chrom=chrom_alias[variant.chrom],
            pos=variant.pos,
            ref=variant.ref,
            alts=variant.alts,
        )
    )


rs_to_pval = {}
for f in os.listdir("../data_prep_new/facial_gwas/white/suggestive_snps"):
    if not f.endswith(".tsv"):
        continue
    with open(
        os.path.join("../data_prep_new/facial_gwas/white/suggestive_snps", f)
    ) as sum:
        for l in sum:
            rs, _, _, _, _, p = l.split()
            if p == "P":
                continue
            p = float(p)
            if rs not in rs_to_pval:
                rs_to_pval[rs] = p
            else:
                rs_to_pval[rs] = min(p, rs_to_pval[rs])

rs_to_max_delta_organoid = {}
for var in delta_ref_alt_organoid:
    abs_max = np.abs(var.deltas).max()
    if var.rsid not in rs_to_max_delta_organoid:
        rs_to_max_delta_organoid[var.rsid] = abs_max
    else:
        rs_to_max_delta_organoid[var.rsid] = max(
            abs_max, rs_to_max_delta_organoid[var.rsid]
        )

rs_to_max_delta_embryo = {}
for var in delta_ref_alt_embryo:
    abs_max = np.abs(var.deltas).max()
    if var.rsid not in rs_to_max_delta_embryo:
        rs_to_max_delta_embryo[var.rsid] = abs_max
    else:
        rs_to_max_delta_embryo[var.rsid] = max(
            abs_max, rs_to_max_delta_embryo[var.rsid]
        )

for var in variants_ucsc:
    var.pval = rs_to_pval.get(var.rsid, None)
    var.max_delta_organoid = rs_to_max_delta_organoid.get(var.rsid, None)
    var.max_delta_embryo = rs_to_max_delta_embryo.get(var.rsid, None)

lead_snps = []
with open("../data_prep_new/facial_gwas/white/203_lead_facial_snps.txt") as f:
    for l in f:
        lead_snps.append(l.strip())

rsid_to_var = {}
for var in variants_ucsc:
    rsid_to_var[var.rsid] = var

rsid_to_seq_ref_alt = {}
for var in seq_ref_alts:
    rsid_to_seq_ref_alt[var.rsid] = var

contrib_var_hit_organoid = []
for var, model_idc, deltas in tqdm(
    [
        (rsid_to_var[v.rsid], abs(v.deltas).argmax(1), abs(v.deltas).max(1))
        for v in delta_ref_alt_organoid
        if abs(v.deltas).max() > 0.5
    ]
):
    model_idx = model_idc[np.argmax(deltas)]
    ref_sequence = rsid_to_seq_ref_alt[var.rsid].ref
    alt_sequence = rsid_to_seq_ref_alt[var.rsid].alts[np.argmax(deltas)]
    seq_oh = np.concatenate(
        [one_hot_encode_sequence(ref_sequence), one_hot_encode_sequence(alt_sequence)]
    )
    explainer = Explainer(model=organoid_model, class_index=model_idx)
    contrib = explainer.integrated_grad(seq_oh)
    ism = explainer.mutagenesis(seq_oh)
    contrib_var_hit_organoid.append(
        (var, model_idx, deltas.max(), seq_oh, contrib, ism)
    )

contrib_var_hit_embryo = []
for var, model_idc, deltas in tqdm(
    [
        (rsid_to_var[v.rsid], abs(v.deltas).argmax(1), abs(v.deltas).max(1))
        for v in delta_ref_alt_embryo
        if abs(v.deltas).max() > 0.5
    ]
):
    model_idx = model_idc[np.argmax(deltas)]
    ref_sequence = rsid_to_seq_ref_alt[var.rsid].ref
    alt_sequence = rsid_to_seq_ref_alt[var.rsid].alts[np.argmax(deltas)]
    seq_oh = np.concatenate(
        [one_hot_encode_sequence(ref_sequence), one_hot_encode_sequence(alt_sequence)]
    )
    explainer = Explainer(model=embryo_model, class_index=model_idx)
    contrib = explainer.integrated_grad(seq_oh)
    ism = explainer.mutagenesis(seq_oh)
    contrib_var_hit_embryo.append((var, model_idx, deltas.max(), seq_oh, contrib, ism))


for var, model_idx, delta, seq_oh, contrib, ism in tqdm(contrib_var_hit_organoid):
    fig, axs = plt.subplots(figsize=(20, 6), nrows=3, sharex=True)
    _ = logomaker.Logo(
        df=pd.DataFrame((contrib * seq_oh)[0], columns=["A", "C", "G", "T"]), ax=axs[0]
    )
    _ = logomaker.Logo(
        df=pd.DataFrame((contrib * seq_oh)[1], columns=["A", "C", "G", "T"]), ax=axs[1]
    )
    for nuc_idx, c in zip(range(4), ["#008001", "#1c00ff", "#ffab10", "#ff0000"]):
        axs[2].scatter(np.arange(500), ism[0][:, nuc_idx], color=c)
    fig.tight_layout()
    fig.savefig(f"draft/{var.rsid}_organoid.pdf")


for var, model_idx, delta, seq_oh, contrib, ism in tqdm(contrib_var_hit_embryo):
    fig, axs = plt.subplots(figsize=(20, 6), nrows=3, sharex=True)
    _ = logomaker.Logo(
        df=pd.DataFrame((contrib * seq_oh)[0], columns=["A", "C", "G", "T"]), ax=axs[0]
    )
    _ = logomaker.Logo(
        df=pd.DataFrame((contrib * seq_oh)[1], columns=["A", "C", "G", "T"]), ax=axs[1]
    )
    for nuc_idx, c in zip(range(4), ["#008001", "#1c00ff", "#ffab10", "#ff0000"]):
        axs[2].scatter(np.arange(500), ism[0][:, nuc_idx], color=c)
    fig.tight_layout()
    fig.savefig(f"draft/{var.rsid}_embryo.pdf")


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
ax_embryo_umap_1 = fig.add_subplot(gs[10:15, 0:5])
ax_embryo_umap_2 = fig.add_subplot(gs[15:20, 0:5])
organoid_cells_both = list(
    set(organoid_neural_crest_cell_topic.index)
    & set(adata_organoid_neural_crest.obs_names)
)
rgb_scatter_plot(
    x=adata_organoid_neural_crest[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_neural_crest[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_organoid_umap_1,
    g_cut=0,
    r_values=organoid_neural_crest_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neural_crest_topics_organoid[0] - 1),
    ].values,
    g_values=organoid_neural_crest_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neural_crest_topics_organoid[1] - 1),
    ].values,
    b_values=organoid_neural_crest_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neural_crest_topics_organoid[2] - 1),
    ].values,
    r_name=model_index_to_topic_name_organoid(
        neural_crest_topics_organoid[0] - 1
    ).replace("neural_crest_", ""),
    g_name=model_index_to_topic_name_organoid(
        neural_crest_topics_organoid[1] - 1
    ).replace("neural_crest_", ""),
    b_name=model_index_to_topic_name_organoid(
        neural_crest_topics_organoid[2] - 1
    ).replace("neural_crest_", ""),
    g_vmin=0,
    g_vmax=0.25,
)
rgb_scatter_plot(
    x=adata_organoid_neural_crest[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_neural_crest[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_organoid_umap_2,
    g_cut=0,
    r_values=organoid_neural_crest_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neural_crest_topics_organoid[3] - 1),
    ].values,
    g_values=organoid_neural_crest_cell_topic.loc[
        organoid_cells_both,
        model_index_to_topic_name_organoid(neural_crest_topics_organoid[4] - 1),
    ].values,
    b_values=np.zeros(len(organoid_cells_both)),
    r_name=model_index_to_topic_name_organoid(
        neural_crest_topics_organoid[3] - 1
    ).replace("neural_crest_", ""),
    g_name=model_index_to_topic_name_organoid(
        neural_crest_topics_organoid[4] - 1
    ).replace("neural_crest_", ""),
)
embryo_cells_both = list(
    set(embryo_neural_crest_cell_topic.index) & set(adata_embryo_neural_crest.obs_names)
)
rgb_scatter_plot(
    x=adata_embryo_neural_crest[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_neural_crest[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_embryo_umap_1,
    g_cut=0,
    r_values=embryo_neural_crest_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neural_crest_topics_embryo[0] - 1),
    ].values,
    g_values=embryo_neural_crest_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neural_crest_topics_embryo[1] - 1),
    ].values,
    b_values=embryo_neural_crest_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neural_crest_topics_embryo[2] - 1),
    ].values,
    r_name=model_index_to_topic_name_embryo(neural_crest_topics_embryo[0] - 1).replace(
        "neural_crest_", ""
    ),
    g_name=model_index_to_topic_name_embryo(neural_crest_topics_embryo[1] - 1).replace(
        "neural_crest_", ""
    ),
    b_name=model_index_to_topic_name_embryo(neural_crest_topics_embryo[2] - 1).replace(
        "neural_crest_", ""
    ),
    r_vmin=0,
    r_vmax=0.25,
    g_vmin=0,
    g_vmax=0.3,
    b_vmin=0,
    b_vmax=0.25,
)
rgb_scatter_plot(
    x=adata_embryo_neural_crest[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_neural_crest[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_embryo_umap_2,
    g_cut=0,
    r_values=np.zeros(len(embryo_cells_both)),
    g_values=embryo_neural_crest_cell_topic.loc[
        embryo_cells_both,
        model_index_to_topic_name_embryo(neural_crest_topics_embryo[3] - 1),
    ].values,
    b_values=np.zeros(len(embryo_cells_both)),
    g_name=model_index_to_topic_name_embryo(neural_crest_topics_embryo[3] - 1).replace(
        "neural_crest_", ""
    ),
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
    print(f"{YMIN}, {YMAX}")
    for ax, (organoid_topic, embryo_topic) in zip(axs, organoid_embryo):
        _ = ax.set_ylim(YMIN, YMAX)
        _ = ax.set_axis_off()

ax_hit_heatmap = fig.add_subplot(gs[0:16, 23:29])
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
ax_organoid_expr_heatmap = fig.add_subplot(gs[0:12, 32:36])
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
ax_network = fig.add_subplot(gs[12 + 2:19, 33:39])
sns.heatmap(
    jaccard_organoid,
    cmap="viridis",
    vmin=0.05,
    vmax=0.2,
    ax=ax_network,
    linecolor="black",
    lw=0.5,
    cbar=False,
    yticklabels=True,
    xticklabels=True,
)
ax_manhattan_all_chrom = fig.add_subplot(gs[22:27, 0:39])
prev = 0
chroms = [*range(1, 23), "X"]
for chrom in chroms:
    var_chrom = sorted(
        [
            var
            for var in variants_ucsc
            if var.chrom == "chr" + str(chrom)
            and var.pval is not None
            and var.max_delta_organoid is not None
        ],
        key=lambda var: var.pos,
    )
    ax_manhattan_all_chrom.scatter(
        x=[var.pos + prev for var in var_chrom],
        y=[-np.log10(var.pval) for var in var_chrom],
        color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
        s=[0.5 if var.rsid not in lead_snps else 5 for var in var_chrom],
        zorder=2,
    )
    ax_manhattan_all_chrom.scatter(
        x=[var.pos + prev for var in var_chrom],
        y=[-var.max_delta_organoid * 100 for var in var_chrom],
        color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
        s=[0.5 if var.rsid not in lead_snps else 5 for var in var_chrom],
        alpha=0.7,
        zorder=2,
    )
    ax_manhattan_all_chrom.scatter(
        x=[var.pos + prev for var in var_chrom],
        y=[-var.max_delta_embryo * 100 for var in var_chrom],
        color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
        s=[5 if var.rsid not in lead_snps else 10 for var in var_chrom],
        marker="x",
        alpha=0.7,
        zorder=2,
    )
    ax_manhattan_all_chrom.text(
        x=np.mean([var.pos + prev for var in var_chrom]), y=80, s=chrom
    )
    prev += var_chrom[-1].pos + 1_000
    ax_manhattan_all_chrom.axvline(prev, color="gray", zorder=1)
ax_manhattan_all_chrom.spines[["bottom", "top", "right"]].set_visible(False)
ax_manhattan_all_chrom.set_xticks([])
ax_manhattan_all_chrom.grid(True)
ax_manhattan_all_chrom.set_ylim(-100, 100)
ax_manhattan_all_chrom.set_yticks(
    (-100, -50, 0, 50, 100), labels=["-1", "-0.5", "0", "50", "100"]
)
ax_manhattan_all_chrom.set_ylabel("delta prediction\n-log10(pval)")
ax_manhattan_chrom_1 = fig.add_subplot(gs[28:33, 0:15])
var, _, delta, seq_oh, contrib, ism = [
    (var, model_idx, delta, seq_oh, contrib, ism)
    for var, model_idx, delta, seq_oh, contrib, ism in contrib_var_hit_organoid
    if var.rsid == "rs1555067"
][0]
chrom = 1
var_chrom = sorted(
    [
        var
        for var in variants_ucsc
        if var.chrom == "chr" + str(chrom)
        and var.pval is not None
        and var.max_delta_organoid is not None
    ],
    key=lambda var: var.pos,
)
ax_manhattan_chrom_1.scatter(
    x=[var.pos for var in var_chrom],
    y=[-np.log10(var.pval) for var in var_chrom],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[0.5 if var.rsid not in lead_snps else 5 for var in var_chrom],
)
ax_manhattan_chrom_1.scatter(
    x=[var.pos for var in var_chrom],
    y=[-var.max_delta_organoid * 100 for var in var_chrom],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[0.5 if var.rsid not in lead_snps else 5 for var in var_chrom],
)
ax_manhattan_chrom_1.scatter(
    x=[var.pos for var in var_chrom],
    y=[-var.max_delta_embryo * 100 for var in var_chrom],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[5 if var.rsid not in lead_snps else 10 for var in var_chrom],
    marker="x",
)
ax_manhattan_chrom_1.grid(True)
ax_manhattan_chrom_1.set_ylim(-100, 100)
# ax.set_axis_off()
ax_manhattan_chrom_1.set_title(chrom)
xmin_in, xmax_in = var.pos - 350_000, var.pos + 350_000
axins = ax_manhattan_chrom_1.inset_axes(
    [0.5, 0.05, 0.1, 0.6],
    xlim=(xmin_in, xmax_in),
    ylim=(-75, 70),
    xticklabels=[],
    yticklabels=[],
)
axins.scatter(
    x=[var.pos for var in var_chrom if var.pos > xmin_in and var.pos < xmax_in],
    y=[
        -np.log10(var.pval)
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[
        0.5 if var.rsid not in lead_snps else 5
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
)
axins.scatter(
    x=[var.pos for var in var_chrom if var.pos > xmin_in and var.pos < xmax_in],
    y=[
        -var.max_delta_organoid * 100
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[
        0.5 if var.rsid not in lead_snps else 5
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
)
axins.scatter(
    x=[var.pos for var in var_chrom if var.pos > xmin_in and var.pos < xmax_in],
    y=[
        -var.max_delta_embryo * 100
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
    color=plt.cm.gnuplot(chroms.index(chrom) / len(chroms)),
    s=[
        5 if var.rsid not in lead_snps else 10
        for var in var_chrom
        if var.pos > xmin_in and var.pos < xmax_in
    ],
    marker="x",
)
axins.scatter(x=[var.pos], y=[-delta * 100], color="red", s=5)
axins.set_title(f"{int(xmax_in - xmin_in)}bps")
axins.grid(True)
axins.set_axisbelow(True)
ax_manhattan_chrom_1.indicate_inset_zoom(axins, edgecolor="red")
ax_manhattan_chrom_1.set_yticks(
    (-100, -50, 0, 50, 100), labels=["-1", "-0.5", "0", "50", "100"]
)
ax_manhattan_chrom_1.set_ylabel("delta prediction\n-log10(pval)")
ax_contrib_ref = fig.add_subplot(gs[28:30, 16:39])
ax_contrib_alt = fig.add_subplot(gs[31:33, 16:39])
# ax_ism_ref = fig.add_subplot(
#    gs[32: 33, 16: 39]
# )
_ = logomaker.Logo(
    df=pd.DataFrame((contrib * seq_oh)[0], columns=["A", "C", "G", "T"]),
    ax=ax_contrib_ref,
    zorder=2,
)
_ = logomaker.Logo(
    df=pd.DataFrame((contrib * seq_oh)[1], columns=["A", "C", "G", "T"]),
    ax=ax_contrib_alt,
    zorder=2,
)
for ax in [ax_contrib_alt, ax_contrib_ref]:
    ax.axvline(250, ls="dashed", color="black", lw=0.5, zorder=1)
    ax.set_xlim(150, 350)
fig.tight_layout()


fig.savefig("Figure_5_v2.png", transparent=False)
fig.savefig("Figure_5_v2.pdf")
