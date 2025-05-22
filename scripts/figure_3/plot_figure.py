import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from pycisTopic.topic_binarization import binarize_topics
import gzip
import os
import pyBigWig
import json
import tensorflow as tf
import crested
import pathlib
from crested.utils._seq_utils import one_hot_encode_sequence
from crested.tl._explainer_tf import Explainer
import logomaker
from dataclasses import dataclass
from typing import Self
import h5py
from tqdm import tqdm
import pickle
from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import torch
from functools import reduce
import seaborn as sns
from scipy.stats import binomtest
import networkx as nx
from scipy.stats import mannwhitneyu
import modiscolite
from scipy import stats
from scipy.stats import fisher_exact

color_dict = json.load(open("../color_maps.json"))

###############################################################################################################
#                                             LOAD DATA                                                       #
###############################################################################################################

# load organoid RNA data and subset for ATAC cells
adata_organoid = sc.read_h5ad("../figure_1/adata_organoid.h5ad")

sample_id_to_num = {
    s: adata_organoid.obs.query("sample_id == @s").index[0].split("-")[-1]
    for s in set(adata_organoid.obs.sample_id)
}

adata_organoid_progenitor = sc.read_h5ad("../figure_1/adata_organoid_progenitor.h5ad")
adata_embryo_progenitor = sc.read_h5ad("../figure_1/adata_embryo_progenitor.h5ad")

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
    "../data_prep_new/organoid_data/ATAC/progenitor_cell_topic_contrib.tsv",
    index_col=0,
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
    "../data_prep_new/embryo_data/ATAC/progenitor_cell_topic_contrib.tsv",
    index_col=0,
)

embryo_progenitor_cell_topic.columns = [
    f"progenitor_Topic_{c.replace('Topic', '')}" for c in embryo_progenitor_cell_topic
]

embryo_progenitor_cell_topic.index = [
    x.split("___")[0] + "-1" + "___" + x.split("___")[1]
    for x in embryo_progenitor_cell_topic.index
]

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




cell_topic_bin_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/cell_bin_topic.tsv"
)

cells, scores, thresholds = binarize_topics(
    embryo_progenitor_cell_topic.to_numpy(), embryo_progenitor_cell_topic.index, "li"
)

cell_topic_bin_embryo = dict(cell_barcode=[], topic_name=[], group=[], topic_prob=[])
for topic_idx in range(len(cells)):
    cell_topic_bin_embryo["cell_barcode"].extend(cells[topic_idx])
    cell_topic_bin_embryo["topic_name"].extend(
        np.repeat(f"Topic{topic_idx + 1}", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["group"].extend(
        np.repeat("progenitor", len(cells[topic_idx]))
    )
    cell_topic_bin_embryo["topic_prob"].extend(scores[topic_idx])

cell_topic_bin_embryo = pd.DataFrame(cell_topic_bin_embryo)

exp_organoid = adata_organoid_progenitor.to_df(layer="log_cpm")

organoid_progenitor_topics_to_show = [33, 38, 36, 54, 48]

SHH_expr_organoid_per_topic = {}
for topic in organoid_progenitor_topics_to_show:
    cell_names = list(
        set(
            [
                atac_to_rna(x)
                for x in cell_topic_bin_organoid.set_index("topic_name")
                .loc[
                    model_index_to_topic_name_organoid(topic - 1)
                    .replace("progenitor", "")
                    .replace("_", ""),
                    "cell_barcode",
                ]
                .values
            ]
        )
        & set(adata_organoid_progenitor.obs_names)
    )
    SHH_expr_organoid_per_topic[topic] = np.exp(
        exp_organoid.loc[cell_names, "SHH"].values
    )

exp_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")

embryo_progenitor_topics_to_show = [34, 38, 79, 88, 58]

SHH_expr_embryo_per_topic = {}
for topic in embryo_progenitor_topics_to_show:
    cell_names = list(
        set(
            cell_topic_bin_embryo.set_index("topic_name")
            .loc[
                model_index_to_topic_name_embryo(topic - 1)
                .replace("progenitor", "")
                .replace("_", ""),
                "cell_barcode",
            ]
            .values
        )
        & set(adata_embryo_progenitor.obs_names)
    )
    SHH_expr_embryo_per_topic[topic] = np.exp(exp_embryo.loc[cell_names, "SHH"].values)

exp_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")
exp_organoid = adata_organoid_progenitor.to_df(layer="log_cpm")

avg_expr_organoid_per_topic = {}
for topic in organoid_progenitor_topics_to_show:
    cell_names = list(
        set(
            [
                atac_to_rna(x)
                for x in cell_topic_bin_organoid.set_index("topic_name")
                .loc[
                    model_index_to_topic_name_organoid(topic - 1)
                    .replace("progenitor", "")
                    .replace("_", ""),
                    "cell_barcode",
                ]
                .values
            ]
        )
        & set(adata_organoid_progenitor.obs_names)
    )
    avg_expr_organoid_per_topic[topic] = exp_organoid.loc[cell_names].mean()

avg_expr_embryo_per_topic = {}
for topic in embryo_progenitor_topics_to_show:
    cell_names = list(
        set(
            cell_topic_bin_embryo.set_index("topic_name")
            .loc[
                model_index_to_topic_name_embryo(topic - 1)
                .replace("progenitor", "")
                .replace("_", ""),
                "cell_barcode",
            ]
            .values
        )
        & set(adata_embryo_progenitor.obs_names)
    )
    avg_expr_embryo_per_topic[topic] = exp_embryo.loc[cell_names].mean()

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


transcript_id = "NM_000193"

transcript = {"exon": [], "3UTR": [], "5UTR": [], "transcript": []}
with gzip.open(
    "/home/VIB.LOCAL/seppe.dewinter/sdewin/resources/hg38/hg38.refGene.gtf.gz"
) as f:
    for line in f:
        line = line.decode().strip()
        if f'transcript_id "{transcript_id}"' not in line:
            continue
        _, _, t, start, stop = line.split()[0:5]
        if t in transcript:
            transcript[t].append((int(start), int(stop)))

transcript_feature_to_size = {"exon": 1, "3UTR": 0.5, "5UTR": 0.5}

bw_dir_organoid = "../data_prep_new/organoid_data/ATAC/bw_per_topic"

topic_organoid_to_bw = {
    topic: os.path.join(
        bw_dir_organoid,
        model_index_to_topic_name_organoid(topic - 1)[::-1].replace("_", "", 1)[::-1]
        + ".fragments.tsv.bigWig",
    )
    for topic in organoid_progenitor_topics_to_show
}

topic_organoid_to_locus_val = {}
for topic in topic_organoid_to_bw:
    print(topic)
    with pyBigWig.open(topic_organoid_to_bw[topic]) as bw:
        topic_organoid_to_locus_val[topic] = np.array(
            list(map(None_to_0, bw.stats(*locus, nBins=nbins)))
        )

bw_dir_embryo = "../data_prep_new/embryo_data/ATAC/bw_per_topic"

topic_embryo_to_bw = {
    topic: os.path.join(
        bw_dir_embryo,
        model_index_to_topic_name_embryo(topic - 1)[::-1].replace("_", "", 1)[::-1]
        + ".fragments.tsv.bigWig",
    )
    for topic in embryo_progenitor_topics_to_show
}

topic_embryo_to_locus_val = {}
for topic in topic_embryo_to_bw:
    print(topic)
    with pyBigWig.open(topic_embryo_to_bw[topic]) as bw:
        topic_embryo_to_locus_val[topic] = np.array(
            list(map(None_to_0, bw.stats(*locus, nBins=nbins)))
        )

shh_locus_de = {"sfpe1": dict(), "sfpe2": dict()}

genome_dir = "../../../../../resources/"
hg38 = crested.Genome(
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
)

path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
embryo_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_embryo_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))

path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
organoid_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_organoid_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
organoid_model.load_weights(os.path.join(path_to_organoid_model, "model_epoch_23.hdf5"))

sfpe_sequences = {
    "sfpe1": hg38.fetch(region=f"chr7:{'-'.join(map(str, SFPE1_hg38_coord))}"),
    "sfpe2": hg38.fetch(region=f"chr7:{'-'.join(map(str, SFPE2_hg38_coord))}"),
}

explainer_organoid = Explainer(
    model=organoid_model, class_index=organoid_progenitor_topics_to_show[0] - 1
)
explainer_embryo = Explainer(
    model=embryo_model, class_index=embryo_progenitor_topics_to_show[0] - 1
)

for sfpe in shh_locus_de:
    print(sfpe)
    oh_sequence = one_hot_encode_sequence(sfpe_sequences[sfpe])
    gradients_integrated = explainer_organoid.integrated_grad(X=oh_sequence).squeeze()
    shh_locus_de[sfpe]["organoid"] = (oh_sequence, gradients_integrated)

for sfpe in shh_locus_de:
    print(sfpe)
    oh_sequence = one_hot_encode_sequence(sfpe_sequences[sfpe])
    gradients_integrated = explainer_embryo.integrated_grad(X=oh_sequence).squeeze()
    shh_locus_de[sfpe]["embryo"] = (oh_sequence, gradients_integrated)


organoid_dv_topics = np.array([8, 16, 13, 9, 11, 19, 25, 1, 29, 23, 3]) + 25

embryo_dv_topics = np.array([4, 44, 57, 8, 49, 21, 58, 28]) + 30


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


def get_value_seqlets(seqlets: list[Seqlet], v: np.ndarray):
    if v.shape[0] != len(seqlets):
        raise ValueError(f"{v.shape[0]} != {len(seqlets)}")
    for i, seqlet in enumerate(seqlets):
        if seqlet.is_revcomp:
            yield v[i, seqlet.start : seqlet.end, :][::-1, ::-1]
        else:
            yield v[i, seqlet.start : seqlet.end, :]


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


organoid_dl_motif_dir = "../data_prep_new/organoid_data/MODELS/modisco/"
embryo_dl_motif_dir = "../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_organoid = []
pattern_names_dl_organoid = []
for topic in tqdm(organoid_dv_topics):
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

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(embryo_dv_topics):
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

pattern_metadata = pd.read_table("draft/motif_metadata.tsv", index_col=0)

if os.path.exists("pattern_to_topic_to_grad_organoid.pkl"):
    pattern_to_topic_to_grad_organoid = pickle.load(
        open("pattern_to_topic_to_grad_organoid.pkl", "rb")
    )
else:
    pattern_to_topic_to_grad_organoid = {}
    for pattern_name in tqdm(pattern_metadata.index):
        pattern = all_patterns[all_pattern_names.index(pattern_name)]
        oh_sequences = np.array(
            [x.region_one_hot for x in pattern.seqlets]
        )  # .astype(np.int8)
        pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
        pattern_to_topic_to_grad_organoid[pattern_name] = {}
        for topic in tqdm(organoid_dv_topics, leave=False):
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

if os.path.exists("pattern_to_topic_to_grad_embryo.pkl"):
    pattern_to_topic_to_grad_embryo = pickle.load(
        open("pattern_to_topic_to_grad_embryo.pkl", "rb")
    )
else:
    pattern_to_topic_to_grad_embryo = {}
    for pattern_name in tqdm(pattern_metadata.index):
        pattern = all_patterns[all_pattern_names.index(pattern_name)]
        oh_sequences = np.array(
            [x.region_one_hot for x in pattern.seqlets]
        )  # .astype(np.int8)
        pattern_ohs = list(get_value_seqlets(pattern.seqlets, oh_sequences))
        pattern_to_topic_to_grad_embryo[pattern_name] = {}
        for topic in tqdm(embryo_dv_topics, leave=False):
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


def allign_patterns_of_cluster_for_topic(
    pattern_to_topic_to_grad: dict[str, dict[int, np.ndarray]],
    pattern_metadata: pd.DataFrame,
    cluster_id: int,
    topic: int,
):
    P = []
    O = []
    cluster_patterns = pattern_metadata.query(
        "hier_cluster == @cluster_id"
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


cluster_to_topic_to_avg_pattern_organoid = {}
for cluster in set(pattern_metadata["hier_cluster"]):
    cluster_to_topic_to_avg_pattern_organoid[cluster] = {}
    for topic in organoid_dv_topics:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_organoid,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        cluster_to_topic_to_avg_pattern_organoid[cluster][topic] = (P * O).mean(0)

cluster_to_topic_to_avg_pattern_embryo = {}
for cluster in set(pattern_metadata["hier_cluster"]):
    cluster_to_topic_to_avg_pattern_embryo[cluster] = {}
    for topic in embryo_dv_topics:
        P, O = allign_patterns_of_cluster_for_topic(
            pattern_to_topic_to_grad=pattern_to_topic_to_grad_embryo,
            pattern_metadata=pattern_metadata,
            cluster_id=cluster,
            topic=topic,
        )
        cluster_to_topic_to_avg_pattern_embryo[cluster][topic] = (P * O).mean(0)

selected_clusters = [6, 7, 12, 15, 10, 18, 1, 19]

cluster_to_tf = {
    6: "FOX((A(1|2))|(P(2|4)))",
    7: "TEAD1",
    12: "RFX(3|4)",
    15: "NKX2-(1|2)",
    10: "PAX(3|6)",
    18: "ZIC1",
    1: "SOX(2|3)",
    19: "ZEB(1|2)",
}

organoid_embryo = [
    (33, 34),
    (38, 38),
    (36, 79),
    (54, 88),
    (48, 58),
]

from scipy import stats

corrs_patterns = []
for cluster in selected_clusters:
    corrs_patterns.append(
        stats.pearsonr(
            [cluster_to_topic_to_avg_pattern_organoid[cluster][o].sum() for o, _ in organoid_embryo],
            [cluster_to_topic_to_avg_pattern_embryo[cluster][e].sum() for _, e in organoid_embryo]
        ).statistic
    )
print(np.mean(corrs_patterns))

n_clusters = len(selected_clusters)

import re

tf_expr_matrix_per_topic_organoid = (
    pd.DataFrame(avg_expr_organoid_per_topic).T
)

tf_expr_matrix_per_topic_organoid = tf_expr_matrix_per_topic_organoid[
    [c for c in tf_expr_matrix_per_topic_organoid.columns if any([re.fullmatch(p, c) for p in cluster_to_tf.values() ])]
]

tf_expr_matrix_per_topic_organoid = tf_expr_matrix_per_topic_organoid[
    tf_expr_matrix_per_topic_organoid.idxmax().sort_values(
        key = lambda X: [[oe[0] for oe in organoid_embryo].index(x) for x in X]).index
]

tf_expr_matrix_per_topic_embryo = (
    pd.DataFrame(avg_expr_embryo_per_topic).T
)

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    [c for c in tf_expr_matrix_per_topic_embryo.columns if any([re.fullmatch(p, c) for p in cluster_to_tf.values() ])]
]

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    tf_expr_matrix_per_topic_organoid.columns
]




# hit scoring


def merge_and_max(left, right, on, max_on, l):
    global a
    if a:
        print(" " * (l - 1) + "|", end="\r", flush=True)
        a = False
    print("x", end="", flush=True)
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).max(1)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1).copy()


def merge_and_absmax(left, right, on, max_on, l):
    global a
    if a:
        print(" " * (l - 1) + "|", end="\r", flush=True)
        a = False
    print("x", end="", flush=True)
    x = pd.merge(left, right, on=on, how="outer")
    x[max_on] = x[[f"{max_on}_x", f"{max_on}_y"]].fillna(0).T.apply(absmax)
    return x.drop([f"{max_on}_x", f"{max_on}_y"], axis=1).copy()


motifs = {
    n: pattern.ppm[pattern_metadata.loc[n].ic_start : pattern_metadata.loc[n].ic_stop].T
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
    hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2))
    hits = pd.concat(hits)
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["cluster"] = [
        pattern_metadata.loc[m, "hier_cluster"] for m in hits["motif_name"]
    ]
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    all_hits_organoid.append(hits)

subset_hits_organoid = []
for topic, _ in organoid_embryo:
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
        pattern_metadata.loc[m, "hier_cluster"] for m in hits["motif_name"]
    ]
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    subset_hits_organoid.append(hits)

a = True
hits_merged_organoid = reduce(
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
            "attribution",
            "p-value",
        ],
        max_on="-logp",
        l=len(all_hits_organoid),
    ),
    all_hits_organoid,
)

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
        l=len(subset_hits_organoid),
    ),
    subset_hits_organoid,
)


def absmax(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum = (
    hits_merged_organoid.groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
    .astype(float)
)


# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum_subset = (
    hits_merged_organoid_subset.groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
    .astype(float)
)

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

pattern_order_organoid = hits_merged_organoid_per_seq_and_cluster_sum.columns.values[
    np.argsort(
        [
            region_order_organoid.index(x)
            for x in hits_merged_organoid_per_seq_and_cluster_sum.idxmax().values
        ]
    )
]

hits_merged_organoid_per_seq_and_cluster_sum_scaled = (
    hits_merged_organoid_per_seq_and_cluster_sum
    / hits_merged_organoid_per_seq_and_cluster_sum.sum()
)

hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset
    / hits_merged_organoid_per_seq_and_cluster_sum_subset.sum()
)

data_organoid = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[
        region_order_organoid_subset, selected_clusters
    ].abs()
    > 0.0008
) * 1
data_organoid.columns = [cluster_to_tf[x] for x in data_organoid.columns]

cooc_organoid = data_organoid.T @ data_organoid
for cluster in selected_clusters:
    tf_name = cluster_to_tf[cluster]
    cooc_organoid.loc[tf_name, tf_name] = 0

cooc_perc_organoid = (cooc_organoid / cooc_organoid.sum()).T

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(selected_clusters), clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set3)

tf_to_color = {
    cluster_to_tf[c]: mapper.to_rgba(i) for i, c in enumerate(selected_clusters)
}

all_hits_embryo = []
for topic in embryo_dv_topics:
    f = f"gradients_Topic_{topic}.npz"
    print(f)
    ohs = np.load(os.path.join(embryo_dl_motif_dir, f))["oh"]
    attr = np.load(os.path.join(embryo_dl_motif_dir, f))["gradients_integrated"]
    region_names = np.load(os.path.join(embryo_dl_motif_dir, f))["region_names"]
    hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2))
    hits = pd.concat(hits)
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["cluster"] = [
        pattern_metadata.loc[m, "hier_cluster"] for m in hits["motif_name"]
    ]
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    all_hits_embryo.append(hits)

subset_hits_embryo = []
for _, topic in organoid_embryo:
    f = f"gradients_Topic_{topic}.npz"
    print(f)
    ohs = np.load(os.path.join(embryo_dl_motif_dir, f))["oh"]
    attr = np.load(os.path.join(embryo_dl_motif_dir, f))["gradients_integrated"]
    region_names = np.load(os.path.join(embryo_dl_motif_dir, f))["region_names"]
    hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2))
    hits = pd.concat(hits)
    hits["attribution"] = extract_signal(
        hits[["sequence_name", "start", "end"]],
        torch.from_numpy(attr.squeeze().swapaxes(1, 2)),
        verbose=True,
    ).sum(dim=1)
    hits["cluster"] = [
        pattern_metadata.loc[m, "hier_cluster"] for m in hits["motif_name"]
    ]
    hits["sequence_name"] = [region_names[x] for x in hits["sequence_name"]]
    hits["-logp"] = -np.log10(hits["p-value"] + 1e-6)
    subset_hits_embryo.append(hits)

a = True
hits_merged_embryo = reduce(
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
        l=len(all_hits_embryo),
    ),
    all_hits_embryo,
)

a = True
hits_merged_embryo_subset = reduce(
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
        l=len(subset_hits_embryo),
    ),
    subset_hits_embryo,
)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum = (
    hits_merged_embryo.groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
    .astype(float)
)

# USE MAX to deal with overlapping hits (variable name says sum but it is max)
hits_merged_embryo_per_seq_and_cluster_sum_subset = (
    hits_merged_embryo_subset.groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
    .astype(float)
)

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

pattern_order_embryo = hits_merged_embryo_per_seq_and_cluster_sum.columns.values[
    np.argsort(
        [
            region_order_embryo.index(x)
            for x in hits_merged_embryo_per_seq_and_cluster_sum.idxmax().values
        ]
    )
]

hits_merged_embryo_per_seq_and_cluster_sum_scaled = (
    hits_merged_embryo_per_seq_and_cluster_sum
    / hits_merged_embryo_per_seq_and_cluster_sum.sum()
)

hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled = (
    hits_merged_embryo_per_seq_and_cluster_sum_subset
    / hits_merged_embryo_per_seq_and_cluster_sum_subset.sum()
)

data_embryo = (
    hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[
        region_order_embryo_subset, selected_clusters
    ].abs()
    > 0.0004
) * 1
data_embryo.columns = [cluster_to_tf[x] for x in data_embryo.columns]

cooc_embryo = data_embryo.T @ data_embryo
for cluster in selected_clusters:
    tf_name = cluster_to_tf[cluster]
    cooc_embryo.loc[tf_name, tf_name] = 0

cooc_perc_embryo = (cooc_embryo / cooc_embryo.sum()).T

G_embryo = nx.DiGraph()
p = 1 / (len(selected_clusters) - 1)
for y, cluster in enumerate(selected_clusters):
    tf_name = cluster_to_tf[cluster]
    left = 0.0
    _sorted_vals = cooc_perc_embryo.loc[tf_name]
    n = sum(cooc_embryo.loc[tf_name])
    for _tf_name, _width in zip(_sorted_vals.index, _sorted_vals.values):
        k = cooc_embryo.loc[tf_name, _tf_name]
        t = binomtest(k, n, p, alternative="greater")
        stat, pval = t.statistic, t.pvalue
        if pval < 1e-6 and (_tf_name != tf_name):
            G_embryo.add_edge(tf_name, _tf_name, weight=-np.log10(pval))


if not os.path.exists("FOX_hits_organoid_embryo.bed"):
    organoid_FOX_hits = set(
        hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[
            hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled[6] > 0.0008
        ].index
    )
    embryo_FOX_hits = set(
        hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[
            hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled[6] > 0.0004
        ].index
    )
    FOX_hits = sorted(
        list(organoid_FOX_hits | embryo_FOX_hits),
        key=lambda r: (
            r.replace(":", "-").split("-")[0],
            int(r.replace(":", "-").split("-")[1]),
        ),
    )
    with open("FOX_hits_organoid_embryo.bed", "wt") as f:
        for region in FOX_hits:
            _ = f.write(region.replace(":", "\t").replace("-", "\t") + "\n")

"""

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/HEPG2_FOXA2_ENCFF466FCB.bed.gz \
    -a FOX_hits_organoid_embryo.bed -F 0.4 -wa \
  > FOX_hits_organoid_embryo_w_HEPG2.bed

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/A549_FOXA2_ENCFF686MSH.bed.gz \
    -a FOX_hits_organoid_embryo.bed -F 0.4 -wa \
  > FOX_hits_organoid_embryo_w_A549.bed

"""

region_names_w_HEPG2 = []
with open("FOX_hits_organoid_embryo_w_HEPG2.bed") as f:
    for l in f:
        chrom, start, end = l.strip().split()
        region_names_w_HEPG2.append(f"{chrom}:{start}-{end}")

region_names_w_A549 = []
with open("FOX_hits_organoid_embryo_w_A549.bed") as f:
    for l in f:
        chrom, start, end = l.strip().split()
        region_names_w_A549.append(f"{chrom}:{start}-{end}")


hits_merged_organoid_subset["overlaps_w_HEPG2"] = False
hits_merged_organoid_subset["overlaps_w_A549"] = False

hits_merged_embryo_subset["overlaps_w_HEPG2"] = False
hits_merged_embryo_subset["overlaps_w_A549"] = False

hits_merged_organoid_subset.loc[
    [r in region_names_w_HEPG2 for r in hits_merged_organoid_subset["sequence_name"]],
    "overlaps_w_HEPG2",
] = True
hits_merged_organoid_subset.loc[
    [r in region_names_w_A549 for r in hits_merged_organoid_subset["sequence_name"]],
    "overlaps_w_A549",
] = True

hits_merged_embryo_subset.loc[
    [r in region_names_w_HEPG2 for r in hits_merged_embryo_subset["sequence_name"]],
    "overlaps_w_HEPG2",
] = True
hits_merged_embryo_subset.loc[
    [r in region_names_w_A549 for r in hits_merged_embryo_subset["sequence_name"]],
    "overlaps_w_A549",
] = True


def get_number_of_non_overlapping_sites(
    df: pd.DataFrame, max_overlap: int, broadcast=False
):
    n = sum(np.diff(df["start"].sort_values()) > max_overlap) + 1
    if not broadcast:
        return n
    else:
        return [n for _ in range(len(df))]


hits_organoid_number_sites_specific = (
    hits_merged_organoid_subset.query(
        "cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

hits_organoid_number_sites_general = (
    hits_merged_organoid_subset.query(
        "cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

hits_embryo_number_sites_specific = (
    hits_merged_embryo_subset.query(
        "cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

hits_embryo_number_sites_general = (
    hits_merged_embryo_subset.query(
        "cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)


def get_non_overlapping_start_end_w_max_attr(df, max_overlap):
    df = df.sort_values("start")
    delta = np.diff(df["start"])
    delta_loc = [0, *np.where(delta > max_overlap)[0] + 1]
    groups = [
        slice(delta_loc[i], delta_loc[i + 1] if (i + 1) < len(delta_loc) else None)
        for i in range(len(delta_loc))
    ]
    return pd.DataFrame(
        [df.iloc[group].iloc[df.iloc[group].attribution.argmax()] for group in groups]
    ).reset_index(drop=True)


hits_organoid_non_overlap_specific = (
    hits_merged_organoid_subset.query(
        "cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10))
    .reset_index(drop=True)
)

hits_organoid_non_overlap_general = (
    hits_merged_organoid_subset.query(
        "cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10))
    .reset_index(drop=True)
)

specific_n_sites_organoid = (
    hits_organoid_non_overlap_specific.groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True))
    .explode()
    .values
)

general_n_sites_organoid = (
    hits_organoid_non_overlap_general.groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True))
    .explode()
    .values
)

hits_embryo_non_overlap_specific = (
    hits_merged_embryo_subset.query(
        "cluster == 6 & overlaps_w_HEPG2 == False & overlaps_w_A549 == False"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10))
    .reset_index(drop=True)
)

hits_embryo_non_overlap_general = (
    hits_merged_embryo_subset.query(
        "cluster == 6 & (overlaps_w_HEPG2 == True | overlaps_w_A549 == True)"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_non_overlapping_start_end_w_max_attr(x, 10))
    .reset_index(drop=True)
)

specific_n_sites_embryo = (
    hits_embryo_non_overlap_specific.groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True))
    .explode()
    .values
)

general_n_sites_embryo = (
    hits_embryo_non_overlap_general.groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10, True))
    .explode()
    .values
)


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
    "n": "n",
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
        _end = start_offset + hit_end + offset_to_root
        to_pad = target_len - (_end - _start)
        # add padding to the start
        _start -= to_pad
        seq = genome.fetch(chrom, _start, _end)
        seq = reverse_complement(seq)
    else:
        # align start
        _start = start_offset + hit_start - offset_to_root
        _end = start_offset + hit_end
        to_pad = target_len - (_end - _start)
        # add padding to end
        _end += to_pad
        seq = genome.fetch(chrom, _start, _end)
    return seq


letter_to_color = {"A": "#008000", "C": "#0000ff", "G": "#ffa600", "T": "#ff0000"}

letter_to_val = {c: v for c, v in zip(list("ACGT"), np.linspace(0, 1, 4))}

nuc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nuc", list(letter_to_color.values()), 4
)

MX = 13
n_sites = 1
pattern_seq = []
for _, hit in (
    hits_organoid_non_overlap_general.loc[general_n_sites_organoid == n_sites]
    .sort_values("sequence_name")
    .iterrows()
):
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30,
    )
    pattern_seq.append(s)

ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
pc = 1e-3
ppm_FOX_organoid_general_counts = ppm + pc
ppm_FOX_organoid_general = (ppm.T / ppm.sum(1)).T
ic_FOX_organoid_general = modiscolite.util.compute_per_position_ic(
    ppm=ppm.to_numpy(), background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
)
FOX_organoid_seq_align_general = np.array(
    [[letter_to_val[nuc.upper()] for nuc in seq] for seq in pattern_seq]
)

pattern_seq = []
for _, hit in (
    hits_organoid_non_overlap_specific.loc[specific_n_sites_organoid == n_sites]
    .sort_values("sequence_name")
    .iterrows()
):
    s = get_sequence_hit(
        hit=hit,
        allignment_info=pattern_metadata.loc[hit.motif_name],
        genome=hg38,
        target_len=30,
    )
    pattern_seq.append(s)

ppm = logomaker.alignment_to_matrix([s.upper() for s in pattern_seq])
pc = 1e-3
ppm_FOX_organoid_specific_counts = ppm + pc
ppm_FOX_organoid_specific = (ppm.T / ppm.sum(1)).T
ic_FOX_organoid_specific = modiscolite.util.compute_per_position_ic(
    ppm=ppm.to_numpy(), background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
)
FOX_organoid_seq_align_specific = np.array(
    [[letter_to_val[nuc.upper()] for nuc in seq] for seq in pattern_seq]
)


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
        max_diff_se = np.max(
            [
                np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
                for p1, p2 in zip(props1, props2)
            ]
        )
        # Store results
        results.append(
            {
                "position": pos,
                "chi2_statistic": chi2,
                "p_value_chi2": p_chi2,
                "cramers_v": cramer_v,
                "max_prop_diff": max_diff,
                "max_prop_diff_se": max_diff_se,
                "n1": n1,
                "n2": n2,
            }
        )
    results_df = pd.DataFrame(results).set_index("position")
    # Add significance levels
    results_df["significance"] = pd.cut(
        results_df["p_value_chi2"],
        bins=[0, 0.001, 0.01, 0.05, 1],
        labels=["***", "**", "*", "ns"],
    )
    return results_df.round(4)


res_ppm = compare_ppm_counts(
    ppm_FOX_organoid_general_counts[0:MX], ppm_FOX_organoid_specific_counts[0:MX]
)

# USE MAX to deal with overlapping hits (variable name says sum, but it is max)
hits_merged_organoid_per_seq_and_cluster_sum_subset_specific = (
    hits_merged_organoid_subset.query(
        "overlaps_w_HEPG2 == False & overlaps_w_A549 == False"
    )
    .groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
)

hits_merged_organoid_per_seq_and_cluster_sum_subset_general = (
    hits_merged_organoid_subset.query(
        "overlaps_w_HEPG2 == True | overlaps_w_A549 == True"
    )
    .groupby(["sequence_name", "cluster"])["attribution"]
    .apply(absmax)
    .reset_index()
    .pivot(index="sequence_name", columns="cluster", values="attribution")
    .fillna(0)
)

hits_merged_organoid_per_seq_and_cluster_sum_subset_specific_scaled = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_specific
    / hits_merged_organoid_per_seq_and_cluster_sum_subset_specific.sum()
)

hits_merged_organoid_per_seq_and_cluster_sum_subset_general_scaled = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_general
    / hits_merged_organoid_per_seq_and_cluster_sum_subset_general.sum()
)

hits_organoid_bin_specific = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_specific_scaled.loc[
        :, selected_clusters
    ].abs()
    > 0.0008
)
hits_organoid_bin_specific.columns = [
    cluster_to_tf[x] for x in hits_organoid_bin_specific.columns
]

def jaccard(s_a: set[str], s_b: set[str]):
    return len(s_a & s_b) / len(s_a | s_b)

jaccard_organoid_specific = pd.DataFrame(
    index=hits_organoid_bin_specific.columns, columns=hits_organoid_bin_specific.columns
).fillna(0)

for tf_1 in jaccard_organoid_specific.columns:
    for tf_2 in jaccard_organoid_specific.index:
        jaccard_organoid_specific.loc[tf_1, tf_2] = jaccard(
            set(hits_organoid_bin_specific.loc[hits_organoid_bin_specific[tf_1]].index),
            set(hits_organoid_bin_specific.loc[hits_organoid_bin_specific[tf_2]].index),
        )


hits_organoid_bin_general = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_general_scaled.loc[
        :, selected_clusters
    ].abs()
    > 0.0008
)
hits_organoid_bin_general.columns = [
    cluster_to_tf[x] for x in hits_organoid_bin_general.columns
]

jaccard_organoid_general = pd.DataFrame(
    index=hits_organoid_bin_general.columns, columns=hits_organoid_bin_general.columns
).fillna(0)

for tf_1 in jaccard_organoid_general.columns:
    for tf_2 in jaccard_organoid_general.index:
        jaccard_organoid_general.loc[tf_1, tf_2] = jaccard(
            set(hits_organoid_bin_general.loc[hits_organoid_bin_general[tf_1]].index),
            set(hits_organoid_bin_general.loc[hits_organoid_bin_general[tf_2]].index),
        )


data = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_specific_scaled[
        selected_clusters
    ]
    > 0.2
) * 1

data.columns = [cluster_to_tf[c] for c in data.columns]

cooc_specific = data.T @ data
for cluster in selected_clusters:
    tf_name = cluster_to_tf[cluster]
    cooc_specific.loc[tf_name, tf_name] = 0

cooc_perc_specific = (cooc_specific / cooc_specific.sum()).T

data = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_general_scaled[
        selected_clusters
    ]
    > 0.2
) * 1

data.columns = [cluster_to_tf[c] for c in data.columns]

cooc_general = data.T @ data
for cluster in selected_clusters:
    tf_name = cluster_to_tf[cluster]
    cooc_general.loc[tf_name, tf_name] = 0

cooc_perc_general = (cooc_general / cooc_general.sum()).T

hits_organoid_bin = (
    hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[
        region_order_organoid_subset, selected_clusters
    ].abs()
    > 0.0008
)
hits_organoid_bin.columns = [cluster_to_tf[x] for x in hits_organoid_bin.columns]

hits_embryo_bin = (
    hits_merged_embryo_per_seq_and_cluster_sum_subset_scaled.loc[
        region_order_embryo_subset, selected_clusters
    ].abs()
    > 0.0004
)
hits_embryo_bin.columns = [cluster_to_tf[x] for x in hits_embryo_bin.columns]


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

jaccard_embryo = pd.DataFrame(
    index=hits_embryo_bin.columns, columns=hits_embryo_bin.columns
).fillna(0)

for tf_1 in jaccard_embryo.columns:
    for tf_2 in jaccard_embryo.index:
        jaccard_embryo.loc[tf_1, tf_2] = jaccard(
            set(hits_embryo_bin.loc[hits_embryo_bin[tf_1]].index),
            set(hits_embryo_bin.loc[hits_embryo_bin[tf_2]].index),
        )


# footprinting

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


def load_pattern_from_modisco_for_topics(
    topics: list[int], pattern_dir: str, prefix: str
) -> tuple[list[ModiscoPattern], list[str]]:
    patterns = []
    pattern_names = []
    for topic in topics:
        with np.load(
            os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz")
        ) as gradients_data:
            ohs = gradients_data["oh"]
            region_names = gradients_data["region_names"]
        print(os.path.join(pattern_dir, f"gradients_Topic_{topic}.npz"))
        for name, pattern in load_pattern_from_modisco(
            filename=os.path.join(pattern_dir, f"patterns_Topic_{topic}.hdf5"),
            ohs=ohs,
            region_names=region_names,
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
    ic_trim_thr: float = 0.2,
):
    # load motifs
    patterns_organoid, pattern_names_organoid = load_pattern_from_modisco_for_topics(
        topics=organoid_topics, pattern_dir=organoid_pattern_dir, prefix="organoid_"
    )
    patterns_embryo, pattern_names_embryo = load_pattern_from_modisco_for_topics(
        topics=embryo_topics, pattern_dir=embryo_pattern_dir, prefix="embryo_"
    )
    all_patterns = [*patterns_organoid, *patterns_embryo]
    all_pattern_names = [*pattern_names_organoid, *pattern_names_embryo]
    pattern_metadata = pd.read_table(pattern_metadata_path, index_col=0)
    motifs = {
        n: pattern.ppm[range(*pattern.ic_trim(ic_trim_thr))].T
        for n, pattern in zip(all_pattern_names, all_patterns)
        if n in pattern_metadata.index
    }
    hits_organoid = []
    region_order_organoid = []
    for topic in organoid_topics:
        hits = get_hit_and_attribution(
            gradients_path=os.path.join(
                organoid_pattern_dir, f"gradients_Topic_{topic}.npz"
            ),
            motifs=motifs,
        )
        hits["topic"] = topic
        hits["cluster"] = [
            pattern_metadata.loc[m, cluster_col] for m in hits["motif_name"]
        ]
        hits = hits.query("cluster in @selected_clusters").reset_index(drop=True).copy()
        hits_organoid.append(hits)
        region_order_organoid.extend(hits["sequence_name"])
    hits_embryo = []
    region_order_embryo = []
    for topic in embryo_topics:
        hits = get_hit_and_attribution(
            gradients_path=os.path.join(
                embryo_pattern_dir, f"gradients_Topic_{topic}.npz"
            ),
            motifs=motifs,
        )
        hits["topic"] = topic
        hits["cluster"] = [
            pattern_metadata.loc[m, cluster_col] for m in hits["motif_name"]
        ]
        hits = hits.query("cluster in @selected_clusters").reset_index(drop=True).copy()
        hits_embryo.append(hits)
        region_order_embryo.extend(hits["sequence_name"])
    if len(organoid_topics) > 0:
        hits_organoid_merged = reduce(
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
                        "score",
                        "p-value",
                        "-logp",
                        "topic",
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
                        "score",
                        "p-value",
                        "-logp",
                        "topic",
                    ]
                ],
                on=[
                    "motif_name",
                    "cluster",
                    "sequence_name",
                    "start",
                    "end",
                    "strand",
                    "score",
                    "p-value",
                    "attribution",
                    "topic",
                ],
                max_on="-logp",
            ),
            hits_organoid,
        )
    else:
        hits_organoid_merged = None
    if len(embryo_topics) > 0:
        hits_embryo_merged = reduce(
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
                        "score",
                        "p-value",
                        "-logp",
                        "topic",
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
                        "score",
                        "p-value",
                        "-logp",
                        "topic",
                    ]
                ],
                on=[
                    "motif_name",
                    "cluster",
                    "sequence_name",
                    "start",
                    "end",
                    "strand",
                    "score",
                    "p-value",
                    "attribution",
                    "topic",
                ],
                max_on="-logp",
            ),
            hits_embryo,
        )
    else:
        hits_embryo_merged = None
    hits_organoid_non_overlap = (
        hits_organoid_merged.groupby("sequence_name")
        .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp"))
        .reset_index(drop=True)
        if len(organoid_topics) > 0
        else None
    )
    hits_embryo_non_overlap = (
        hits_embryo_merged.groupby("sequence_name")
        .apply(lambda x: get_non_overlapping_start_end_w_max_score(x, 10, "-logp"))
        .reset_index(drop=True)
        if len(embryo_topics) > 0
        else None
    )
    return (
        (hits_organoid_merged, hits_organoid_non_overlap, region_order_organoid),
        (hits_embryo_merged, hits_embryo_non_overlap, region_order_embryo),
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
    pattern_metadata_path="../figure_3/draft/motif_metadata.tsv",
    cluster_col="hier_cluster",
    selected_patterns=[6, 7, 12, 15, 10, 18, 1, 19],
)

progenitor_ap_clustering_result = ModiscoClusteringResult(
    organoid_topics=[],
    embryo_topics=[61, 59, 31, 62, 70, 52, 71],
    pattern_metadata_path="../figure_4/motif_metadata.tsv",
    cluster_col="hier_cluster",
    selected_patterns=[1, 4, 8, 7, 6],
)

neural_crest_clustering_result = ModiscoClusteringResult(
    organoid_topics=[62, 60, 65, 59, 58],
    embryo_topics=[103, 105, 94, 91],
    pattern_metadata_path="../figure_5/draft/pattern_metadata.tsv",
    cluster_col="cluster_sub_cluster",
    selected_patterns=[3.0, 13.1, 9.2, 14.0, 11.1, 9.1, 10.2, 2.2, 2.1, 13.2],
)

neuron_clustering_result = ModiscoClusteringResult(
    organoid_topics=[6, 4, 23, 24, 13, 2],
    embryo_topics=[10, 8, 13, 24, 18, 29],
    pattern_metadata_path="../figure_6/draft/pattern_metadata.tsv",
    cluster_col="cluster_sub_cluster",
    selected_patterns=[1.1, 2.1, 2.2, 3.1, 5.1, 5.2, 6.0, 7.3, 7.5, 8.0],
)

cell_type_to_modisco_result = {
    "progenitor_dv": progenitor_dv_clustering_result,
    "progenitor_ap": progenitor_ap_clustering_result,
    "neural_crest": neural_crest_clustering_result,
    "neuron": neuron_clustering_result,
}

organoid_dl_motif_dir = "../data_prep_new/organoid_data/MODELS/modisco/"
embryo_dl_motif_dir = "../data_prep_new/embryo_data/MODELS/modisco/"

for cell_type in cell_type_to_modisco_result:
    print(cell_type)
    (
        (hits_organoid, hits_organoid_non_overlap, region_order_organoid),
        (hits_embryo, hits_embryo_non_overlap, region_order_embryo),
    ) = get_hits_for_topics(
        organoid_pattern_dir=organoid_dl_motif_dir,
        embryo_pattern_dir=embryo_dl_motif_dir,
        ic_trim_thr=0.2,
        organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
        embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
        selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
        pattern_metadata_path=cell_type_to_modisco_result[
            cell_type
        ].pattern_metadata_path,
        cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
    )
    cell_type_to_modisco_result[cell_type].hits_organoid = hits_organoid
    cell_type_to_modisco_result[
        cell_type
    ].hits_organoid_non_overlap = hits_organoid_non_overlap
    cell_type_to_modisco_result[cell_type].region_order_organoid = region_order_organoid
    cell_type_to_modisco_result[cell_type].hits_embryo = hits_embryo
    cell_type_to_modisco_result[
        cell_type
    ].hits_embryo_non_overlap = hits_embryo_non_overlap
    cell_type_to_modisco_result[cell_type].region_order_embryo = region_order_embryo


def get_chrom_start_stop_topic_hit(r):
    for _, h in r.iterrows():
        chrom, start, _ = h["sequence_name"].replace("-", ":").split(":")
        start = int(start)
        # end = int(end)
        start += h["start"]
        # end = start + h["end"]
        yield chrom, int(start), int(start), h["topic"]


import pyBigWig


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
    topic_name_to_model_index_organoid(f.replace("Topic", "Topic_").split(".")[0])
    + 1: pyBigWig.open(f"../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/{f}")
    for f in os.listdir("../data_prep_new/organoid_data/ATAC/bw_per_topic_cut/")
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
    topic_to_region_name_to_grad_organoid[
        int(f.split(".")[0].replace("gradients_Topic_", ""))
    ] = dict(zip(region_names, attr.squeeze() * oh))

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

hit_per_seq_organoid = pd.concat(hit_per_seq_organoid).reset_index(drop=True)

max_hit_per_seq_organoid = hit_per_seq_organoid.loc[
    hit_per_seq_organoid.groupby(["sequence_name", "topic"])["attribution"].idxmax()
]

min_hit_per_seq_organoid = hit_per_seq_organoid.loc[
    hit_per_seq_organoid.groupby(["sequence_name", "topic"])["attribution"].idxmin()
]


pattern = "progenitor_dv_6"

D = max_hit_per_seq_organoid.query("cluster == @pattern")
x = get_chrom_start_stop_topic_hit(D)

FOX_organoid_cov = np.zeros((len(D), S), dtype=float)

FOX_organoid_attr = np.full((len(D), S), np.nan)

for i, (chrom, start, end, topic) in tqdm(
    enumerate(x), total=FOX_organoid_cov.shape[0]
):
    FOX_organoid_cov[i] = np.array(
        bw_per_topic_organoid[topic].values(
            chrom,
            start - S // 2,
            end + S // 2,
        ),
        dtype=float,
    )
    mid = S // 2
    seq_name = D.iloc[i]["sequence_name"]
    start = mid - int(D.iloc[i]["start"])
    end = start + 500
    l_start = max(0, start)
    l_end = min(S, end)
    x_start = max(0, -start)
    x_end = x_start + (l_end - l_start)
    FOX_organoid_attr[i][l_start:l_end] = topic_to_region_name_to_grad_organoid[topic][
        seq_name
    ][x_start:x_end, :].sum(1)


FOX_organoid_cov_mn = np.nanmean(
    np.lib.stride_tricks.sliding_window_view(FOX_organoid_cov, bin_size, axis=1), axis=2
)
FOX_organoid_attr_mn = np.nan_to_num(
    np.nanmean(
        np.lib.stride_tricks.sliding_window_view(FOX_organoid_attr, bin_size, axis=1),
        axis=2,
    )
)


def scale(X):
    return (X - X.min()) / (X.max() - X.min())


print("GENERATING FIGURE 3")

N_PIXELS_PER_GRID = 50

plt.style.use(
    "/data/projects/c20/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/paper.mplstyle"
)

fig = plt.figure()
width, height = fig.get_size_inches()
n_w_pixels = fig.get_dpi() * width
n_h_pixels = fig.get_dpi() * height
ncols = int((n_w_pixels) // N_PIXELS_PER_GRID)
nrows = int((n_h_pixels) // N_PIXELS_PER_GRID)
gs = fig.add_gridspec(
    nrows, ncols, wspace=0.05, hspace=0.1, left=0.05, right=0.97, bottom=0.05, top=0.95
)
"""
umap
"""
ax_topic_umap_organoid_1 = fig.add_subplot(gs[0:5, 0:5])
ax_topic_umap_organoid_2 = fig.add_subplot(gs[0:5, 6:11])
ax_topic_umap_embryo_1 = fig.add_subplot(gs[6:11, 0:5])
ax_topic_umap_embryo_2 = fig.add_subplot(gs[6:11, 6:11])
organoid_cells_both = list(
    set(organoid_progenitor_cell_topic.index) & set(adata_organoid_progenitor.obs_names)
)
r_values = organoid_progenitor_cell_topic.loc[
    organoid_cells_both, model_index_to_topic_name_organoid(33 - 1)
].values
g_values = organoid_progenitor_cell_topic.loc[
    organoid_cells_both, model_index_to_topic_name_organoid(38 - 1)
].values
b_values = organoid_progenitor_cell_topic.loc[
    organoid_cells_both, model_index_to_topic_name_organoid(36 - 1)
].values
rgb_scatter_plot(
    x=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_topic_umap_organoid_1,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    # r_vmin=0,
    # r_vmax=1.5,
    g_vmin=0,
    g_vmax=0.1,
    # b_vmin=0,
    # b_vmax=1.5,
    r_name="Topic33",
    g_name="Topic38",
    b_name="Topic36",
)
organoid_cells_both = list(
    set(organoid_progenitor_cell_topic.index) & set(adata_organoid_progenitor.obs_names)
)
r_values = organoid_progenitor_cell_topic.loc[
    organoid_cells_both, model_index_to_topic_name_organoid(54 - 1)
].values
g_values = organoid_progenitor_cell_topic.loc[
    organoid_cells_both, model_index_to_topic_name_organoid(48 - 1)
].values
b_values = np.zeros_like(g_values)
rgb_scatter_plot(
    x=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 0],
    y=adata_organoid_progenitor[organoid_cells_both].obsm["X_umap"][:, 1],
    ax=ax_topic_umap_organoid_2,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    r_vmin=0,
    r_vmax=0.2,
    # g_vmin=0,
    # g_vmax=0.1,
    # b_vmin=0,
    # b_vmax=1.5,
    r_name="Topic54",
    g_name="Topic48",
)
embryo_cells_both = list(
    set(embryo_progenitor_cell_topic.index) & set(adata_embryo_progenitor.obs_names)
)
r_values = embryo_progenitor_cell_topic.loc[
    embryo_cells_both, model_index_to_topic_name_embryo(34 - 1)
].values
g_values = embryo_progenitor_cell_topic.loc[
    embryo_cells_both, model_index_to_topic_name_embryo(38 - 1)
].values
b_values = embryo_progenitor_cell_topic.loc[
    embryo_cells_both, model_index_to_topic_name_embryo(79 - 1)
].values
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 2],
    ax=ax_topic_umap_embryo_1,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    # r_vmin=0,
    # r_vmax=1.5,
    g_vmin=0,
    g_vmax=0.2,
    b_vmin=0,
    b_vmax=0.1,
    r_name="Topic34",
    g_name="Topic38",
    b_name="Topic79",
)
embryo_cells_both = list(
    set(embryo_progenitor_cell_topic.index) & set(adata_embryo_progenitor.obs_names)
)
r_values = embryo_progenitor_cell_topic.loc[
    embryo_cells_both, model_index_to_topic_name_embryo(88 - 1)
].values
g_values = embryo_progenitor_cell_topic.loc[
    embryo_cells_both, model_index_to_topic_name_embryo(58 - 1)
].values
b_values = np.zeros_like(g_values)
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 2],
    ax=ax_topic_umap_embryo_2,
    g_cut=0,
    r_values=r_values,
    g_values=g_values,
    b_values=b_values,
    # r_vmin=0,
    # r_vmax=0.1,
    # g_vmin=0,
    # g_vmax=0.1,
    # b_vmin=0,
    # b_vmax=1.5,
    r_name="Topic88",
    g_name="Topic58",
)
"""
boxplot
"""
ax_shh_boxplot_organoid = fig.add_subplot(gs[0:5, 13:17])
ax_shh_boxplot_embryo = fig.add_subplot(gs[6:11, 13:17])
_ = ax_shh_boxplot_organoid.boxplot(
    SHH_expr_organoid_per_topic.values(),
    labels=SHH_expr_organoid_per_topic.keys(),
    flierprops=dict(markersize=2),
    medianprops=dict(color="black"),
)
ax_shh_boxplot_organoid.set_yticks(np.arange(0, 16, 5))
ax_shh_boxplot_organoid.spines[["right", "top"]].set_visible(False)
ax_shh_boxplot_organoid.set_ylabel("SHH CPM")
_ = ax_shh_boxplot_embryo.boxplot(
    SHH_expr_embryo_per_topic.values(),
    labels=SHH_expr_embryo_per_topic.keys(),
    flierprops=dict(markersize=2),
    medianprops=dict(color="black"),
)
ax_shh_boxplot_embryo.set_yticks(np.arange(0, 16, 5))
ax_shh_boxplot_embryo.spines[["right", "top"]].set_visible(False)
ax_shh_boxplot_embryo.set_ylabel("SHH CPM")
ax_shh_boxplot_embryo.set_xlabel("Topic")
"""
tracks
"""
for i, topic in enumerate(topic_organoid_to_locus_val):
    ax = fig.add_subplot(gs[i, 18:ncols])
    _ = ax.fill_between(
        np.arange(locus[1], locus[2]),
        np.zeros(locus[2] - locus[1]),
        topic_organoid_to_locus_val[topic],
        color=color_dict["progenitor_topics"][f"organoid_{topic}"],
    )
    _ = ax.set_ylim(0, 7)
    _ = ax.set_xlim(locus[1], locus[2])
    ax.set_axis_off()
    highlight = matplotlib.patches.Rectangle(
        xy=(SFPE1_hg38_coord[0], 0),
        width=SFPE1_hg38_coord[1] - SFPE1_hg38_coord[0],
        height=7,
        fill=True,
        facecolor="#008bd8",
        alpha=0.5,
        lw=0,
    )
    ax.add_patch((highlight))
    highlight = matplotlib.patches.Rectangle(
        xy=(SFPE2_hg38_coord[0], 0),
        width=SFPE2_hg38_coord[1] - SFPE2_hg38_coord[0],
        height=7,
        fill=True,
        facecolor="#008bd8",
        alpha=0.5,
        lw=0,
    )
    ax.add_patch(highlight)
for i, topic in enumerate(topic_embryo_to_locus_val):
    ax = fig.add_subplot(gs[i + 6, 18:ncols])
    _ = ax.fill_between(
        np.arange(locus[1], locus[2]),
        np.zeros(locus[2] - locus[1]),
        topic_embryo_to_locus_val[topic],
        color=color_dict["progenitor_topics"][f"embryo_{topic}"],
    )
    _ = ax.set_ylim(0, 5)
    _ = ax.set_xlim(locus[1], locus[2])
    ax.set_axis_off()
    highlight = matplotlib.patches.Rectangle(
        xy=(SFPE1_hg38_coord[0], 0),
        width=SFPE1_hg38_coord[1] - SFPE1_hg38_coord[0],
        height=7,
        fill=True,
        facecolor="#008bd8",
        alpha=0.5,
        lw=0,
    )
    ax.add_patch((highlight))
    highlight = matplotlib.patches.Rectangle(
        xy=(SFPE2_hg38_coord[0], 0),
        width=SFPE2_hg38_coord[1] - SFPE2_hg38_coord[0],
        height=7,
        fill=True,
        facecolor="#008bd8",
        alpha=0.5,
        lw=0,
    )
    ax.add_patch(highlight)
ax = fig.add_subplot(gs[i + 6 + 1, 18:ncols])
for k in transcript:
    if k == "transcript":
        for start, stop in transcript[k]:
            ax.plot([start, stop], [0, 0], color="black", lw=1)
    else:
        size = transcript_feature_to_size[k]
        for start, stop in transcript[k]:
            p = matplotlib.patches.Rectangle(
                (start, -size / 2),
                stop - start,
                size,
                facecolor="black",
                lw=0,
                fill=True,
            )
            ax.add_patch(p)
_ = ax.set_xlim(locus[1], locus[2])
_ = ax.set_ylim(-1, 1)
ax.set_axis_off()
"""
Deep explainer
"""
ax_de_sfpe2_organoid = fig.add_subplot(gs[13:15, 0:23])
ax_de_sfpe1_organoid = fig.add_subplot(gs[13:15, 24:ncols])
ax_de_sfpe2_embryo = fig.add_subplot(gs[16:18, 0:23])
ax_de_sfpe1_embryo = fig.add_subplot(gs[16:18, 24:ncols])
_ = logomaker.Logo(
    pd.DataFrame(
        np.multiply(*shh_locus_de["sfpe1"]["organoid"])
        .squeeze()
        .astype(float)[140:300],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_sfpe1_organoid,
)
_ = logomaker.Logo(
    pd.DataFrame(
        np.multiply(*shh_locus_de["sfpe2"]["organoid"])
        .squeeze()
        .astype(float)[150:460],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_sfpe2_organoid,
)
_ = logomaker.Logo(
    pd.DataFrame(
        np.multiply(*shh_locus_de["sfpe1"]["embryo"]).squeeze().astype(float)[140:300],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_sfpe1_embryo,
)
_ = logomaker.Logo(
    pd.DataFrame(
        np.multiply(*shh_locus_de["sfpe2"]["embryo"]).squeeze().astype(float)[150:460],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_sfpe2_embryo,
)
_ = ax_de_sfpe1_organoid.set_xticks([])
_ = ax_de_sfpe2_organoid.set_xticks([])
for ax in [
    ax_de_sfpe1_organoid,
    ax_de_sfpe1_embryo,
    ax_de_sfpe2_organoid,
    ax_de_sfpe2_embryo,
]:
    ax.spines[["right", "top"]].set_visible(False)
"""
code table
"""
x_current = 0
y_current = 20
pattern_height = 2
pattern_width = 4
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
            )
            ymn, ymx = ax.get_ylim()
            YMIN = min(ymn, YMIN)
            YMAX = max(ymx, YMAX)
        if topic_embr is not None:
            pwm = cluster_to_topic_to_avg_pattern_embryo[cluster][topic_embr]
            _ = logomaker.Logo(
                pd.DataFrame(-pwm, columns=["A", "C", "G", "T"]),
                ax=ax,
                edgecolor="black",
                edgewidth=0.4,
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
        _, xmax = ax.get_xlim()
"""
hit heatmap
"""
x_current = 21
ax_hit_hm = fig.add_subplot(gs[y_current : y_current + 15, x_current : x_current + 5])
tmp = hits_merged_organoid_per_seq_and_cluster_sum_subset_scaled.loc[
    region_order_organoid_subset, selected_clusters
]
tmp.columns = [cluster_to_tf[x] for x in tmp.columns]
sns.heatmap(
    tmp,
    yticklabels=False,
    xticklabels=True,
    robust=True,
    ax=ax_hit_hm,
    cbar=False,
    cmap="bwr",
    vmin=-0.0008,
    vmax=0.0008,
)
#######
# bars
######
x_current = 27
ax_organoid_expr_heatmap = fig.add_subplot(
    gs[y_current : y_current + 8, x_current + 2 : x_current + 7]
)
ax_embryo_expr_heatmap = fig.add_subplot(
    gs[y_current : y_current + 8, x_current + 7 : ncols]
)
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
ax_nx_organoid = fig.add_subplot(
    gs[y_current + 10 : y_current + 15, x_current + 2 : x_current + 7]
)
ax_nx_embryo = fig.add_subplot(
    gs[y_current + 10 : y_current + 15, x_current + 7 : ncols]
)
sns.heatmap(
    jaccard_organoid,
    cmap="viridis",
    vmin=0,
    vmax=0.3,
    ax=ax_nx_organoid,
    linecolor="black",
    lw=0.5,
    cbar=False,
    yticklabels=True
)
sns.heatmap(
    jaccard_embryo,
    cmap="viridis",
    vmin=0.1,
    vmax=0.3,
    ax=ax_nx_embryo,
    linecolor="black",
    lw=0.5,
    yticklabels=False,
    cbar=False,
)
y_current = 38
ax_profile = fig.add_subplot(gs[y_current : y_current + 3, 0:5])
ax_hm = fig.add_subplot(gs[y_current + 3 : nrows, 0:5])
ax_profile.plot(
    np.arange(FOX_organoid_cov_mn.shape[1]),
    scale(np.nanmean(FOX_organoid_cov_mn, axis=0)),
    color="black",
    label="cut sites",
    zorder=2,
)
ax_profile.plot(
    np.arange(FOX_organoid_attr_mn.shape[1]),
    scale(FOX_organoid_attr_mn.mean(0)),
    color="red",
    label="attribution",
    zorder=1,
)
ax_profile.legend(loc="upper left")
sns.heatmap(
    np.nan_to_num(
        FOX_organoid_cov_mn[
            np.argsort(np.nan_to_num(FOX_organoid_cov_mn).mean(1))[::-1]
        ]
    ),
    ax=ax_hm,
    cmap="Spectral_r",
    robust=True,
    cbar=False,
    yticklabels=False,
)
_ = ax_hm.set_xticks(
    np.arange(0, S, 10),
)
ax_profile.grid(True)
ax_profile.set_axisbelow(True)
###
# hits barchart
###
x_current = 8
ax_bar_n_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 6]
)
pval = mannwhitneyu(
    hits_organoid_number_sites_general, hits_organoid_number_sites_specific
).pvalue
max_sites = 10
for n in range(max_sites):
    ax_bar_n_spec.bar(
        x=n,
        height=sum(hits_organoid_number_sites_specific == n + 1)
        / len(hits_organoid_number_sites_specific),
        width=0.5,
        align="edge",
        color="lightcoral",
        edgecolor="black",
    )
    ax_bar_n_spec.bar(
        x=n + 0.5,
        height=sum(hits_embryo_number_sites_specific == n + 1)
        / len(hits_embryo_number_sites_specific),
        width=0.5,
        align="edge",
        color="deepskyblue",
        edgecolor="black",
    )
ax_bar_n_spec.set_xticks(np.arange(max_sites), labels=np.arange(max_sites) + 1)
ax_bar_n_spec.set_ylim(0, 0.5)
ax_bar_n_spec.grid()
ax_bar_n_spec.set_axisbelow(True)
ax_bar_n_spec.text(
    0.3, 0.8, s=f"pval. = 1e{int((np.log10(pval)))}", transform=ax_bar_n_spec.transAxes
)
ax_bar_n_spec.set_ylabel("Fraction of regions")
ax_bar_n_general = fig.add_subplot(gs[nrows - 4 : nrows, x_current : x_current + 6])
max_sites = 10
for n in range(max_sites):
    ax_bar_n_general.bar(
        x=n,
        height=sum(hits_organoid_number_sites_general == n + 1)
        / len(hits_organoid_number_sites_general),
        width=0.5,
        align="edge",
        color="lightcoral",
        edgecolor="black",
    )
    ax_bar_n_general.bar(
        x=n + 0.5,
        height=sum(hits_embryo_number_sites_general == n + 1)
        / len(hits_embryo_number_sites_general),
        width=0.5,
        align="edge",
        color="deepskyblue",
        edgecolor="black",
    )
ax_bar_n_general.set_xticks(np.arange(max_sites), labels=np.arange(max_sites) + 1)
ax_bar_n_general.set_ylim(0, 0.5)
ax_bar_n_general.grid()
ax_bar_n_general.set_axisbelow(True)
ax_bar_n_general.set_xlabel("Number of FOX sites")
ax_bar_n_general.set_ylabel("Fraction of regions")
#
max_sites = 5
y_current = 38
x_current = 17
ax_bplot_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 4]
)
ax_bplot_general = fig.add_subplot(gs[nrows - 4 : nrows, x_current : x_current + 4])
a = -np.log10(
    hits_organoid_non_overlap_general.loc[general_n_sites_organoid == 1, "p-value"]
    + 0.000001
)
b = -np.log10(
    hits_organoid_non_overlap_specific.loc[specific_n_sites_organoid == 1, "p-value"]
    + 0.000001
)
pval = mannwhitneyu(a, b).pvalue
ax_bplot_spec.boxplot(
    [
        -np.log10(
            hits_organoid_non_overlap_specific.loc[
                specific_n_sites_organoid == (n + 1), "p-value"
            ]
            + 0.000001
        )
        for n in range(max_sites)
    ],
    labels=[n + 1 for n in range(max_sites)],
    flierprops=dict(markersize=2),
    medianprops=dict(color="lightcoral"),
    boxprops=dict(color="lightcoral"),
    whiskerprops=dict(color="lightcoral"),
    capprops=dict(color="lightcoral"),
)
ax_bplot_general.boxplot(
    [
        -np.log10(
            hits_organoid_non_overlap_general.loc[
                general_n_sites_organoid == (n + 1), "p-value"
            ]
            + 0.000001
        )
        for n in range(max_sites)
    ],
    labels=[n + 1 for n in range(max_sites)],
    flierprops=dict(markersize=2),
    medianprops=dict(color="lightcoral"),
    boxprops=dict(color="lightcoral"),
    whiskerprops=dict(color="lightcoral"),
    capprops=dict(color="lightcoral"),
)
ax_bplot_spec.text(
    0, 0, s=f"pval = 1e{int(np.log10(pval))}", transform=ax_bplot_spec.transAxes
)
ax_bplot_spec.set_ylabel("$-log_{10}(pval)$")
ax_bplot_general.set_ylabel("$-log_{10}(pval)$")
ax_bplot_spec.set_xlabel("Number of FOX sites")
x_current = 23
a = -np.log10(
    hits_embryo_non_overlap_general.loc[general_n_sites_embryo == 1, "p-value"]
    + 0.000001
)
b = -np.log10(
    hits_embryo_non_overlap_specific.loc[specific_n_sites_embryo == 1, "p-value"]
    + 0.000001
)
pval = mannwhitneyu(a, b).pvalue
ax_bplot_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 4]
)
ax_bplot_general = fig.add_subplot(gs[nrows - 4 : nrows, x_current : x_current + 4])
ax_bplot_spec.boxplot(
    [
        -np.log10(
            hits_embryo_non_overlap_specific.loc[
                specific_n_sites_embryo == (n + 1), "p-value"
            ]
            + 0.000001
        )
        for n in range(max_sites)
    ],
    labels=[n + 1 for n in range(max_sites)],
    flierprops=dict(markersize=2),
    medianprops=dict(color="deepskyblue"),
    boxprops=dict(color="deepskyblue"),
    whiskerprops=dict(color="deepskyblue"),
    capprops=dict(color="deepskyblue"),
)
ax_bplot_general.boxplot(
    [
        -np.log10(
            hits_embryo_non_overlap_general.loc[
                general_n_sites_embryo == (n + 1), "p-value"
            ]
            + 0.000001
        )
        for n in range(max_sites)
    ],
    labels=[n + 1 for n in range(max_sites)],
    flierprops=dict(markersize=2),
    medianprops=dict(color="deepskyblue"),
    boxprops=dict(color="deepskyblue"),
    whiskerprops=dict(color="deepskyblue"),
    capprops=dict(color="deepskyblue"),
)
ax_bplot_spec.text(
    0, 0, s=f"pval = 1e{int(np.log10(pval))}", transform=ax_bplot_spec.transAxes
)
#
x_current = 30
ax_FOX_logo_organoid = fig.add_subplot(
    gs[y_current : y_current + 2, x_current : x_current + 4]
)
ax_FOX_logo_organoid_hm = fig.add_subplot(
    gs[y_current + 4 : nrows, x_current : x_current + 4]
)
_ = logomaker.Logo(
    (ppm_FOX_organoid_specific * ic_FOX_organoid_specific[:, None]),
    ax=ax_FOX_logo_organoid,
)
sns.heatmap(
    FOX_organoid_seq_align_specific,
    cmap=nuc_cmap,
    ax=ax_FOX_logo_organoid_hm,
    cbar=False,
    yticklabels=False,
)
ax_FOX_logo_organoid_hm.set_ylabel("motif instances")
ax_FOX_logo_organoid.set_ylabel("bits")
ax_FOX_logo_organoid_hm.set_xlabel("position")
ax_FOX_logo_organoid.set_ylim((0, 2))
ax_FOX_logo_organoid_hm.set_xticks(
    np.arange(FOX_organoid_seq_align_specific.shape[1]), labels=[]
)
ax_FOX_logo_organoid.set_xticks(
    np.arange(FOX_organoid_seq_align_specific.shape[1]) - 0.5,
    labels=np.arange(FOX_organoid_seq_align_specific.shape[1]),
)
ax_FOX_logo_organoid.set_xlim(-0.5, MX + 0.5)
ax_FOX_logo_organoid_hm.set_xlim(0, MX + 1)
#
x_current = 35
ax_FOX_logo_organoid = fig.add_subplot(
    gs[y_current : y_current + 2, x_current : x_current + 4]
)
ax_FOX_logo_organoid_hm = fig.add_subplot(
    gs[y_current + 4 : nrows, x_current : x_current + 4]
)
_ = logomaker.Logo(
    (ppm_FOX_organoid_general * ic_FOX_organoid_general[:, None]),
    ax=ax_FOX_logo_organoid,
)
sns.heatmap(
    FOX_organoid_seq_align_general,
    cmap=nuc_cmap,
    ax=ax_FOX_logo_organoid_hm,
    cbar=False,
    yticklabels=False,
)
ax_FOX_logo_organoid_hm.set_xlabel("position")
ax_FOX_logo_organoid.set_ylim((0, 2))
ax_FOX_logo_organoid_hm.set_xticks(
    np.arange(FOX_organoid_seq_align_general.shape[1]), labels=[]
)
ax_FOX_logo_organoid.set_xticks(
    np.arange(FOX_organoid_seq_align_general.shape[1]) - 0.5,
    labels=np.arange(FOX_organoid_seq_align_general.shape[1]),
)
ax_FOX_logo_organoid.set_xlim(-0.5, MX + 0.5)
ax_FOX_logo_organoid_hm.set_xlim(0, MX + 1)
#


fig.tight_layout()
fig.savefig("Figure_3_v2.png", transparent=False)
fig.savefig("Figure_3_v2.pdf")


fig, ax = plt.subplots()
sns.heatmap(
    np.nan_to_num(
        FOX_organoid_cov_mn[
            np.argsort(np.nan_to_num(FOX_organoid_cov_mn).mean(1))[::-1]
        ]
    ),
    ax=ax,
    cmap="Spectral_r",
    robust=True,
    cbar=True,
    yticklabels=False,
)
_ = ax_hm.set_xticks(
    np.arange(0, S, 10),
)
fig.savefig("footprint.png")

