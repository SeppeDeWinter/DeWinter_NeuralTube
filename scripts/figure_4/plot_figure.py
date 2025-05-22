import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
from crested.utils._seq_utils import one_hot_encode_sequence
import pysam
import tensorflow as tf
import os
from dataclasses import dataclass
import h5py
import logomaker
from typing import Self
from tqdm import tqdm
import torch
from tangermeme.tools.tomtom import tomtom
from tangermeme.tools.fimo import fimo
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
import pickle
import random
from pycisTopic.topic_binarization import binarize_topics

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


def replace_hit(
    hit, alignment_info, genome, replace_to, verbose=False, add_extra_to_start=False
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
        hit_end = region_start + hit_end - offset_to_root
    else:
        # align start
        hit_start = region_start + hit_start + offset_to_root
        hit_end = region_start + hit_end
    orig_pattern = genome.fetch(chrom, hit_start, hit_end)
    delta = len(orig_pattern) - len(replace_to)
    if delta < 0:
        raise ValueError("replace_to is shorter than orig_pattern")
    if is_rc_hit:
        if not add_extra_to_start:
            replace_to += "".join(
                [COMPLEMENT.get(n, "") for n in orig_pattern[0:delta]]
            ).lower()
        else:
            replace_to = (
                "".join(
                    [COMPLEMENT.get(n, "") for n in orig_pattern[::-1][0:delta]]
                ).lower()
                + replace_to
            )
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
            f">{chrom}:{region_start}-{region_end}\t"
            + f"[{hit_start_relative}, {hit_end_relative}]\t{is_rc_hit}\n\t{orig_pattern}\n\t{replace_to}"
        )
    return (
        f"{chrom}:{region_start}-{region_end}",
        [hit_start_relative, hit_end_relative],
        replace_to,
    )


def get_non_overlapping_start_end_w_max_score(df, max_overlap, score_col):
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


# load embryo data and subset for ATAC cells

adata_embryo_progenitor = sc.read_h5ad("../figure_1/adata_embryo_progenitor.h5ad")

# load cell topic

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

ap_topics = np.array([31, 29, 1, 32, 40, 22, 41]) + 30

dv_topics = np.array([4, 44, 57, 8, 49, 21, 58, 28]) + 30

region_topic = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv", index_col=0
)

selected_regions = set()
top_n = 1_000
for topic in [*ap_topics, *dv_topics]:
    print(topic)
    topic_name = model_index_to_topic_name_embryo(topic - 1).replace("progenitor_", "")
    selected_regions = selected_regions | set(
        region_topic[topic_name].sort_values(ascending=False).head(top_n).index
    )

ap_dv_topics_names = [
    model_index_to_topic_name_embryo(topic - 1).replace("progenitor_", "")
    for topic in [*ap_topics, *dv_topics]
]

corr_region_topic = region_topic.loc[list(selected_regions), ap_dv_topics_names].corr()
corr_region_topic.index = [x.replace("Topic_", "") for x in corr_region_topic.index]
corr_region_topic.columns = [x.replace("Topic_", "") for x in corr_region_topic.columns]

# load human model
path_to_human_model = "../data_prep_new/embryo_data/MODELS/"

model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_human_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)

model.load_weights(os.path.join(path_to_human_model, "model_epoch_36.hdf5"))

selected_regions_ap = []
top_n = 3_000
for topic in ap_topics:
    print(topic)
    topic_name = model_index_to_topic_name_embryo(topic - 1).replace("progenitor_", "")
    for r in region_topic[topic_name].sort_values(ascending=False).head(top_n).index:
        if r not in selected_regions_ap:
            selected_regions_ap.append(r)

hg38 = pysam.FastaFile("../../../../../resources/hg38/hg38.fa")

selected_regions_ap_onehot = np.array(
    [
        one_hot_encode_sequence(
            hg38.fetch(*region_to_chrom_start_end(r)), expand_dim=False
        )
        for r in tqdm(selected_regions_ap)
    ]
)

selected_regions_ap_prediction_score = model.predict(
    selected_regions_ap_onehot, verbose=1
)

embryo_dl_motif_dir = "../data_prep_new/embryo_data/MODELS/modisco/"

patterns_dl_embryo = []
pattern_names_dl_embryo = []
for topic in tqdm(ap_topics):
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

pattern_topic = np.array([x.split("_")[3] for x in pattern_names_dl_embryo])

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

linkage = hierarchy.linkage(distance.pdist(dat), method="average")

clusters = hierarchy.fcluster(linkage, t=10, criterion="maxclust")

clusters_to_show = np.array([1, 4, 8, 7, 6])

seqlet_count_per_cluster = np.zeros((10, len(ap_topics)))
for pattern, cluster, name in zip(
    patterns_to_cluster, clusters, patterns_to_cluster_names
):
    topic = int(name.split("_")[3])
    j = np.where(ap_topics == topic)[0][0]
    seqlet_count_per_cluster[cluster - 1, j] += len(pattern.seqlets)

motif_metadata = pd.DataFrame(index=patterns_to_cluster_names)

motif_metadata["hier_cluster"] = clusters

motif_metadata["ic_start"] = [
    pattern.ic_trim(0.2)[0] for pattern in patterns_to_cluster
]
motif_metadata["ic_stop"] = [pattern.ic_trim(0.2)[1] for pattern in patterns_to_cluster]
motif_metadata["is_root_motif"] = False
motif_metadata["is_rc_to_root"] = False
motif_metadata["offset_to_root"] = 0

for cluster in set(clusters):
    cluster_idc = np.where(clusters == cluster)[0]
    print(f"cluster: {cluster}")
    _, _, o, _, s = tomtom(
        [torch.from_numpy(patterns_to_cluster[m].ppm).T for m in cluster_idc],
        [torch.from_numpy(patterns_to_cluster[m].ppm).T for m in cluster_idc],
    )
    for i, m in enumerate(cluster_idc):
        print(i)
        rel = np.argmax([np.mean(patterns_to_cluster[m].ic()) for m in cluster_idc])
        motif_metadata.loc[
            patterns_to_cluster_names[cluster_idc[rel]], "is_root_motif"
        ] = True
        is_rc = s[i, rel] == 1
        offset = int(o[i, rel].numpy())
        motif_metadata.loc[patterns_to_cluster_names[m], "is_rc_to_root"] = bool(is_rc)
        motif_metadata.loc[patterns_to_cluster_names[m], "offset_to_root"] = offset

selected_regions_topic_31_1 = list(
    set(region_topic["Topic_1"].sort_values(ascending=False).head(3_000).index)
    | set(region_topic["Topic_31"].sort_values(ascending=False).head(3_000).index)
)

ohs_topic_31_1 = np.array(
    [
        one_hot_encode_sequence(
            hg38.fetch(r.split(":")[0], *map(int, r.split(":")[1].split("-"))),
            expand_dim=False,
        )
        for r in selected_regions_topic_31_1
    ]
)

motifs = {
    **{
        patterns_to_cluster_names[pattern_idx]: patterns_to_cluster[pattern_idx]
        .ppm[range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))]
        .T
        for pattern_idx in np.where(clusters == 4)[0]
    },
    **{
        patterns_to_cluster_names[pattern_idx]: patterns_to_cluster[pattern_idx]
        .ppm[range(*patterns_to_cluster[pattern_idx].ic_trim(0.1))]
        .T
        for pattern_idx in np.where(clusters == 8)[0]
    },
}

hits = pd.concat(
    fimo(motifs=motifs, sequences=ohs_topic_31_1.swapaxes(1, 2), threshold=0.0001)
)

hits["cluster"] = [motif_metadata.loc[n, "hier_cluster"] for n in hits["motif_name"]]
hits["-log(p-value)"] = -np.log(hits["p-value"] + 1e-6)
hits["sequence_name"] = [selected_regions_topic_31_1[x] for x in hits["sequence_name"]]

hits_non_overlap = (
    hits.groupby(["sequence_name", "cluster"])
    .apply(
        lambda hit_region_cluster: get_non_overlapping_start_end_w_max_score(
            hit_region_cluster, 10, "-log(p-value)"
        )
    )
    .reset_index(drop=True)
)

modifications_per_sequence_4 = {}
for _, hit in (
    hits_non_overlap.query("cluster == 4").sort_values("sequence_name").iterrows()
):
    region_name, start_end, replace_to = replace_hit(
        hit, motif_metadata.loc[hit.motif_name], hg38, "AGGGATTAG", verbose=True
    )
    if region_name not in modifications_per_sequence_4:
        modifications_per_sequence_4[region_name] = []
    modifications_per_sequence_4[region_name].append((start_end, replace_to))

modifications_per_sequence_8 = {}
for _, hit in (
    hits_non_overlap.query("cluster == 8").sort_values("sequence_name").iterrows()
):
    region_name, start_end, replace_to = replace_hit(
        hit,
        motif_metadata.loc[hit.motif_name],
        hg38,
        "CTAATTAG",
        verbose=True,
        add_extra_to_start=True,
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
    orig_seq = hg38.fetch(chrom, start, end)
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
    orig_seq = hg38.fetch(chrom, start, end)
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

prediction_score_cluster_4_orig = model.predict(
    np.array(
        [one_hot_encode_sequence(s, expand_dim=False) for s in cluster_4_seq_orig]
    ),
    verbose=True,
)

prediction_score_cluster_4_modi = model.predict(
    np.array(
        [one_hot_encode_sequence(s, expand_dim=False) for s in cluster_4_seq_modi]
    ),
    verbose=True,
)

prediction_score_cluster_8_orig = model.predict(
    np.array(
        [one_hot_encode_sequence(s, expand_dim=False) for s in cluster_8_seq_orig]
    ),
    verbose=True,
)

prediction_score_cluster_8_modi = model.predict(
    np.array(
        [one_hot_encode_sequence(s, expand_dim=False) for s in cluster_8_seq_modi]
    ),
    verbose=True,
)

data_4 = (
    pd.concat(
        [
            pd.DataFrame(
                np.concatenate(
                    [
                        prediction_score_cluster_4_orig[:, ap_topics - 1],
                        np.repeat("wt", prediction_score_cluster_4_orig.shape[0])[
                            :, None
                        ],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "condition",
                ],
                index=modifications_per_sequence_4.keys(),
            ),
            pd.DataFrame(
                np.concatenate(
                    [
                        prediction_score_cluster_4_modi[:, ap_topics - 1],
                        np.repeat("alt", prediction_score_cluster_4_modi.shape[0])[
                            :, None
                        ],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "condition",
                ],
                index=modifications_per_sequence_4.keys(),
            ),
        ]
    )
    .melt(id_vars="condition", ignore_index=False)
    .rename({"variable": "Topic", "value": "prediction score"}, axis=1)
)

data_8 = (
    pd.concat(
        [
            pd.DataFrame(
                np.concatenate(
                    [
                        prediction_score_cluster_8_orig[:, ap_topics - 1],
                        np.repeat("wt", prediction_score_cluster_8_orig.shape[0])[
                            :, None
                        ],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "condition",
                ],
                index=modifications_per_sequence_8,
            ),
            pd.DataFrame(
                np.concatenate(
                    [
                        prediction_score_cluster_8_modi[:, ap_topics - 1],
                        np.repeat("alt", prediction_score_cluster_8_modi.shape[0])[
                            :, None
                        ],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "condition",
                ],
                index=modifications_per_sequence_8,
            ),
        ]
    )
    .melt(id_vars="condition", ignore_index=False)
    .rename({"variable": "Topic", "value": "prediction score"}, axis=1)
)

data_4["prediction score"] = data_4["prediction score"].astype(float)
data_8["prediction score"] = data_8["prediction score"].astype(float)


hits_no_thr = pd.concat(
    fimo(motifs=motifs, sequences=ohs_topic_31_1.swapaxes(1, 2), threshold=0.5)
)

hits_no_thr["cluster"] = [
    clusters[patterns_to_cluster_names.index(name)]
    for name in tqdm(hits_no_thr["motif_name"])
]

hits_no_thr["-log(p-value)"] = -np.log(hits_no_thr["p-value"] + 1e-6)
max_hits_no_thr_per_seq = (
    hits_no_thr.groupby(["sequence_name", "cluster"])[["score", "-log(p-value)"]]
    .max()
    .reset_index()
)

hits_no_thr["sequence_name"] = [
    selected_regions_topic_31_1[x] for x in hits_no_thr["sequence_name"]
]

y = (
    np.log(region_topic.loc[selected_regions_topic_31_1, "Topic_1"].values + 1e-6)
    - np.log(region_topic.loc[selected_regions_topic_31_1, "Topic_31"].values + 1e-6)
).reshape(-1, 1)

cluster_4_score = (
    max_hits_no_thr_per_seq.set_index("cluster")
    .loc[4]
    .reset_index(drop=True)
    .set_index("sequence_name")
    .iloc[np.arange(len(y))]["-log(p-value)"]
    .values
)
cluster_8_score = (
    max_hits_no_thr_per_seq.set_index("cluster")
    .loc[8]
    .reset_index(drop=True)
    .set_index("sequence_name")
    .iloc[np.arange(len(y))]["-log(p-value)"]
    .values
)

X = np.array([cluster_4_score, cluster_8_score]).T

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
reg.score(X_test, (y_test > 0).ravel())

precision, recall, threshold = precision_recall_curve(
    (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1]
)

fpr, tpr, thresholds = roc_curve((y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1])


from sklearn.metrics import auc

auc(recall, precision)
auc(fpr, tpr)

motif_embedding_results = pickle.load(open("draft/motif_embedding.pkl", "rb"))

n_seq = 200
data_motif_embedding = (
    pd.concat(
        [
            pd.DataFrame(
                np.concatenate(
                    [
                        np.array(
                            [
                                motif_embedding_results[experiment][0][i][
                                    "predictions"
                                ][iter][ap_topics - 1]
                                for i in range(n_seq)
                            ]
                        ),
                        np.repeat(experiment, n_seq)[:, None],
                        np.repeat(iter, n_seq)[:, None],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "condition",
                    "iter",
                ],
            )
            for experiment in motif_embedding_results
            for iter in range(6)
        ]
    )
    .melt(id_vars=["condition", "iter"])
    .rename({"variable": "Topic", "value": "prediction score"}, axis=1)
)


data_motif_embedding["iter"] = data_motif_embedding["iter"].astype(int)
data_motif_embedding["prediction score"] = data_motif_embedding[
    "prediction score"
].astype(float)

random.seed(123)
motif_to_put = "CTAATTAG"
ctaag_rand_prediction_scores = np.empty((n_seq, 6, model.output.shape[1]))
for i in tqdm(range(n_seq)):
    seqs = np.empty((6, 500, 4))
    seqs[0] = one_hot_encode_sequence(
        motif_embedding_results["CTAATTAG_Topic_31_5"][0][i]["initial_sequence"],
        expand_dim=False,
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
        seqs[j][loc : loc + len(motif_to_put)] = one_hot_encode_sequence(
            motif_to_put, expand_dim=False
        )
    ctaag_rand_prediction_scores[i] = model.predict(seqs, verbose=0)

random.seed(123)
motif_to_put = "GGATTAG"
ggattag_rand_prediction_scores = np.empty((n_seq, 6, model.output.shape[1]))
for i in tqdm(range(n_seq)):
    seqs = np.empty((6, 500, 4))
    seqs[0] = one_hot_encode_sequence(
        motif_embedding_results["GGATTAG_Topic_1_5"][0][i]["initial_sequence"],
        expand_dim=False,
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
        seqs[j][loc : loc + len(motif_to_put)] = one_hot_encode_sequence(
            motif_to_put, expand_dim=False
        )
    ggattag_rand_prediction_scores[i] = model.predict(seqs, verbose=0)

data_ctaag = (
    pd.concat(
        [
            pd.DataFrame(
                np.concatenate(
                    [
                        ctaag_rand_prediction_scores[:, i, ap_topics - 1],
                        np.repeat(i, n_seq)[:, None],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "iter",
                ],
            )
            for i in range(6)
        ]
    )
    .melt(id_vars="iter")
    .rename({"variable": "Topic", "value": "prediction score"}, axis=1)
)

data_ggattag = (
    pd.concat(
        [
            pd.DataFrame(
                np.concatenate(
                    [
                        ggattag_rand_prediction_scores[:, i, ap_topics - 1],
                        np.repeat(i, n_seq)[:, None],
                    ],
                    axis=1,
                ),
                columns=[
                    *[
                        model_index_to_topic_name_embryo(x - 1).replace(
                            "progenitor_", ""
                        )
                        for x in ap_topics
                    ],
                    "iter",
                ],
            )
            for i in range(6)
        ]
    )
    .melt(id_vars="iter")
    .rename({"variable": "Topic", "value": "prediction score"}, axis=1)
)


fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=data_ctaag,
    x="iter",
    y="prediction score",
    hue="Topic",
    ax=ax,
    palette="Spectral",
    legend=False,
)
fig.savefig(f"plots/motif_embedding_random_CTAATTAG.pdf")


fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(
    [
        motif_embedding_results["CTAATTAG_Topic_31_5"][0][i]["predictions"][-1]
        for i in range(200)
    ],
    ax=ax,
    xticklabels=True,
    yticklabels=False,
    vmin=0,
    vmax=1,
)
fig.savefig("CTAATAG_all_pred_test.pdf")

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(
    [
        motif_embedding_results["GGATTAG_Topic_1_5"][0][i]["predictions"][-1]
        for i in range(200)
    ],
    ax=ax,
    vmin=0,
    vmax=1,
    xticklabels=True,
    yticklabels=False,
)
fig.savefig("GGATTAG_all_pred_test.pdf")

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(
    [
        motif_embedding_results["SOX_GGATTAG_Topic_1_5"][0][i]["predictions"][-1]
        for i in range(200)
    ],
    ax=ax,
    vmin=0,
    vmax=1,
    xticklabels=True,
    yticklabels=False,
)
fig.savefig("SOX_GGATTAG_all_pred_test.pdf")

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(
    ctaag_rand_prediction_scores[:, -1, :],
    ax=ax,
    vmin=0,
    vmax=0.7,
    xticklabels=True,
    yticklabels=False,
)
fig.savefig("CTAATAG_all_pred_randon_test.pdf")

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(
    ggattag_rand_prediction_scores[:, -1, :],
    ax=ax,
    vmin=0,
    vmax=0.7,
    xticklabels=True,
    yticklabels=False,
)
fig.savefig("GGATTAG_all_pred_randon_test.pdf")


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



exp_embryo = adata_embryo_progenitor.to_df(layer="log_cpm")

avg_expr_embryo_per_topic = {}
for topic in ap_topics:
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

import re

cluster_to_TF = {
    1: "SOX(2|3)",
    4: "(LHX(2|9))|(RAX)|(SIX(3|6))",
    8: "(OTX(1|2))|(EMX2)|(DMBX1)|(LMX1A)",
    7: "(PAX(5|8))|(EN(1|2))",
    6: "(MEIS(1|2))|(PBX(1|3))|(HOXB3)"
}

tf_expr_matrix_per_topic_embryo = (
    pd.DataFrame(avg_expr_embryo_per_topic).T
)

tf_expr_matrix_per_topic_embryo = tf_expr_matrix_per_topic_embryo[
    [
        "SOX2", "SOX3", "LHX2", "LHX9", "RAX", "SIX3", "SIX6",
        "OTX1", "OTX2", "EMX2", "DMBX1", "LMX1A",
        "PAX5", "PAX8", "EN1", "EN2",
        "MEIS1", "MEIS2", "PBX1", "PBX3", "HOXB3"
    ]
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
# Cell topic UMAP
ax_topic_umap_embryo_1 = fig.add_subplot(gs[0:5, 0:5])
ax_topic_umap_embryo_2 = fig.add_subplot(gs[6:11, 0:5])
embryo_cells_both = list(
    set(embryo_progenitor_cell_topic.index) & set(adata_embryo_progenitor.obs_names)
)
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_topic_umap_embryo_1,
    g_cut=0,
    r_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[0] - 1)
    ].values,
    g_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[1] - 1)
    ].values,
    b_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[2] - 1)
    ].values,
    r_name=model_index_to_topic_name_embryo(ap_topics[0] - 1).replace(
        "progenitor_", ""
    ),
    g_name=model_index_to_topic_name_embryo(ap_topics[1] - 1).replace(
        "progenitor_", ""
    ),
    b_name=model_index_to_topic_name_embryo(ap_topics[2] - 1).replace(
        "progenitor_", ""
    ),
)
rgb_scatter_plot(
    x=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 0],
    y=adata_embryo_progenitor[embryo_cells_both].obsm["X_umap"][:, 1],
    ax=ax_topic_umap_embryo_2,
    g_cut=0,
    r_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[3] - 1)
    ].values,
    g_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[4] - 1)
    ].values,
    b_values=embryo_progenitor_cell_topic.loc[
        embryo_cells_both, model_index_to_topic_name_embryo(ap_topics[6] - 1)
    ].values,
    r_name=model_index_to_topic_name_embryo(ap_topics[3] - 1).replace(
        "progenitor_", ""
    ),
    g_name=model_index_to_topic_name_embryo(ap_topics[4] - 1).replace(
        "progenitor_", ""
    ),
    b_name=model_index_to_topic_name_embryo(ap_topics[6] - 1).replace(
        "progenitor_", ""
    ),
)
# Region topic corr
ax_rt_colors = fig.add_subplot(gs[0, 8:18])
sns.heatmap(
    np.array([*np.repeat(1, len(ap_topics)), *np.repeat(2, len(dv_topics))])[None, :],
    cmap="Set1",
    xticklabels=False,
    yticklabels=False,
    ax=ax_rt_colors,
    vmin=1,
    vmax=10,
)
ax_rt_corr = fig.add_subplot(gs[1:10, 8:18], sharex=ax_rt_colors)
sns.heatmap(
    corr_region_topic,
    annot=np.round(corr_region_topic, 1).mask(corr_region_topic < 0.15).fillna(""),
    fmt="",
    cmap="Spectral_r",
    vmin=0,
    vmax=0.6,
    linewidths=1,
    linecolor="black",
    xticklabels=True,
    yticklabels=True,
    cbar_kws=dict(shrink=0.5, ticks=[0, 0.3, 0.6]),
    annot_kws=dict(fontsize=5),
)
# region topic hm
#ax_hm_rt_contrib = fig.add_subplot(gs[0:10, 19:23])
tmp = region_topic.loc[
    selected_regions_ap,
    [
        model_index_to_topic_name_embryo(x - 1).replace("progenitor_", "")
        for x in ap_topics
    ],
]
tmp.columns = [x.replace("Topic_", "") for x in tmp.columns]
ax_hm_rt_pred = fig.add_subplot(gs[0:10, 25 - 6:29 - 6])
sns.heatmap(
    selected_regions_ap_prediction_score[:, ap_topics - 1],
    ax=ax_hm_rt_pred,
    yticklabels=False,
    xticklabels=tmp.columns,
    cbar_kws=dict(
        shrink=0.5,
        format=lambda x, pos: "{:.0e}".format(x),
        ticks=[0, 0.25, 0.5],
    ),
    cmap="binary_r",
    vmin=0,
    vmax=0.5,
)
#
ax_n_seqlets = fig.add_subplot(gs[0:10, 31-6:34-6])
sns.heatmap(
    np.log10(seqlet_count_per_cluster[np.array(clusters_to_show) - 1] + 1),
    ax=ax_n_seqlets,
    yticklabels=False,
    xticklabels=tmp.columns,
    cmap="magma",
    cbar=False,
    robust=True,
    vmax=2.8,
    vmin=0,
)
for i, cluster in enumerate(clusters_to_show):
    ax_pattern = fig.add_subplot(gs[2 * i : 2 * i + 2, 35-6:39-6])
    ppm = patterns_to_cluster[np.where(clusters == cluster)[0].min()].ppm
    ic = patterns_to_cluster[np.where(clusters == cluster)[0].min()].ic()
    ic_range = range(
        *patterns_to_cluster[np.where(clusters == cluster)[0].min()].ic_trim(0.2)
    )
    _ = logomaker.Logo(
        pd.DataFrame((ppm * ic[:, None])[ic_range], columns=["A", "C", "G", "T"]),
        ax=ax_pattern,
    )
    ax_pattern.set_ylim(0, 2)
    ax_pattern.spines[["right", "top", "left", "bottom"]].set_visible(False)
    ax_pattern.set_xticks([])
    ax_pattern.set_yticks([])
#

ax_tf_gex = fig.add_subplot(gs[0:10, 39-3:])
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
    ax=ax_tf_gex,
    xticklabels=True,
    yticklabels=True,
    cbar=False,
    vmin=0,
    vmax=1,
    lw=0.5,
    linecolor="black",
)
ax_mod_box_1 = fig.add_subplot(gs[13:18, 0:9])
ax_mod_box_2 = fig.add_subplot(gs[20:25, 0:9], sharex=ax_mod_box_1)
data = data_4
regions = data.loc[data["prediction score"] >= 0.4].index.drop_duplicates()
sns.boxplot(
    data=data.loc[regions],
    x="condition",
    y="prediction score",
    hue="Topic",
    ax=ax_mod_box_1,
    palette="Spectral",
    legend=False,
    flierprops=dict(markersize=3, marker="."),
)
ax_mod_box_1.grid(True)
ax_mod_box_1.set_title("CTAATTAG -> AGGGATTAG")
data = data_8
regions = data.loc[data["prediction score"] >= 0.4].index.drop_duplicates()
sns.boxplot(
    data=data.loc[regions],
    x="condition",
    y="prediction score",
    hue="Topic",
    ax=ax_mod_box_2,
    palette="Spectral",
    legend=False,
    flierprops=dict(markersize=3, marker="."),
)
ax_mod_box_2.grid(True)
ax_mod_box_2.set_title("CTAATTAG -> AGGGATTAG")
#
ax_ROC = fig.add_subplot(gs[13:18, 11:16])
ax_PR = fig.add_subplot(gs[20:25, 11:16])
_ = ax_ROC.plot(fpr, tpr, color="black")
_ = ax_PR.plot(recall, precision, color="black")
_ = ax_ROC.set_xlabel("FPR")
_ = ax_ROC.set_ylabel("TPR")
_ = ax_PR.set_xlabel("Recall")
_ = ax_PR.set_ylabel("Precision")
for ax in [ax_ROC, ax_PR]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=[0, "", "", "", 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=[0, "", "", "", 1])
#
ax_motif_embedding_1 = fig.add_subplot(gs[13:18, 17:27])
ax_motif_embedding_2 = fig.add_subplot(gs[20:25, 17:27])
for experiment, ax in zip(
    ["CTAATTAG_Topic_31_5", "GGATTAG_Topic_1_5"],
    [ax_motif_embedding_1, ax_motif_embedding_2],
):
    print(experiment)
    sns.boxplot(
        data=data_motif_embedding.query("condition == @experiment"),
        x="iter",
        y="prediction score",
        hue="Topic",
        ax=ax,
        palette="Spectral",
        legend=False,
        flierprops=dict(markersize=3, marker="."),
    )
    ax.grid(True)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
#
ax_motif_embedding_random_1 = fig.add_subplot(gs[13:18, 29:39])
ax_motif_embedding_random_2 = fig.add_subplot(gs[20:25, 29:39])
sns.boxplot(
    data=data_ctaag,
    x="iter",
    y="prediction score",
    hue="Topic",
    ax=ax_motif_embedding_random_1,
    palette="Spectral",
    legend=False,
    flierprops=dict(markersize=3, marker="."),
)
ax_motif_embedding_random_1.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax_motif_embedding_random_1.grid(True)
sns.boxplot(
    data=data_ggattag,
    x="iter",
    y="prediction score",
    hue="Topic",
    ax=ax_motif_embedding_random_2,
    palette="Spectral",
    legend=False,
    flierprops=dict(markersize=3, marker="."),
)
ax_motif_embedding_random_2.grid(True)
ax_motif_embedding_random_2.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("Figure_4.png", transparent=False)
fig.savefig("Figure_4.pdf")
