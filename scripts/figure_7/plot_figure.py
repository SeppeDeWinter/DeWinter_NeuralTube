import os
import matplotlib.pyplot as plt
import matplotlib
import scanpy as sc
import pandas as pd
import numpy as np
from dataclasses import dataclass
import h5py
from typing import Self
from tqdm import tqdm
import pickle
import logomaker
from functools import reduce
import torch
import seaborn as sns
from scipy.stats import binomtest
import json
from matplotlib.lines import Line2D
import pysam
from crested.utils._seq_utils import one_hot_encode_sequence
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy.stats import norm

##########################################################################################
#
#
#                                           FUNCTIONS
#
#
#########################################################################################


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


def calc_ic(ppm, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
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
    selected_patterns,
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
    if "ic_start" not in pattern_metadata.columns:
        pattern_metadata["ic_start"] = 0
    if "ic_stop" not in pattern_metadata.columns:
        pattern_metadata["ic_stop"] = 30
    for cluster_id in selected_patterns:
        cluster_patterns = pattern_metadata.loc[
            pattern_metadata[cluster_col] == cluster_id
        ].index.to_list()
        P = []
        for pattern_name in cluster_patterns:
            if pattern_name not in all_pattern_names:
                continue
            pattern = all_patterns[all_pattern_names.index(pattern_name)]
            ic_start, ic_end, is_rc_to_root, offset_to_root = pattern_metadata.loc[
                pattern_name, ["ic_start", "ic_stop", "is_rc_to_root", "offset_to_root"]
            ]
            pattern_ic = [s.ppm[ic_start:ic_end] for s in pattern.seqlets]
            if is_rc_to_root:
                pattern_ic = [s[::-1, ::-1] for s in pattern_ic]
            if offset_to_root > 0:
                pattern_ic = [
                    np.concatenate([np.zeros((offset_to_root, 4)), s])
                    for s in pattern_ic
                ]
            elif offset_to_root < 0:
                pattern_ic = [s[abs(offset_to_root) :, :] for s in pattern_ic]
            P.extend(pattern_ic)
        max_len = max([p.shape[0] for p in P])
        P = np.array(
            [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in P]
        )
        P += 1e-6
        P = (P.sum(0).T / P.sum(0).sum(1)).T
        yield (cluster_id, P[range(*ic_trim(calc_ic(P), 0.2))])


def scale(X):
    return (X - X.min()) / (X.max() - X.min())


def get_pred_and_l2_for_cell_type(
    result,
    cell_type,
):
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
    ic_trim_thr: float = 0.2,
):
    print("One hot encoding ... ")
    ohs = np.array(
        [
            one_hot_encode_sequence(
                genome.fetch(r.split(":")[0], *map(int, r.split(":")[1].split("-"))),
                expand_dim=False,
            )
            for r in tqdm(region_names)
        ]
    )
    print("loading data")
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
        if (
            n in pattern_metadata.index
            and pattern_metadata.loc[n, cluster_col] in selected_clusters
        )
    }
    print("Scoring hits ...")
    l_hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2), threshold=0.5)
    for i in tqdm(range(len(l_hits)), desc="getting max"):
        l_hits[i]["-logp"] = -np.log10(l_hits[i]["p-value"] + 1e-6)
        l_hits[i] = (
            l_hits[i]
            .groupby(["sequence_name", "motif_name"])[["score", "-logp"]]
            .max()
            .reset_index()
        )
    hits = pd.concat(l_hits)
    hits["sequence_name"] = [
        region_names[x] for x in tqdm(hits["sequence_name"], desc="sequence name")
    ]
    hits["cluster"] = [
        pattern_metadata.loc[m, cluster_col]
        for m in tqdm(hits["motif_name"], desc="cluster")
    ]
    return hits


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
    fg: np.ndarray, bg: np.ndarray, X: np.ndarray, seed: int = 123
) -> ClassificationResult:
    y = (fg - bg).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )
    reg = LogisticRegression().fit(X_train, (y_train > 0).ravel())
    precision, recall, threshold = precision_recall_curve(
        (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1]
    )
    fpr, tpr, thresholds = roc_curve(
        (y_test > 0).ravel(), reg.predict_proba(X_test)[:, 1]
    )
    return ClassificationResult(
        model=reg,
        precision=precision,
        recall=recall,
        fpr=fpr,
        tpr=tpr,
        auc_roc=auc(fpr, tpr),
        auc_pr=auc(recall, precision),
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


window = 10
thr = 0.01

all_hits_organoid = []
for cell_type in cell_type_to_modisco_result:
    if cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap is None:
        continue
    tmp = cell_type_to_modisco_result[cell_type].hits_organoid_non_overlap.copy()
    tmp["cell_type"] = cell_type
    all_hits_organoid.append(tmp)

all_hits_organoid = pd.concat(all_hits_organoid).sort_values("start")

all_hits_organoid["passing_thr"] = abs(all_hits_organoid["attribution"]) >= thr

all_hits_organoid["start_bins"] = pd.cut(
    all_hits_organoid["start"], bins=range(-1, 500, window)
)

n_hits_per_bin_organoid = all_hits_organoid.groupby("start_bins")["passing_thr"].sum()

n_hits_per_bin_mean_organoid = (
    np.sum(
        [
            iv.mid * count
            for iv, count in zip(
                n_hits_per_bin_organoid.index, n_hits_per_bin_organoid.values
            )
        ]
    )
    / n_hits_per_bin_organoid.values.sum()
)

n_hits_per_bin_var_organoid = (
    np.sum(
        [
            count * (iv.mid - n_hits_per_bin_mean_organoid) ** 2
            for iv, count in zip(
                n_hits_per_bin_organoid.index, n_hits_per_bin_organoid.values
            )
        ]
    )
    / n_hits_per_bin_organoid.values.sum()
)

n_hits_per_bin_std_organoid = np.sqrt(n_hits_per_bin_var_organoid)

all_hits_embryo = []
for cell_type in cell_type_to_modisco_result:
    if cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap is None:
        continue
    tmp = cell_type_to_modisco_result[cell_type].hits_embryo_non_overlap.copy()
    tmp["cell_type"] = cell_type
    all_hits_embryo.append(tmp)

all_hits_embryo = pd.concat(all_hits_embryo).sort_values("start")

all_hits_embryo["passing_thr"] = abs(all_hits_embryo["attribution"]) >= thr

all_hits_embryo["start_bins"] = pd.cut(
    all_hits_embryo["start"], bins=range(-1, 500, window)
)

n_hits_per_bin_embryo = all_hits_embryo.groupby("start_bins")["passing_thr"].sum()

n_hits_per_bin_mean_embryo = (
    np.sum(
        [
            iv.mid * count
            for iv, count in zip(
                n_hits_per_bin_embryo.index, n_hits_per_bin_embryo.values
            )
        ]
    )
    / n_hits_per_bin_embryo.values.sum()
)

n_hits_per_bin_var_embryo = (
    np.sum(
        [
            count * (iv.mid - n_hits_per_bin_mean_embryo) ** 2
            for iv, count in zip(
                n_hits_per_bin_embryo.index, n_hits_per_bin_embryo.values
            )
        ]
    )
    / n_hits_per_bin_embryo.values.sum()
)

n_hits_per_bin_std_embryo = np.sqrt(n_hits_per_bin_var_embryo)


##

cell_type_to_pattern_consensus = {}
for cell_type in cell_type_to_modisco_result:
    organoid_topics = cell_type_to_modisco_result[cell_type].organoid_topics
    embryo_topics = cell_type_to_modisco_result[cell_type].embryo_topics
    organoid_pattern_dir = "../data_prep_new/organoid_data/MODELS/modisco/"
    embryo_pattern_dir = "../data_prep_new/embryo_data/MODELS/modisco/"
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
        selected_patterns,
    ):
        cell_type_to_pattern_consensus[cell_type][cluster] = consensus

cell_type_organoid_to_coef = {
    cell_type: pd.read_table(
        f"draft/organoid_{cell_type}_pattern_coef.tsv", index_col=0
    )
    for cell_type in cell_type_to_modisco_result
    if os.path.exists(f"draft/organoid_{cell_type}_pattern_coef.tsv")
}

cell_type_embryo_to_coef = {
    cell_type: pd.read_table(f"draft/embryo_{cell_type}_pattern_coef.tsv", index_col=0)
    for cell_type in cell_type_to_modisco_result
    if os.path.exists(f"draft/embryo_{cell_type}_pattern_coef.tsv")
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
            cell_type in cell_type_organoid_to_coef
            and cell_type_organoid_to_coef[cell_type].loc[pattern].max() > 0
        ) or (
            cell_type in cell_type_embryo_to_coef
            and cell_type_embryo_to_coef[cell_type].loc[pattern].max() > 0
        ):
            cell_type_to_pattern_consensus_filtered[cell_type][pattern] = (
                cell_type_to_pattern_consensus[cell_type][pattern]
            )
            all_patterns.append(cell_type_to_pattern_consensus[cell_type][pattern])
            cell_type_to_pattern_consensus_index[cell_type][pattern] = i
            index_to_cell_type_pattern[i] = (cell_type, pattern)
            i += 1


pattern_metadata = pd.read_table("draft/pattern_metadata.tsv", index_col=0)

all_patterns_names = [
    " ".join(map(str, index_to_cell_type_pattern[i]))
    for i in index_to_cell_type_pattern
]

cluster_to_avg_pattern = {}
for cluster in set(pattern_metadata["cluster_sub_cluster"]):
    cluster_idc = np.where(pattern_metadata["cluster_sub_cluster"] == cluster)[0]
    pwms_aligned = []
    for i, m in enumerate(cluster_idc):
        offset = pattern_metadata.loc[all_patterns_names[m], "offset_to_root"]
        is_rc = pattern_metadata.loc[all_patterns_names[m], "is_rc_to_root"]
        pwm = all_patterns[m]
        ic = calc_ic(all_patterns[m])[:, None]
        if is_rc:
            pwm = pwm[::-1, ::-1]
            ic = ic[::-1, ::-1]
        if offset > 0:
            pwm = np.concatenate([np.zeros((offset, 4)), pwm])
            ic = np.concatenate([np.zeros((offset, 1)), ic])
        elif offset < 0:
            pwm = pwm[abs(offset) :, :]
            ic = ic[abs(offset) :, :]
        pwms_aligned.append(pwm)
    max_len = max([x.shape[0] for x in pwms_aligned])
    pwm_avg = np.array(
        [np.concatenate([p, np.zeros((max_len - p.shape[0], 4))]) for p in pwms_aligned]
    ).mean(0)
    ic = calc_ic(pwm_avg)
    cluster_to_avg_pattern[cluster] = pwm_avg


pattern_code = pd.DataFrame(
    index=pattern_metadata["cluster_sub_cluster"].unique(),
    columns=[
        *[
            "O_" + topic
            for cell_type in cell_type_organoid_to_coef
            for topic in cell_type_organoid_to_coef[cell_type]
        ],
        *[
            "E_" + topic
            for cell_type in cell_type_embryo_to_coef
            for topic in cell_type_embryo_to_coef[cell_type]
        ],
    ],
).fillna(0)

for cell_type in cell_type_organoid_to_coef:
    for topic in cell_type_organoid_to_coef[cell_type].columns:
        patterns = (
            cell_type_organoid_to_coef[cell_type]
            .loc[cell_type_organoid_to_coef[cell_type][topic] > 0]
            .index.to_list()
        )
        if len(patterns) == 2:
            patterns = [*patterns, *patterns]
        elif len(patterns) == 1:
            patterns = [*patterns, *patterns, *patterns, *patterns]
        for pattern in patterns:
            pattern_code.loc[
                pattern_metadata.loc[
                    all_patterns_names[
                        cell_type_to_pattern_consensus_index[cell_type][pattern]
                    ],
                    "cluster_sub_cluster",
                ],
                "O_" + topic,
            ] += 1

for cell_type in cell_type_embryo_to_coef:
    for topic in cell_type_embryo_to_coef[cell_type].columns:
        patterns = (
            cell_type_embryo_to_coef[cell_type]
            .loc[cell_type_embryo_to_coef[cell_type][topic] > 0]
            .index.to_list()
        )
        if len(patterns) == 2:
            patterns = [*patterns, *patterns]
        elif len(patterns) == 1:
            patterns = [*patterns, *patterns, *patterns, *patterns]
        for pattern in patterns:
            pattern_code.loc[
                pattern_metadata.loc[
                    all_patterns_names[
                        cell_type_to_pattern_consensus_index[cell_type][pattern]
                    ],
                    "cluster_sub_cluster",
                ],
                "E_" + topic,
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

sorted_patterns = (
    pattern_code.T.idxmax()
    .sort_values(key=lambda X: [topic_order.index(x) for x in X])
    .index[::-1]
)

results_per_experiment_organoid = pickle.load(
    open("draft/motif_embedding_organoid.pkl", "rb")
)

results_per_experiment_embryo = pickle.load(
    open("draft/motif_embedding_embryo.pkl", "rb")
)

cell_type_to_pred_l2_organoid = {
    "O_" + cell_type.split("_")[-1]: get_pred_and_l2_for_cell_type(
        results_per_experiment_organoid, cell_type
    )
    for cell_type in results_per_experiment_organoid
}

cell_type_to_pred_l2_embryo = {
    "E_" + cell_type.split("_")[-1]: get_pred_and_l2_for_cell_type(
        results_per_experiment_embryo, cell_type
    )
    for cell_type in results_per_experiment_embryo
}

color_dict = json.load(open("../color_maps.json"))["cell_type_classes"]

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
    "E_29": "neuron",
}


max_hits_per_seq_progenitor_organoid = pd.read_table(
    "draft/max_hits_per_seq_progenitor_organoid.tsv",
)
max_hits_per_seq_neural_crest_organoid = pd.read_table(
    "draft/max_hits_per_seq_neural_crest_organoid.tsv",
)
max_hits_per_seq_neuron_organoid = pd.read_table(
    "draft/max_hits_per_seq_neuron_organoid.tsv",
)
max_hits_per_seq_progenitor_dv_embryo = pd.read_table(
    "draft/max_hits_per_seq_progenitor_dv_embryo.tsv",
)
max_hits_per_seq_progenitor_ap_embryo = pd.read_table(
    "draft/max_hits_per_seq_progenitor_ap_embryo.tsv",
)
max_hits_per_seq_neural_crest_embryo = pd.read_table(
    "draft/max_hits_per_seq_neural_crest_embryo.tsv",
)
max_hits_per_seq_neuron_embryo = pd.read_table(
    "draft/max_hits_per_seq_neuron_embryo.tsv",
)

progenitor_region_topic_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
    index_col=0,
)

selected_regions_progenitor_organoid = list(
    set.union(
        *[
            set(
                progenitor_region_topic_organoid[f"Topic{topic - 25}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["progenitor_dv"].organoid_topics
        ]
    )
)

selected_regions_progenitor_organoid = [
    r for r in selected_regions_progenitor_organoid if r.startswith("chr")
]

neural_crest_region_topic_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
    index_col=0,
)

selected_regions_neural_crest_organoid = list(
    set.union(
        *[
            set(
                neural_crest_region_topic_organoid[f"Topic{topic - 55}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["neural_crest"].organoid_topics
        ]
    )
)

selected_regions_neural_crest_organoid = [
    r for r in selected_regions_neural_crest_organoid if r.startswith("chr")
]

neuron_region_topic_organoid = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv", index_col=0
)

selected_regions_neuron_organoid = list(
    set.union(
        *[
            set(
                neuron_region_topic_organoid[f"Topic{topic}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["neuron"].organoid_topics
        ]
    )
)

selected_regions_neuron_organoid = [
    r for r in selected_regions_neuron_organoid if r.startswith("chr")
]

progenitor_region_topic_embryo = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv", index_col=0
)

selected_regions_progenitor_dv_embryo = list(
    set.union(
        *[
            set(
                progenitor_region_topic_embryo[f"Topic_{topic - 30}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["progenitor_dv"].embryo_topics
        ]
    )
)

selected_regions_progenitor_ap_embryo = list(
    set.union(
        *[
            set(
                progenitor_region_topic_embryo[f"Topic_{topic - 30}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["progenitor_ap"].embryo_topics
        ]
    )
)

neural_crest_region_topic_embryo = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/neural_crest_region_topic_contrib.tsv",
    index_col=0,
)

selected_regions_neural_crest_embryo = list(
    set.union(
        *[
            set(
                neural_crest_region_topic_embryo[f"Topic_{topic - 90}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["neural_crest"].embryo_topics
        ]
    )
)

selected_regions_neural_crest_embryo = [
    r for r in selected_regions_neural_crest_embryo if r.startswith("chr")
]

neuron_region_topic_embryo = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/neuron_region_topic_contrib.tsv", index_col=0
)

selected_regions_neuron_embryo = list(
    set.union(
        *[
            set(
                neuron_region_topic_embryo[f"Topic_{topic}"]
                .sort_values(ascending=False)
                .head(1_000)
                .index
            )
            for topic in cell_type_to_modisco_result["neuron"].embryo_topics
        ]
    )
)

selected_regions_neuron_embryo = [
    r for r in selected_regions_neuron_embryo if r.startswith("chr")
]


cell_type_to_hits_offset_region_topic = {
    "organoid_progenitor_dv": (
        max_hits_per_seq_progenitor_organoid,
        25,
        progenitor_region_topic_organoid,
        selected_regions_progenitor_organoid,
    ),
    "organoid_neural_crest": (
        max_hits_per_seq_neural_crest_organoid,
        55,
        neural_crest_region_topic_organoid,
        selected_regions_neural_crest_organoid,
    ),
    "organoid_neuron": (
        max_hits_per_seq_neuron_organoid,
        0,
        neuron_region_topic_organoid,
        selected_regions_neuron_organoid,
    ),
    "embryo_progenitor_dv": (
        max_hits_per_seq_progenitor_dv_embryo,
        30,
        progenitor_region_topic_embryo,
        selected_regions_progenitor_dv_embryo,
    ),
    "embryo_progenitor_ap": (
        max_hits_per_seq_progenitor_ap_embryo,
        30,
        progenitor_region_topic_embryo,
        selected_regions_progenitor_ap_embryo,
    ),
    "embryo_neural_crest": (
        max_hits_per_seq_neural_crest_embryo,
        90,
        neural_crest_region_topic_embryo,
        selected_regions_neural_crest_embryo,
    ),
    "embryo_neuron": (
        max_hits_per_seq_neuron_embryo,
        0,
        neuron_region_topic_embryo,
        selected_regions_neuron_embryo,
    ),
}

cell_type_to_classification_result = {}
for cell_type in cell_type_to_hits_offset_region_topic:
    print(cell_type)
    max_hits, t_offset, region_topic, selected_regions = (
        cell_type_to_hits_offset_region_topic[cell_type]
    )
    topics = (
        cell_type_to_modisco_result[cell_type.split("_", 1)[1]].organoid_topics
        if cell_type.split("_")[0] == "organoid"
        else cell_type_to_modisco_result[cell_type.split("_", 1)[1]].embryo_topics
    )
    X = max_hits.pivot(index="sequence_name", columns=["cluster"], values="-logp").loc[
        selected_regions
    ]
    feature_names = list(X.columns)
    X = X.to_numpy()
    topic_to_classification_result = {}
    for topic in topics:
        print(topic)
        if cell_type.split("_")[0] == "organoid":
            fg = set([f"Topic{topic - t_offset}"])
            bg = set(region_topic.columns) - fg
        else:
            fg = set([f"Topic_{topic - t_offset}"])
            bg = set(region_topic.columns) - fg
        fg_y = np.log(region_topic.loc[selected_regions, list(fg)].max(1).values + 1e-6)
        bg_y = np.log(region_topic.loc[selected_regions, list(bg)].values.max(1) + 1e-6)
        topic_to_classification_result[topic] = get_classification_results(
            fg=fg_y,
            bg=bg_y,
            X=X,
        )
    cell_type_to_classification_result[cell_type] = (
        topic_to_classification_result,
        feature_names,
    )


organoid_topics = [int(x.split("_")[1]) for x in topic_order if x.startswith("O_")]

confusion_organoid = np.array(
    [
        np.mean(
            [
                results_per_experiment_organoid[cell_type][0][seq]["predictions"][-1][
                    [t - 1 for t in organoid_topics]
                ]
                for seq in range(200)
            ],
            axis=0,
        )
        for cell_type in results_per_experiment_organoid
    ]
)

df_confusion_organoid = pd.DataFrame(
    confusion_organoid,
    index=[
        int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_organoid
    ],
    columns=organoid_topics,
).loc[organoid_topics]

embryo_topics = [int(x.split("_")[1]) for x in topic_order if x.startswith("E_")]

confusion_embryo = np.array(
    [
        np.mean(
            [
                results_per_experiment_embryo[cell_type][0][seq]["predictions"][-1][
                    [t - 1 for t in embryo_topics]
                ]
                for seq in range(200)
            ],
            axis=0,
        )
        for cell_type in results_per_experiment_embryo
    ]
)

df_confusion_embryo = pd.DataFrame(
    confusion_embryo,
    index=[
        int(cell_type.split("_")[-1]) for cell_type in results_per_experiment_embryo
    ],
    columns=embryo_topics,
).loc[embryo_topics]


N_PIXELS_PER_GRID = 25

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
ax = fig.add_subplot(gs[0:4, 8:78])
for x, topic in enumerate(topic_order):
    auc_pr = cell_type_to_classification_result[
        {"O": "organoid_", "E": "embryo_"}[topic.split("_")[0]] + topic_to_ct[topic]
    ][0][int(topic.split("_")[1])].auc_pr
    auc_roc = cell_type_to_classification_result[
        {"O": "organoid_", "E": "embryo_"}[topic.split("_")[0]] + topic_to_ct[topic]
    ][0][int(topic.split("_")[1])].auc_roc
    _ = ax.scatter(x, auc_pr, color=color_dict[topic_to_ct[topic]], s=20)
    _ = ax.scatter(
        x, auc_roc, color=color_dict[topic_to_ct[topic]], s=20, edgecolor="black", lw=1
    )
_ = ax.grid(True)
_ = ax.set_axisbelow(True)
_ = ax.set_ylim(0, 1)
_ = ax.set_xlim(-0.5, x + 0.5)
_ = ax.set_xticks(
    np.arange(pattern_code.shape[1]),
    labels=["" for _ in range(pattern_code.shape[1])],
    rotation=90,
)
_ = ax.set_yticks(np.arange(0, 1.25, 0.25), labels=[0.0, "", 0.5, "", 1.0])
_ = ax.set_ylabel("$AUC_{PR/ROC}$")
current_y = 5
for i, pattern_name in enumerate(sorted_patterns[::-1]):
    ax = fig.add_subplot(gs[current_y + 3 * i : current_y + 3 * i + 3, 0:6])
    pattern = cluster_to_avg_pattern[pattern_name]
    _ = logomaker.Logo(
        pd.DataFrame(
            pattern * calc_ic(pattern)[:, None],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.set_axis_off()
ax = fig.add_subplot(gs[current_y : current_y + len(sorted_patterns) * 3, 8:78])
XX, YY = np.meshgrid(np.arange(pattern_code.shape[1]), np.arange(pattern_code.shape[0]))
for j in range(pattern_code.shape[1]):
    idx = np.where(pattern_code.loc[sorted_patterns, topic_order[j]].values)[0]
    _ = ax.scatter(
        XX[idx, j],
        YY[idx, j],
        s=pattern_code.loc[sorted_patterns, topic_order[j]].values[idx] * 20,
        color=color_dict[topic_to_ct[topic_order[j]]],
        zorder=3,
        edgecolor="black",
        lw=1,
    )
    _ = ax.plot(
        [XX[idx[YY[idx, j].argmin()], j], XX[idx[YY[idx, j].argmax()], j]],
        [YY[idx, j].min(), YY[idx, j].max()],
        color=color_dict[topic_to_ct[topic_order[j]]],
        zorder=2,
    )
ax.legend(
    handles=[
        Line2D(
            [0], [0], color=color_dict[ct], markerfacecolor="o", label=ct, markersize=10
        )
        for ct in color_dict
    ],
    loc="lower left",
    fontsize=8,
)
_ = ax.set_xlim(-0.5, XX.max() + 0.5)
_ = ax.set_xticks(
    np.arange(pattern_code.shape[1]),
    labels=["" for _ in range(pattern_code.shape[1])],
    rotation=90,
)
_ = ax.set_yticks(
    np.arange(pattern_code.shape[0]), labels=["" for _ in range(pattern_code.shape[0])]
)
ax.grid(True)
ax.set_axisbelow(True)
ax = fig.add_subplot(
    gs[
        current_y + len(sorted_patterns) * 3 + 1 : current_y
        + len(sorted_patterns) * 3
        + 6,
        8:78,
    ]
)
bplots = ax.boxplot(
    [
        {**cell_type_to_pred_l2_organoid, **cell_type_to_pred_l2_embryo}[t][0]
        for t in topic_order
    ],
    patch_artist=True,
    medianprops=dict(color="black"),
    flierprops=dict(markersize=1),
)
for bplot_patch, topic in zip(bplots["boxes"], topic_order):
    color = color_dict[topic_to_ct[topic]]
    bplot_patch.set_facecolor(color)
_ = ax.set_xticks(np.arange(pattern_code.shape[1]) + 1, labels=topic_order, rotation=90)
_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.arange(0, 1.25, 0.25), labels=[0, "", 0.5, "", 1])
ax.grid(True)
ax.set_axisbelow(True)
ax.set_ylabel("pred. score")
current_y = current_y + len(sorted_patterns) * 3 + 11
ax = fig.add_subplot(gs[current_y:, 0 : nrows - current_y])
sns.heatmap(
    df_confusion_organoid,
    xticklabels=True,
    yticklabels=True,
    cmap="viridis",
    linewidths=0.5,
    linecolor="black",
    cbar=False,
    vmin=0,
    vmax=1,
)
ax = fig.add_subplot(
    gs[current_y:, nrows - current_y + 3 : (nrows - current_y) * 2 + 6]
)
sns.heatmap(
    df_confusion_embryo,
    xticklabels=True,
    yticklabels=True,
    cmap="viridis",
    linewidths=0.5,
    linecolor="black",
    vmin=0,
    vmax=1,
    cbar_kws=dict(shrink=0.5),
)
ax = fig.add_subplot(gs[current_y : current_y + 11, 44 : 44 + 15])
ax.scatter(
    all_hits_organoid.start,
    all_hits_organoid.attribution,
    c=all_hits_organoid["-logp"],
    s=1,
)
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(0, 550, 50), labels=["" for _ in np.arange(0, 550, 50)])
_ = ax.set_ylabel("Contribution")
ax = fig.add_subplot(gs[current_y + 12 :, 44 : 44 + 15])
ax_pdf = ax.twinx()
mean_organoid = n_hits_per_bin_mean_organoid
std_organoid = n_hits_per_bin_std_organoid
ax.bar(
    x=[x.left for x in n_hits_per_bin_organoid.index],
    height=n_hits_per_bin_organoid.values,
    width=window,
    color=[
        "white"
        if iv.mid > mean_organoid + std_organoid * 2
        or iv.mid < mean_organoid - std_organoid * 2
        else "darkgray"
        if iv.mid > mean_organoid + std_organoid * 1
        or iv.mid < mean_organoid - std_organoid * 1
        else "dimgray"
        for iv in n_hits_per_bin_organoid.index
    ],
    edgecolor=[
        "lightgray"
        if iv.mid > mean_organoid + std_organoid * 2
        or iv.mid < mean_organoid - std_organoid * 2
        else "gray"
        if iv.mid > mean_organoid + std_organoid * 1
        or iv.mid < mean_organoid - std_organoid * 1
        else "black"
        for iv in n_hits_per_bin_organoid.index
    ],
    lw=1,
)
ax_pdf.plot(
    [x.left for x in n_hits_per_bin_organoid.index],
    norm(loc=mean_organoid, scale=std_organoid).pdf(
        [x.left for x in n_hits_per_bin_organoid.index]
    ),
    color="black",
)
ax_pdf.text(
    0.05,
    0.5,
    s=f"μ = {int(mean_organoid)}\nσ = {int(std_organoid)}",
    transform=ax_pdf.transAxes,
)
ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1])
ax_pdf.set_yticks([])
_ = ax.set_xticks(
    np.arange(0, 550, 50), ["", 50, "", 150, "", 250, "", 350, "", 450, ""]
)
_ = ax.set_xlabel("Start position")
ax.grid(True)
ax.set_axisbelow(True)
ax = fig.add_subplot(gs[current_y : current_y + 11, 63 : 63 + 15])
ax.scatter(
    all_hits_embryo.start, all_hits_embryo.attribution, c=all_hits_embryo["-logp"], s=1
)
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xticks(np.arange(0, 550, 50), labels=["" for _ in np.arange(0, 550, 50)])
# _ = ax.set_ylabel("Contribution")
ax = fig.add_subplot(gs[current_y + 12 :, 63 : 63 + 15])
ax_pdf = ax.twinx()
mean_embryo = n_hits_per_bin_mean_embryo
std_embryo = n_hits_per_bin_std_embryo
ax.bar(
    x=[x.left for x in n_hits_per_bin_embryo.index],
    height=n_hits_per_bin_embryo.values,
    width=window,
    color=[
        "white"
        if iv.mid > mean_embryo + std_embryo * 2
        or iv.mid < mean_embryo - std_embryo * 2
        else "darkgray"
        if iv.mid > mean_embryo + std_embryo * 1
        or iv.mid < mean_embryo - std_embryo * 1
        else "dimgray"
        for iv in n_hits_per_bin_embryo.index
    ],
    edgecolor=[
        "lightgray"
        if iv.mid > mean_embryo + std_embryo * 2
        or iv.mid < mean_embryo - std_embryo * 2
        else "gray"
        if iv.mid > mean_embryo + std_embryo * 1
        or iv.mid < mean_embryo - std_embryo * 1
        else "black"
        for iv in n_hits_per_bin_embryo.index
    ],
    lw=1,
)
ax_pdf.plot(
    [x.left for x in n_hits_per_bin_embryo.index],
    norm(loc=mean_embryo, scale=std_embryo).pdf(
        [x.left for x in n_hits_per_bin_embryo.index]
    ),
    color="black",
)
ax_pdf.text(
    0.05,
    0.5,
    s=f"μ = {int(mean_embryo)}\nσ = {int(std_embryo)}",
    transform=ax_pdf.transAxes,
)
ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1])
ax_pdf.set_yticks([])
_ = ax.set_xticks(
    np.arange(0, 550, 50), ["", 50, "", 150, "", 250, "", 350, "", 450, ""]
)
_ = ax.set_xlabel("Start position")
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Figure_7.png", transparent=False)
fig.savefig("Figure_7.pdf")
