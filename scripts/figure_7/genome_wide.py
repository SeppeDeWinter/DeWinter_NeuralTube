import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
import h5py
from typing import Self
from tqdm import tqdm
import pickle
from functools import reduce
import torch
import json
import pysam
from crested.utils._seq_utils import one_hot_encode_sequence
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tangermeme.tools.fimo import fimo
from tangermeme.utils import extract_signal
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib


##########################################################################################
#
#
#                                           FUNCTIONS
#
#
#########################################################################################

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
    organoid_pattern_dir,
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
    organoid_topics: list[int],
    embryo_topics: list[int],
    selected_clusters: list[float],
    organoid_pattern_dir: str,
    embryo_pattern_dir: str,
    pattern_metadata_path: str,
    cluster_col: str,
    ic_trim_thr: float = 0.2,
    fimo_threshold: float = 0.5,
    region_names: list[str] | None = None,
    genome: pysam.FastaFile | None = None,
    ohs: np.ndarray | None = None
):
    if ohs is None:
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
    l_hits = fimo(motifs=motifs, sequences=ohs.swapaxes(1, 2), threshold=fimo_threshold)
    for i in tqdm(range(len(l_hits)), desc="getting max"):
        l_hits[i]["-logp"] = -np.log10(l_hits[i]["p-value"] + 1e-6)
        l_hits[i] = (
            l_hits[i]
            .groupby(["sequence_name", "motif_name"])[["score", "-logp"]]
            .max()
            .reset_index()
        )
    hits = pd.concat(l_hits)
    if region_names is not None:
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
    selected_regions = list(set(selected_regions) & set(max_hits["sequence_name"]))
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

import pickle

nonAugmented_data_dict_organoid = pickle.load(open("../data_prep_new/organoid_data/MODELS/nonAugmented_data_dict.pkl", "rb"))
data_set = "organoid"
cell_type_to_genome_wide_classification = {}
for cell_type in cell_type_to_modisco_result:
    print(cell_type)
    cell_type_to_genome_wide_classification[f"{data_set}_{cell_type}"] = {}
    hits = get_hit_score_for_topics(
        ohs = nonAugmented_data_dict_organoid["test_data"],
        embryo_pattern_dir=embryo_dl_motif_dir ,
        organoid_pattern_dir=organoid_dl_motif_dir ,
        ic_trim_thr=0.2,
        fimo_threshold = 0.0001,
        embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
        organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
        selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
        pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
        cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
    )
    max_hits_per_seq = hits \
        .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()
    if f"{data_set}_{cell_type}" in cell_type_to_hits_offset_region_topic:
        X = max_hits_per_seq  \
            .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
            .loc[:, cell_type_to_classification_result[f"{data_set}_{cell_type}"][1]] \
            .fillna(0)
        topics = cell_type_to_modisco_result[cell_type].organoid_topics if data_set == "organoid" \
            else cell_type_to_modisco_result[cell_type].embryo_topics
        for topic in topics:
            y = nonAugmented_data_dict_organoid["y_test"][X.index][:, topic - 1]
            reg = cell_type_to_classification_result[f"{data_set}_{cell_type}"][0][topic].model
            precision, recall, threshold = precision_recall_curve(
                y, reg.predict_log_proba(X.values)[:, 1]
            )
            fpr, tpr, thresholds = roc_curve(
                y, reg.predict_log_proba(X.values)[:, 1]
            )
            cell_type_to_genome_wide_classification[f"{data_set}_{cell_type}"][topic] = ClassificationResult(
                model=reg,
                precision=precision,
                recall=recall,
                fpr=fpr,
                tpr=tpr,
                auc_roc=auc(fpr, tpr),
                auc_pr=auc(recall, precision),
            )
            fig, axs = plt.subplots(ncols = 2, figsize = (8,4))
            axs[0].plot(
                fpr, tpr
            )
            axs[1].plot(
                recall, precision
            )
            fig.tight_layout()
            fig.savefig(f"genome_wide_scoring/{data_set}_{cell_type}_{topic}.png")
            plt.close(fig)

import gzip

nonAugmented_data_dict_embryo = pickle.load(gzip.open("../data_prep_new/embryo_data/MODELS/nonAugmented_data_dict.pkl.gz", "rb"))
data_set = "embryo"
for cell_type in cell_type_to_modisco_result:
    print(cell_type)
    cell_type_to_genome_wide_classification[f"{data_set}_{cell_type}"] = {}
    hits = get_hit_score_for_topics(
        ohs = nonAugmented_data_dict_embryo["test_data"],
        embryo_pattern_dir=embryo_dl_motif_dir ,
        organoid_pattern_dir=organoid_dl_motif_dir ,
        ic_trim_thr=0.2,
        fimo_threshold = 0.0001,
        embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
        organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
        selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
        pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
        cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
    )
    max_hits_per_seq = hits \
        .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()
    if f"{data_set}_{cell_type}" in cell_type_to_hits_offset_region_topic:
        X = max_hits_per_seq  \
            .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp") \
            .loc[:, cell_type_to_classification_result[f"{data_set}_{cell_type}"][1]] \
            .fillna(0)
        topics = cell_type_to_modisco_result[cell_type].organoid_topics if data_set == "organoid" \
            else cell_type_to_modisco_result[cell_type].embryo_topics
        for topic in topics:
            y = nonAugmented_data_dict_embryo["y_test"][X.index][:, topic - 1]
            reg = cell_type_to_classification_result[f"{data_set}_{cell_type}"][0][topic].model
            precision, recall, threshold = precision_recall_curve(
                y, reg.predict_log_proba(X.values)[:, 1]
            )
            fpr, tpr, thresholds = roc_curve(
                y, reg.predict_log_proba(X.values)[:, 1]
            )
            cell_type_to_genome_wide_classification[f"{data_set}_{cell_type}"][topic] = ClassificationResult(
                model=reg,
                precision=precision,
                recall=recall,
                fpr=fpr,
                tpr=tpr,
                auc_roc=auc(fpr, tpr),
                auc_pr=auc(recall, precision),
            )
            fig, axs = plt.subplots(ncols = 2, figsize = (8,4))
            axs[0].plot(
                fpr, tpr
            )
            axs[1].plot(
                recall, precision
            )
            fig.tight_layout()
            fig.savefig(f"genome_wide_scoring/{data_set}_{cell_type}_{topic}.png")
            plt.close(fig)


cell_type = "progenitor_dv"
hits = get_hit_score_for_topics(
    ohs = nonAugmented_data_dict_organoid["train_data"],
    embryo_pattern_dir=embryo_dl_motif_dir ,
    organoid_pattern_dir=organoid_dl_motif_dir ,
    ic_trim_thr=0.2,
    fimo_threshold = 0.0001,
    embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
    organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
    selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
    pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
    cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
)

topic = cell_type_to_modisco_result[cell_type].organoid_topics[0]

max_hits_per_seq = hits \
    .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()

X = max_hits_per_seq  \
    .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp").fillna(0)


hits_test = get_hit_score_for_topics(
    ohs = nonAugmented_data_dict_organoid["test_data"],
    embryo_pattern_dir=embryo_dl_motif_dir ,
    organoid_pattern_dir=organoid_dl_motif_dir ,
    ic_trim_thr=0.2,
    fimo_threshold = 0.0001,
    embryo_topics=cell_type_to_modisco_result[cell_type].embryo_topics,
    organoid_topics=cell_type_to_modisco_result[cell_type].organoid_topics,
    selected_clusters=cell_type_to_modisco_result[cell_type].selected_patterns,
    pattern_metadata_path=cell_type_to_modisco_result[cell_type].pattern_metadata_path,
    cluster_col=cell_type_to_modisco_result[cell_type].cluster_col,
)

max_hits_per_seq_test = hits_test \
    .groupby(["sequence_name", "cluster"])[["score", "-logp"]].max().reset_index()

X_test = max_hits_per_seq_test  \
    .pivot(index = "sequence_name", columns = ["cluster"], values = "-logp").fillna(0)

y = nonAugmented_data_dict_organoid["y_train"][X.index][:, topic - 1]
y_test =nonAugmented_data_dict_organoid["y_test"][X_test.index][:, topic - 1]

reg = LogisticRegression()

reg.fit(X, y)

precision, recall, threshold = precision_recall_curve(
    y_test, reg.predict_log_proba(X_test.values)[:, 1]
)
fpr, tpr, thresholds = roc_curve(
    y_test, reg.predict_log_proba(X_test.values)[:, 1]
)
fig, axs = plt.subplots(ncols = 2, figsize = (8,4))
axs[0].plot(
    fpr, tpr
)
axs[1].plot(
    recall, precision
)
fig.tight_layout()
fig.savefig(f"genome_wide_scoring/refit_{data_set}_{cell_type}_{topic}.png")

pickle.dump(
    cell_type_to_genome_wide_classification,
    open("cell_type_to_genome_wide_classification.pkl", "wb")
)

cell_type_to_genome_wide_classification = pickle.load(
    open("cell_type_to_genome_wide_classification.pkl", "rb")
)


matplotlib.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots()
X = [
    cell_type_to_classification_result[cell_type][0][topic].auc_pr 
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
Y = [
    cell_type_to_genome_wide_classification[cell_type][topic].auc_pr 
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
cell_types = [
    ("E_" if "embryo" in cell_type else "O_") + str(topic)
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
ax.scatter(
    x = X,
    y = Y,
    color = [
        color_dict[cell_type.split("_", 1)[1]] for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]
    ],
    lw = [1 if "organoid" in cell_type else 0 for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]],
    edgecolor = "black"
)
texts = []
for x, y, s in zip(X, Y, cell_types):
    texts.append(
        ax.text(
            x, y, s
        )
    )
_ = adjust_text(
    texts,
    arrowprops = dict(arrowstyle = "-", color = "black")
)
ax.set_xlabel("Contrast based aucPR")
ax.set_ylabel("Genome wide aucPR")
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("genome_wide_scoring/aucPR_contrast_v_genome_wide.pdf")

fig, ax = plt.subplots()
X = [
    cell_type_to_classification_result[cell_type][0][topic].auc_roc
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
Y = [
    cell_type_to_genome_wide_classification[cell_type][topic].auc_roc
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
cell_types = [
    ("E_" if "embryo" in cell_type else "O_") + str(topic)
    for cell_type in cell_type_to_classification_result 
    for topic in cell_type_to_classification_result[cell_type][0]
]
ax.scatter(
    x = X,
    y = Y,
    color = [
        color_dict[cell_type.split("_", 1)[1]] for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]
    ],
    lw = [1 if "organoid" in cell_type else 0 for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]],
    edgecolor = "black"
)
texts = []
for x, y, s in zip(X, Y, cell_types):
    texts.append(
        ax.text(
            x, y, s
        )
    )
_ = adjust_text(
    texts,
    arrowprops = dict(arrowstyle = "-", color = "black")
)
ax.set_xlabel("Contrast based aucROC")
ax.set_ylabel("Genome wide aucROC")
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("genome_wide_scoring/aucROC_contrast_v_genome_wide.pdf")




fig, ax = plt.subplots()
ax.scatter(
    x = [cell_type_to_classification_result[cell_type][0][topic].auc_roc for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]],
    y = [cell_type_to_genome_wide_classification[cell_type][topic].auc_roc for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]],
    color = [
        color_dict[cell_type.split("_", 1)[1]] for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]
    ],
    lw = [1 if "organoid" in cell_type else 0 for cell_type in cell_type_to_classification_result for topic in cell_type_to_classification_result[cell_type][0]],
    edgecolor = "black"
)
ax.set_xlabel("Contrast based aucROC")
ax.set_ylabel("Genome wide aucROC")
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("genome_wide_scoring/aucROC_contrast_v_genome_wide.pdf")



