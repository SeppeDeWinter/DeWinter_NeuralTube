# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "matplotlib",
#   "scanpy",
#   "numpy",
#   "pandas",
#   "tqdm",
#   "tangermeme",
#   "torch",
#   "scipy",
#   "harmonypy",
#   "pycistarget @ git+https://github.com/aertslab/pycistarget",
#   "setuptools",
#   "tables",
#   "h5py",
#   "igraph",
#   "leidenalg",
#   "crested",
#   "scikit-learn",
#   "tensorflow[and-cuda]"
# ]
# ///

print("importing modules...")
from matplotlib import axes as mpl_axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
import crested
import pathlib
import os
from tqdm import tqdm
from functools import cache
import sklearn
import json
from pycistarget.input_output import read_hdf5
from tangermeme.tools.tomtom import tomtom
import logomaker
import h5py
import modiscolite
import scanpy as sc
import torch
from crested.utils._seq_utils import one_hot_encode_sequence
from crested.tl._explainer_tf import Explainer
import tensorflow as tf
import requests
from PIL import Image
import io

sc.settings.figdir = os.path.join(os.getcwd(), "tmp_figs")


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


def get_sequence_and_metadata(
    vista_experiment: pd.Series,
    organism_to_genome: dict[str, crested.Genome],
    target_size: int,
    debug=False,
):
    experiment_id = vista_experiment["exp_hier"]
    sequence = vista_experiment["seq"]
    organism = vista_experiment["organism"]
    genomic_coordinate = vista_experiment["coord"]
    strand = vista_experiment["strand"]
    if not isinstance(sequence, str):
        if organism == "Human":
            sequence = vista_experiment["seq_hg38"]
        elif organism == "Mouse":
            sequence = vista_experiment["seq_mm10"]
        else:
            yield None
    if len(sequence) == target_size:
        yield experiment_id, sequence, vista_experiment
    elif len(sequence) < target_size:
        if organism not in organism_to_genome:
            yield None
        chrom, start, end = genomic_coordinate.replace(":", "-").split("-")
        start = int(start)
        end = int(end)
        upstream_to_add = downstream_to_add = (target_size - (end - start)) // 2
        if (target_size - (end - start)) % 2 != 0:
            upstream_to_add += 1
        new_start = int(start) - upstream_to_add
        new_end = int(end) + downstream_to_add
        slopped_seq = organism_to_genome[organism].fetch(
            chrom, new_start, new_end, strand
        )
        if len(slopped_seq) != target_size:
            raise RuntimeError(f"{len(slopped_seq)}")
        idx_seq = 0
        _max = 0
        for i in range(target_size - len(sequence)):
            n_nuc_overlap = sum(
                [a == b for a, b in zip(sequence, slopped_seq[i : i + len(sequence)])]
            )
            if n_nuc_overlap > _max:
                _max = n_nuc_overlap
                idx_seq = i
        if debug:
            print("\033[0;32m" + slopped_seq)
            print("\033[0;31m" + "-" * idx_seq + sequence)
        slopped_seq = list(slopped_seq)
        for i, nuc in enumerate(sequence):
            slopped_seq[i + idx_seq] = nuc
        slopped_seq = "".join(slopped_seq)
        if debug:
            print("\033[0;34m" + slopped_seq + "\033[0m")
            input()
        yield experiment_id, slopped_seq, vista_experiment
    elif len(sequence) > target_size:
        for i in range(len(sequence) - target_size):
            yield experiment_id, sequence[i : i + target_size], vista_experiment


class Grouper:
    def __init__(self, ids: list[str]):
        self._groups: set[str] = set()
        self._grp_to_idx: dict[str, list[int]] = dict()
        for i, _id in enumerate(ids):
            if _id not in self._grp_to_idx:
                self._grp_to_idx[_id] = [i]
                self._groups.add(_id)
            else:
                self._grp_to_idx[_id].append(i)
    def __repr__(self) -> str:
        return f"Grouper with {len(self._groups)} elements."
    def get_groups(self, verbose=True):
        for group in tqdm(self._groups, disable=not verbose):
            yield group, self._grp_to_idx[group]


def get_max_pred_per_experiment(
    experiment_ids: list[str], metadata: list[pd.Series], prediction_scores: np.ndarray
):
    if len(experiment_ids) != len(metadata):
        raise ValueError("lenght of experiment ids does not match with metadata!")
    if len(experiment_ids) != prediction_scores.shape[0]:
        raise ValueError(
            "Length of experiment ids does not match with prediction scores!"
        )
    grouper = Grouper(experiment_ids)
    for group, indices in grouper.get_groups():
        if not all([metadata[i].exp_hier == group for i in indices]):
            raise ValueError("Metadata does not match with group indices!")
        yield (
            group,
            prediction_scores[indices].max(0),
            prediction_scores[indices].argmax(0),
            metadata[indices[0]],
        )


def get_experiments_positive_for_tissue(
    experiments: pd.DataFrame, tissue: str, min_frac: float, min_embryo: int
):
    experiments_for_tissue = (
        experiments.loc[
            [
                tissue in x.tissue.split(";") if isinstance(x.tissue, str) else False
                for _, x in experiments.iterrows()
            ]
        ]
        .query("denominator > @min_embryo")
        .copy()
    )
    for _, (_id, tissues, positives, denominator) in experiments_for_tissue[
        ["exp_hier", "tissue", "tissue_positive", "denominator"]
    ].iterrows():
        tissues = tissues.split(";")
        positives = positives.split(";")
        if int(positives[tissues.index(tissue)]) / denominator >= min_frac:
            yield _id


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
                )


def get_dataset(d):
    if "organoid" in d and "embryo" in d:
        return "both"
    elif "organoid" in d:
        return "organoid"
    elif "embryo" in d:
        return "embryo"
    else:
        raise ValueError(d)


def calculate_overlap_vectorized(sequences, ref_idc, alt_idc):
    # Convert sequences to numpy array for vectorization
    ref_seqs = np.array([list(sequences[i]) for i in ref_idc])
    alt_seqs = np.array([list(sequences[i]) for i in alt_idc])
    # Reshape for broadcasting
    ref_seqs = ref_seqs[:, np.newaxis, :]  # Shape: (len(ref_idc), 1, seq_length)
    alt_seqs = alt_seqs[np.newaxis, :, :]  # Shape: (1, len(alt_idc), seq_length)
    # Calculate overlap using vectorized operations
    nuc_overlap = np.sum(ref_seqs == alt_seqs, axis=2)
    return nuc_overlap


def get_start_end_max_stretch(arr):
    breaks = np.where(np.diff(arr) != 1)[0]
    all_breaks = np.concatenate(([0], breaks + 1, [len(arr)]))
    l = np.diff(all_breaks)
    return all_breaks[np.argmax(l) : np.argmax(l) + 2]


@cache
def prepare_vista_data():
    genome_dir = "../../../../../resources/"
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    prediction_score_organoid = np.load(
        "../data_prep_new/organoid_data/MODELS/VISTA_VALIDATION/prediction_score_organoid.npz"
    )["prediction_score"]
    prediction_score_embryo = np.load(
        "../data_prep_new/embryo_data/MODELS/VISTA_VALIDATION/prediction_score_embryo.npz"
    )["prediction_score"]
    (
        experiment_ids_organoid,
        prediction_scores_organoid_max,
        prediction_scores_organoid_argmax,
        metadata_organoid,
    ) = [], [], [], []
    for _id, _pred, _argmax, _meta in get_max_pred_per_experiment(
        experiment_ids, metadata, prediction_score_organoid
    ):
        experiment_ids_organoid.append(_id)
        prediction_scores_organoid_max.append(_pred)
        prediction_scores_organoid_argmax.append(_argmax)
        metadata_organoid.append(_meta)
    prediction_scores_organoid_max = np.array(prediction_scores_organoid_max)
    prediction_scores_organoid_argmax = np.array(prediction_scores_organoid_argmax)
    (
        experiment_ids_embryo,
        prediction_scores_embryo_max,
        prediction_scores_embryo_argmax,
        metadata_embryo,
    ) = [], [], [], []
    for _id, _pred, _argmax, _meta in get_max_pred_per_experiment(
        experiment_ids, metadata, prediction_score_embryo
    ):
        experiment_ids_embryo.append(_id)
        prediction_scores_embryo_max.append(_pred)
        prediction_scores_embryo_argmax.append(_argmax)
        metadata_embryo.append(_meta)
    prediction_scores_embryo_max = np.array(prediction_scores_embryo_max)
    prediction_scores_embryo_argmax = np.array(prediction_scores_embryo_argmax)
    neural_tube_pos_enhancers = [
        experiment_ids_organoid.index(x)
        for x in get_experiments_positive_for_tissue(VISTA_experiments, "nt", 0.5, 3)
    ]
    facial_mesenchyme_pos_enhancers = [
        experiment_ids_organoid.index(x)
        for x in get_experiments_positive_for_tissue(VISTA_experiments, "fm", 0.5, 3)
    ]
    return (
        prediction_scores_organoid_max,
        prediction_scores_embryo_max,
        neural_tube_pos_enhancers,
        facial_mesenchyme_pos_enhancers,
    )


@cache
def prepare_motif_clustering_data():
    print("reading ctx")
    menr_organoid = read_hdf5(
        "../data_prep_new/organoid_data/ATAC/motif_enrichment_training_data.h5ad"
    )
    menr_embryo = read_hdf5(
        "../data_prep_new/embryo_data/ATAC/motif_enrichment_training_data.h5ad"
    )
    all_motifs = []
    motif_sub_names = []
    motif_names = []
    for topic in tqdm(topics_to_show_organoid):
        for motif_name in menr_organoid[
            f"training_data_Topic_{topic}"
        ].motif_enrichment.index:
            if motif_name in motif_names:
                continue
            _motifs, _m_sub_names = load_motif(
                motif_name,
                "/data/projects/c20/sdewin/PhD/motif_collection/cluster_buster",
            )
            all_motifs.extend(_motifs)
            motif_sub_names.extend(_m_sub_names)
            motif_names.extend(np.repeat(motif_name, len(_motifs)))
    for topic in tqdm(topics_to_show_embryo):
        for motif_name in menr_embryo[
            f"training_data_Topic_{topic}"
        ].motif_enrichment.index:
            if motif_name in motif_names:
                continue
            _motifs, _m_sub_names = load_motif(
                motif_name,
                "/data/projects/c20/sdewin/PhD/motif_collection/cluster_buster",
            )
            all_motifs.extend(_motifs)
            motif_sub_names.extend(_m_sub_names)
            motif_names.extend(np.repeat(motif_name, len(_motifs)))
    all_motif_enrichment_res = []
    for topic in topics_to_show_organoid:
        m = menr_organoid[f"training_data_Topic_{topic}"].motif_enrichment
        m["data_set"] = "organoid"
        all_motif_enrichment_res.append(m)
    for topic in topics_to_show_embryo:
        m = menr_embryo[f"training_data_Topic_{topic}"].motif_enrichment
        m["data_set"] = "embryo"
        all_motif_enrichment_res.append(m)
    all_motif_enrichment_res = pd.concat(all_motif_enrichment_res)
    motif_name_to_max_NES = (
        all_motif_enrichment_res.reset_index()
        .pivot_table(index=["data_set", "index"], columns="Region_set", values="NES")
        .fillna(0)
        .max(1)
    )
    motif_name_to_dataset = (
        all_motif_enrichment_res.reset_index()
        .groupby("index")["data_set"]
        .apply(lambda x: get_dataset(list(x)))
    )
    print("reading patterns")
    organoid_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/DEEPTOPIC_w_20221004/tfmodisco_new_all_topics/outs"
    embryo_dl_motif_dir = "/data/projects/c20/sdewin/PhD/De_Winter_hNTorg/EMBRYO_ANALYSIS/DEEPTOPIC/tfmodisco_all_topics/outs"
    all_motifs_dl_organoid = []
    motif_names_dl_organoid = []
    is_motif_pos_dl_organoid = []
    ic_motifs_dl_organoid = []
    for topic in tqdm(topics_to_show_organoid):
        for name, is_pos, ppm, ic in load_motif_from_modisco(
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
    all_motifs_dl_embryo = []
    motif_names_dl_embryo = []
    is_motif_pos_dl_embryo = []
    ic_motifs_dl_embryo = []
    for topic in tqdm(topics_to_show_embryo):
        for name, is_pos, ppm, ic in load_motif_from_modisco(
            filename=os.path.join(embryo_dl_motif_dir, f"patterns_Topic_{topic}.hdf5"),
            ic_thr=0.2,
            avg_ic_thr=0.5,
        ):
            all_motifs_dl_embryo.append(ppm)
            motif_names_dl_embryo.append("embryo_" + name)
            is_motif_pos_dl_embryo.append(is_pos)
            ic_motifs_dl_embryo.append(ic)
    motif_metadata = pd.DataFrame(
        index=motif_sub_names,
        data=dict(
            motif_name=motif_names,
            max_NES_organoid=pd.DataFrame(index=list(set(motif_names)))
            .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["organoid"]}),
                left_index=True,
                right_index=True,
                how="left",
            )
            .loc[motif_names]["NES"]
            .values,
            max_NES_embryo=pd.DataFrame(index=list(set(motif_names)))
            .merge(
                pd.DataFrame({"NES": motif_name_to_max_NES.loc["embryo"]}),
                left_index=True,
                right_index=True,
                how="left",
            )
            .loc[motif_names]["NES"]
            .values,
            data_set=motif_name_to_dataset.loc[motif_names].values,
            method=np.repeat("cisTarget", len(motif_names)),
            max_ic=np.repeat(np.nan, len(motif_names)),
            avg_ic=np.repeat(np.nan, len(motif_names)),
        ),
    )
    motif_metadata_dl_organoid = pd.DataFrame(
        index=motif_names_dl_organoid,
        data=dict(
            motif_name=motif_names_dl_organoid,
            max_NES_organoid=np.repeat(np.nan, len(motif_names_dl_organoid)),
            max_NES_embryo=np.repeat(np.nan, len(motif_names_dl_organoid)),
            data_set=np.repeat("organoid", len(motif_names_dl_organoid)),
            method=np.repeat("deep learning", len(motif_names_dl_organoid)),
            max_ic=[ic.max() for ic in ic_motifs_dl_organoid],
            avg_ic=[ic.mean() for ic in ic_motifs_dl_organoid],
        ),
    )
    motif_metadata_dl_embryo = pd.DataFrame(
        index=motif_names_dl_embryo,
        data=dict(
            motif_name=motif_names_dl_embryo,
            max_NES_organoid=np.repeat(np.nan, len(motif_names_dl_embryo)),
            max_NES_embryo=np.repeat(np.nan, len(motif_names_dl_embryo)),
            data_set=np.repeat("embryo", len(motif_names_dl_embryo)),
            method=np.repeat("deep learning", len(motif_names_dl_embryo)),
            max_ic=[ic.max() for ic in ic_motifs_dl_embryo],
            avg_ic=[ic.mean() for ic in ic_motifs_dl_embryo],
        ),
    )
    motif_names_dl_ctx = [
        *motif_names,
        *motif_names_dl_organoid,
        *motif_names_dl_embryo,
    ]
    motif_sub_names_dl_ctx = [
        *motif_sub_names,
        *motif_names_dl_organoid,
        *motif_names_dl_embryo,
    ]
    all_motifs_dl_ctx = [*all_motifs, *all_motifs_dl_organoid, *all_motifs_dl_embryo]
    motif_metadata_dl_ctx = pd.concat(
        [motif_metadata, motif_metadata_dl_organoid, motif_metadata_dl_embryo]
    ).loc[motif_sub_names_dl_ctx]
    print("clustering")
    t_all_motifs_dl_ctx = [torch.from_numpy(m).T for m in tqdm(all_motifs_dl_ctx)]
    motif_metadata_dl_ctx["motif_length"] = [m.shape[1] for m in t_all_motifs_dl_ctx]
    pvals, scores, offsets, overlaps, strands = tomtom(
        t_all_motifs_dl_ctx, t_all_motifs_dl_ctx
    )
    evals = pvals.numpy() * len(all_motifs_dl_ctx)
    adata_motifs_dl_ctx = sc.AnnData(evals, obs=motif_metadata_dl_ctx)
    sc.tl.pca(adata_motifs_dl_ctx)
    sc.pp.neighbors(adata_motifs_dl_ctx)
    sc.tl.tsne(adata_motifs_dl_ctx)
    sc.tl.leiden(adata_motifs_dl_ctx, resolution=2)
    sc.pl.tsne(
        adata_motifs_dl_ctx,
        color=["leiden"],
        save="_leiden_motifs_dl_ctx.pdf",
        legend_loc="on data",
    )
    return adata_motifs_dl_ctx, t_all_motifs_dl_ctx


@cache
def prepare_ref_alt_data_vista():
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    neural_tube_topics_organoid = [
        "neural_crest_Topic_5",
        "neural_crest_Topic_7",
        "neuron_Topic_1",
        "neuron_Topic_2",
        "neuron_Topic_3",
        "neuron_Topic_4",
        "neuron_Topic_6",
        "neuron_Topic_10",
        "neuron_Topic_11",
        "neuron_Topic_12",
        "neuron_Topic_13",
        "neuron_Topic_15",
        "neuron_Topic_16",
        "neuron_Topic_18",
        "neuron_Topic_19",
        "neuron_Topic_20",
        "neuron_Topic_21",
        "neuron_Topic_23",
        "neuron_Topic_24",
        "neuron_Topic_25",
        "progenitor_Topic_1",
        "progenitor_Topic_3",
        "progenitor_Topic_8",
        "progenitor_Topic_9",
        "progenitor_Topic_11",
        "progenitor_Topic_13",
        "progenitor_Topic_14",
        "progenitor_Topic_16",
        "progenitor_Topic_19",
        "progenitor_Topic_21",
        "progenitor_Topic_23",
        "progenitor_Topic_24",
        "progenitor_Topic_25",
        "progenitor_Topic_29",
        "progenitor_Topic_30",
    ]
    classes_to_consider = [
        topic_name_to_model_index_organoid(x) for x in neural_tube_topics_organoid
    ]
    genome_dir = "../../../../../resources"
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    oh_sequences = np.array(
        [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
    )
    ref_alt_exp = (
        pd.read_table(
            "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_allelic_curations.tsv.gz"
        )
        .query("refExp != altExp")
        .set_index("refExp")["altExp"]
    )
    prediction_score_organoid = np.load(
        "../data_prep_new/organoid_data/MODELS/VISTA_VALIDATION/prediction_score_organoid.npz"
    )["prediction_score"]
    grp_to_idx = Grouper(experiment_ids)._grp_to_idx
    neural_tube_pos_enhancers = list(
        get_experiments_positive_for_tissue(VISTA_experiments, "nt", 0.5, 3)
    )
    neural_tube_pos_enhancers_w_alt = list(
        set(neural_tube_pos_enhancers) & set(ref_alt_exp.index)
    )
    ref_alt_to_data = {}
    for ref_enhancer in neural_tube_pos_enhancers_w_alt:
        print(f"ref: {ref_enhancer}")
        alt_names = (
            ref_alt_exp.loc[ref_enhancer]
            if not isinstance(ref_alt_exp.loc[ref_enhancer], str)
            else [ref_alt_exp.loc[ref_enhancer]]
        )
        for alt_enhancer in alt_names:
            print("\t################")
            print(f"\talt: {alt_enhancer}")
            ref_enhancer_idc = np.array(grp_to_idx[ref_enhancer])
            alt_enhancer_idc = np.array(grp_to_idx[alt_enhancer])
            nuc_overlap = calculate_overlap_vectorized(
                sequences, ref_enhancer_idc, alt_enhancer_idc
            )
            ref_overlap, alt_overlap = np.where(nuc_overlap > 250)
            ref_start, ref_end = get_start_end_max_stretch(ref_overlap)
            alt_start, alt_end = get_start_end_max_stretch(alt_overlap)
            n_seq = min((alt_end - alt_start), (ref_end - ref_start))
            if n_seq == 0:
                print("\tno overalp :(")
                continue
            print(f"\t{n_seq} overlap.")
            ref_enhancer_idc = ref_enhancer_idc[ref_overlap[ref_start:ref_end]][0:n_seq]
            alt_enhancer_idc = alt_enhancer_idc[alt_overlap[alt_start:alt_end]][0:n_seq]
            idx_w_alt = []
            for i, (ref, alt) in enumerate(zip(ref_enhancer_idc, alt_enhancer_idc)):
                if (
                    sum(oh_sequences[ref].argmax(1) == oh_sequences[alt].argmax(1))
                    != 500
                ):
                    idx_w_alt.append(i)
            if len(idx_w_alt) == 0:
                print("\tno differences...")
                continue
            ref_enhancer_idc = ref_enhancer_idc[idx_w_alt]
            alt_enhancer_idc = alt_enhancer_idc[idx_w_alt]
            prediction_scores_ref = prediction_score_organoid[ref_enhancer_idc][
                :, classes_to_consider
            ]
            prediction_scores_alt = prediction_score_organoid[alt_enhancer_idc][
                :, classes_to_consider
            ]
            relative_window, relative_class_idx = np.unravel_index(
                prediction_scores_ref.argmax(), prediction_scores_ref.shape
            )
            ref_pred = prediction_scores_ref[relative_window, relative_class_idx]
            alt_pred = prediction_scores_alt[relative_window, relative_class_idx]
            print(f"\tref pred: {ref_pred}")
            print(f"\talt pred: {alt_pred}")
            if max(ref_pred, alt_pred) >= 0.5 and abs(ref_pred - alt_pred) >= 0.2:
                print("\t😊😊😊😊")
                ref_alt_to_data[(ref_enhancer, alt_enhancer)] = dict(
                    ref_enhancer_idx=ref_enhancer_idc[relative_window],
                    alt_enhancer_idx=alt_enhancer_idc[relative_window],
                    class_idx=classes_to_consider[relative_class_idx],
                )
    return ref_alt_to_data


color_dict = json.load(open("../color_maps.json"))

topics_to_show_organoid = []
with open(
    "../data_prep_new/organoid_data/ATAC/topics_hq_motif_enrichment.txt", "rt"
) as f:
    for l in f:
        topics_to_show_organoid.append(int(l.strip()))

topics_to_show_embryo = []
with open(
    "../data_prep_new/embryo_data/ATAC/topics_hq_motif_enrichment.txt", "rt"
) as f:
    for l in f:
        topics_to_show_embryo.append(int(l.strip()))


ax_cache = {}


def restore_plot(fig, gs, y, x, key):
    print("using cache for: " + key)
    ax = ax_cache[key]
    old_fig = ax.figure
    ax.remove()
    ax.figure = fig
    fig.axes.append(ax)
    fig.add_axes(ax)
    dummy_ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    ax.set_position(dummy_ax.get_position())
    dummy_ax.remove()
    plt.close(old_fig)


def plot_w_cache(key):
    def decorator(plot_func):
        def wrapper(fig, gs, y, x):
            if key in ax_cache:
                restore_plot(fig, gs, y, x, key)
            else:
                ax = plot_func(fig, gs, y, x)
                ax_cache[key] = ax
        return wrapper
    return decorator


@plot_w_cache(key="region_topic_organoid")
def plot_region_topic_organoid(fig, gs, y, x):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    print("Plotting region topic organoid.")
    print("\tReading data...")
    organoid_neural_crest_region_topic = pd.read_table(
        "../data_prep_new/organoid_data/ATAC/neural_crest_region_topic_contrib.tsv",
        index_col=0,
    )
    organoid_neuron_region_topic = pd.read_table(
        "../data_prep_new/organoid_data/ATAC/neuron_region_topic_contrib.tsv",
        index_col=0,
    )
    organoid_progenitor_region_topic = pd.read_table(
        "../data_prep_new/organoid_data/ATAC/progenitor_region_topic_contrib.tsv",
        index_col=0,
    )
    organoid_pluripotent_region_topic = pd.read_table(
        "../data_prep_new/organoid_data/ATAC/pluripotent_region_topic_contrib.tsv",
        index_col=0,
    )
    organoid_neural_crest_region_topic.columns = [
        f"neural_crest_Topic_{topic.replace('Topic', '')}"
        for topic in organoid_neural_crest_region_topic.columns
    ]
    organoid_neuron_region_topic.columns = [
        f"neuron_Topic_{topic.replace('Topic', '')}"
        for topic in organoid_neuron_region_topic.columns
    ]
    organoid_progenitor_region_topic.columns = [
        f"progenitor_Topic_{topic.replace('Topic', '')}"
        for topic in organoid_progenitor_region_topic.columns
    ]
    organoid_pluripotent_region_topic.columns = [
        f"pluripotent_Topic_{topic.replace('Topic', '')}"
        for topic in organoid_pluripotent_region_topic.columns
    ]
    organoid_region_topic = pd.concat(
        [
            organoid_neural_crest_region_topic,
            organoid_neuron_region_topic,
            organoid_progenitor_region_topic,
            organoid_pluripotent_region_topic,
        ],
        axis=1,
    ).fillna(0)
    organoid_selected_topics = [
        model_index_to_topic_name_organoid(x - 1) for x in topics_to_show_organoid
    ]
    organoid_selected_regions = set()
    for topic in organoid_selected_topics:
        organoid_selected_regions.update(
            organoid_region_topic.sort_values(topic, ascending=False).head(500).index
        )
    organoid_selected_regions = list(organoid_selected_regions)
    mat_to_plot = organoid_region_topic.loc[
        organoid_selected_regions, organoid_selected_topics
    ]
    print("\tClustering regions...")
    row_linkage = hierarchy.linkage(
        distance.pdist(X=np.asarray(mat_to_plot)), method="average"
    )
    row_dendrogram = hierarchy.dendrogram(
        row_linkage, no_plot=True, color_threshold=-np.inf
    )
    region_ind = row_dendrogram["leaves"]
    print("\tClustering topics...")
    col_linkage = hierarchy.linkage(
        distance.pdist(X=np.asarray(mat_to_plot).T), method="average"
    )
    col_dendrogram = hierarchy.dendrogram(
        col_linkage, no_plot=True, color_threshold=-np.inf
    )
    topic_ind = col_dendrogram["leaves"]
    region_order = [organoid_selected_regions[i] for i in region_ind]
    topic_order = [organoid_selected_topics[i] for i in topic_ind]
    print("\tPlotting...")
    sns.heatmap(
        data=mat_to_plot.loc[region_order, topic_order],
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        cbar=False,
        cmap="viridis",
        vmin=0,
        vmax=7e-5,
    )
    return ax


@plot_w_cache(key="region_topic_embryo")
def plot_region_topic_embryo(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    print("Plotting region topic embryo.")
    print("\tReading data...")
    embryo_neural_crest_region_topic = pd.read_table(
        "../data_prep_new/embryo_data/ATAC/neural_crest_region_topic_contrib.tsv",
        index_col=0,
    )
    embryo_neuron_region_topic = pd.read_table(
        "../data_prep_new/embryo_data/ATAC/neuron_region_topic_contrib.tsv",
        index_col=0,
    )
    embryo_progenitor_region_topic = pd.read_table(
        "../data_prep_new/embryo_data/ATAC/progenitor_region_topic_contrib.tsv",
        index_col=0,
    )
    embryo_neural_crest_region_topic.columns = [
        f"neural_crest_Topic_{topic.replace('Topic_', '')}"
        for topic in embryo_neural_crest_region_topic.columns
    ]
    embryo_neuron_region_topic.columns = [
        f"neuron_Topic_{topic.replace('Topic_', '')}"
        for topic in embryo_neuron_region_topic.columns
    ]
    embryo_progenitor_region_topic.columns = [
        f"progenitor_Topic_{topic.replace('Topic_', '')}"
        for topic in embryo_progenitor_region_topic.columns
    ]
    embryo_region_topic = pd.concat(
        [
            embryo_neural_crest_region_topic,
            embryo_neuron_region_topic,
            embryo_progenitor_region_topic,
        ],
        axis=1,
    ).fillna(0)
    embryo_selected_topics = [
        model_index_to_topic_name_embryo(x - 1) for x in topics_to_show_embryo
    ]
    embryo_selected_regions = set()
    for topic in embryo_selected_topics:
        embryo_selected_regions.update(
            embryo_region_topic.sort_values(topic, ascending=False).head(500).index
        )
    embryo_selected_regions = list(embryo_selected_regions)
    mat_to_plot = embryo_region_topic.loc[
        embryo_selected_regions, embryo_selected_topics
    ]
    print("\tClustering regions...")
    row_linkage = hierarchy.linkage(
        distance.pdist(X=np.asarray(mat_to_plot)), method="average"
    )
    row_dendrogram = hierarchy.dendrogram(
        row_linkage, no_plot=True, color_threshold=-np.inf
    )
    region_ind = row_dendrogram["leaves"]
    print("\tClustering topics...")
    col_linkage = hierarchy.linkage(
        distance.pdist(X=np.asarray(mat_to_plot).T), method="average"
    )
    col_dendrogram = hierarchy.dendrogram(
        col_linkage, no_plot=True, color_threshold=-np.inf
    )
    topic_ind = col_dendrogram["leaves"]
    topic_order = [embryo_selected_topics[i] for i in topic_ind]
    region_order = [embryo_selected_regions[i] for i in region_ind]
    print("\tPlotting...")
    sns.heatmap(
        data=mat_to_plot.loc[region_order, topic_order],
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        cbar=False,
        cmap="viridis",
        vmin=0,
        vmax=7e-5,
    )
    return ax


@plot_w_cache(key="PR_vista_neural_tube")
def plot_PR_vista_neural_tube(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    neural_tube_topics_organoid = [
        "neural_crest_Topic_5",
        "neural_crest_Topic_7",
        "neuron_Topic_1",
        "neuron_Topic_2",
        "neuron_Topic_3",
        "neuron_Topic_4",
        "neuron_Topic_6",
        "neuron_Topic_10",
        "neuron_Topic_11",
        "neuron_Topic_12",
        "neuron_Topic_13",
        "neuron_Topic_15",
        "neuron_Topic_16",
        "neuron_Topic_18",
        "neuron_Topic_19",
        "neuron_Topic_20",
        "neuron_Topic_21",
        "neuron_Topic_23",
        "neuron_Topic_24",
        "neuron_Topic_25",
        "progenitor_Topic_1",
        "progenitor_Topic_3",
        "progenitor_Topic_8",
        "progenitor_Topic_9",
        "progenitor_Topic_11",
        "progenitor_Topic_13",
        "progenitor_Topic_14",
        "progenitor_Topic_16",
        "progenitor_Topic_19",
        "progenitor_Topic_21",
        "progenitor_Topic_23",
        "progenitor_Topic_24",
        "progenitor_Topic_25",
        "progenitor_Topic_29",
        "progenitor_Topic_30",
    ]
    neural_tube_topics_embryo = [
        "neuron_Topic_1",  #
        "neuron_Topic_3",  #
        "neuron_Topic_5",  #
        "neuron_Topic_6",  #
        "neuron_Topic_7",
        "neuron_Topic_8",  #
        "neuron_Topic_9",  #
        "neuron_Topic_10",
        "neuron_Topic_11",  #
        "neuron_Topic_12",
        "neuron_Topic_13",  #
        "neuron_Topic_14",
        "neuron_Topic_15",  #
        "neuron_Topic_17",  #
        "neuron_Topic_18",  #
        "neuron_Topic_19",  #
        "neuron_Topic_22",  #
        "neuron_Topic_24",  #
        "neuron_Topic_26",  #
        "neuron_Topic_27",  #
        "neuron_Topic_29",
        "neuron_Topic_30",
        "progenitor_Topic_1",  #
        "progenitor_Topic_3",
        "progenitor_Topic_4",  #
        "progenitor_Topic_8",  #
        "progenitor_Topic_10",
        "progenitor_Topic_11",
        "progenitor_Topic_16",
        "progenitor_Topic_17",
        "progenitor_Topic_21",  #
        "progenitor_Topic_22",  #
        "progenitor_Topic_28",  #
        "progenitor_Topic_29",  #
        "progenitor_Topic_31",  #
        "progenitor_Topic_32",  #
        "progenitor_Topic_36",
        "progenitor_Topic_40",  #
        "progenitor_Topic_41",  #
        "progenitor_Topic_44",  #
        "progenitor_Topic_45",
        "progenitor_Topic_49",  #
        "progenitor_Topic_51",
        "progenitor_Topic_57",  #
        "progenitor_Topic_58",  #
        "progenitor_Topic_59",
        "neural_crest_Topic_12",  #
        "neural_crest_Topic_13",  #
        "neural_crest_Topic_15",
    ]
    (
        prediction_scores_organoid_max,
        prediction_scores_embryo_max,
        neural_tube_pos_enhancers,
        _,
    ) = prepare_vista_data()
    y = np.zeros(prediction_scores_organoid_max.shape[0])
    y[neural_tube_pos_enhancers] = 1
    prec_organoid, rec_organoid, _ = sklearn.metrics.precision_recall_curve(
        y,
        prediction_scores_organoid_max[
            :,
            [
                topic_name_to_model_index_organoid(t)
                for t in neural_tube_topics_organoid
            ],
        ].max(1),
    )
    pr_organoid_auc = sklearn.metrics.auc(rec_organoid, prec_organoid)
    prec_embryo, rec_embryo, _ = sklearn.metrics.precision_recall_curve(
        y,
        prediction_scores_embryo_max[
            :, [topic_name_to_model_index_embryo(t) for t in neural_tube_topics_embryo]
        ].max(1),
    )
    pr_embryo_auc = sklearn.metrics.auc(rec_embryo, prec_embryo)
    ax.plot(
        rec_organoid,
        prec_organoid,
        color=color_dict["model_validation"]["organoid"],
        label=f"organoid (AUC = {round(pr_organoid_auc, 3)})",
    )
    ax.plot(
        rec_embryo,
        prec_embryo,
        color=color_dict["model_validation"]["embryo"],
        label=f"embryo (AUC = {round(pr_embryo_auc, 3)})",
    )
    ax.grid("gray")
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return ax

(
    prediction_scores_organoid_max,
    prediction_scores_embryo_max,
    neural_tube_pos_enhancers,
    facial_mesenchyme_pos_enhancers,
) = prepare_vista_data()

organoid_topics = np.array([
    6, 4, 23, 24, 13, 2, 33, 38, 36, 54, 48, 62, 60, 65, 59, 58
]) - 1

embryo_topics = np.array(
    [10, 8, 13, 24, 18, 29, 34, 38, 79, 88, 58, 61, 59, 31, 62, 70, 52, 71, 103, 105, 94, 91]
) - 1

np.unique(organoid_topics[prediction_scores_organoid_max[neural_tube_pos_enhancers, :][:, organoid_topics].argmax(1)] + 1, return_counts = True)

np.unique(embryo_topics[prediction_scores_embryo_max[neural_tube_pos_enhancers, :][:, embryo_topics].argmax(1)] + 1, return_counts = True)

np.logical_and(np.isin(embryo_topics[prediction_scores_embryo_max[neural_tube_pos_enhancers, :][:, embryo_topics].argmax(1)] + 1, [103, 105, 94]), np.isin(organoid_topics[prediction_scores_organoid_max[neural_tube_pos_enhancers, :][:, organoid_topics].argmax(1)]+ 1, [62, 60, 65])).sum() / len(neural_tube_pos_enhancers)


@plot_w_cache(key="ROC_vista_neural_tube")
def plot_ROC_vista_neural_tube(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    neural_tube_topics_organoid = [
        "neural_crest_Topic_5",
        "neural_crest_Topic_7",
        "neuron_Topic_1",
        "neuron_Topic_2",
        "neuron_Topic_3",
        "neuron_Topic_4",
        "neuron_Topic_6",
        "neuron_Topic_10",
        "neuron_Topic_11",
        "neuron_Topic_12",
        "neuron_Topic_13",
        "neuron_Topic_15",
        "neuron_Topic_16",
        "neuron_Topic_18",
        "neuron_Topic_19",
        "neuron_Topic_20",
        "neuron_Topic_21",
        "neuron_Topic_23",
        "neuron_Topic_24",
        "neuron_Topic_25",
        "progenitor_Topic_1",
        "progenitor_Topic_3",
        "progenitor_Topic_8",
        "progenitor_Topic_9",
        "progenitor_Topic_11",
        "progenitor_Topic_13",
        "progenitor_Topic_14",
        "progenitor_Topic_16",
        "progenitor_Topic_19",
        "progenitor_Topic_21",
        "progenitor_Topic_23",
        "progenitor_Topic_24",
        "progenitor_Topic_25",
        "progenitor_Topic_29",
        "progenitor_Topic_30",
    ]
    neural_tube_topics_embryo = [
        "neuron_Topic_1",  #
        "neuron_Topic_3",  #
        "neuron_Topic_5",  #
        "neuron_Topic_6",  #
        "neuron_Topic_7",
        "neuron_Topic_8",  #
        "neuron_Topic_9",  #
        "neuron_Topic_10",
        "neuron_Topic_11",  #
        "neuron_Topic_12",
        "neuron_Topic_13",  #
        "neuron_Topic_14",
        "neuron_Topic_15",  #
        "neuron_Topic_17",  #
        "neuron_Topic_18",  #
        "neuron_Topic_19",  #
        "neuron_Topic_22",  #
        "neuron_Topic_24",  #
        "neuron_Topic_26",  #
        "neuron_Topic_27",  #
        "neuron_Topic_29",
        "neuron_Topic_30",
        "progenitor_Topic_1",  #
        "progenitor_Topic_3",
        "progenitor_Topic_4",  #
        "progenitor_Topic_8",  #
        "progenitor_Topic_10",
        "progenitor_Topic_11",
        "progenitor_Topic_16",
        "progenitor_Topic_17",
        "progenitor_Topic_21",  #
        "progenitor_Topic_22",  #
        "progenitor_Topic_28",  #
        "progenitor_Topic_29",  #
        "progenitor_Topic_31",  #
        "progenitor_Topic_32",  #
        "progenitor_Topic_36",
        "progenitor_Topic_40",  #
        "progenitor_Topic_41",  #
        "progenitor_Topic_44",  #
        "progenitor_Topic_45",
        "progenitor_Topic_49",  #
        "progenitor_Topic_51",
        "progenitor_Topic_57",  #
        "progenitor_Topic_58",  #
        "progenitor_Topic_59",
        "neural_crest_Topic_12",  #
        "neural_crest_Topic_13",  #
        "neural_crest_Topic_15",
    ]
    (
        prediction_scores_organoid_max,
        prediction_scores_embryo_max,
        neural_tube_pos_enhancers,
        _,
    ) = prepare_vista_data()
    y = np.zeros(prediction_scores_organoid_max.shape[0])
    y[neural_tube_pos_enhancers] = 1
    fpr_organoid, tpr_organoid, _ = sklearn.metrics.roc_curve(
        y,
        prediction_scores_organoid_max[
            :,
            [
                topic_name_to_model_index_organoid(t)
                for t in neural_tube_topics_organoid
            ],
        ].max(1),
    )
    ROC_organoid_auc = sklearn.metrics.auc(fpr_organoid, tpr_organoid)
    fpr_embryo, tpr_embryo, _ = sklearn.metrics.roc_curve(
        y,
        prediction_scores_embryo_max[
            :, [topic_name_to_model_index_embryo(t) for t in neural_tube_topics_embryo]
        ].max(1),
    )
    ROC_embryo_auc = sklearn.metrics.auc(fpr_embryo, tpr_embryo)
    ax.plot(
        fpr_organoid,
        tpr_organoid,
        color=color_dict["model_validation"]["organoid"],
        label=f"organoid (AUC = {round(ROC_organoid_auc, 3)})",
    )
    ax.plot(
        fpr_embryo,
        tpr_embryo,
        color=color_dict["model_validation"]["embryo"],
        label=f"embryo (AUC = {round(ROC_embryo_auc, 3)})",
    )
    ax.plot([0, 1], [0, 1], color="darkgray", ls="dashed")
    ax.grid("gray")
    ax.set_axisbelow(True)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    return ax


@plot_w_cache(key="PR_vista_facial_mesenchyme")
def plot_PR_vista_facial_mesenchyme(
    fig,
    gs,
    y,
    x,
):
    facial_mesenchyme_topics_organoid = ["neural_crest_Topic_3"]
    facial_mesenchyme_topics_embryo = [
        "neural_crest_Topic_1",
        "neural_crest_Topic_19",
        "neural_crest_Topic_21",
        "neural_crest_Topic_29",
    ]
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    (
        prediction_scores_organoid_max,
        prediction_scores_embryo_max,
        _,
        facial_mesenchyme_pos_enhancers,
    ) = prepare_vista_data()
    y = np.zeros(prediction_scores_organoid_max.shape[0])
    y[facial_mesenchyme_pos_enhancers] = 1
    prec_organoid, rec_organoid, _ = sklearn.metrics.precision_recall_curve(
        y,
        prediction_scores_organoid_max[
            :,
            [
                topic_name_to_model_index_organoid(t)
                for t in facial_mesenchyme_topics_organoid
            ],
        ].max(1),
    )
    pr_organoid_auc = sklearn.metrics.auc(rec_organoid, prec_organoid)
    prec_embryo, rec_embryo, _ = sklearn.metrics.precision_recall_curve(
        y,
        prediction_scores_embryo_max[
            :,
            [
                topic_name_to_model_index_embryo(t)
                for t in facial_mesenchyme_topics_embryo
            ],
        ].max(1),
    )
    pr_embryo_auc = sklearn.metrics.auc(rec_embryo, prec_embryo)
    ax.plot(
        rec_organoid,
        prec_organoid,
        color=color_dict["model_validation"]["organoid"],
        label=f"organoid (AUC = {round(pr_organoid_auc, 3)})",
    )
    ax.plot(
        rec_embryo,
        prec_embryo,
        color=color_dict["model_validation"]["embryo"],
        label=f"embryo (AUC = {round(pr_embryo_auc, 3)})",
    )
    ax.grid("gray")
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return ax


@plot_w_cache(key="ROC_vista_facial_mesenchyme")
def plot_ROC_vista_facial_mesenchyme(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    facial_mesenchyme_topics_organoid = ["neural_crest_Topic_3"]
    facial_mesenchyme_topics_embryo = [
        "neural_crest_Topic_1",
        "neural_crest_Topic_19",
        "neural_crest_Topic_21",
        "neural_crest_Topic_29",
    ]
    (
        prediction_scores_organoid_max,
        prediction_scores_embryo_max,
        _,
        facial_mesenchyme_pos_enhancers,
    ) = prepare_vista_data()
    y = np.zeros(prediction_scores_organoid_max.shape[0])
    y[facial_mesenchyme_pos_enhancers] = 1
    fpr_organoid, tpr_organoid, _ = sklearn.metrics.roc_curve(
        y,
        prediction_scores_organoid_max[
            :,
            [
                topic_name_to_model_index_organoid(t)
                for t in facial_mesenchyme_topics_organoid
            ],
        ].max(1),
    )
    ROC_organoid_auc = sklearn.metrics.auc(fpr_organoid, tpr_organoid)
    fpr_embryo, tpr_embryo, _ = sklearn.metrics.roc_curve(
        y,
        prediction_scores_embryo_max[
            :,
            [
                topic_name_to_model_index_embryo(t)
                for t in facial_mesenchyme_topics_embryo
            ],
        ].max(1),
    )
    ROC_embryo_auc = sklearn.metrics.auc(fpr_embryo, tpr_embryo)
    ax.plot(
        fpr_organoid,
        tpr_organoid,
        color=color_dict["model_validation"]["organoid"],
        label=f"organoid (AUC = {round(ROC_organoid_auc, 3)})",
    )
    ax.plot(
        fpr_embryo,
        tpr_embryo,
        color=color_dict["model_validation"]["embryo"],
        label=f"embryo (AUC = {round(ROC_embryo_auc, 3)})",
    )
    ax.plot([0, 1], [0, 1], color="darkgray", ls="dashed")
    ax.grid("gray")
    ax.set_axisbelow(True)
    ax.set_xlabel("FPR")
    # ax.set_ylabel("TPR")
    ax.legend()
    return ax


@plot_w_cache(key="motif_clust_leiden")
def plot_motif_cluster_leiden(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    motif_anndata, _ = prepare_motif_clustering_data()
    sc.pl.tsne(
        motif_anndata,
        color="leiden",
        frameon=False,
        title="",
        ax=ax,
        legend_loc="on_data",
    )
    tsne = motif_anndata.obsm["X_tsne"]
    for c in motif_anndata.obs["leiden"].unique():
        x, y = tsne[motif_anndata.obs.leiden == c].mean(0)
        ax.text(x, y, c, weight="bold", fontsize=8)
    return ax


@plot_w_cache(key="motif_clust_data_set")
def plot_motif_cluster_dataset(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    motif_anndata, _ = prepare_motif_clustering_data()
    motif_anndata_ctx = motif_anndata[motif_anndata.obs.method == "cisTarget"]
    sc.pl.tsne(
        motif_anndata_ctx,
        color="data_set",
        frameon=False,
        title="",
        ax=ax,
        palette=color_dict["motif_clustering"],
        legend_loc="lower center",
        size=7,
    )
    return ax


@plot_w_cache(key="motif_clust_method")
def plot_motif_cluster_method(
    fig,
    gs,
    y,
    x,
):
    ax: mpl_axes.Axes = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    motif_anndata, _ = prepare_motif_clustering_data()
    sc.pl.tsne(
        motif_anndata[motif_anndata.obs.method == "cisTarget"],
        color="method",
        frameon=False,
        title="",
        ax=ax,
        palette=color_dict["motif_clustering"],
        legend_loc="lower center",
        size=7,
    )
    sc.pl.tsne(
        motif_anndata[motif_anndata.obs.method == "deep learning"],
        color="method",
        frameon=False,
        title="",
        ax=ax,
        palette=color_dict["motif_clustering"],
        legend_loc="lower center",
        size=7,
    )
    return ax


def plot_top_motif_for_cluster(cluster, ax1, ax2):
    motif_anndata, all_motifs = prepare_motif_clustering_data()
    top_motif_idx_ctx = np.where(
        motif_anndata.obs_names
        == motif_anndata.obs.query("leiden == @cluster & method == 'cisTarget'")[
            ["max_NES_organoid", "max_NES_embryo"]
        ]
        .fillna(0)
        .max(1)
        .sort_values(ascending=False)
        .index[0]
    )[0][0]
    top_motif_idx_dl = np.where(
        motif_anndata.obs_names
        == motif_anndata.obs.query("leiden == @cluster & method == 'deep learning'")[
            "avg_ic"
        ]
        .sort_values(ascending=False)
        .index[0]
    )[0][0]
    ppm_ctx = all_motifs[top_motif_idx_ctx].numpy().T
    ppm_dl = all_motifs[top_motif_idx_dl].numpy().T
    ic_ctx = modiscolite.util.compute_per_position_ic(
        ppm=ppm_ctx, background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
    )
    ic_dl = modiscolite.util.compute_per_position_ic(
        ppm=ppm_dl, background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
    )
    _ = logomaker.Logo(
        pd.DataFrame(ppm_ctx * ic_ctx[:, None], columns=["A", "C", "G", "T"]), ax=ax1
    )
    _ = logomaker.Logo(
        pd.DataFrame(ppm_dl * ic_dl[:, None], columns=["A", "C", "G", "T"]), ax=ax2
    )
    for i, (ax, ppm) in enumerate(zip([ax1, ax2], [ppm_ctx, ppm_dl])):
        _ = ax.set_ylim((0, 2))
        if i == 0:
            _ = ax.set_yticks(
                ticks=np.arange(0, 2.2, 0.4),
                labels=[
                    int(y) if y == 0 or y == 1 or y == 2 else ""
                    for y in np.arange(0, 2.2, 0.4)
                ],
            )
        else:
            _ = ax.set_yticks(
                ticks=np.arange(0, 2.2, 0.4),
                labels=["" for y in np.arange(0, 2.2, 0.4)],
            )
        _ = ax.set_xticks(
            ticks=np.arange(0, ppm.shape[0]),
            labels=[x + 1 if x % 2 == 0 else "" for x in np.arange(0, ppm.shape[0])],
        )
        ax.spines[["right", "top"]].set_visible(False)


@plot_w_cache(key="DE_ref_org")
def plot_DE_ref_org(
    fig,
    gs,
    y,
    x,
):
    print("plotting DE ref org")
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
    organoid_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_organoid_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    organoid_model.load_weights(
        os.path.join(path_to_organoid_model, "model_epoch_23.hdf5")
    )
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    # oh_sequences = np.array(
    #    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
    # )
    prediction_score_organoid = np.load(
        "../data_prep_new/organoid_data/MODELS/VISTA_VALIDATION/prediction_score_organoid.npz"
    )["prediction_score"]
    ref_alt_to_data = prepare_ref_alt_data_vista()
    ref_enhancer, alt_enhancer = "001700010002", "001700060001"
    vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
        0
    ]
    ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
    class_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["class_idx"]
    pred_score = prediction_score_organoid[ref_enhancer_idx, class_idx]
    explainer = Explainer(model=organoid_model, class_index=int(class_idx))
    oh_sequence = one_hot_encode_sequence(sequences[ref_enhancer_idx], expand_dim=False)
    ref_gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (ref_gradients_integrated * oh_sequence).astype(float)[150:350],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.01,
        y=0.8,
        s=f"{vista_id}: Organoid {model_index_to_topic_name_organoid(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


@plot_w_cache(key="DE_alt_org")
def plot_DE_alt_org(
    fig,
    gs,
    y,
    x,
):
    print("plotting DE alt org")
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
    organoid_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_organoid_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    organoid_model.load_weights(
        os.path.join(path_to_organoid_model, "model_epoch_23.hdf5")
    )
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    # oh_sequences = np.array(
    #    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
    # )
    prediction_score_organoid = np.load(
        "../data_prep_new/organoid_data/MODELS/VISTA_VALIDATION/prediction_score_organoid.npz"
    )["prediction_score"]
    ref_alt_to_data = prepare_ref_alt_data_vista()
    ref_enhancer, alt_enhancer = "001700010002", "001700060001"
    vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
        0
    ]
    alt_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["alt_enhancer_idx"]
    ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
    class_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["class_idx"]
    pred_score = prediction_score_organoid[alt_enhancer_idx, class_idx]
    explainer = Explainer(model=organoid_model, class_index=int(class_idx))
    oh_sequence = one_hot_encode_sequence(sequences[alt_enhancer_idx], expand_dim=False)
    alt_gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (alt_gradients_integrated * oh_sequence).astype(float)[150:350],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.01,
        y=0.8,
        s=f"{vista_id}: Organoid {model_index_to_topic_name_organoid(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


@plot_w_cache(key="DE_ref_embr")
def plot_DE_ref_embr(
    fig,
    gs,
    y,
    x,
):
    print("plotting DE ref embr")
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
    embryo_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_embryo_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    # oh_sequences = np.array(
    #    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
    # )
    prediction_score_embryo = np.load(
        "../data_prep_new/embryo_data/MODELS/VISTA_VALIDATION/prediction_score_embryo.npz"
    )["prediction_score"]
    ref_alt_to_data = prepare_ref_alt_data_vista()
    ref_enhancer, alt_enhancer = "001700010002", "001700060001"
    vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
        0
    ]
    ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
    class_idx = prediction_score_embryo[ref_enhancer_idx].argmax()
    pred_score = prediction_score_embryo[ref_enhancer_idx, class_idx]
    explainer = Explainer(model=embryo_model, class_index=int(class_idx))
    oh_sequence = one_hot_encode_sequence(sequences[ref_enhancer_idx], expand_dim=False)
    ref_gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (ref_gradients_integrated * oh_sequence).astype(float)[150:350],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.02,
        y=0.8,
        s=f"{vista_id}: embryo {model_index_to_topic_name_embryo(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


@plot_w_cache(key="DE_alt_emb")
def plot_DE_alt_embr(
    fig,
    gs,
    y,
    x,
):
    print("plotting DE alt embr")
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
    embryo_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_embryo_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    mm10 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
        pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
    )
    VISTA_experiments = pd.read_table(
        "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
    )
    experiment_ids, sequences, metadata = [], [], []
    for _, vista_experiment in tqdm(
        VISTA_experiments.iterrows(),
        total=len(VISTA_experiments),
    ):
        for result in get_sequence_and_metadata(
            vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
        ):
            if result is not None:
                experiment_ids.append(result[0])
                sequences.append(result[1])
                metadata.append(result[2])
    # oh_sequences = np.array(
    #    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
    # )
    prediction_score_embryo = np.load(
        "../data_prep_new/embryo_data/MODELS/VISTA_VALIDATION/prediction_score_embryo.npz"
    )["prediction_score"]
    ref_alt_to_data = prepare_ref_alt_data_vista()
    ref_enhancer, alt_enhancer = "001700010002", "001700060001"
    vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
        0
    ]
    alt_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["alt_enhancer_idx"]
    ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
    class_idx = prediction_score_embryo[ref_enhancer_idx].argmax()
    pred_score = prediction_score_embryo[alt_enhancer_idx, class_idx]
    explainer = Explainer(model=embryo_model, class_index=int(class_idx))
    oh_sequence = one_hot_encode_sequence(sequences[alt_enhancer_idx], expand_dim=False)
    alt_gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (alt_gradients_integrated * oh_sequence).astype(float)[150:350],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.02,
        y=0.8,
        s=f"{vista_id}: embryo {model_index_to_topic_name_embryo(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


@plot_w_cache(key="DE_NC_org")
def plot_DE_NC_org(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
    organoid_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_organoid_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    organoid_model.load_weights(
        os.path.join(path_to_organoid_model, "model_epoch_23.hdf5")
    )
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    sequence = hg38.fetch(region="chr1:161200798-161201298")
    oh_sequence = one_hot_encode_sequence(sequence, expand_dim=False)
    class_idx = organoid_model.predict(oh_sequence[None]).argmax()
    pred_score = organoid_model.predict(oh_sequence[None]).max()
    explainer = Explainer(model=organoid_model, class_index=int(class_idx))
    gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (gradients_integrated * oh_sequence).astype(float)[0:200],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.02,
        y=0.8,
        s=f"NC: organoid {model_index_to_topic_name_organoid(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


@plot_w_cache(key="DE_NC_embr")
def plot_DE_NC_embr(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
    embryo_model = tf.keras.models.model_from_json(
        open(os.path.join(path_to_embryo_model, "model.json")).read(),
        custom_objects={"Functional": tf.keras.models.Model},
    )
    genome_dir = "/data/projects/c20/sdewin/resources/"
    embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))
    hg38 = crested.Genome(
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
        pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
    )
    sequence = hg38.fetch(region="chr1:161200798-161201298")
    oh_sequence = one_hot_encode_sequence(sequence, expand_dim=False)
    class_idx = embryo_model.predict(oh_sequence[None]).argmax()
    pred_score = embryo_model.predict(oh_sequence[None]).max()
    explainer = Explainer(model=embryo_model, class_index=int(class_idx))
    gradients_integrated = explainer.integrated_grad(X=oh_sequence[None]).squeeze()
    _ = logomaker.Logo(
        pd.DataFrame(
            (gradients_integrated * oh_sequence).astype(float)[0:200],
            columns=["A", "C", "G", "T"],
        ),
        ax=ax,
    )
    ax.spines[["right", "top"]].set_visible(False)
    _ = ax.text(
        x=0.02,
        y=0.8,
        s=f"NC: embryo {model_index_to_topic_name_embryo(class_idx).replace('_', ' ')}\n{round(float(pred_score), 3)}",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    return ax


def plot_wt_vista_enhancer(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    r = requests.get(
        "https://mouse.lbl.gov/api/containers/0017000100020001/image-asset/001700010002000100010002/default"
    )
    img = np.asarray(Image.open(io.BytesIO(r.content)))
    ax.imshow(img)
    ax.set_axis_off()


def plot_alt_vista_enhancer(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    r = requests.get(
        "https://mouse.lbl.gov/api/containers/0017000600010002/image-asset/001700060001000200010002/default"
    )
    img = np.asarray(Image.open(io.BytesIO(r.content)))
    ax.imshow(img)
    ax.set_axis_off()


def plot_nc_enhancer(
    fig,
    gs,
    y,
    x,
):
    ax = fig.add_subplot(gs[y[0] : y[1], x[0] : x[1]])
    img = np.asarray(
        Image.open(
            "../data_prep_new/images/65_exovo_day1_e1_Orthogonal Projection-01_wo_brightfield.png"
        )
    )
    ax.imshow(img.swapaxes(0, 1))
    ax.set_axis_off()


print("GENERATING FIGURE 2")

N_PIXELS_PER_GRID = 50

plt.style.use(
    "/data/projects/c20/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/paper.mplstyle"
)

ax_cache = {}
fig = plt.figure()
width, height = fig.get_size_inches()
n_w_pixels = fig.get_dpi() * width
n_h_pixels = fig.get_dpi() * height
ncols = int((n_w_pixels) // N_PIXELS_PER_GRID)
nrows = int((n_h_pixels) // N_PIXELS_PER_GRID)
gs = fig.add_gridspec(
    nrows, ncols, wspace=0.05, hspace=0.1, left=0.05, right=0.97, bottom=0.05, top=0.95
)
# ax for schematic
current_x = 0
current_y = 0
ax_panel_a = fig.add_subplot(
    gs[current_y : current_y + nrows // 5, current_x : current_x + nrows // 4]
)
ax_panel_a.set_axis_off()
current_x += nrows // 4 + 6
plot_ROC_vista_neural_tube(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + nrows // 5),
    x=(current_x, current_x + nrows // 5),
)
current_x += nrows // 5 + 3
plot_ROC_vista_facial_mesenchyme(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + nrows // 5),
    x=(current_x, current_x + nrows // 5),
)
current_x = 0
current_y += nrows // 5 + 2
plot_motif_cluster_leiden(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + ncols // 2),
    x=(current_x, current_x + ncols // 2),
)
current_x += ncols // 2 - 1
plot_motif_cluster_dataset(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + ncols // 4),
    x=(current_x, current_x + ncols // 4),
)
current_y += ncols // 4
plot_motif_cluster_method(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + ncols // 4),
    x=(current_x, current_x + ncols // 4),
)
current_y -= ncols // 4 - 1
current_x += ncols // 4
ax_m_ctx_1 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_1 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("0", ax_m_ctx_1, ax_m_dl_1)
current_y += 3
ax_m_ctx_2 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_2 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("1", ax_m_ctx_2, ax_m_dl_2)
current_y += 3
ax_m_ctx_3 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_3 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("2", ax_m_ctx_3, ax_m_dl_3)
current_y += 3
ax_m_ctx_4 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_4 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("6", ax_m_ctx_4, ax_m_dl_4)
current_y += 3
ax_m_ctx_5 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_5 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("31", ax_m_ctx_5, ax_m_dl_5)
current_y += 3
ax_m_ctx_6 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 1 : current_x + 6]
)
ax_m_dl_6 = fig.add_subplot(
    gs[current_y : current_y + 2, current_x + 7 : current_x + 12]
)
plot_top_motif_for_cluster("33", ax_m_ctx_6, ax_m_dl_6)
current_y += 4
current_x = ((3 * ncols) // 4) - 1
plot_wt_vista_enhancer(
    fig=fig, gs=gs, y=(current_y, current_y + 12), x=(current_x, current_x + 5)
)
current_x += 5
plot_alt_vista_enhancer(
    fig=fig, gs=gs, y=(current_y, current_y + 12), x=(current_x, current_x + 6)
)
current_x = 0
plot_DE_ref_org(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
current_y += 3
plot_DE_alt_org(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
current_y += 3
plot_DE_ref_embr(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
current_y += 3
plot_DE_alt_embr(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
current_y += 3
current_x = ((3 * ncols) // 4) - 2
plot_nc_enhancer(fig=fig, gs=gs, y=(current_y, current_y + 6), x=(current_x, ncols))
current_x = 0
plot_DE_NC_org(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
current_y += 3
plot_DE_NC_embr(
    fig=fig,
    gs=gs,
    y=(current_y, current_y + 2),
    x=(current_x, current_x + ((3 * ncols) // 4) - 2),
)
fig.tight_layout()
fig.savefig("Figure_2.png", transparent=False)
fig.savefig("Figure_2.pdf")



path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
organoid_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_organoid_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
genome_dir = "../../../../../resources"
organoid_model.load_weights(
    os.path.join(path_to_organoid_model, "model_epoch_23.hdf5")
)
hg38 = crested.Genome(
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
)
mm10 = crested.Genome(
    pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
    pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
)

VISTA_experiments = pd.read_table(
    "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
)
experiment_ids, sequences, metadata = [], [], []
for _, vista_experiment in tqdm(
    VISTA_experiments.iterrows(),
    total=len(VISTA_experiments),
):
    for result in get_sequence_and_metadata(
        vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
    ):
        if result is not None:
            experiment_ids.append(result[0])
            sequences.append(result[1])
            metadata.append(result[2])
# oh_sequences = np.array(
#    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
# )

prediction_score_organoid = np.load(
    "../data_prep_new/organoid_data/MODELS/VISTA_VALIDATION/prediction_score_organoid.npz"
)["prediction_score"]

ref_alt_to_data = prepare_ref_alt_data_vista()

ref_enhancer, alt_enhancer = "001700010002", "001700060001"
vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
    0
]
ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
class_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["class_idx"]
pred_score = prediction_score_organoid[ref_enhancer_idx, class_idx]
explainer = Explainer(model=organoid_model, class_index=int(class_idx))
oh_sequence = one_hot_encode_sequence(sequences[ref_enhancer_idx], expand_dim=False)
ref_gradients_integrated_organoid = explainer.integrated_grad(X=oh_sequence[None]).squeeze()

organoid_de = oh_sequence  * ref_gradients_integrated_organoid 


path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
embryo_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_embryo_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))
hg38 = crested.Genome(
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.fa")),
    pathlib.Path(os.path.join(genome_dir, "hg38/hg38.chrom.sizes")),
)
mm10 = crested.Genome(
    pathlib.Path(os.path.join(genome_dir, "mm10/mm10.fa")),
    pathlib.Path(os.path.join(genome_dir, "mm10/mm10.chrom.sizes")),
)
VISTA_experiments = pd.read_table(
    "../data_prep_new/validation_data/VISTA_VALIDATION/VISTA_experiments.tsv.gz"
)
experiment_ids, sequences, metadata = [], [], []
for _, vista_experiment in tqdm(
    VISTA_experiments.iterrows(),
    total=len(VISTA_experiments),
):
    for result in get_sequence_and_metadata(
        vista_experiment, {"Human": hg38, "Mouse": mm10}, 500
    ):
        if result is not None:
            experiment_ids.append(result[0])
            sequences.append(result[1])
            metadata.append(result[2])
# oh_sequences = np.array(
#    [one_hot_encode_sequence(s, expand_dim=False) for s in tqdm(sequences)]
# )
prediction_score_embryo = np.load(
    "../data_prep_new/embryo_data/MODELS/VISTA_VALIDATION/prediction_score_embryo.npz"
)["prediction_score"]
ref_alt_to_data = prepare_ref_alt_data_vista()
ref_enhancer, alt_enhancer = "001700010002", "001700060001"
vista_id = VISTA_experiments.query("exp_hier == @ref_enhancer")["vista_id"].values[
    0
]
ref_enhancer_idx = ref_alt_to_data[(ref_enhancer, alt_enhancer)]["ref_enhancer_idx"]
class_idx = prediction_score_embryo[ref_enhancer_idx].argmax()
pred_score = prediction_score_embryo[ref_enhancer_idx, class_idx]
explainer = Explainer(model=embryo_model, class_index=int(class_idx))
oh_sequence = one_hot_encode_sequence(sequences[ref_enhancer_idx], expand_dim=False)
ref_gradients_integrated_embryo = explainer.integrated_grad(X=oh_sequence[None]).squeeze()


embryo_de = oh_sequence  * ref_gradients_integrated_embryo 


from scipy.stats import pearsonr

print( pearsonr(embryo_de.sum(1), organoid_de.sum(1)))
