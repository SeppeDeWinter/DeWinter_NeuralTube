
"""

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/HEPG2_FOXA2_ENCFF466FCB.bed.gz \
    -a ../data_prep_new/organoid_data/ATAC/topics/training_data/Topic_33.bed -F 0.4 -wa \
  > organoid_topic_33_w_HEPG2.bed

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/A549_FOXA2_ENCFF686MSH.bed.gz \
    -a ../data_prep_new/organoid_data/ATAC/topics/training_data/Topic_33.bed -F 0.4 -wa \
  > organoid_topic_33_w_A549.bed

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/HEPG2_FOXA2_ENCFF466FCB.bed.gz \
    -a ../data_prep_new/embryo_data/ATAC/topics/training_data/Topic_34.bed -F 0.4 -wa \
  > embryo_topic_34_w_HEPG2.bed

bedtools intersect \
    -b ../data_prep_new/validation_data/FOXA2_ChIP/A549_FOXA2_ENCFF686MSH.bed.gz \
    -a ../data_prep_new/embryo_data/ATAC/topics/training_data/Topic_34.bed -F 0.4 -wa \
  > embryo_topic_34_w_A549.bed

"""

import numpy as np
from dataclasses import dataclass
import h5py
from typing import Self
from tqdm import tqdm
import os
import pandas as pd
from crested._genome import Genome
from crested.utils._seq_utils import one_hot_encode_sequence
from tangermeme.tools.fimo import fimo
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import matplotlib
import modiscolite
import logomaker
import seaborn as sns

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
                "ppm does not match onehot\n"
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

organoid_fp_regions = pd.read_table(
    "../data_prep_new/organoid_data/ATAC/topics/training_data/Topic_33.bed",
    names = ["chrom", "start", "end"]
)
embryo_fp_regions = pd.read_table(
    "../data_prep_new/embryo_data/ATAC/topics/training_data/Topic_34.bed",
    names = ["chrom", "start", "end", "name", "score"]
)

organoid_fp_w_hepg2 = set(pd.read_table(
    "organoid_topic_33_w_HEPG2.bed",
    names = ["chrom", "start", "end"]
).apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values)
organoid_fp_w_a549 = set(pd.read_table(
    "organoid_topic_33_w_A549.bed",
    names = ["chrom", "start", "end"]
).apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values)
embryo_fp_w_hepg2 = set(pd.read_table(
    "embryo_topic_34_w_HEPG2.bed",
    names = ["chrom", "start", "end", "name", "score"]
).apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values)
embryo_fp_w_a549 = set(pd.read_table(
    "embryo_topic_34_w_A549.bed",
    names = ["chrom", "start", "end", "name", "score"]
).apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values)


hg38 = Genome(
    fasta="../../../../../resources/hg38/hg38.fa",
    chrom_sizes="../../../../../resources/hg38/hg38.chrom.sizes"
)

organoid_fp_oh = np.array(
    [
        one_hot_encode_sequence(hg38.fetch(chrom, start, end), expand_dim=False)
        for _, (chrom, start, end) in tqdm(organoid_fp_regions.iterrows(), total = len(organoid_fp_regions))
    ]
)

embryo_fp_oh = np.array(
    [
        one_hot_encode_sequence(hg38.fetch(chrom, start, end), expand_dim=False)
        for _, (chrom, start, end, _, _) in tqdm(embryo_fp_regions.iterrows(), total = len(embryo_fp_regions))
    ]
)

fox_motifs_names = pattern_metadata.query("hier_cluster == 6").index.to_list()

fox_patterns = {
    n: p.ppm[pattern_metadata.loc[n].ic_start : pattern_metadata.loc[n].ic_stop].T 
    for n, p in zip(all_pattern_names, all_patterns)
    if n in fox_motifs_names
}

fox_hits_organoid = pd.concat(
        fimo(motifs = fox_patterns, sequences = organoid_fp_oh.swapaxes(1,2), threshold=0.0001)).reset_index(drop = True)
fox_hits_embryo = pd.concat(
        fimo(motifs = fox_patterns, sequences = embryo_fp_oh.swapaxes(1,2), threshold=0.0001)).reset_index(drop = True)

fox_hits_organoid["sequence_name"] = organoid_fp_regions.loc[
    fox_hits_organoid["sequence_name"]
].apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values

fox_hits_embryo["sequence_name"] = embryo_fp_regions.loc[
    fox_hits_embryo["sequence_name"]
].apply(lambda row: row.chrom + ":" + str(row.start) + "-" + str(row.end), axis = 1).values

fox_hits_organoid["overlap_w_hepg2"] = [r in organoid_fp_w_hepg2 for r in fox_hits_organoid["sequence_name"]]
fox_hits_organoid["overlap_w_a549"] = [r in organoid_fp_w_a549 for r in fox_hits_organoid["sequence_name"]]

fox_hits_embryo["overlap_w_hepg2"] = [r in embryo_fp_w_hepg2 for r in fox_hits_embryo["sequence_name"]]
fox_hits_embryo["overlap_w_a549"] = [r in embryo_fp_w_a549 for r in fox_hits_embryo["sequence_name"]]

def get_number_of_non_overlapping_sites(
    df: pd.DataFrame, max_overlap: int, broadcast=False
):
    n = sum(np.diff(df["start"].sort_values()) > max_overlap) + 1
    if not broadcast:
        return n
    else:
        return [n for _ in range(len(df))]

organoid_number_sites_specific = (
    fox_hits_organoid.query(
        "not overlap_w_hepg2 & not overlap_w_a549"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

organoid_number_sites_general = (
    fox_hits_organoid.query(
        "overlap_w_hepg2 | overlap_w_a549"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

embryo_number_sites_specific = (
    fox_hits_embryo.query(
        "not overlap_w_hepg2 & not overlap_w_a549"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
)

embryo_number_sites_general = (
    fox_hits_embryo.query(
        "overlap_w_hepg2 | overlap_w_a549"
    )
    .groupby("sequence_name")
    .apply(lambda x: get_number_of_non_overlapping_sites(x, 10))
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
    fox_hits_organoid.set_index("sequence_name").loc[
        organoid_number_sites_general.index[organoid_number_sites_general == n_sites]] \
        .reset_index() \
        .sort_values("p-value") \
        .drop_duplicates(["sequence_name"]) \
        .sort_values("motif_name") \
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
    fox_hits_organoid.set_index("sequence_name").loc[
        organoid_number_sites_specific.index[organoid_number_sites_specific == n_sites]] \
        .reset_index() \
        .sort_values("p-value") \
        .drop_duplicates(["sequence_name"]) \
        .sort_values("motif_name") \
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
x_current = 8
y_current = 38
ax_bar_n_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 6]
)

pval = mannwhitneyu(
    organoid_number_sites_general, organoid_number_sites_specific
).pvalue

max_sites = 10
for n in range(max_sites):
    ax_bar_n_spec.bar(
        x=n,
        height=sum(organoid_number_sites_specific == n + 1)
        / len(organoid_number_sites_specific),
        width=0.5,
        align="edge",
        color="lightcoral",
        edgecolor="black",
    )
    ax_bar_n_spec.bar(
        x=n + 0.5,
        height=sum(embryo_number_sites_specific == n + 1)
        / len(embryo_number_sites_specific),
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
        height=sum(organoid_number_sites_general == n + 1)
        / len(organoid_number_sites_general),
        width=0.5,
        align="edge",
        color="lightcoral",
        edgecolor="black",
    )
    ax_bar_n_general.bar(
        x=n + 0.5,
        height=sum(embryo_number_sites_general == n + 1)
        / len(embryo_number_sites_general),
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
ax_line = fig.add_subplot(
    gs[y_current : nrows, x_current : 27]
)
ax_line.plot(
    np.arange(max_sites) + 1,
    [
        -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_specific.index[organoid_number_sites_specific == (n + 1)]]["p-value"]
            + 1e-6
        ).mean()
       for n in range(max_sites)
    ],
    ls = "dotted",
    label="organoid specific",
    color = "lightcoral"
)

for n in range(max_sites):
    v = -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_specific.index[organoid_number_sites_specific == (n + 1)]]["p-value"]
            + 1e-6
        )
    v_other = -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_general.index[organoid_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        )
    print(f"{n}: {len(v)}  {len(v_other)}") 
    ax_line.text(
        n + 1,
        4.57,
        "*" if mannwhitneyu(v, v_other).pvalue < 1e-6 else "",
        color = "lightcoral"
    )
    CI = 1.96 * v.std() / np.sqrt(len(v))
    ax_line.plot(
        [n + 1, n + 1],
        [v.mean() - CI, v.mean() + CI],
        color = "black"
    )

ax_line.plot(
    np.arange(max_sites) + 1,
    [
        -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_general.index[organoid_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        ).mean()
       for n in range(max_sites)
    ],
    label="organoid general",
    color = "lightcoral"
)
for n in range(max_sites):
    v = -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_general.index[organoid_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        )
    CI = 1.96 * v.std() / np.sqrt(len(v))
    ax_line.plot(
        [n + 1, n + 1],
        [v.mean() - CI, v.mean() + CI],
        color = "black"
    )
ax_line.plot(
    np.arange(max_sites) + 1,
    [
        -np.log10(
            fox_hits_embryo.set_index("sequence_name") \
                .loc[embryo_number_sites_specific.index[embryo_number_sites_specific == (n + 1)]]["p-value"]
            + 1e-6
        ).mean()
       for n in range(max_sites)
    ],
    ls = "dotted",
    label="embryo specific",
    color = "deepskyblue"
)
for n in range(max_sites):
    v = -np.log10(
            fox_hits_embryo.set_index("sequence_name") \
                .loc[embryo_number_sites_specific.index[embryo_number_sites_specific == (n + 1)]]["p-value"]
            + 1e-6
        )
    v_other = -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_general.index[organoid_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        ) 
    ax_line.text(
        n + 1,
        4.55,
        "*" if mannwhitneyu(v, v_other).pvalue < 1e-6 else "",
        color = "deepskyblue"
    )
    CI = 1.96 * v.std() / np.sqrt(len(v))
    ax_line.plot(
        [n + 1, n + 1],
        [v.mean() - CI, v.mean() + CI],
        color = "black"
    )
ax_line.plot(
    np.arange(max_sites) + 1,
    [
        -np.log10(
            fox_hits_embryo.set_index("sequence_name") \
                .loc[embryo_number_sites_general.index[embryo_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        ).mean()
       for n in range(max_sites)
    ],
    label="embryo general",
    color = "deepskyblue"
)
for n in range(max_sites):
    v = -np.log10(
            fox_hits_embryo.set_index("sequence_name") \
                .loc[embryo_number_sites_general.index[embryo_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
        )
    CI = 1.96 * v.std() / np.sqrt(len(v))
    ax_line.plot(
        [n + 1, n + 1],
        [v.mean() - CI, v.mean() + CI],
        color = "black"
    )
ax_line.legend()
ax_line.grid(True)
ax_line.set_axisbelow(True)
ax_line.set_ylabel("$-log_{10}(pval)$")
ax_line.set_xlabel("Number of FOX sites")
"""
max_sites = 5
y_current = 38
x_current = 17
ax_bplot_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 4]
)
ax_bplot_general = fig.add_subplot(gs[nrows - 4 : nrows, x_current : x_current + 4])
a = -np.log10(
    fox_hits_organoid.set_index("sequence_name") \
        .loc[organoid_number_sites_general.index[organoid_number_sites_general == 1]]["p-value"]
    + 1e-6
)
b = -np.log10(
    fox_hits_organoid.set_index("sequence_name") \
        .loc[organoid_number_sites_specific.index[organoid_number_sites_specific == 1]]["p-value"]
    + 1e-6
)
pval = mannwhitneyu(a, b).pvalue
ax_bplot_spec.boxplot(
    [
        -np.log10(
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_specific.index[organoid_number_sites_specific == (n + 1)]]["p-value"]
            + 1e-6
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
            fox_hits_organoid.set_index("sequence_name") \
                .loc[organoid_number_sites_general.index[organoid_number_sites_general == (n + 1)]]["p-value"]
            + 1e-6
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
    fox_hits_embryo.set_index("sequence_name") \
        .loc[embryo_number_sites_general.index[embryo_number_sites_general == 1]]["p-value"]
    + 1e-6
)
b = -np.log10(
    fox_hits_embryo.set_index("sequence_name") \
        .loc[embryo_number_sites_specific.index[embryo_number_sites_specific == 1]]["p-value"]
    + 1e-6
)
pval = mannwhitneyu(a, b).pvalue
ax_bplot_spec = fig.add_subplot(
    gs[y_current : y_current + 4, x_current : x_current + 4]
)
ax_bplot_general = fig.add_subplot(gs[nrows - 4 : nrows, x_current : x_current + 4])
ax_bplot_spec.boxplot(
    [
        -np.log10(fox_hits_embryo.set_index("sequence_name") \
            .loc[embryo_number_sites_specific.index[embryo_number_sites_specific == (n + 1)]]["p-value"]
        + 1e-6
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
        -np.log10(fox_hits_embryo.set_index("sequence_name") \
            .loc[embryo_number_sites_general.index[embryo_number_sites_general == (n + 1)]]["p-value"]
        + 1e-6
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
"""
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
fig.savefig("figure_3_fox_rerun.png",  transparent=False)
fig.savefig("figure_3_fox_rerun.pdf")
plt.close(fig)


