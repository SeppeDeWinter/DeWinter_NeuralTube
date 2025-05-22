import numpy as np
import matplotlib.pyplot as plt
import logomaker
import pathlib
import os
import crested
import tensorflow as tf
import pandas as pd
import json

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

enhancer_A2_region = "chr18:77367159-77367659"

attr_a2_organoid, oh_a2_organoid = crested.tl.contribution_scores(
    input=enhancer_A2_region,
    target_idx=32,
    model=organoid_model,
    genome=hg38
)
attr_a2_embryo, oh_a2_embryo = crested.tl.contribution_scores(
    input=enhancer_A2_region,
    target_idx=33,
    model=embryo_model,
    genome=hg38
)

p_embryo = crested.tl.predict(enhancer_A2_region, embryo_model, hg38)
p_organoid = crested.tl.predict(enhancer_A2_region, organoid_model, hg38)

organoid_progenitor_topics_to_show = np.array([33, 38, 36, 54, 48]) - 1
embryo_progenitor_topics_to_show = np.array([34, 38, 79, 88, 58]) - 1


N_PIXELS_PER_GRID = 50

plt.style.use(
    "/data/projects/c20/sdewin/PhD/papers/NeuralTubeOrganoids_DeWinter_etal_2023/figures_new/paper.mplstyle"
)

color_dict = json.load(open("../color_maps.json"))

fig = plt.figure()
width, height = fig.get_size_inches()
n_w_pixels = fig.get_dpi() * width
n_h_pixels = fig.get_dpi() * height
ncols = int((n_w_pixels) // N_PIXELS_PER_GRID)
nrows = int((n_h_pixels) // N_PIXELS_PER_GRID)
gs = fig.add_gridspec(
    nrows, ncols, wspace=0.05, hspace=0.1, left=0.05, right=0.97, bottom=0.05, top=0.95
)
y_current = 38
ax_de_a2_organoid = fig.add_subplot(
    gs[y_current: y_current + 3, 0: 23]
)
_ = logomaker.Logo(
    pd.DataFrame(
        (attr_a2_organoid * oh_a2_organoid).squeeze()
        .astype(float)[80:400],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_a2_organoid,
)
ax_pred_organoid = fig.add_subplot(
    gs[y_current: y_current + 4, 25: 29]
)
ax_pred_organoid.barh(
    np.arange(organoid_progenitor_topics_to_show.shape[0]),
    p_organoid[0, organoid_progenitor_topics_to_show[::-1]],
    color = [
        color_dict["progenitor_topics"][f"organoid_{c + 1}"]
        for c in organoid_progenitor_topics_to_show[::-1]
    ],
    lw = 1, edgecolor = "black"
)
ax_pred_organoid.set_yticks(
    np.arange(organoid_progenitor_topics_to_show.shape[0]),
    organoid_progenitor_topics_to_show[::-1] + 1
)
ax_pred_organoid.set_xticks(
    np.arange(0, 7, 2) / 10
)
ax_pred_organoid.grid(True)
ax_pred_organoid.set_axisbelow(True)
ax_de_a2_embryo = fig.add_subplot(
    gs[y_current + 5: y_current + 8, 0: 23]
)
_ = logomaker.Logo(
    pd.DataFrame(
        (attr_a2_embryo * oh_a2_embryo).squeeze()
        .astype(float)[80:400],
        columns=["A", "C", "G", "T"],
    ),
    ax=ax_de_a2_embryo,
)
ax_pred_embryo = fig.add_subplot(
    gs[y_current + 5: y_current + 9, 25: 29]
)
ax_pred_embryo.barh(
    np.arange(embryo_progenitor_topics_to_show.shape[0]),
    p_embryo[0, embryo_progenitor_topics_to_show[::-1]],
    color = [
        color_dict["progenitor_topics"][f"embryo_{c + 1}"]
        for c in embryo_progenitor_topics_to_show[::-1]
    ],
    lw = 1, edgecolor = "black"
)
ax_pred_embryo.set_yticks(
    np.arange(embryo_progenitor_topics_to_show.shape[0]),
    embryo_progenitor_topics_to_show[::-1] + 1
)
ax_pred_embryo.set_xticks(
    np.arange(0, 7, 2) / 10
)
ax_pred_embryo.grid(True)
ax_pred_embryo.set_axisbelow(True)
fig.savefig("figure_3_w_A2.png",  transparent=False)
fig.savefig("figure_3_w_A2.pdf")
plt.close(fig)


