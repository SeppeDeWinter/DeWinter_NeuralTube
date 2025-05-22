
import os
import tensorflow as tf
from sklearn import metrics
import pickle
import gzip
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

path_to_organoid_model = "../data_prep_new/organoid_data/MODELS/"
organoid_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_organoid_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
organoid_model.load_weights(
    os.path.join(path_to_organoid_model, "model_epoch_23.hdf5")
)

path_to_embryo_model = "../data_prep_new/embryo_data/MODELS/"
embryo_model = tf.keras.models.model_from_json(
    open(os.path.join(path_to_embryo_model, "model.json")).read(),
    custom_objects={"Functional": tf.keras.models.Model},
)
embryo_model.load_weights(os.path.join(path_to_embryo_model, "model_epoch_36.hdf5"))


organoid_non_augmented_data = pickle.load(
    open("../data_prep_new/organoid_data/MODELS/nonAugmented_data_dict.pkl", "rb")
)

embryo_non_augmented_data = pickle.load(
    gzip.open("../data_prep_new/embryo_data/MODELS/nonAugmented_data_dict.pkl.gz", "rb")
)

y_score_organoid = organoid_model.predict(organoid_non_augmented_data["test_data"])
y_true_organoid = organoid_non_augmented_data["y_test"]

y_score_embryo = embryo_model.predict(embryo_non_augmented_data["test_data"])
y_true_embryo = embryo_non_augmented_data["y_test"]

color_dict = json.load(open("../color_maps.json"))

organoid_topics = np.array(
    [
        6, 4, 23, 24, 13, 2,
        33, 38, 36, 54, 48,
        62, 60, 65, 59, 58
    ]
) - 1

colors_organoid = [
    *np.repeat(color_dict["cell_type_classes"]["neuron"], 6),
    *np.repeat(color_dict["cell_type_classes"]["progenitor_dv"], 5),
    *np.repeat(color_dict["cell_type_classes"]["neural_crest"], 5)
]


classes_organoid = [
    *np.repeat("neuron", 6),
    *np.repeat("progenitor_dv", 5),
    *np.repeat("neural_crest", 5)
]

classes_organoid = np.array(classes_organoid)

embryo_topics = np.array(
    [
        10, 8, 13, 24, 18, 29,
        34, 38, 79, 88, 58,
        61, 59, 31, 62, 70, 52, 71,
        103, 105, 94, 91
    ]
) - 1

colors_embryo = [
    *np.repeat(color_dict["cell_type_classes"]["neuron"], 6),
    *np.repeat(color_dict["cell_type_classes"]["progenitor_dv"], 5),
    *np.repeat(color_dict["cell_type_classes"]["progenitor_ap"], 7),
    *np.repeat(color_dict["cell_type_classes"]["neural_crest"], 4)
]

classes_embryo = [
    *np.repeat("neuron", 6),
    *np.repeat("progenitor_dv", 5),
    *np.repeat("progenitor_ap", 7),
    *np.repeat("neural_crest", 4)
]

classes_embryo = np.array(classes_embryo)

prec_organoid, rec_organoid = [], []
pr_auc_organoid = []
fpr_organoid, tpr_organoid = [], []
roc_auc_organoid = []
for topic in organoid_topics:
    p, r, _ = metrics.precision_recall_curve(
        y_true_organoid[:, topic], y_score_organoid[:, topic],
        drop_intermediate = False
    )
    pr_auc_organoid.append(metrics.auc(r, p))
    f, t, _ = metrics.roc_curve(
        y_true_organoid[:, topic], y_score_organoid[:, topic],
        drop_intermediate = False
    )
    roc_auc_organoid.append(metrics.auc(f, t))
    prec_organoid.append(p)
    rec_organoid.append(r)
    fpr_organoid.append(f)
    tpr_organoid.append(t)

pr_auc_organoid = np.array(pr_auc_organoid)
roc_auc_organoid = np.array(roc_auc_organoid)


prec_embryo, rec_embryo = [], []
pr_auc_embryo = []
fpr_embryo, tpr_embryo = [], []
roc_auc_embryo = []
for topic in embryo_topics:
    p, r, _ = metrics.precision_recall_curve(
        y_true_embryo[:, topic], y_score_embryo[:, topic],
        drop_intermediate = False
    )
    pr_auc_embryo.append(metrics.auc(r, p))
    f, t, _ = metrics.roc_curve(
        y_true_embryo[:, topic], y_score_embryo[:, topic],
        drop_intermediate = False
    )
    roc_auc_embryo.append(metrics.auc(f, t))
    prec_embryo.append(p)
    rec_embryo.append(r)
    fpr_embryo.append(f)
    tpr_embryo.append(t)

pr_auc_embryo = np.array(pr_auc_embryo)
roc_auc_embryo = np.array(roc_auc_embryo)



matplotlib.rcParams["pdf.fonttype"] = 42

fig, ax = plt.subplots()
for (rec, prec), color in zip(zip(rec_organoid, prec_organoid), colors_organoid):
    _ = ax.plot(
        rec, prec, color = color
    )
for i, c in enumerate(np.unique(classes_organoid)):
    ax.text(
        0.6,
        1 - (i / 10),
        f"AUpr {c} = {np.round(pr_auc_organoid[np.where(classes_organoid == c)[0]].mean(), 3)}",
        color = color_dict["cell_type_classes"][c]
    )
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xlabel("Recall")
_ = ax.set_ylabel("Precision")
fig.savefig("PR_organoid.pdf")
plt.close(fig)

fig, ax = plt.subplots()
for (fpr, tpr), color in zip(zip(fpr_organoid, tpr_organoid), colors_organoid):
    _ = ax.plot(
        fpr, tpr, color = color
    )
for i, c in enumerate(np.unique(classes_organoid)):
    ax.text(
        0.6,
        0.7 - (i / 10),
        f"AUroc {c} = {np.round(roc_auc_organoid[np.where(classes_organoid == c)[0]].mean(), 3)}",
        color = color_dict["cell_type_classes"][c]
    )
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xlabel("FPR")
_ = ax.set_ylabel("TPR")
fig.savefig("roc_organoid.pdf")
plt.close(fig)

fig, ax = plt.subplots()
for (rec, prec), color in zip(zip(rec_embryo, prec_embryo), colors_embryo):
    _ = ax.plot(
        rec, prec, color = color
    )
for i, c in enumerate(np.unique(classes_embryo)):
    ax.text(
        0.6,
        1 - (i / 10),
        f"AUpr {c} = {np.round(pr_auc_embryo[np.where(classes_embryo == c)[0]].mean(), 3)}",
        color = color_dict["cell_type_classes"][c]
    )
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xlabel("Recall")
_ = ax.set_ylabel("Precision")
fig.savefig("PR_embryo.pdf")
plt.close(fig)

fig, ax = plt.subplots()
for (fpr, tpr), color in zip(zip(fpr_embryo, tpr_embryo), colors_embryo):
    _ = ax.plot(
        fpr, tpr, color = color
    )
for i, c in enumerate(np.unique(classes_embryo)):
    ax.text(
        0.6,
        0.7 - (i / 10),
        f"AUroc {c} = {np.round(roc_auc_embryo[np.where(classes_embryo == c)[0]].mean(), 3)}",
        color = color_dict["cell_type_classes"][c]
    )
ax.grid(True)
ax.set_axisbelow(True)
_ = ax.set_xlabel("FPR")
_ = ax.set_ylabel("TPR")
fig.savefig("roc_embryo.pdf")
plt.close(fig)


