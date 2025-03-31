```python

import scanpy as sc
import matplotlib.pyplot as plt
import json
import matplotlib
from scipy import stats
from itertools import combinations
import numpy as np


adata_organoid = sc.read_h5ad("../../figure_1/adata_organoid.h5ad")
color_dict = json.load(open("../../color_maps.json"))

matplotlib.rcParams['pdf.fonttype'] = 42

fig, axs = plt.subplots(ncols = 3, figsize = (12, 4))
for i, line in enumerate(set(adata_organoid.obs["SNG.BEST.GUESS"])):
  ax = axs[i]
  ax.set_axis_off()
  ax.scatter(
    adata_organoid[adata_organoid.obs["SNG.BEST.GUESS"] == line].obsm["X_umap"][:, 0],
    adata_organoid[adata_organoid.obs["SNG.BEST.GUESS"] == line].obsm["X_umap"][:, 1],
    c = [
      color_dict["all_cells"][x] for x in adata_organoid[adata_organoid.obs["SNG.BEST.GUESS"] == line].obs.COMMON_ANNOTATION
    ],
    s = 0.5
  )
  _ = ax.set_title(line)
fig.tight_layout()
fig.savefig("organoid_umap_per_cell_line.png")
fig.savefig("organoid_umap_per_cell_line.pdf")
plt.close(fig)

avg_exp_per_line = {
  line: adata_organoid[adata_organoid.obs["SNG.BEST.GUESS"] == line] \
    .to_df(layer = "log_cpm") \
    .groupby(adata_organoid[adata_organoid.obs["SNG.BEST.GUESS"] == line].obs.COMMON_ANNOTATION).mean()
  for line in set(adata_organoid.obs["SNG.BEST.GUESS"])
}

line_combos = list(combinations(avg_exp_per_line.keys(), 2))

genes = sc.pp.highly_variable_genes(adata_organoid, layer = "log_cpm", inplace = False).query("highly_variable").index

fig, axs = plt.subplots(
  ncols = len(line_combos), nrows = len(set(adata_organoid.obs["COMMON_ANNOTATION"])),
  figsize = (len(line_combos) * 2, len(set(adata_organoid.obs["COMMON_ANNOTATION"])) * 2 ),
  sharex = True, sharey = True
)
for i, (line_1, line_2) in enumerate(line_combos):
  for j, cell_type in enumerate(set(adata_organoid.obs["COMMON_ANNOTATION"])):
    ax = axs[j, i]
    if i == 0:
      _ = ax.set_ylabel(cell_type)
    if j == 0:
      _ = ax.set_title(f"{line_1} {line_2}")
    _ = ax.scatter(
      x = avg_exp_per_line[line_1].loc[cell_type, genes],
      y = avg_exp_per_line[line_2].loc[cell_type, genes],
      color = "black",
      s = 0.5
    )
    _ = ax.text(
      x = 0.1,
      y = 0.8,
      s = np.round(
        stats.pearsonr(avg_exp_per_line[line_1].loc[cell_type, genes], avg_exp_per_line[line_2].loc[cell_type, genes]).statistic,
        2
      ),
      transform = ax.transAxes
    )
fig.tight_layout()
fig.savefig("line_corrs.png")
fig.savefig("line_corss.pdf")
plt.close(fig)

annotations = list(set(adata_organoid.obs.COMMON_ANNOTATION))

for line_1, line_2 in line_combos:
  fig, axs = plt.subplots(
    ncols = len(annotations), nrows = len(annotations),
    figsize = (len(annotations) * 2, len(annotations) * 2 ),
    sharex = True, sharey = True
  )
  for i, cell_type_1 in enumerate(annotations):
    for j, cell_type_2 in enumerate(annotations):
      ax = axs[j, i]
      if i == 0:
        _ = ax.set_ylabel(f"{cell_type_2} {line_2[0:3]}")
      if j == 0:
        _ = ax.set_title(f"{cell_type_1} {line_1[0:3]}")
      _ = ax.scatter(
        x = avg_exp_per_line[line_1].loc[cell_type_1, genes],
        y = avg_exp_per_line[line_2].loc[cell_type_2, genes],
        color = "black" if cell_type_1 == cell_type_2 else "darkgray",
        s = 0.5
      )
      _ = ax.text(
        x = 0.1,
        y = 0.8,
        s = np.round(
          stats.pearsonr(
            avg_exp_per_line[line_1].loc[cell_type_1, genes],
            avg_exp_per_line[line_2].loc[cell_type_2, genes]
          ).statistic,
          2
        ),
        transform = ax.transAxes
      )
  fig.tight_layout()
  fig.savefig(f"line_corrs_pw_{line_1}_{line_2}.png")
  fig.savefig(f"line_corrs_pw_{line_1}_{line_2}.pdf")
  plt.close(fig)


```
