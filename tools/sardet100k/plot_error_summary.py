"""Draw the confusion matrix from error_gallery.py's report.json. Usage: plot_error_summary.py REPORT_JSON OUT_PNG [--score score_0.3]"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RAMP = LinearSegmentedColormap.from_list('blue', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#104281'])

parser = argparse.ArgumentParser()
parser.add_argument('report')
parser.add_argument('out')
parser.add_argument('--score', default='score_0.3')
args = parser.parse_args()
point = json.load(open(args.report, encoding='utf8'))['operating_points'][args.score]
labels, matrix = point['confusion']['labels'], np.array(point['confusion']['matrix'])
shade = np.log10(matrix + 1)

fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=160)
ax.imshow(shade, cmap=RAMP, vmin=0, vmax=shade.max())
ax.set_xticks(range(len(labels)), ['missed (FN)' if l == 'background' else l for l in labels], rotation=30, ha='right')
ax.set_yticks(range(len(labels)), ['not a GT (FP)' if l == 'background' else l for l in labels])
ax.set_xlabel('Predicted')
ax.set_ylabel('Ground truth')
ax.set_title(f"SARDet-100K val, E000 baseline: score >= {point['score']}, IoU >= {point['iou']}\nCell colour is log-scaled; each cell shows its count", fontsize=11, loc='left')
for i in range(len(labels)):
    total = matrix[i].sum()
    for j in range(len(labels)):
        value = matrix[i, j]
        if i == len(labels) - 1 and j == len(labels) - 1:
            continue
        share = f'\n{value / total:.1%}' if i < len(labels) - 1 and value else ''
        ink = 'white' if shade[i, j] / shade.max() > .55 else '#222222'
        ax.text(j, i, f'{value}{share}', ha='center', va='center', color=ink if value else '#a0a0a0', fontsize=9)
ax.set_xticks(np.arange(-.5, len(labels)), minor=True)
ax.set_yticks(np.arange(-.5, len(labels)), minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig(args.out)
