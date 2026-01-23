# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt


time_windows = [0.5, 1.0, 2.0, 3.0, 4.0]
N_subjects = 10

loso_acc = [0, 0, 0, 0, 0]
loso_acc_std = [0, 0, 0, 0, 0]
loso_itr = [0, 0, 0, 0, 0]
loso_itr_std = [0, 0, 0, 0, 0]

# 95% (CI): 1.96 * (SD / sqrt(N))
loso_acc_ci = 1.96 * (loso_acc_std / np.sqrt(N_subjects))
loso_itr_ci = 1.96 * (loso_itr_std / np.sqrt(N_subjects))

save_dir = "/home/brainlab/Workspace/jycha/SSVEP/result"

c_loso = '#E62727'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Accuracy Plot
ax1.errorbar(time_windows, loso_acc, yerr=loso_acc_ci, label='LOOCV',
             fmt='-s', capsize=3, color=c_loso, linewidth=1.5, markersize=4)

ax1.set_xlabel('Time Window (s)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('(a) Accuracy vs. Time Window', fontsize=13)
ax1.set_xticks(time_windows)

ax1.set_ylim(0, 100)
ax1.set_yticks(np.arange(0, 101, 10))
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=9, loc='lower right')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ITR Plot
ax2.errorbar(time_windows, loso_itr, yerr=loso_itr_ci, label='LOOCV',
             fmt='-s', capsize=3, color=c_loso, linewidth=1.5, markersize=4)

ax2.set_xlabel('Time Window (s)', fontsize=12)
ax2.set_ylabel('ITR (bits/min)', fontsize=12)
ax2.set_title('(b) ITR vs. Time Window', fontsize=13)
ax2.set_xticks(time_windows)

ax2.set_ylim(0, 100)
ax2.set_yticks(np.arange(0, 101, 10))
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=9)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

save_path = os.path.join(save_dir, 'time_window_analysis_nakanishi.eps')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[Success] Figure saved to: {save_path}")

plt.show()
plt.close