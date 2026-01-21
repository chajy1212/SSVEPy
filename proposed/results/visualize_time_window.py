# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt


time_windows = [0.5, 1.0, 2.0, 3.0, 4.0]
N_subjects = 54

# Session Split
ss_acc = [47.22, 74.74, 87.61, 91.06, 93.02]
ss_acc_std = [13.11, 16.71, 15.56, 15.82, 14.92]
ss_itr = [26.00, 53.72, 41.71, 31.51, 25.29]
ss_itr_std = [24.73, 29.42, 15.20, 10.58, 8.00]

# LOSO
loso_acc = [48.24, 81.18, 92.21, 94.81, 96.46]
loso_acc_std = [12.11, 14.65, 10.64, 9.83, 8.71]
loso_itr = [26.88, 66.32, 47.06, 34.25, 27.14]
loso_itr_std = [25.19, 29.69, 12.76, 8.00, 5.54]

# 95% (CI): 1.96 * (SD / sqrt(N))
ss_acc_ci = 1.96 * (ss_acc_std / np.sqrt(N_subjects))
loso_acc_ci = 1.96 * (loso_acc_std / np.sqrt(N_subjects))
ss_itr_ci = 1.96 * (ss_itr_std / np.sqrt(N_subjects))
loso_itr_ci = 1.96 * (loso_itr_std / np.sqrt(N_subjects))

save_dir = "/home/brainlab/Workspace/jycha/SSVEP/result"

c_ss = '#3D74B6'
c_loso = '#E62727'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Accuracy Plot
ax1.errorbar(time_windows, ss_acc, yerr=ss_acc_ci, label='Session Split',
             fmt='-o', capsize=3, color=c_ss, linewidth=1.5, markersize=4)
ax1.errorbar(time_windows, loso_acc, yerr=loso_acc_ci, label='LOOCV',
             fmt='-s', capsize=3, color=c_loso, linewidth=1.5, markersize=4)

ax1.set_xlabel('Time Window (s)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('(a) Accuracy vs. Time Window', fontsize=13)
ax1.set_xticks(time_windows)

ax1.set_ylim(40, 100)
ax1.set_yticks(np.arange(40, 101, 10))
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=9, loc='lower right')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ITR Plot
ax2.errorbar(time_windows, ss_itr, yerr=ss_itr_ci, label='Session Split',
             fmt='-o', capsize=3, color=c_ss, linewidth=1.5, markersize=4)
ax2.errorbar(time_windows, loso_itr, yerr=loso_itr_ci, label='LOOCV',
             fmt='-s', capsize=3, color=c_loso, linewidth=1.5, markersize=4)

ax2.set_xlabel('Time Window (s)', fontsize=12)
ax2.set_ylabel('ITR (bits/min)', fontsize=12)
ax2.set_title('(b) ITR vs. Time Window', fontsize=13)
ax2.set_xticks(time_windows)

ax2.set_ylim(10, 80)
ax2.set_yticks(np.arange(10, 81, 10))
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=9)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

save_path = os.path.join(save_dir, 'time_window_analysis.eps')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[Success] Figure saved to: {save_path}")

plt.show()
plt.close