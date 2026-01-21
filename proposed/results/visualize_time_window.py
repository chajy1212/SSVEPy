# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

save_dir = "/home/brainlab/Workspace/jycha/SSVEP/result"
time_windows = [0.5, 1.0, 2.0, 3.0, 4.0]

# Session Split
ss_acc = [47.22, 74.74, 87.61, 91.06, 93.02]
ss_acc_std = [13.11, 16.71, 15.56, 15.82, 14.92]
ss_itr = [26.00, 53.72, 41.71, 31.51, 25.29]

# LOSO
loso_acc = [48.24, 81.18, 92.21, 94.81, 96.46]
loso_acc_std = [12.11, 14.65, 10.64, 9.83, 8.71]
loso_itr = [26.88, 66.32, 47.06, 34.25, 27.14]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy Plot
ax1.errorbar(time_windows, ss_acc, yerr=ss_acc_std, label='Session Split',
             fmt='-o', capsize=4, color='tab:blue', linewidth=2)
ax1.errorbar(time_windows, loso_acc, yerr=loso_acc_std, label='LOSO',
             fmt='-s', capsize=4, color='tab:red', linewidth=2)
ax1.set_xlabel('Time Window (s)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('(a) Accuracy vs. Time Window', fontsize=14)
ax1.set_xticks(time_windows)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# ITR Plot
ax2.plot(time_windows, ss_itr, label='Session Split', marker='o', color='tab:blue', linewidth=2)
ax2.plot(time_windows, loso_itr, label='LOSO', marker='s', color='tab:red', linewidth=2)
ax2.set_xlabel('Time Window (s)', fontsize=12)
ax2.set_ylabel('ITR (bits/min)', fontsize=12)
ax2.set_title('(b) ITR vs. Time Window', fontsize=14)
ax2.set_xticks(time_windows)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig('performance_trend.png', dpi=300)
plt.show()
plt.close