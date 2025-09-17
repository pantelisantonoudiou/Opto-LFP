# power_ratio_plots.py
# -*- coding: utf-8 -*-
"""
Quick seaborn plots for power ratio (stim / baseline).

Assumptions & Data Contract
---------------------------
- Input CSV (default: power_ratio_results.csv) is in root_dir and includes:
    required:  stim_hz, power_ratio, condition
    recommended: animal_id OR filename, brain_region
- If 'animal_id' is missing but 'filename' is present, we use filename as animal_id.
- 'condition' is used to facet columns with col/col_wrap.
- No files are written; figures are just shown.

Minimal columns used by this script
-----------------------------------
- stim_hz (numeric), power_ratio (numeric), condition (str)
- animal_id (str) (or filename), brain_region (str)  [optional but useful]
"""

# ----------------------------- imports ----------------------------- #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", font_scale=1.5)
# ------------------------------------------------------------------- #


if __name__ == "__main__":
    
    # power_ratio csv path
    load_path = r"R:\Pantelis\for analysis\new LFP data for analysis\new opto trains\to_analyze\power_ratio_results.csv"
    df = pd.read_csv(load_path)

    # summary plot with indivudal lines
    g = sns.relplot(df, x="stim_hz", y="power_ratio", kind="line", errorbar="se",
        row="condition", style='filename', col='brain_region', hue='condition')
    for ax in g.axes.flatten():
        ax.axhline(1.0, ls="--", lw=1, c="gray")
        ax.set_xlabel("stimulation frequency (Hz)")
        ax.set_ylabel("power ratio (stim / baseline)")
    g.figure.suptitle("power ratio by condition", y=1.02)
    
    
    # summary plot
    g = sns.relplot(df, x="stim_hz", y="power_ratio", kind="line", errorbar="se",
          col='brain_region', hue='condition')
    for ax in g.axes.flatten():
        ax.axhline(1.0, ls="--", lw=1, c="gray")
        ax.set_xlabel("stimulation frequency (Hz)")
        ax.set_ylabel("power ratio (stim / baseline)")
    g.figure.suptitle("power ratio by condition", y=1.02)
    
    plt.show()
    
    
    
