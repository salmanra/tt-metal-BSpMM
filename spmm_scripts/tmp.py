import numpy as np
import matplotlib.pyplot as plt

# ---- Example inputs ----
categories = ["A", "B", "C", "D"]
groups = ["G1", "G2"]
data = np.array([
    [5, 7],
    [3, np.nan],  # Missing value
    [6, 4],
    [np.nan, 8]   # Missing value
])
# ------------------------

bar_width = 0.35
x = np.arange(len(categories))

# Distinct colors for each group
group_colors = ["steelblue", "darkorange"]

fig, ax = plt.subplots()

# Max real value to set the full height for missing bars
max_val = np.nanmax(data) * 1.05

for i, group in enumerate(groups):
    vals = data[:, i]

    # Plot non-missing
    mask_ok = ~np.isnan(vals)
    ax.bar(
        x[mask_ok] + i * bar_width,
        vals[mask_ok],
        width=bar_width,
        color=group_colors[i],
        label=f"{group} (data)",
    )

    # Plot missing as full-height, styled differently
    mask_nan = np.isnan(vals)
    missing_bars = ax.bar(
        x[mask_nan] + i * bar_width,
        np.full(mask_nan.sum(), max_val),
        width=bar_width,
        color="lightgray",
        edgecolor="black",
        hatch="//",
        alpha=0.5,
        label=f"{group} (missing)",
    )

    # Add "Missing" text centered inside missing bars
    for bar in missing_bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            "Missing",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            rotation=90,
            color="black",
        )

# Axes formatting
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(categories)
ax.set_ylabel("Value")
ax.set_ylim(0, max_val * 1.1)
ax.legend()
ax.set_title("Grouped bars with distinct colors and missing value placeholders")

plt.show()



profiles_dir = "/home/user/tt-metal/profiles/"
png_output_dir = profiles_dir + "pngs/"
plt.savefig(png_output_dir + "tmp.png")