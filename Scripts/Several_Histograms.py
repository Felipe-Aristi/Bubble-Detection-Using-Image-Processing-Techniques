import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------------------------------------------------
# 1. Set your folder path
# ------------------------------------------------------------
folder_path = r"C:\Users\juanp\Desktop\Research project\TEST3\0.8\Diameters"

# ------------------------------------------------------------
# 2. Group diameters by RPM
# ------------------------------------------------------------
rpm_diameters = defaultdict(list)

for fname in os.listdir(folder_path):
    if fname.endswith(".txt"):
        try:
            rpm = int(fname.split("_")[0])  # Example: 1100_0.8_0001.txt
        except (ValueError, IndexError):
            print(f" Skipping invalid file name: {fname}")
            continue

        with open(os.path.join(folder_path, fname), "r") as f:
            values = [float(line.strip()) for line in f if line.strip()]
            rpm_diameters[rpm].extend(values)

# ------------------------------------------------------------
# 3. Plot one histogram per RPM
# ------------------------------------------------------------
all_diameters = [d for lst in rpm_diameters.values() for d in lst]
min_val, max_val = min(all_diameters), max(all_diameters)
bins = 31

# Define the x-axis ticks (adjust as needed)
xtick_step = 0.5  # mm step size
xticks = np.arange(round(min_val, 1), round(max_val + xtick_step, 1), xtick_step)

for rpm in sorted(rpm_diameters):
    diameters = rpm_diameters[rpm]
    mean_d = np.mean(diameters)
    
    Prove_rpm = 1750
    ml = 400
    time= 5.30 *60  #min
    Flow_rate = ml/time
    
    New_flowr = (rpm * Flow_rate )/Prove_rpm #ml/s


    print(f" RPM: {rpm}")
    print(f" → Mean diameter: {mean_d:.4f} mm")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(diameters, bins=bins, edgecolor='black', color='skyblue')
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Frequency')
    plt.title(f'Bubble Diameters at {New_flowr:.1f} mL/s – Initial Bubble Size: 0.8 mm')
    plt.xticks(xticks, rotation=0)  # Set and rotate x-axis labels
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

