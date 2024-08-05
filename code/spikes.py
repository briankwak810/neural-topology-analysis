import pandas as pd
import csv, yaml, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
from utils import rat_trajectory, calculate_border_score
from matplotlib.colors import Normalize

cell_num = 12 # to visualize
cell_name = f' C{cell_num:02}'

with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
spk_filename = config['inscopix_files']['spk']
trc_filename = config['inscopix_files']['trc']

os.makedirs(f"../output-{config.get('experiment_name')}/ratemaps", exist_ok=True)

# Read the excel file
df = pd.read_csv(f'../data/{trc_filename}', header=None)
if df.iloc[1].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all():
    df_start = 1
else:
    df_start = 2

first_column = df.iloc[:, 0]
expected_sequence = pd.Series(range(1, len(first_column) + 1))

if not first_column.equals(expected_sequence):
    df_time = 1
else:
    df_time = 0

# Get cell numbers
cell_numbers = df.iloc[0].dropna().tolist()[1:]
spike_times = {cell: [] for cell in cell_numbers}

# Get time data
time_data = [float(i) for i in df.iloc[df_start:, df_time].tolist()]
start_time = time_data[0]
time_data = [time_data[i] - start_time for i in range(len(time_data))]

with open(f'../data/{spk_filename}', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # Skip the header row
    for row in csv_reader:
        if len(row) == 4:
            start = 1
        elif len(row) == 3:
            start = 0
        break
    for row in csv_reader:
        time = float(row[start]) - start_time
        cell = row[start + 1]
        spike_times[cell].append(time)

traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory()

## NaN interpolation ##
for pos in [x_pos, y_pos, head_dir, velocity]:
    invalid = np.isnan(pos)
    pos_interp = pos.copy()
    valid = np.where(~invalid)[0]
    pos_interp[invalid] = np.interp(np.where(invalid)[0], valid, pos[valid])
    pos[:] = pos_interp

# discard index where velocity < threshold
threshold = 2
moving_idx = [i for i, x in enumerate(velocity) if x > threshold]

border_indx = []
border_name = []

for i, cell_name in enumerate(spike_times.keys()): ###########
    fire_xpos = []
    fire_ypos = []
    fire_hd = []
    for spike_time in spike_times[cell_name]:
        closest_index = np.argmin(np.abs(np.array(traj_time) - spike_time))
        # discard spikes where velocity < threshold
        if closest_index in moving_idx:
            fire_xpos.append(x_pos[closest_index])
            fire_ypos.append(y_pos[closest_index])
            fire_hd.append(head_dir[closest_index])

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Spike position diagram with color-coded head direction
    ax1.plot(x_pos, y_pos, color='lightgray', alpha=0.7, linewidth=1)  # Trajectory in light gray

    cmap = plt.get_cmap('rainbow')
    norm = Normalize(vmin=min(fire_hd), vmax=max(fire_hd))
    scatter = ax1.scatter(fire_xpos, fire_ypos, c=fire_hd, cmap=cmap, norm=norm, s=20, zorder=2)

    ax1.set_title(f'Rat Trajectory and Spike Positions for {cell_name}')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axis('equal')  # To ensure the aspect ratio is 1:1

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Head Direction')

    # 2. Rate map
    num_bins = 50

    # Create 2D histograms
    position_hist, x_edges, y_edges = np.histogram2d(x_pos[moving_idx], y_pos[moving_idx], bins=num_bins)
    spike_hist, _, _ = np.histogram2d(fire_xpos, fire_ypos, bins=[x_edges, y_edges])

    # Calculate rate map (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        firing_map = np.divide(spike_hist, position_hist)
        firing_map[np.isinf(firing_map)] = 0
        firing_map[np.isnan(firing_map)] = 0

    # Create 2D histogram of spike positions
    sigma = 1.3  # Adjust this value to control the amount of smoothing
    firing_map_smooth = gaussian_filter(firing_map, sigma)
    firing_map_smooth = np.ma.array(firing_map_smooth.T)

    bin_size = (x_edges[1] - x_edges[0])  # Assuming square bins
    border_score = calculate_border_score(firing_map_smooth, bin_size)

    if border_score > 0.1:
        border_indx.append(i)
        border_name.append(cell_name)

    # # Add border score to the plot
    # ax2.text(0.05, 0.95, f'Border Score: {border_score:.2f}', 
    #         transform=ax2.transAxes, fontsize=10, 
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # im = ax2.imshow(firing_map_smooth, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    #                 cmap='jet', interpolation='bilinear') # norm=LogNorm(vmin=1)

    # ax2.plot(x_pos, y_pos, color='gray', alpha=0.5, linewidth=0.5)
    # # ax2.scatter(fire_xpos, fire_ypos, color='red', s=10, alpha=0.5)

    # # Add colorbar
    # cbar = fig.colorbar(im, ax=ax2)
    # cbar.set_label('Number of Spikes')

    # ax2.set_title(f'Firing Map and Trajectory for {cell_name}')
    # ax2.set_xlabel('X Position')
    # ax2.set_ylabel('Y Position')

    # ax2.set_aspect('equal', adjustable='box')

    # # Adjust layout and display
    # plt.tight_layout()
    # plt.savefig(f"../output-{config.get('experiment_name')}/ratemaps/{cell_name}")

print(border_name, border_indx)