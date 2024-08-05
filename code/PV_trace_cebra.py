import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
from downsampling import *
from scipy.ndimage import gaussian_filter1d
import os, yaml

import cebra
from cebra import CEBRA

#########################################################################
########################### Configuration  ##############################
#########################################################################


with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Use configuration values
mouse_num = config['mouse_num']
maxdim = config['maxdim']
seed = config['seed']
importance_threshold = config['importance_threshold']
discard_percent = config['discard_low_fire']
usespikes = config['usespikes']

ply_name = config['experiment_name']
os.makedirs(f"../output-{ply_name}", exist_ok=True)
os.makedirs(f"../models", exist_ok=True)

print(f"Starting CEBRA analysis of experiment {ply_name}.")

index = config['cell_index']
spk_filename = config['inscopix_files']['spk']
trc_filename = config['inscopix_files']['trc']
dims = config['dims']
max_iterations = config['max_iterations']
save_models = config['save_models']
model_architecture = config['model_architecture']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
temperature_mode = config['temperature_mode']
temperature = config['temperature']
distance = config['distance']
conditional = config['conditional']
time_offsets = config['time_offsets']
hybrid = config['hybrid']

use_downsampling = config['use_downsampling']

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
trace_per_cell = {cell: [] for cell in cell_numbers}
spike_times = {cell: [] for cell in cell_numbers}
nn_trace_per_cell = {cell: [] for cell in cell_numbers}

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

# get trace data from each cell + interpolate
for i in range(len(cell_numbers)):
    cell_name = np.asarray(cell_numbers)[i]
    trace_data = [float(i) for i in df.iloc[1:, i+2].tolist()]
    time_interp = interp1d(time_data, trace_data, bounds_error=False, fill_value="extrapolate")

    trace_data = time_interp(traj_time)
    nn_trace_per_cell[cell_name] = trace_data.tolist()
    trace_data = (trace_data - np.mean(trace_data)) / np.std(trace_data)
    trace_per_cell[cell_name] = trace_data.tolist()

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

total_trace = np.zeros((len(traj_time), len(cell_numbers)))

# Iterate through cells and their spike times
for i, cell in enumerate(cell_numbers):
    total_trace[:, i] = np.asarray(trace_per_cell[cell])

if usespikes:
    # Get spike data -> nearest interpolation
    spiking_data = np.zeros((len(traj_time), len(cell_numbers)))

    for i, cell in enumerate(cell_numbers):
        for spike_time in spike_times[cell]:
            nearest_index = find_nearest_time_index(spike_time, traj_time)
            if nearest_index in moving_idx:
                spiking_data[nearest_index, i] += 1

    # Apply Gaussian filter
    sigma_ms = 0.2  # sigma in seconds
    sigma_samples = sigma_ms / (traj_time[1] - traj_time[0])  # convert sigma to samples
    spiking_data = gaussian_filter1d(spiking_data, sigma=sigma_samples, axis=0)

    total_spk = spiking_data

if discard_percent != 0 and index is not None:
    # Not-Normalized total trace to discard lower percent
    nn_total_trace = np.zeros((len(traj_time), len(cell_numbers)))
    for i, cell in enumerate(cell_numbers):
        nn_total_trace[:, i] = np.asarray(nn_trace_per_cell[cell])

    # Get population
    higher_pop = get_higher_percent_indices(nn_total_trace, index, discard_percent)
    total_trace = total_trace[higher_pop, :]
    total_spk = spiking_data[higher_pop, :]

if not usespikes:
    if index is not None:
        neural = total_trace[:, index]
    else:
        neural = total_trace[:, :]
else:
    if index is not None:
        neural = total_spk[:, index]
    else:
        neural = total_spk[:, :]

########################################################################
##########################  Hypothesis Setting #########################
## Use the auxilary variables that you want to use to train the model ##
# Variables available are: traj_time, x_pos, y_pos, head_dir, velocity #
####### You can add more! --Default set to distance from boundary ######
########################################################################

# normalize (for square)
x_pos = x_pos / 10
y_pos = y_pos / 10
min_x = np.min(x_pos)
max_x = np.max(x_pos)
min_y = np.min(y_pos)
max_y = np.max(y_pos)

continuous_index = np.column_stack((x_pos-min_x, max_x-x_pos, y_pos-min_y, max_y-y_pos))

if discard_percent != 0:
    continuous_index = continuous_index[higher_pop, :]

########################### CEBRA training ##############################

cebra_models = []
embeddings = []
for dim in dims:
    model = CEBRA(
        model_architecture=model_architecture,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature_mode=temperature_mode,
        temperature=temperature,
        min_temperature=0.1,
        output_dimension=dim,
        max_iterations=max_iterations,
        distance=distance,
        conditional=conditional,
        device='cuda_if_available',
        verbose=True,
        time_offsets=time_offsets,
        hybrid=hybrid
    )
    cebra_models.append(model)

for dim, model in zip(dims, cebra_models):
    model.fit(neural, continuous_index)
    if save_models:
        model.save(f"../models/{ply_name}_model_{dim}.pt")

for model in cebra_models:
    result = model.transform(neural)
    embeddings.append(result)

print("CEBRA embedding for neural data complete.")

os.makedirs(f'../output-{ply_name}/arrays', exist_ok=True)
embedding_dict = {}
for i, (dim, embedding) in enumerate(zip(dims, embeddings)):
    embedding_dict[f'embedding_{dim}'] = embedding
np.savez(f'../output-{ply_name}/arrays/embedding_arrays.npz', **embedding_dict)

print(f"\nAll {len(embeddings)} arrays saved successfully.")

########################################################################
####################### VISUALIZE to PLY file ##########################
########### Change `embedding_labels` to change ply color ##############
########################################################################

embedding_labels = continuous_index[:, 0]

norm = plt.Normalize(embedding_labels.min(), embedding_labels.max())
normalized_index = norm(embedding_labels)
cmap = plt.get_cmap('plasma')  # or 'viridis'
colors_float = cmap(normalized_index)[:, :3]
colors = (colors_float * 255).astype(int)

# Write the colored PLY file
os.makedirs(f'../output-{ply_name}/ply', exist_ok=True)
for dim, points in zip(dims, embeddings):
    write_colored_ply(f"../output-{ply_name}/ply/embedding_{dim}.ply", points, colors)

print("\nEmbedding to ply complete.")

########### SHUFFLE for TOPOLOGY ############

shuffled_index = np.random.permutation(continuous_index)

shuffled_cebra_models = []
shuffled_embeddings = []
for dim in dims:
    model = CEBRA(
        model_architecture=model_architecture,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature_mode=temperature_mode,
        temperature=temperature,
        min_temperature=0.1,
        output_dimension=dim,
        max_iterations=max_iterations,
        distance=distance,
        conditional=conditional,
        device='cuda_if_available',
        verbose=True,
        time_offsets=time_offsets,
        hybrid=hybrid
    )
    shuffled_cebra_models.append(model)

for dim, model in zip(dims, shuffled_cebra_models):
    model.fit(neural, shuffled_index)
    if save_models:
        model.save(f"../models/{ply_name}_shuffled_model_{dim}.pt")

for model in shuffled_cebra_models:
    result = model.transform(neural)
    shuffled_embeddings.append(result)

print("\nCEBRA embedding for shuffled neural data complete.")

shuffled_embedding_dict = {}
for i, (dim, embedding) in enumerate(zip(dims, shuffled_embeddings)):
    shuffled_embedding_dict[f'shuffled_embedding_{dim}'] = embedding
np.savez(f'../output-{ply_name}/arrays/shuffled_embedding_arrays.npz', **embedding_dict)

print(f"\nAll {len(embeddings)} shuffled arrays saved successfully.")

########################################################################
############### Downsampling & cleaning of Embeddings ##################
########################################################################

if use_downsampling:
    embedding_arrays = np.load(f'../output-{ply_name}/arrays/embedding_arrays.npz')
    downsampled_arrays = {}
    adjusted_shuffled_arrays = {}
    downsampled_list = []
    adjusted_shuffled_list = []

    for dim, array, shuffled_array in zip(dims, embeddings, shuffled_embeddings):
        downsampled, adjusted_shuffled = process_array(array, shuffled_array, name=dim, n_neighbors=5, std_ratio=2.0, initial_importance_threshold=importance_threshold)
        downsampled_arrays[f'embedding_{dim}'] = downsampled
        downsampled_list.append(downsampled)
        adjusted_shuffled_arrays[f'shuffled_embedding_{dim}'] = adjusted_shuffled
        adjusted_shuffled_list.append(adjusted_shuffled)

    # Save the downsampled and adjusted shuffled arrays
    np.savez(f'../output-{ply_name}/arrays/downsampled_embedding_arrays.npz', **downsampled_arrays)
    np.savez(f'../output-{ply_name}/arrays/adjusted_shuffled_embedding_arrays.npz', **adjusted_shuffled_arrays)
    print("\nAll downsampled and adjusted shuffled arrays saved successfully.")

    # Print summary
    print("\nSummary:")
    for name, array in downsampled_arrays.items():
        original_size = len(embedding_arrays[name])
        downsampled_size = len(array)
        shuffled_size = len(adjusted_shuffled_arrays[f"shuffled_{name}"])
        percentage = downsampled_size / original_size
        print(f"{name}: Original {original_size}, Downsampled {downsampled_size}, Adjusted Shuffled {shuffled_size} ({percentage:.2%})")

else:
    print(f"\nTopology analysis will be done without downsampling...")

########################################################################
################# Calculate Topology of Embeddings #####################
########################################################################

# Variables: ply_name, dims, save_bar_graph, embeddings, shuffled_embeddings, seed, maxdim
if use_downsampling:
    topology_calls = [
        (
            ply_name,
            dims,
            True,
            downsampled_list,
            adjusted_shuffled_list,
            seed,
            maxdim
        )
    ]
else:
    topology_calls = [
        (
            ply_name,
            dims,
            True, 
            embeddings,
            shuffled_embeddings,
            seed,
            maxdim
        )
    ]

for args in topology_calls:
    drawTopology(*args)