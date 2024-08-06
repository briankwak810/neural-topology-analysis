## Neural topology analysis

A fast end-to-end data analysis pipeline that utilizes CEBRA, to analyze neural data and find underlying topologies.
The purpose of this code is to easily use topological data analysis tools, along with data pre-/post-processing 

### USAGE
Put data under `data/` folder, change `config.yaml` file to change settings & directories. And then run

```console
python PV_trace_cebra.py
```

to run topological data analysis, and run

```console
python spikes.py
```

to get raw firing map images.

### Data

Neural data(spike, trace) from **Inscopix** and behaviorial data from **Ethovision** have compatible data structures.
Data files should be under the directory `data/` to avoid directory crashes.
Other data is possible, change `code` according to data structure.

### Configuration

Change `config.yaml` file to tune model according to specific task.

Analysis mode parameters, Neural data, CEBRA parameters, Persistent homology(ripser) parameters are given as basic configuration parameters.

### Training

Training is done with auxillary variables(`pos`, `head_dir`, `vel`), which are used as variables to calculate self-supervised distance in CEBRA. See the [official CEBRA document](https://cebra.ai/docs/) for additional information on selecting variables for specific tasks.

### Data processing

Spike data is convoluted with 1d gaussian filter, and trace data is normalized as z-score. Raw data is fed into the model, without PCA whitening, but indices with overall lower activity can be discarded.

Embedded points are postprocessed with radial filter and a fuzzy filter algorithm that filters out outliers and density-based clustering.
