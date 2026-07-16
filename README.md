# STDL-MP-ACOPF

Learning-side implementation for the paper:

> Jihun Kim, Sojin Park, Dongwoo Kang, and Hunyoung Shin,  
> **“Spatio-Temporal Deep Learning-Assisted Multi-Period AC Optimal Power Flow,”**  
> *Electronics*, 15(4), 761, 2026.  
> [https://doi.org/10.3390/electronics15040761](https://doi.org/10.3390/electronics15040761)

STDL-MP-ACOPF trains a Graph Attention Network–Temporal Convolutional Network
(GAT-TCN) to predict multi-period bus voltage magnitudes `V` and voltage angles
`θ`. Predictions for cases that did not converge with the ordinary solver
initialization are inverse-normalized and passed back to the MP-ACOPF solver as
an AI-assisted initial solution.

## End-to-end workflow

This repository is the learning part of a pipeline shared with the separate
private `acopf-code` solver repository.

```text
acopf-code
  1. Generate multi-period Pd profiles
  2. Run MP-ACOPF for 1,500 cases per system/period condition
  3. Separate converged and failed/difficult cases
  4. Export Pd and converged V/θ results
        │
        ▼
STDL-MP-ACOPF
  5. Convert converged cases into graph samples
  6. Train GAT-TCN on 1,000 selected converged cases
  7. Convert failed-case Pd profiles into failed graph samples
  8. Predict V/θ for failed cases
  9. Apply inverse min-max normalization
 10. Export failed_predictions_*.txt
        │
        ▼
acopf-code
 11. Read predicted V/θ as the solver initial solution
 12. Re-run MP-ACOPF
 13. Compare convergence rate and solution time
```

The solver and learning environments remain separate. AMPL, Knitro, other
commercial solver binaries, licenses, activation files, and solver executables
are not included here.

## Experiment conditions

| System | Buses | Generators | Branches | Periods | Generated | Training dataset |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic South Carolina | 500 | 90 | 597 | 8 | 1,500 | 1,000 converged cases |
| Synthetic South Carolina | 500 | 90 | 597 | 24 | 1,500 | 1,000 converged cases |
| PEGASE | 1,354 | 260 | 1,991 | 8 | 1,500 | 1,000 converged cases |
| PEGASE | 1,354 | 260 | 1,991 | 24 | 1,500 | 1,000 converged cases |

The 8-period case uses 15-minute intervals over two hours. The 24-period case
uses hourly intervals over 24 hours.

## Graph dataset

- Node: bus
- Edge: branch
- Node feature `x`: multi-period active-power demand `Pd`
- Target `y[..., 0]`: voltage magnitude `V`
- Target `y[..., 1]`: voltage angle `θ`
- Shared topology: sparse/dense adjacency matrix under `AM/`
- Edge features `e`: branch information retained for edge-aware comparison layers
- Target preprocessing: separate min-max normalization for `V` and `θ`

Each `graph_data/GNN_<index>.npz` contains:

```text
x: (n_bus, n_period)
a: graph adjacency
e: edge-feature tensor
y: (n_bus, n_period, 2)
```

The 1354-bus case uses `Data/1354bus_mapping.npy` to convert the original
PEGASE bus identifiers to contiguous graph node indices.

## Paper model

- Three GAT layers
- Three attention heads per GAT layer
- Causal TCN
- 8 periods: kernel size 2 and dilations `(1, 2, 4)`
- 24 periods: kernel size 3 and dilations `(1, 3, 9)`
- Dense output layers
- Joint `V` and `θ` prediction
- MSE training loss
- MSE and SMAPE evaluation
- 70%/15%/15% train/validation/test split

The scripts retain alternative GNN/RNN branches from comparison experiments.
For the paper configuration, use:

```python
GNN_types = ["GATConv"]
RNN_types = ["TCN"]
```

## Source files

### `500bus_train.py`

Main 500-bus pipeline. Its `mode` controls successful graph generation,
training, evaluation, failed-case graph generation/prediction, and spreadsheet
export.

### `1354bus_train.py`

Main 1354-bus pipeline. It performs the same stages as `500bus_train.py` and
adds PEGASE bus-ID mapping.

### `500bus_split_train.py`

Comparison code that trains separate voltage-magnitude and voltage-angle
models and evaluates alternative GNN/RNN combinations. It is not required for
the joint-output GAT-TCN reproduction, but is retained as an ablation record.

## Data layout

```text
Data/
├── 1354bus_mapping.npy
├── same proportions/
│   ├── 500bus/
│   │   ├── 8/
│   │   └── 24/
│   └── 1354bus/
│       ├── 8/
│       └── 24/
└── 25_1017 access review/
    ├── 500bus/
    └── 1354bus/
```

For each system/period directory:

```text
case*_edgedata.dat          branch/topology input
demand*all.txt              flattened Pd for converged cases
*_ma_values_all.txt         flattened converged V/θ solver results
demand*failed_all.txt       flattened Pd for failed/difficult cases
*_ma_values_failed_all.txt  placeholder/label data used by the current loader
AM/                         shared adjacency matrices
graph_data/                 converged graph samples
failed_graph_data/          failed-case graph samples
GNN_trained_model/          TensorFlow SavedModels
result/                     training/evaluation results
failed_predictions/         inverse-normalized solver initial solutions
```

Large datasets, raw solver results, graph archives, predictions, and trained
weights are research artifacts and should not be committed to Git.

The repository includes only the small static inputs required to reconstruct
the graph topology: `1354bus_mapping.npy` and the four
`case*_edgedata.dat` files for the 500-bus and 1354-bus, 8-period and
24-period configurations. The demand profiles and V/θ solver outputs must be
generated with `acopf-code` or supplied separately using the layout above.

## Execution order

The main scripts currently select work through variables near the beginning of
the file. Configure `nPrd_list`, `GNN_types`, and `RNN_types`, then select one
of the following modes.

### 1. Generate successful graph data

Prerequisites supplied by `acopf-code`/preprocessing:

- `case*_edgedata.dat`
- `demand*all.txt`
- `*_ma_values_all.txt`
- `Data/1354bus_mapping.npy` for PEGASE

Set:

```python
mode = "data gen"
nPrd_list = [8]  # or [24]
GNN_types = ["GATConv"]
RNN_types = ["TCN"]
```

Run:

```bash
python 500bus_train.py
# or
python 1354bus_train.py
```

Outputs are written to `AM/` and `graph_data/`.

### 2. Train the paper model

Set:

```python
mode = "train"
nPrd_list = [8]  # or [24]
GNN_types = ["GATConv"]
RNN_types = ["TCN"]
```

Run the corresponding main script. The scripts load `graph_data`, split the
1,000 samples into train/validation/test subsets, train with early stopping,
restore the best validation weights, and save the model under
`GNN_trained_model/`.

### 3. Evaluate the trained model

Set:

```python
mode = "test"
```

The corresponding script loads the selected SavedModel and evaluates the
training, validation, test, and full datasets. Result summaries are written
under `Data/25_1017 access review/<system>/<period>/`.

### 4. Generate AI initial solutions for failed cases

Set:

```python
mode = "save_output"
```

Use the system-specific main script:

```bash
python 500bus_train.py
# or
python 1354bus_train.py
```

The `save_output` path performs all failed-case learning-side steps:

1. read `demand*failed_all.txt`;
2. build `failed_graph_data/GNN_*.npz`;
3. load the trained GAT-TCN;
4. predict normalized `V` and `θ`;
5. inverse-normalize using the successful-dataset ranges; and
6. write `failed_predictions/failed_predictions_*.txt`.

Prediction file format:

```text
voltages and angles:
bus 1 M <voltage> A <angle> k 0
bus 1 M <voltage> A <angle> k 1
...
```

These files are copied or referenced by `acopf-code` and supplied to the
MP-ACOPF solver as initial voltage magnitude/angle values.

## Installation

Create a separate Python 3.8 environment for the learning code:

```bash
python -m pip install -r requirements.txt
```

The recorded environment uses TensorFlow 2.8.0. GPU execution requires a
compatible CUDA/cuDNN installation.

## Portable paths

All main scripts derive the project directory from their own file location:

```python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
```

No path depends on the former repository name or a specific user home
directory. The project can therefore be moved or cloned under the
`STDL-MP-ACOPF` name without editing data paths.

## Provenance

The graph dataset and spatio-temporal training workflow was adapted from the
RPGLab `GNN-LSTM_C-V-R-SCUC` project by Arun Venkatesh Ramesh and Xingpeng Li.
This project applies and extends that workflow for MP-ACOPF, joint voltage
magnitude/angle prediction, failed-case inference, inverse normalization, and
solver initialization. Appropriate upstream attribution should be retained for
derived code.

## Citation

```bibtex
@article{kim2026spatiotemporal,
  title   = {Spatio-Temporal Deep Learning-Assisted Multi-Period AC Optimal Power Flow},
  author  = {Kim, Jihun and Park, Sojin and Kang, Dongwoo and Shin, Hunyoung},
  journal = {Electronics},
  volume  = {15},
  number  = {4},
  pages   = {761},
  year    = {2026},
  doi     = {10.3390/electronics15040761}
}
```
