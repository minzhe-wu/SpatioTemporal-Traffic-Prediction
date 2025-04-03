# GWTformer: Graph + Transformer for Spatio-Temporal Traffic Forecasting

**GWTformer** is a spatio-temporal deep learning model designed to predict traffic conditions based on both spatial graph structures and long-range temporal patterns.  
It integrates **Graph Convolutional Networks (GCN)** for spatial dependencies and **Transformer encoder** for temporal sequence modeling.

This implementation is based on PyTorch and tested on the METR-LA traffic dataset.

---

## Requirements

- Python 3.7+
- PyTorch 1.4+
- NumPy, Pandas, h5py

You can install the necessary packages via:

```bash
pip install torch numpy pandas h5py

---

## Data Preparation

### Step 1: Download METR-LA dataset  
You will need the following data file manually placed in the correct folder:

- `data/METR-LA/metr-la.h5`  
  > Please obtain this file from public resources or the dataset author.

- The graph structure `adj_mx.pkl` is already included in this repository:
```bash
data/METR-LA/sensor_graph/adj_mx.pkl

data/
└── METR-LA/
    ├── metr-la.h5
    └── sensor_graph/
        └── adj_mx.pkl

---

## Train Commands

```bash
python train.py --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/METR-LA/sensor_graph/adj_mx.pkl \
  --gcn_bool --addaptadj --randomadj \
