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

