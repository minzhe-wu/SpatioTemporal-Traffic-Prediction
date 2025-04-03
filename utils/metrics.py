# utils
import torch
import numpy as np

def masked_mse(preds, labels, null_val=float('nan')):
    mask = ~torch.isnan(labels) if torch.isnan(torch.tensor(null_val)) else (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)  # normalize mask
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=float('nan')):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mae(preds, labels, null_val=float('nan')):
    mask = ~torch.isnan(labels) if torch.isnan(torch.tensor(null_val)) else (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real, null_val=0.0):
    mae = masked_mae(pred, real, null_val).item()
    mape = masked_mae(pred, real, null_val=np.nan).item()  
    rmse = masked_rmse(pred, real, null_val).item()
    return mae, mape, rmse

