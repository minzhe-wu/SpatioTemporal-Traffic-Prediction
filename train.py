# train.py
import argparse
import time
import torch
import numpy as np

from model.gwtformer_model import GWTformer
from data import data_loader  # import data loading functions
from utils import metrics

def main():
    parser = argparse.ArgumentParser()
    # Basic runtime parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Computing device (e.g. "cuda:0" or "cpu")')
    parser.add_argument('--data', type=str, default='data/METR-LA', help='Data directory for the METR-LA dataset')
    parser.add_argument('--adjdata', type=str, default='data/METR-LA/sensor_graph/adj_mx.pkl', help='Path to adjacency matrix pickle')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='Adjacency matrix type: e.g. transition, doubletransition, symnadj, etc.')
    # Graph adjacency options
    parser.add_argument('--gcn_bool', action='store_true', help='Use graph convolution (spatial)?')
    parser.add_argument('--addaptadj', action='store_true', help='Add adaptive adjacency matrix (learnable)?')
    parser.add_argument('--aptonly', action='store_true', help='Use only adaptive adjacency (ignore given adj)?')
    parser.add_argument('--randomadj', action='store_true', help='Randomly initialize adaptive adj (instead of using given adj matrix)?')
    # Model hyperparameters
    parser.add_argument('--seq_length', type=int, default=12, help='Sequence input length (and output horizon length)')
    parser.add_argument('--in_dim', type=int, default=2, help='Number of input features (e.g. 2 for speed + time-of-day)')
    parser.add_argument('--nhid', type=int, default=32, help='Number of hidden units (channels) for residual and dilation')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in multi-head attention')
    parser.add_argument('--ffn_dim', type=int, default=64, help='Hidden dimension of the Transformer FFN (position-wise)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_nodes', type=int, default=207, help='Number of nodes (e.g., 207 for METR-LA)')  # ✅ 加了这一行
    # Training options
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--print_every', type=int, default=50, help='Print training info every these many iterations')
    parser.add_argument('--save', type=str, default='./garage/gwtf', help='Path prefix for saving model checkpoints')
    parser.add_argument('--expid', type=int, default=1, help='Experiment ID (for distinguishing multiple runs)')
    # Data options
    parser.add_argument('--dow', action='store_true', help='Add day-of-week as feature (third input dimension)')
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load adjacency matrices
    sensor_ids, sensor_id_to_ind, adj_mats = data_loader.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(mat).to(device) for mat in adj_mats]
    if args.aptonly:
        supports = []

    # Load data
    data = data_loader.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size,
                                    add_time_in_day=True, add_day_in_week=args.dow)
    scaler = data['scaler']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']

    aptinit = None if args.randomadj or len(supports) == 0 else supports[0]

    model = GWTformer(device, args.num_nodes, args.dropout,
                      supports=supports if len(supports) > 0 else None,
                      gcn_bool=args.gcn_bool, addaptadj=args.addaptadj,
                      aptinit=aptinit,
                      in_dim=args.in_dim, out_dim=args.seq_length,
                      residual_channels=args.nhid, dilation_channels=args.nhid,
                      skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                      num_heads=args.num_heads, ffn_dim=args.ffn_dim)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = metrics.masked_mae

    print(f"Start training (Experiment ID={args.expid}) ...")
    his_loss = []
    val_times = []
    train_times = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loader.shuffle()
        t_epoch_start = time.time()
        batch_iter = 0
        for x_batch, y_batch in train_loader.get_iterator():
            batch_iter += 1
            x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output = model(x_tensor.permute(0, 3, 2, 1))
            output = output.transpose(1, 3)
            predict = scaler.inverse_transform(output)
            real = scaler.inverse_transform(y_tensor.transpose(1, 2))
            real = real.transpose(1, 2)
            loss = loss_fn(predict, real, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if batch_iter % args.print_every == 0:
                train_mae = metrics.masked_mae(predict, real, 0.0).item()
                train_rmse = metrics.masked_rmse(predict, real, 0.0).item()
                train_mape = metrics.masked_mae(predict, real, null_val=np.nan).item()
                print(f"Iter {batch_iter:03d} (Epoch {epoch}) - Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
        t_epoch_end = time.time()
        train_times.append(t_epoch_end - t_epoch_start)

        # Validation
        model.eval()
        val_losses, val_maes, val_mapes, val_rmses = [], [], [], []
        t_val_start = time.time()
        with torch.no_grad():
            for x_batch, y_batch in val_loader.get_iterator():
                x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y_batch, dtype=torch.float32).to(device)
                output = model(x_tensor.permute(0, 3, 2, 1))  
                output = output.transpose(1, 3)
                predict = scaler.inverse_transform(output)
                real = scaler.inverse_transform(y_tensor.transpose(1, 2)).transpose(1, 2)
                val_loss = loss_fn(predict, real, 0.0)
                val_losses.append(val_loss.item())
                val_maes.append(metrics.masked_mae(predict, real, 0.0).item())
                val_rmses.append(metrics.masked_rmse(predict, real, 0.0).item())
                val_mapes.append(metrics.masked_mae(predict, real, null_val=np.nan).item())
        t_val_end = time.time()
        val_times.append(t_val_end - t_val_start)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(val_maes)
        val_rmse = np.mean(val_rmses)
        val_mape = np.mean(val_mapes)
        his_loss.append(val_loss)
        print(f"Epoch {epoch:03d}/{args.epochs} - Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.save}_best_epoch{epoch}.pth")

    print("Training finished")
    print(f"Average Training Time per Epoch: {np.mean(train_times):.2f} secs")
    print(f"Average Validation Time per Epoch: {np.mean(val_times):.2f} secs")

    # Load best model and evaluate on test data
    model.load_state_dict(torch.load(f"{args.save}_best_epoch{best_epoch}.pth"))
    model.eval()
    test_maes, test_mapes, test_rmses = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader.get_iterator():
            x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.float32).to(device)
            output = model(x_tensor.permute(0, 3, 2, 1)).transpose(1, 3)
            predict = scaler.inverse_transform(output)
            real = scaler.inverse_transform(y_tensor.transpose(1, 2)).transpose(1, 2)
            mae, mape, rmse = metrics.metric(predict, real, null_val=0.0)
            test_maes.append(mae); test_mapes.append(mape); test_rmses.append(rmse)
    print(f"Test MAE: {np.mean(test_maes):.4f}, Test RMSE: {np.mean(test_rmses):.4f}, Test MAPE: {np.mean(test_mapes):.4f}")

if __name__ == "__main__":
    main()

