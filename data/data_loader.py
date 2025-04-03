# data
import os
import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoader:
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding:
                x_pad = np.repeat(xs[-1:], num_padding, axis=0)
                y_pad = np.repeat(ys[-1:], num_padding, axis=0)
                xs = np.concatenate([xs, x_pad], axis=0)
                ys = np.concatenate([ys, y_pad], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // batch_size)
        self.current_ind = 0
        self.xs = xs
        self.ys = ys
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]
    def get_iterator(self):
        self.current_ind = 0
        def _iterator():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = start_ind + self.batch_size
                x_batch = self.xs[start_ind:end_ind, ...]
                y_batch = self.ys[start_ind:end_ind, ...]
                yield (x_batch, y_batch)
                self.current_ind += 1
        return _iterator()

def load_adj(pkl_path, adj_type):
    import pickle
    with open(pkl_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    def asym_adj(adj):
        adj = adj.copy()
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1, where=rowsum!=0)
        d_inv[d_inv == np.inf] = 0
        D_inv = np.diag(d_inv)
        return D_inv @ adj
    def sym_adj(adj):
        adj = adj.copy()
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum!=0)
        d_inv_sqrt[d_inv_sqrt == np.inf] = 0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt
    if adj_type == "scalap":
        from scipy.sparse import csr_matrix
        import scipy.sparse.linalg as sla
        adj = sym_adj(adj_mx)
        lap = np.identity(adj.shape[0]) - adj
        try:
            lambda_max = sla.eigs(csr_matrix(lap), k=1, which='LR')[0].real
        except Exception:
            lambda_max = 2
        adj_list = [ (2 / lambda_max * lap - np.identity(adj.shape[0])).astype(np.float32) ]
    elif adj_type == "normlap":
        from scipy.sparse.csgraph import laplacian
        lap = laplacian(adj_mx, normed=True)
        adj_list = [np.array(lap).astype(np.float32)]
    elif adj_type == "symnadj":
        adj_list = [sym_adj(adj_mx).astype(np.float32)]
    elif adj_type == "transition":
        adj_list = [asym_adj(adj_mx).astype(np.float32)]
    elif adj_type == "doubletransition":
        adj_list = [asym_adj(adj_mx).astype(np.float32),
                    asym_adj(adj_mx.T).astype(np.float32)]
    elif adj_type == "identity":
        adj_list = [np.eye(adj_mx.shape[0], dtype=np.float32)]
    else:
        raise ValueError(f"adj_type '{adj_type}' is not defined.")
    return sensor_ids, sensor_id_to_ind, adj_list

def load_dataset(data_dir, batch_size, valid_batch_size=None, test_batch_size=None, add_time_in_day=True, add_day_in_week=False):
    data = {}
    if valid_batch_size is None: valid_batch_size = batch_size
    if test_batch_size is None: test_batch_size = batch_size
    data_dir = data_dir.rstrip(os.sep)
    data_path = os.path.join(data_dir, '')
    train_fp = os.path.join(data_path, 'train.npz')
    if os.path.exists(train_fp):
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(data_path, f"{category}.npz"))
            data[f'x_{category}'] = cat_data['x']
            data[f'y_{category}'] = cat_data['y']
        print(f"Loaded preprocessed data from {data_dir}")
    else:
        dataset_name = os.path.basename(data_dir)
        raw_h5 = os.path.join(os.path.dirname(data_dir), dataset_name.lower() + '.h5')
        if not os.path.exists(raw_h5):
            raise FileNotFoundError(f"Data file {raw_h5} not found. Please provide the METR-LA .h5 data file.")
        df = pd.read_hdf(raw_h5)
        seq_length_x = 12
        seq_length_y = 12
        x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
        y_offsets = np.arange(1, seq_length_y + 1, 1)
        data_times = df.index.values
        data_values = np.expand_dims(df.values, axis=-1)
        feature_list = [data_values]
        if add_time_in_day:
            time_ind = (data_times - data_times.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, (df.shape[1], 1)).T
            time_in_day = np.expand_dims(time_in_day, axis=-1)
            feature_list.append(time_in_day)
        if add_day_in_week:
            day_of_week = df.index.dayofweek
            dow = np.tile(day_of_week.values, (df.shape[1], 1)).T
            dow = np.expand_dims(dow, axis=-1)
            feature_list.append(dow)
        data_combined = np.concatenate(feature_list, axis=-1)
        x_samples, y_samples = [], []
        min_t = int(abs(x_offsets[0]))
        max_t = int(df.shape[0] - abs(y_offsets[-1]))
        for t in range(min_t, max_t):
            x_t = data_combined[t + x_offsets, ...]
            y_t = data_combined[t + y_offsets, ...]
            x_samples.append(x_t)
            y_samples.append(y_t)
        x_samples = np.stack(x_samples, axis=0)
        y_samples = np.stack(y_samples, axis=0)
        y_samples = y_samples[..., :1]
        num_samples = x_samples.shape[0]
        num_train = int(round(num_samples * 0.7))
        num_test = int(round(num_samples * 0.2))
        num_val = num_samples - num_train - num_test
        train_indices = range(0, num_train)
        val_indices = range(num_train, num_train + num_val)
        test_indices = range(num_train + num_val, num_samples)
        data['x_train'] = x_samples[train_indices]
        data['y_train'] = y_samples[train_indices]
        data['x_val'] = x_samples[val_indices]
        data['y_val'] = y_samples[val_indices]
        data['x_test'] = x_samples[test_indices]
        data['y_test'] = y_samples[test_indices]
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(os.path.join(data_dir, "train.npz"), x=data['x_train'], y=data['y_train'])
        np.savez_compressed(os.path.join(data_dir, "val.npz"), x=data['x_val'], y=data['y_val'])
        np.savez_compressed(os.path.join(data_dir, "test.npz"), x=data['x_test'], y=data['y_test'])
        print(f"Processed data from {raw_h5} and saved numpy arrays to {data_dir}")
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data[f'x_{category}'][..., 0] = scaler.transform(data[f'x_{category}'][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader']   = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader']  = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



