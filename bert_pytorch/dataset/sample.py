from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    id_seq = []
    time_seq = []
    for i in range(0, len(line), window_size):
        id_seq.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return id_seq, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None, seq_len=None, min_len=0):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    id_seq = []
    time_seq = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        ids, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        id_seq += ids
        time_seq += times

    id_seq = np.array(id_seq)
    time_seq = np.array(time_seq)

    id_trainset, id_validset, time_trainset, time_validset = train_test_split(id_seq,
                                                                                      time_seq,
                                                                                      test_size=test_size,
                                                                                      random_state=1234)

    # sort seq_pairs by seq len
    train_len = list(map(len, id_trainset))
    valid_len = list(map(len, id_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    id_trainset = id_trainset[train_sort_index]
    id_validset = id_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(id_trainset))
    print("Num of valid seqs", len(id_validset))
    print("="*40)

    return id_trainset, id_validset, time_trainset, time_validset