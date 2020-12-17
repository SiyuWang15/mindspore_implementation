import numpy as np 
import os

def process_H(H): # bs x 2 x 4 x 32
    # 真实的频域信道，获取标签
    batch_size = len(H)
    Hf_train = np.array(H)[:, 0, :] + 1j * np.array(H)[:, 1, :]
    Hf_train = np.fft.fft(Hf_train, 256)/20 # 4*256
    Hf = np.stack([Hf_train.real, Hf_train.imag], axis = 1) # bsx2x4x256
#     return ms.Tensor(Hf, dtype = ms.float16)
    return Hf.reshape(batch_size, -1)

def get_test_data(Pn):
    tag = 1 if Pn == 32 else 2
    dp = os.path.join(f'../dataset/Y_{tag}.csv')
    print(f'loading test data from {dp}')
    Y = np.loadtxt(dp, dtype = np.str, delimiter=',')
    Y = Y.astype(np.float32) # 10000x2048
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order = 'F')
    return Y 