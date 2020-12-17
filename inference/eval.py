import numpy as np 
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import os
from src.models import *
from src.dataset import *
context.set_context(device_target='Ascend')


def infer_P8():
    N = 10000
    y8 = get_test_data(8)
#     yp4fc = ms.Tensor(y8[:, :, 0, :].reshape(N, -1), dtype=ms.float32)
#     yp4cnn = ms.Tensor(y8[:, :, 0, :], dtype=ms.float32)
    
    FC = MS_FC_Estimation(1024, 4096, 2048, 2)
    CNN = CNN_Estimation()
    SD = DeepRx_SD()

    fc_param = load_checkpoint('./checkpoints_my/fc.ckpt')
    cnn_param = load_checkpoint('./checkpoints_my/cnn.ckpt')
    sd_param = load_checkpoint('./checkpoints_my/sd.ckpt')

    load_param_into_net(FC, fc_param)
    load_param_into_net(CNN, cnn_param)
    load_param_into_net(SD, sd_param)
    FC.set_train(False)
    CNN.set_train(False)
    SD.set_train(False)
    
    bs = 1000
    X = []
    for i in range(10):
        yy8 = y8[bs*i:bs*(i+1)]
        yp4fc = ms.Tensor(yy8[:,:,0,:,:].reshape(bs,-1), dtype = ms.float32)
        yp4cnn = ms.Tensor(yy8[:,:,0,:,:], dtype = ms.float32)
        yd = ms.Tensor(yy8[:,:,1,:,:], dtype = ms.float32)
        hf = FC(yp4fc)
        hf = P.Reshape()(hf, (bs, 2, 4, 256))
        cnn_input = P.Concat(axis=2)((yd, yp4cnn, hf))
        ht = CNN(cnn_input)
        ht = P.Reshape()(ht, (bs, 2, 4, 32))
        
        ht = ht.asnumpy()
        hf = process_H(ht).reshape(bs, 2, 4, 256)
        hf = ms.Tensor(hf, dtype=ms.float32)
        sd_input = P.Concat(axis=2)((yd, hf))
        sd_input = P.Reshape()(sd_input, (bs, 1, 12, 256))
        x = SD(sd_input).asnumpy()
        X.append(x)
        print(f'{i+1}/10 complete.')
    X = np.concatenate(X, axis = 0)
    X = (X>0.5)
    X.tofile('./X_pre_2.bin')
    label = np.fromfile('./checkpoints_my/X_pre_2.bin', dtype = np.bool)
    label = label.reshape([10000, 1024])
    print(f'Pn=8, similarity between mindspore and torch inference: {(X==label).mean()}')

def infer_P32():
    N = 10000
    y32 = get_test_data(32)
#     yp4fc = ms.Tensor(y8[:, :, 0, :].reshape(N, -1), dtype=ms.float32)
#     yp4cnn = ms.Tensor(y8[:, :, 0, :], dtype=ms.float32)
    
    FC = MS_FC_Estimation(1024, 4096, 2048, 2)
    CNN = CNN_Estimation()
    SD = DeepRx_SD()

    fc_param = load_checkpoint('./checkpoints_wxh/fc.ckpt')
    cnn_param = load_checkpoint('./checkpoints_wxh/cnn.ckpt')
    sd_param = load_checkpoint('./checkpoints_wxh/sd.ckpt')

    load_param_into_net(FC, fc_param)
    load_param_into_net(CNN, cnn_param)
    load_param_into_net(SD, sd_param)
    FC.set_train(False)
    CNN.set_train(False)
    SD.set_train(False)
    
    bs = 1000
    X = []
    for i in range(10):
        yy32 = y32[bs*i:bs*(i+1)]
        yp4fc = ms.Tensor(yy32[:,:,0,:,:].reshape(bs,-1), dtype = ms.float32)
        yp4cnn = ms.Tensor(yy32[:,:,0,:,:], dtype = ms.float32)
        yd = ms.Tensor(yy32[:,:,1,:,:], dtype = ms.float32)
        hf = FC(yp4fc)
        hf = P.Reshape()(hf, (bs, 2, 4, 256))
        cnn_input = P.Concat(axis=2)((yd, yp4cnn, hf))
        ht = CNN(cnn_input)
        ht = P.Reshape()(ht, (bs, 2, 4, 32))
        
        ht = ht.asnumpy()
        hf = process_H(ht).reshape(bs, 2, 4, 256)
        hf = ms.Tensor(hf, dtype=ms.float32)
        sd_input = P.Concat(axis=2)((yd, hf))
        sd_input = P.Reshape()(sd_input, (bs, 1, 12, 256))
        x = SD(sd_input).asnumpy()
        X.append(x)
        print(f'{i+1}/10 complete.')
    X = np.concatenate(X, axis = 0)
    X = (X>0.5)
    X.tofile('./X_pre_1.bin')
    label = np.fromfile('./checkpoints_wxh/X_pre_1.bin', dtype = np.bool)
    label = label.reshape([10000, 1024])
    print(f'Pn=32, similarity between mindspore and torch inference: {(X==label).mean()}')

if __name__ == '__main__':
    infer_P32()
    infer_P8()