import numpy as np 
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import os
context.set_context(device_target='Ascend')

class BasicBlock(nn.Cell):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = stride, padding=1, pad_mode='pad',  has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = 1, padding=1, pad_mode = 'pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            ])
    
    def construct(self, x):
        out = P.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = P.ReLU()(out)
        return out

class CNN_Estimation(nn.Cell):
    def __init__(self, block = BasicBlock, num_blocks = [2,2,2,2], num_classes = 256):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride = 1, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Dense(512*1*32, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        out = P.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = P.Reshape()(out, (len(out), -1))
        out = self.linear(out)
        return out

class DeepRx_SD(nn.Cell):
    def __init__(self, block = BasicBlock, num_blocks = [2,2,2,2], num_classes = 1024):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride = 1, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Dense(512*2*32, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        out = P.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = P.Reshape()(out, (len(out), -1))
        out = self.linear(out)
        return out

class MS_FC_Estimation(nn.Cell):
    def __init__(self, in_dim, h_dim, out_dim, n_blocks):
        super().__init__()
        self.input_layer = nn.SequentialCell(
            nn.Dense(in_dim, h_dim),
            nn.ELU(),
            nn.BatchNorm1d(h_dim)
        )
        hidden_layers = []
        for i in range(n_blocks):
            hidden_layers.extend([
                    nn.Dense(h_dim, h_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(h_dim)]
            )
        self.hidden_layers = nn.CellList(hidden_layers)
        self.output_layer = nn.Dense(h_dim, out_dim)
    
    def construct(self, x):
        out = self.input_layer(x)
#         for l in self.input_layer:
#             out = l(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

def NMSE(h1, h2):
    mse = ((h1-h2)**2).sum()
    norm = (h2**2).sum()
    return mse / norm 

def process_H(H): # bs x 2 x 4 x 32
    # 真实的频域信道，获取标签
    batch_size = len(H)
    Hf_train = np.array(H)[:, 0, :] + 1j * np.array(H)[:, 1, :]
    Hf_train = np.fft.fft(Hf_train, 256)/20 # 4*256
    Hf = np.stack([Hf_train.real, Hf_train.imag], axis = 1) # bsx2x4x256
#     return ms.Tensor(Hf, dtype = ms.float16)
    return Hf.reshape(batch_size, -1)

def get_val_data():
    d = np.load('../evaluation.npy', allow_pickle = True)
    a = d.item()
    X = a['x']
    Y = a['y']
    H = a['h']
    return X, Y, H

def val():
    FC = MS_FC_Estimation(1024, 4096, 2048, 2)
    CNN = CNN_Estimation()

    fc_param = load_checkpoint('../from_tc.ckpt')
    cnn_param = load_checkpoint('../checkpoints_my/cnn.ckpt')

    load_param_into_net(FC, fc_param)
    load_param_into_net(CNN, cnn_param)
    FC.set_train(False)
    CNN.set_train(False)

    X, Y, H = get_val_data() # Y: bsx2x2x2x256
    n = 500
    h = H[:n, :]

    hf_label = process_H(h)

    yp4fc = ms.Tensor(Y[:n,:,0,:].reshape(n, -1), dtype=ms.float32)
    yp4cnn = ms.Tensor(Y[:n, :, 0, :, :], dtype=ms.float32)
    yd = ms.Tensor(Y[:n,:,1,:,:], dtype=ms.float32)
    hf = FC(yp4fc)

    print(NMSE(hf.asnumpy(), hf_label))

    hf = P.Reshape()(hf, (n, 2, 4, 256))
    cnn_input = P.Concat(axis = 2)((yd, yp4cnn, hf))
    print(cnn_input.shape)
    ht = CNN(cnn_input)
    ht = P.Reshape()(ht, (n, 2, 4, 32)).asnumpy()
    with open('./ht.txt', 'w') as f:
        f.write(str(ht[:2,:]))

    print(NMSE(ht, h))

def get_test_data(Pn):
    tag = 1 if Pn == 32 else 2
    dp = os.path.join(f'../dataset/Y_{tag}.csv')
    print(f'loading test data from {dp}')
    Y = np.loadtxt(dp, dtype = np.str, delimiter=',')
    Y = Y.astype(np.float32) # 10000x2048
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order = 'F')
    return Y 

def test_P8():
    N = 10000
    y8 = get_test_data(8)
#     yp4fc = ms.Tensor(y8[:, :, 0, :].reshape(N, -1), dtype=ms.float32)
#     yp4cnn = ms.Tensor(y8[:, :, 0, :], dtype=ms.float32)
    
    FC = MS_FC_Estimation(1024, 4096, 2048, 2)
    CNN = CNN_Estimation()
    SD = DeepRx_SD()

    fc_param = load_checkpoint('../checkpoints_my/fc.ckpt')
    cnn_param = load_checkpoint('../checkpoints_my/cnn.ckpt')
    sd_param = load_checkpoint('../checkpoints_my/sd.ckpt')

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
    label = np.fromfile('./checkpoints_my.bin', dtype = np.bool)
    label = label.reshape([10000, 1024])
    print(f'Pn=8, similarity between mindspore and torch inference: {(X==label).mean()}')

def test_P32():
    N = 10000
    y32 = get_test_data(32)
#     yp4fc = ms.Tensor(y8[:, :, 0, :].reshape(N, -1), dtype=ms.float32)
#     yp4cnn = ms.Tensor(y8[:, :, 0, :], dtype=ms.float32)
    
    FC = MS_FC_Estimation(1024, 4096, 2048, 2)
    CNN = CNN_Estimation()
    SD = DeepRx_SD()

    fc_param = load_checkpoint('../checkpoints_wxh/fc.ckpt')
    cnn_param = load_checkpoint('../checkpoints_wxh/cnn.ckpt')
    sd_param = load_checkpoint('../checkpoints_wxh/sd.ckpt')

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
    label = np.fromfile('../checkpoints_wxh/X_pre_1.bin', dtype = np.bool)
    label = label.reshape([10000, 1024])
    print(f'Pn=32, similarity between mindspore and torch inference: {(X==label).mean()}')

if __name__ == '__main__':
    test_P32()
    test_P8()