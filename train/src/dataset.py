import numpy as np 
import random
import math 
import torch
import datetime
from torch.utils.data import Dataset
import struct
import time
import math
import os

mu = 2
K = 256
CP = 32


def print_something():
    print('utils.py has been loaded perfectly')


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x) ** 2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB


def Modulation(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return 0.7071 * (2 * bit_r[:, 0] - 1) + 0.7071j * (2 * bit_r[:, 1] - 1)  # This is just for QAM modulation


def deModulation(Q):
    Qr=np.real(Q)
    Qi=np.imag(Q)
    bits=np.zeros([64,2])
    bits[:,0]=Qr>0
    bits[:,1]=Qi>0
    return bits.reshape([-1])  # This is just for QAM modulation

def Modulation1(bits, mu):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (bit_r[:, 0]) + 1j * (bit_r[:, 1])


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]  # take the last CP samples ...
    # cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):

    convolved = np.convolve(signal, channelResponse)

    sigma2 = 0.35 * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))



def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                  Clipping_Flag):

    # --- training inputs ----

    CR=1
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, 2 * K)
    # OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)  # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,  K)
    OFDM_RX_noCP = np.fft.fft(OFDM_RX_noCP)
    # OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword, mu)
    if len(codeword_qam) != K:
        print('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    # OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)  # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP,  K)
    OFDM_RX_noCP_codeword = np.fft.fft(OFDM_RX_noCP_codeword)
    AA = np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))


    CC=OFDM_RX_noCP/np.max(AA)
    BB = np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))

    return np.concatenate((AA, BB)), CC  # sparse_mask


def MIMO(X, HMIMO, SNRdb,flag,P):
    P = P * 2
    Pilot_file_name = '../dataset/Pilot_' + str(P)
    bits = np.loadtxt(Pilot_file_name, delimiter = ',')
    # if os.path.isfile(Pilot_file_name):
    #     bits = np.loadtxt(Pilot_file_name, delimiter=',')
    # else:
    #     print('Invalid Pilot file.')
    #     bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
    #     np.savetxt(Pilot_file_name, bits, delimiter=',')
    pilotValue = Modulation(bits, mu)


    if flag==1:
        cpflag, CR = 0, 0
    elif flag==2:
        cpflag, CR = 0, 1
    else:
        cpflag, CR = 1, 0
    allCarriers = np.arange(K)
    pilotCarriers = np.arange(0, K, K // P)
    dataCarriers = [val for val in allCarriers if not (val in pilotCarriers)]



    bits0=X[0]
    bits1=X[1]
    pilotCarriers1 = pilotCarriers[0:P:2]
    pilotCarriers2 = pilotCarriers[1:P:2]
    signal_output00, para = ofdm_simulate(bits0, HMIMO[0,:], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                                    pilotCarriers1, dataCarriers, CR)
    signal_output01, para = ofdm_simulate(bits0, HMIMO[1, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[0:P:2],
                                          pilotCarriers1, dataCarriers, CR)
    signal_output10, para = ofdm_simulate(bits1, HMIMO[2, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)
    signal_output11, para = ofdm_simulate(bits1, HMIMO[3, :], SNRdb, mu, cpflag, K, P, CP, pilotValue[1:P:2],
                                          pilotCarriers2, dataCarriers, CR)

    signal_output0=signal_output00+signal_output10
    signal_output1=signal_output01+signal_output11
    output=np.concatenate((signal_output0, signal_output1))
    output=np.transpose(np.reshape(output,[8,-1]),[1,0])

    #print(np.shape(signal_output00))
    return np.reshape(output,[-1])

class RandomDataset():
    def __init__(self, H, Pilot_num, SNRdb=-1, mode=0):
        super().__init__()
        self.H = H
        self.Pilot_num = Pilot_num
        self.SNRdb = SNRdb
        self.mode = mode

    def __getitem__(self, index):
        # return 
        # Yp: 1024 can be directedly fed into FC 
        # Yp4cnn: 2x2x256 used for cnn net, can be directedly cat with hf and Yd
        # Yd: 2x2x256  used for cnn net, can be directedly cat with hf and Yp4cnn
        HH = self.H[index]
        seed = math.floor(math.modf(time.time())[0]*500*320000)**2 % (2**32 - 2)
        np.random.seed(seed)
        # print(seed)
        # print(np.random.randn())
        bits0 = np.random.binomial(1, 0.5, size=(128 * 4,))
        bits1 = np.random.binomial(1, 0.5, size=(128 * 4,))
        SS = self.SNRdb
        mm = self.mode
        if self.SNRdb == -1:
            SS =  np.random.uniform(8, 12)
        if self.mode == -1:
            mm = np.random.randint(0, 3)
        YY = MIMO([bits0, bits1], HH, SS, mm, self.Pilot_num)/20
        YY = np.reshape(YY, [2, 2, 2, 256], order = 'F').astype('float32')
        Yp = YY[:, 0, :, :]
        Yd = YY[:, 1, :, :]
        Yp4cnn = Yp.copy()
        Yp4fc = Yp.reshape(-1)
        XX = np.concatenate([bits0, bits1], 0).astype('float32')
        newHH = np.stack([HH.real, HH.imag], axis = 0).astype('float32')  # 2x4x32
        return Yp4fc, Yp4cnn, Yd, XX, newHH
    def __len__(self):
        return len(self.H)
        # return 2000

def get_YH_data_random(mode, Pn):
    N1 = 320000
    data1 = open('../dataset/H.bin','rb')
    H1 = struct.unpack('f'*2*2*2*32*N1,data1.read(4*2*2*2*32*N1))
    H1 = np.reshape(H1,[N1,2,4,32])
    H_tra = H1[:,1,:,:]+1j*H1[:,0,:,:]   # time-domain channel for training 

    data2 = open('../dataset/H_val.bin','rb')
    H2 = struct.unpack('f'*2*2*2*32*2000,data2.read(4*2*2*2*32*2000))
    H2 = np.reshape(H2,[2000,2,4,32])
    H_val = H2[:,1,:,:]+1j*H2[:,0,:,:] 
    trainset = RandomDataset(H_tra, Pilot_num=Pn, mode = mode)
    valset = RandomDataset(H_val, Pilot_num=Pn, mode=mode)
    return trainset, valset

def get_val_data(Pn, mode):
    H_path = os.path.join('../dataset/H_val.bin')
    H_data = open(H_path, 'rb')
    H = struct.unpack('f'*2*2*2*32*2000, H_data.read(4*2*2*2*32*2000))
    H = np.reshape(H, [2000, 2, 4, 32]).astype('float32')
    # H_label = np.reshape(H, [len(H), -1])
    H = H[:, 1, :, :] + 1j*H[:, 0, :, :]
    val_set = RandomDataset(H, Pilot_num=Pn, mode = mode)
    return val_set
