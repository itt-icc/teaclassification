import spectral.io.envi as envi
from spectral import *
import os
import numpy as np
import cv2
import wx
import matplotlib.pyplot as plt
import pandas as pd

#查看光谱的函数
def pltrgb(S):
    for i in range(S):
        plt.plot(s[i], label=str(i))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('wavelength', fontsize=18)
    plt.ylabel('ref', fontsize=18)
    plt.legend(fontsize=15)
    plt.show()

#观察掩膜
def showmask(mask):
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("mask", mask)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# 求均值的函数
def calmean(matrix):
    return 1.0*np.sum(matrix)/np.sum(np.array(matrix > 0))

#计算掩膜划分区域
def getsample(CUBE, HDR):
    sample = []
    for index in range(len(CUBE)):
        cue, hdr = CUBE[index], HDR[index]
        print(cue, hdr)
        hsp = envi.open(str(os.path.join(path, hdr)),
                        str(os.path.join(path, cue)))
        hyp = hsp[:, :]
        filename1 = 'rgb_'+cue[:-4]+'.jpg'
        save_rgb(filename1, hsp[:, :], [1, hsp[:, :].shape[2]-5, 40])
        img1 = cv2.imread(filename1)
        # 设置阈值
        thup = np.array(hsp[:, :, 138] > 2500, dtype='uint8')
        thdown = np.array(hsp[:, :, 5] < 2000, dtype='uint8')
        img = img1[:, :, 1]*thup[:, :, 0]*thdown[:, :, 0]
        # 闭运算
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
        # 开运算
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)
        # 腐蚀
        eroded = cv2.erode(opened, kernel2)
        # 显示一秒的掩膜
        showmask(eroded)

        mask = np.array(eroded > 0, dtype='uint8')
        hpspectra = np.zeros(hyp.shape, np.float)
        for i in range(hpspectra.shape[2]):
            hpspectra[:, :, i] = hyp[:, :, i]*mask
        s = []
        _col = mask.shape[1]
        # 每张图片采样20块区域
        for j in range(20):
            _region = hpspectra[:, int(_col/20*j):int(_col/20*(j+1)), 4:-5]
            _s = np.zeros((_region.shape[2]), np.float)
            for i in range(_region.shape[2]):
                _s[i] = calmean(_region[:, :, i])/10000
            s.append(_s)
        spec_df = pd.DataFrame(s, columns=name[4:-5])
        spec_df['year'] = year[Map[index]-1]
        sample.append(spec_df)
    return pd.concat(sample, axis=0)

#查看光谱
def PlotSpectrum(spec):
    plt.figure(figsize=(5.2, 3.1), dpi=600)
    col = spec.columns.values.tolist()
    x = np.linspace(float(col[0]), float(col[-1]), len(col))
    for i in range(spec.shape[0]):
        plt.plot(x, spec.iloc[i, :], linewidth=1)
    fonts = 10
    plt.xlim(float(col[0]), float(col[-1]))
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('absorbance (AU)', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)
    plt.tight_layout(pad=0.3)
    plt.grid(True)
    return plt


if __name__ == '__main__':
    #程序运行需要指定高光谱文件所在路径
    path = r"G:\光谱数据\短波"  # 待读取的文件夹
    df = pd.read_excel('波段名称.xlsx', header=0, index_col=0)
    name = df.columns.tolist()
    df1 = pd.read_excel('file.xlsx', header=0,
                        index_col=0)  # file这个文件夹是为了对应样本与年份
    year = df1['year'].tolist()
    Map = df1['map'].tolist()
    # 文件读取
    path_list = os.listdir(path)
    path_list.sort()
    for filename in path_list:print(os.path.join(path, filename))
    L_cue, L_hdr = [], []
    for i in path_list:
        if i[-1] == 'e':L_cue.append(i)
        else:L_hdr.append(i)
    L_cue.sort(), L_hdr.sort()
    Sample = getsample(CUBE=L_cue, HDR=L_hdr)
    Sample.reset_index(drop=True, inplace=True)
    Sample.to_csv('yangben.csv')
    Sample = pd.read_csv('yangben.csv', header=0, index_col=0)
    PlotSpectrum(Sample.iloc[:, 0:Sample.shape[1] - 1]).show()
