import os
import sys
import math
import numpy as np

precision = 0.02
# path1 = './paddle' + '.txt'
path1 = './cpu1'
path2 = './result'
shader = np.loadtxt(path1,delimiter=',').flatten().tolist()
result = np.loadtxt(path2,delimiter=',').flatten().tolist()

if len(shader) != len(result):
    print("数据错误:paddle:" + str(len(shader)) + " fluid:" + str(len(result)))
else:
    print(path1)
    print(path2)
    print("数据长度：" + str(len(shader)))
    noError = True
    maxPrecision = 0
    maxIndex = 0

    for i in range(0, len(shader)):
        if (noError):
            if abs(shader[i] - result[i]) > precision:
                print("数据不对齐:" + str(i))
                print(shader[i])
                print(result[i])
                noError = False

        if (abs(shader[i] - result[i]) > maxPrecision):
            maxPrecision = abs(shader[i] - result[i])
            maxIndex = i

    if noError:
        print("数据完全对齐")
    else:
        print('最大精度误差为：', maxPrecision)
        print('index为', maxIndex)
