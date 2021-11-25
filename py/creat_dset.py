import copy
import numpy as np


def loadDataSet(fileName):
    # 创建空特征矩阵
    feature_mat = []  # 装有整个 29*29 的特征map
    feature_r = []  # 一个一个位的装 装完一行存入 feature_row
    feature_row = []  # 一行一行的装 装到 29 行存入 feature_mat
    # 创建空标签向量
    label_mat = []
    # 定义CAN ID的帧数 29
    can_frame = 29
    # 打开文件
    fr = open(fileName)
    # 按行遍历读取文件
    count = 0
    for line in fr.readlines():
        count += 1
        print(count)
        # 每一行先去掉回车换行符，再以Tab键为元素之间的分割符号，把每一行分割成若干元素
        lineArr = line.strip()
        lineArr = list(lineArr)
        # print('当前行是：', lineArr)
        # 向特征矩阵featureMat添加元素，即lineArr当前行的第0个元素和第一个元素
        # 特征矩阵featureMat实际上是二维列表，注意添加元素的方法和一维列表稍有不同
        # for j in range(can_frame):
        for i in lineArr:  # 添加一行 29 个二进制位
            feature_r.append(int(i))  # 共有 29 个二进制位
        feature_row.append(copy.deepcopy(feature_r))  # 存入一行
        feature_r.clear()
        if count == can_frame:  # 当收集 29 行之后
            feature_mat.append(copy.deepcopy(feature_row))
            feature_row.clear()
            count = 0  # 重新开始从 0 计数
            # print('当前的特征矩阵featureMat是：', feature_mat)
            # 向标签向量labelMat添加元素，即 正常报文为 0 DOS 异常报文为 1
            label_mat.append([1])
    # print('本数据文件的标签为：', label_mat)
    # print('本数据文件标签个数：', len(label_mat))

    # 所有行读取完毕后 关闭文件
    fr.close()
    # 函数返回特征矩阵列表 feature_mat 和向量标签
    return feature_mat, label_mat


if __name__ == '__main__':
    fileName = 'dataset/DoS_attack_656579.txt'

    feature_mat, label_mat = loadDataSet(fileName)
    feature_mat = np.array(feature_mat)  # 转换为 numpy 格式
    label_mat = np.array(label_mat)  # 转换为 numpy 格式
    print(feature_mat)
    print(label_mat)

    feature_size = len(feature_mat)
    label_size = len(label_mat)
    np.savez("dataset/DoS_attack_feature_%d.npz" % feature_size, feature_mat)
    np.savez("dataset/DoS_attack_label_%d.npz" % label_size, label_mat)

