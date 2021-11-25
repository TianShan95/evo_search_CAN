import numpy as np
import random

enc0 = np.load("dataset/Attack_free_feature_81703.npz")
a0 = enc0['arr_0']
enc0_label = np.load("dataset/Attack_free_label_81703.npz")
a0_label = enc0_label['arr_0']
print("插入之前的大小", len(a0))



enc1 = np.load("dataset/DoS_attack_feature_22640.npz")
b0 = enc1['arr_0']
enc1_label = np.load("dataset/DoS_attack_label_22640.npz")
b0_label = enc1_label['arr_0']
b0_split = np.split(b0, 20)  # 把入侵报文 特征 平均分为一定份数
b0_label_split = np.split(b0_label, 20)  # 把入侵报文 特征 平均分为一定份数

random.seed(42)
count = 0

for index, value in enumerate(b0_split):
    ins_num = random.randint(0, 81703)
    print('ins_num:', ins_num)
    a0 = np.insert(a0, ins_num, value, axis=0)  # 入侵报文特征 插入 正常报文特征
    a0_label = np.insert(a0_label, ins_num, b0_label_split[index], axis=0)  # 入侵报文标签 插入 正常报文标签

    count += 1
    print(len(value))
    print(count)

a0_size = len(a0)
a0_label_size = len(a0_label)
print('全部插入完成之后的 特征大小：', a0_size)
print('全部插入完成之后的 标签大小：', a0_label_size)

np.savez("dataset/mix_dos_feature_dset_%d" % a0_size, a0)
np.savez("dataset/mix_dos_label_dset_%d" % a0_label_size, a0_label)