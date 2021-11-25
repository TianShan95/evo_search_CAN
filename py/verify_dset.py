import numpy as np
np.set_printoptions(threshold=np.inf)


# enc0 = np.load("dataset/Attack_free_feature_81703.npz")
# a0 = enc0['arr_0']
# a = a0[0:2]
# print(a)
# print(len(a0))  # Attack_free feature 中共有多少个数据
#
#
# enc1 = np.load("dataset/DoS_attack_feature_22640.npz")
# b0 = enc1['arr_0']
# b = b0[0]
# print(b)
# print(len(b0))  # Dos_attack feature 中共有多少个数据

# enc1_label = np.load("dataset/DoS_attack_label_22640.npz")
# b0_label = enc1_label['arr_0']
#
# print(len(b0_label))  # Dos_attack feature 中共有多少个数据

# # 把 b 插入 a
# c = np.insert(a, 1, b, axis=0)
# print(c)

# 校验 入侵报文 在 正常报文 中的 分布
enc0 = np.load("dataset/mix_dos_label_dset_104343.npz")
a0 = enc0['arr_0']
train_count = 0
valid_count = 0
test_count = 0
for index, value in enumerate(a0):
    # print(value)
    if value[0] == 1:
        if index <= 66780:
            train_count += 1
        elif index <= 83474:
            valid_count += 1
        else:
            test_count += 1

print(train_count/66780)
print(valid_count/(83474 - 66780))
print(test_count/(104343 - 83474))