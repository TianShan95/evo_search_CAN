import re
import io
import numpy as np

can_id_list = list()
file = 'dataset/DoS_attack_dataset.txt'
with io.open(file, 'rt', encoding='UTF-8') as fd:
    for line in fd:
        can_id_hex = re.findall(r"ID: (.*)    DLC:", line, re.M)[0]  # 把can id字符串取出
        can_id_hex = can_id_hex.replace(' ', '')
        can_id_bin = (bin(int(can_id_hex, 16)))[2:].zfill(29)
        # print(can_id_bin)
        # print(can_id_hex)
        can_id_list.append(can_id_bin)

len_can_id_list = len(can_id_list)
print(len_can_id_list)

can_id_file = 'dataset/DoS_attack_%d.txt' % len_can_id_list
with io.open(can_id_file, "w") as f:
    for i in can_id_list:
        f.writelines(i)
        f.writelines('\n')