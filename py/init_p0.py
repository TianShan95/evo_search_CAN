import random
import copy


def init_p(K):
    '''
    :param K 需要生成个体的数量:
    :return: 初始化个体的列表
    population_list : 群体列表 这个列表存有本代的所有个体 共有 K 个个体
    individual_list : 个体列表 这个列表存有本个体的所有基因 每个个体有四个块列表
    block_list : 块的列表 每个块列表由4个基因组成 共有7中基因类型 分别用 0 - 6 7个数字表示

    '''
    random.seed(42)
    population_list = list()
    individual_list = list()
    block_list = list()

    # block_type_index = 4  # 设置两个类型的block 两个res 两个reduction
    node_type_index_list = [3, 1, 1]  # 基因数量：resA 3 redA 1 resB 1
    node_type_redB_index = 3  # redB 单独随机生成基因 只有在 【2， 4】中产生
    node_type_redB_list = [2, 4]  # redB 只有选择 3 种基因
    gene_type_index_redB = 2  # redB 有 2 种基因类型 dw3 3*3
    gene_type_index = 5  # 有 5 种基因类型 sw3 sw5 dw3 dw5 3*3

    rand_min = 0  # 基因产生器中最小的数字

    for k in range(K):  # 初始化每个群体

        for n in node_type_index_list:  # 每个块每个块的初始化基因
            for s in range(n):  # 初始化每个基因

                block_list.append(random.randint(rand_min, rand_min + gene_type_index - 1))
            individual_list.append(copy.deepcopy(block_list))
            block_list.clear()
        # print(individual_list)
        population_list.append(copy.deepcopy(individual_list))
        # print(population_list)
        individual_list.clear()

    for i in population_list:  # 给种群种每个个体添加 redB
        for j in range(node_type_redB_index):  # redB 有 3 个基因
            rand = random.randint(rand_min, rand_min + gene_type_index_redB - 1)
            block_list.append(node_type_redB_list[rand])
        i.append(copy.deepcopy(block_list))
        block_list.clear()

    return population_list


if __name__ == '__main__':
    print(init_p(3))