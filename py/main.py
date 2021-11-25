import init_p0
import structure_net2
import copy
import hiddenlayer as hl
import torch
import torch.nn as nn
import data
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import time

if __name__ == '__main__':
    p0_list = init_p0.init_p(1)  # 初始种群有 K 个个体
    net_list = list()
    individual_list = list()
    population_list = list()
    for i in p0_list:  # 遍历每个个体
        for j in i:  # 遍历每个块
            for g in j:  # 遍历每个基因
                individual_list.append(g)
        population_list.append(copy.deepcopy(individual_list))
        individual_list.clear()

    print(population_list)
    data.printA()
    MyData = data.MyData()
    for index, value in enumerate(population_list):
        # 根据基因 生成 网络
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        net = structure_net2.StrNN(value)

        #  打印进化好的网络 基因序列
        print('网络基因序列_%d' % index, population_list[index])
        # print(net)

        #  画图出基因的网络结构
        hl_graph = hl.build_graph(net, torch.zeros([1, 1, 29, 29]))
        hl_graph.theme = hl.graph.THEMES['blue'].copy()
        hl_graph.save('net_%s' % index, format='png')

        # 设置网络超参数
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss().to(DEVICE)  # 二分类损失函数
        #  记录训练过程指标
        history1 = hl.History()
        #  使用Canvas进行可视化
        canvas1 = hl.Canvas()
        print_step = 25
        # 对训练模型进行迭代训练，对所有的数据训练 epoch 轮

        # 创建日志文件
        file = open('evo_search_log.txt', "w+")
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        file.write(str((population_list[index])) + '\n')

        net = net.to(DEVICE)  # 送入cuda
        for epoch in range(15):
            # 对训练数据的加载器进行迭代计算
            for step, (b_x, b_y) in enumerate(MyData.train_loader):
                # 计算每个batch的损失
                # print(b_x)
                net.train()  # 训练模式
                b_x = Variable(b_x).to(DEVICE)
                b_y = Variable(torch.squeeze(b_y)).to(DEVICE)

                print("====================")
                print('b_x.is_cuda: ', b_x.is_cuda)
                print('b_y.is_cuda: ', b_y.is_cuda)

                output = net(b_x).to(DEVICE)  # net 在训练 batch 上的输出
                print('output.is_cude: ', output.is_cuda)
                # output = torch.squeeze(output)

                # print('step: ', step)
                print('b_y: ', b_y)
                # print('type(b_x): ', type(b_x))
                # print('b_x.shape: ', b_x.shape)
                # print('----------------------')
                # print('type(b_y): ', type(b_y))
                # print('b_y.shape: ', b_y.shape)
                # print('=======================')
                # print('type(output): ', type(output))
                # print('output.shape: ', output.shape)
                # print('=-=-=-=-=-=-=-=-=-=-=-=')

                train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
                optimizer.zero_grad()  # 每个迭代步的梯度初始化为 0
                train_loss.backward()  # 损失的后向传播，计算梯度
                optimizer.step()  # 使用梯度进行优化

                niter = epoch*len(MyData.train_loader)+step+1
                # 计算每经过 print_step 次迭代后的输出
                if niter % print_step == 0:
                    print("= = = = = = = = = = =")
                    # print('MyData.x_valid_data.shape: ', MyData.x_valid_data.shape)
                    net.eval()
                    output = net(MyData.x_valid_data)
                    _, pre_lab = torch.max(output, 1)
                    # print('pre_lab: ', pre_lab)
                    pre_lab = pre_lab.cpu()
                    valid_accuracy = accuracy_score(MyData.y_valid_data, pre_lab)
                    # 为 history 添加 epoch 损失和精度
                    history1.log(niter, train_loss=train_loss, valid_accuracy=valid_accuracy)
                    # 使用两个突袭那个可视化损失函数和精度
                    with canvas1:
                        canvas1.draw_plot(history1["train_loss"])
                        canvas1.draw_plot(history1["valid_accuracy"])
                    log = "Epoch {:03d}: step {:07d}, Loss {:.4f}, TrainAcc {:.4}".format(epoch, step, train_loss.item(), valid_accuracy)
                    print(log)
                    file.write(log + '\n')

        file.close()