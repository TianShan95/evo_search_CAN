import torch.nn as nn
import torch
from torch.autograd import Variable

operation_list1 = ['sw3', 'sw5', 'sw7', 'dw3', 'dw5', '3*3', None]
operation_list = ['sw3', 'sw5', 'sw7', 'dw3', 'dw5', '3*3', None]


class StrNN(nn.Module):
    def __init__(self, individual_list):
        super(StrNN, self).__init__()

        self.individual_list = individual_list
        # 定义 stem
        self.stem = nn.Sequential(  # 输入是：29 * 29 * 1 输出：13 * 13 * 128 最后 nn.ReLu 非线性
            nn.Conv2d(  # stem1 3*3 卷积 channel: 1->32
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                bias=True,
                stride=(1, 1),
                padding=1
            ),  # 输出：29 * 29 * 32
            nn.Conv2d(  # stem2 3*3 卷积 stride=1 but no padding channel: 32->32
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                bias=True,
                stride=(1, 1),
                padding=0
            ),  # 输出：27 * 27 * 32
            nn.MaxPool2d(  # stem3 3*3 MaxPool stride=2
                kernel_size=3,
                stride=2,
            ),  # 输出：13 * 13 * 32
            nn.Conv2d(  # stem4 1*1 卷积 channel: 32->64
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 1),
                bias=True,
                stride=(1, 1),
                padding=0
            ),  # 输出：13 * 13 * 64
            nn.Conv2d(  # stem5 3*3 卷积 channel: 64->128
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                bias=True,
                stride=(1, 1),
                padding=1
            ),  # 输出：13 * 13 * 128
            nn.Conv2d(  # stem6  3*3 卷积 channel: 128->128
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                bias=True,
                stride=(1, 1),
                padding=1
            ),  # 输出：13 * 13 * 128
            nn.ReLU(inplace=True)  # resA 开始的relu 输出：13 * 13 * 128
        )

        # 定义 general
        # resA 3 个基因 预定义 3个【1 * 1 卷积】 和 1个【1 * 1 Linear】  ==> 4 个卷积
        # 连接 Linear
        self.resA_conv1_1 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=32,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：13 * 13 * 32
        # 连接 基因0
        self.resA_conv1_2 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=32,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：13 * 13 * 32
        # 连接 基因1
        self.resA_conv1_3 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=32,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：13 * 13 * 32
        # resA Linear
        self.resA_linear = nn.Conv2d(  # 输入 13 * 13 * 96
            in_channels=96,
            out_channels=128,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：13 * 13 * 128
        # resA 输出的relu
        self.resA_relu = nn.ReLU(inplace=True)

        # resB 1 个基因 预定义 2个【1 * 1 卷积】 和 1个【1 * 1 Linear】==> 3 个卷积
        # 连接 Linear
        self.resB_conv1_1 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=64,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 64
        # 连接 基因3
        self.resB_conv1_2 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=64,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 64
        # resB Linear
        self.resB_linear = nn.Conv2d(  # 输入 6 * 6 * 128
            in_channels=128,
            out_channels=448,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 448
        # resB 输出的relu
        self.resB_relu = nn.ReLU(inplace=True)

        # redA 1 个基因 预定义 1个【3 * 3 MaxPool】 和 2个【3 * 3 卷积】 和 1个【1 * 1 卷积】==> 4 个卷积
        # 连接 redA filter concat
        self.redA_maxp = nn.MaxPool2d(  # 输入：13 * 13 * 128
            kernel_size=3,
            stride=2,
        )  # 输出：6 * 6 * 128
        # 连接 redA filter concat
        self.redA_conv3_192 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=192,
            kernel_size=(3, 3),
            bias=True,
            stride=(2, 2),
            padding=0
        )  # 输出：6 * 6 * 192
        # 连接 基因4
        self.redA_conv1 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=96,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：13 * 13 * 96
        # 连接 redA filter concat
        self.redA_conv3_128 = nn.Conv2d(  # 输入：13 * 13 * 96
            in_channels=96,
            out_channels=128,
            kernel_size=(3, 3),
            bias=True,
            stride=(2, 2),
            padding=0
        )  # 输出：6 * 6 * 128
        # redA 输出的relu resB 输入的relu
        self.redA_relu = nn.ReLU(inplace=True)

        # redB 3 个基因 预定义 1个【3 * 3 MaxPool】和 1个【3 * 3 卷积】 和 3个【1 * 1 卷积】==> 5 个卷积
        # 连接 redB concat
        self.redB_maxp = nn.MaxPool2d(  # 输入：6 * 6 * 448
            kernel_size=3,
            stride=2,
        )  # 输出：2 * 2 * 448
        # 连接 redB concat
        self.redB_conv3 = nn.Conv2d(  # 输入：6 * 6 * 128
            in_channels=128,
            out_channels=192,
            kernel_size=(3, 3),
            bias=True,
            stride=(2, 2),
            padding=0
        )  # 输出：2 * 2 * 192
        # 连接 redB 3*3 卷积
        self.redB_conv1_1 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 128
        # 连接 基因5
        self.redB_conv1_2 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 128
        # 连接 基因6
        self.redB_conv1_3 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=(1, 1),
            bias=True,
        )  # 输出：6 * 6 * 128

        # 根据传进来的list 构建 基因部分的 component
        for index, value in enumerate(self.individual_list):
            size = None
            channel = None
            padding = None
            stride = 1  # 默认步幅为 1 只有 基因5 基因7 存在 stride 为 3
            if value == 0 or value == 2 or value == 4:
                size = 3  # sw3 dw3 3*3
                padding = 1  # 尺寸3 padding = 1 相当于 same
            if value == 1 or value == 3:
                size = 5  # sw5 dw5
                padding = 2  # 尺寸5 padding = 2 相当于 same

            if index <= 2:  # resA
                channel = 32  # resA 输入和输出通道 都是 32
            elif index <= 3:  # redA
                channel = 96  # redA 输入和输出通道 都是 96
            elif index <= 4:  # resB
                channel = 64  # resB 输入和输出通道 都是 64
            elif index <= 7:  # redB 输入和输出通道 都是 128
                channel = 128
                if index == 5 or index == 7:  # redB 基因5 和 基因7 的padding 是 0
                    padding = 0
                    stride = 3
            if value <= 1:  # sw3 and sw5
                # globals()['gen%s_op' % index] = nn.Sequential(
                #     nn.Conv2d(
                #         in_channels=channel,
                #         out_channels=channel,
                #         kernel_size=(1, size),
                #         bias=False,
                #         padding=padding,
                #         stride=stride
                #     ),
                #     nn.Conv2d(
                #         in_channels=channel,
                #         out_channels=channel,
                #         kernel_size=(size, 1),
                #         bias=False,
                #         padding=padding,
                #         stride=stride
                #     )
                # )
                exec(f'''self.gen{index}_op = nn.Sequential(
                    nn.Conv2d(
                        in_channels={channel},
                        out_channels={channel},
                        kernel_size=(1, {size}),
                        bias=True,
                        padding=(0, {padding}),
                        stride={stride}
                    ),
                    nn.Conv2d(
                        in_channels={channel},
                        out_channels={channel},
                        kernel_size=({size}, 1),
                        bias=True,
                        padding=({padding}, 0),
                        stride={stride}
                    )
                )''')
                continue  # sw3 or sw5
            if value <= 3:  # dw3 and dw5
                # globals()['gen%s_op' % index] = nn.Sequential(
                #     nn.Conv2d(
                #         in_channels=channel,
                #         out_channels=channel,
                #         kernel_size=(size, size),
                #         bias=False,
                #         padding=padding,
                #         groups=channel,
                #         stride=stride
                #
                # )
                # )
                exec(f'''self.gen{index}_op = nn.Sequential(
                    nn.Conv2d(
                        in_channels={channel},
                        out_channels={channel},
                        kernel_size=({size}, {size}),
                        bias=True,
                        padding={padding},
                        groups={channel},
                        stride={stride}
                )
                )''')
                continue  # dw3 or dw5
            if value == 4:  # 3*3 卷积
                # globals()['gen%s_op' % index] = nn.Sequential(
                #     nn.Conv2d(
                #         in_channels=channel,
                #         out_channels=channel,
                #         kernel_size=(size, size),
                #         bias=False,
                #         padding=padding,
                #         stride=stride
                #     )
                # )
                exec(f'''self.gen{index}_op = nn.Sequential(
                    nn.Conv2d(
                        in_channels={channel},
                        out_channels={channel},
                        kernel_size=({size}, {size}),
                        bias=True,
                        padding={padding},
                        stride={stride}
                    )
                )''')
                continue  # 3*3

        self.bottom = nn.Sequential(  # redB输入bottom：2 * 2 * 896 bottom输出：1
            nn.AvgPool2d(
                kernel_size=2,
                stride=1
            ),
            nn.Dropout(p=0.5),

            # nn.Linear(896, 2),
            # nn.Softmax(dim=1)  # 通道上做 softmax
        )

        self.classifier = nn.Sequential(  # 输入：2 输出： 2
            nn.Linear(896, 2, bias=True),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):  # 输入 x： 29 * 29 * 1

        x = self.stem(x)  # stem 输出(已经过relu) x： 13 * 13 * 128

        # return x
        # print('=======')
        # print(dir(self))

        # resA 输入：13 * 13 * 128 输出：13 * 13 * 128
        x_resA_conv1_1 = self.resA_conv1_1(x)  # 1*1 卷积 直连 Linear
        x_resA_conv1_2 = self.resA_conv1_2(x)  # 1*1 卷积 作为 基因1 输入
        x_resA_conv1_3 = self.resA_conv1_3(x)  # 1*1 卷积 作为 基因2 输入

        x_resA_gen0_return = self.gen0_op(x_resA_conv1_2)  # 基因0
        x_resA_gen1_return = self.gen1_op(x_resA_conv1_3)  # 基因1
        x_resA_gen2_return = self.gen2_op(x_resA_gen1_return)  # 基因2
        # print('x_resA_conv1_1.shape :', x_resA_conv1_1.shape)
        # print('x_resA_gen0_return.shape :', x_resA_gen0_return.shape)
        # print('x_resA_gen2_return.shape :', x_resA_gen2_return.shape)
        resA_out = torch.cat((x_resA_conv1_1, x_resA_gen0_return, x_resA_gen2_return), 1)
        resA_out = self.resA_linear(resA_out)
        resA_out += x
        resA_out = self.resA_relu(resA_out)

        # redA 输入：13 * 13 * 128 输出：6 * 6 * 448
        x_redA_MaxPool_return = self.redA_maxp(resA_out)
        x_redA_conv3_192_return = self.redA_conv3_192(resA_out)
        x_redA_conv1_return = self.redA_conv1(resA_out)  # 1*1 卷积 作为 基因3 输入

        x_redA_gen3_return = self.gen3_op(x_redA_conv1_return)  # 基因3
        x_redA_conv3_128_return = self.redA_conv3_128(x_redA_gen3_return)
        redA_out = torch.cat((x_redA_MaxPool_return, x_redA_conv3_192_return, x_redA_conv3_128_return), 1)
        redA_out = self.redA_relu(redA_out)

        # resB 输入：6 * 6 * 448 输出：6 * 6 * 448
        x_resB_conv1_1 = self.resB_conv1_1(redA_out)
        x_resB_conv1_2 = self.resB_conv1_2(redA_out)

        x_resB_gen4_return = self.gen4_op(x_resB_conv1_2)  # 基因4
        resB_out = torch.cat((x_resB_conv1_1, x_resB_gen4_return), 1)
        resB_out = self.resB_linear(resB_out)
        resB_out += redA_out
        resB_out = self.resB_relu(resB_out)

        # redB 输入：6 * 6 * 448 输出：2 * 2 * 896
        x_redB_MaxPool_return = self.redB_maxp(resB_out)
        x_redB_conv1_1 = self.redB_conv1_1(resB_out)
        x_redB_conv1_2 = self.redB_conv1_2(resB_out)
        x_redB_conv1_3 = self.redB_conv1_3(resB_out)
        x_redB_conv3_return = self.redB_conv3(x_redB_conv1_1)

        # print('x_redB_conv1_2.shape :', x_redB_conv1_2.shape)
        # print('x_redB_conv1_3.shape :', x_redB_conv1_3.shape)


        x_redB_gen5_return = self.gen5_op(x_redB_conv1_2)  # 基因5
        x_redB_gen6_return = self.gen6_op(x_redB_conv1_3)  # 基因6
        # print('x_redB_gen6_return.shape :', x_redB_gen6_return.shape)
        x_redB_gen7_return = self.gen7_op(x_redB_gen6_return)  # 基因7

        redB_out = torch.cat((x_redB_MaxPool_return, x_redB_conv3_return, x_redB_gen5_return, x_redB_gen7_return), 1)

        out = self.bottom(redB_out)
        out = Variable(out.view(-1, 896))
        # print('out::::\n', out)
        # print(len(out))

        out = self.classifier(out)
        # print(out)
        # print(out.shape)
        # out = out[:, 0]
        # print('out1', out[:, 0])
        # out = torch.max(out, dim=1)[1].data
        # out = out[0].data
        # out = torch.unsqueeze(out, dim=1).float()
        # print('out', out)
        return out