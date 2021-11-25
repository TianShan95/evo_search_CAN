import torch.nn as nn
import torch

operation_list1 = ['sw3', 'sw5', 'sw7', 'dw3', 'dw5', '3*3', None]
operation_list = ['sw3', 'sw5', 'sw7', 'dw3', 'dw5', '3*3', None]


class StrNN(nn.Module):
    def __init__(self, individual_list):
        super(StrNN, self).__init__()

        self.individual_list = individual_list

        # for index, value in enumerate(individual_list):  # 遍历这个 个体 列表 内有 4 个块 共 16 个基因
        #     block_index = int(index/4)  # 确定此时的通道数 ResA 32 ReductionA 96 ResB 64 ReductionB 128
        #     channels = None
        #     if block_index == 0: channels = 32
        #     if block_index == 0: channels = 96
        #     if block_index == 0: channels = 64
        #     if block_index == 0: channels = 128
        # 定义 stem
        self.stem = nn.Sequential(  # 输入是：29 * 29 * 1 输出：13 * 13 * 128 最后 nn.ReLu 非线性
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                bias=False,
                stride=1,
                padding='same'
            ),  # 输出：29 * 29 * 32
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                bias=False,
                stride=1,
                padding=0
            ),  # 输出：27 * 27 * 32
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),  # 输出：13 * 13 * 32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                bias=False,
                stride=1,
                padding='same'
            ),  # 输出：13 * 13 * 64
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                bias=False,
                stride=1,
                padding='same'
            ),  # 输出：13 * 13 * 128
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                bias=False,
                stride=1,
                padding='same'
            ),  # 输出：13 * 13 * 128
            nn.ReLU(inplace=True)
        )

        # 定义 general
        # resA 3 个基因 预定义 3个【1 * 1 卷积】 和 1个【1 * 1 Linear】  ==> 4 个卷积
        self.resA_conv1_1 = nn.Conv2d(  # 输入：13 * 13 * 128
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                bias=False,
                stride=1,
                padding='same'
            )  # 输出：13 * 13 * 32
        self.resA_conv1_2 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=32,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：13 * 13 * 32
        self.resA_conv1_3 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=32,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：13 * 13 * 32
        self.resA_linear = nn.Conv2d(  # 输入 13 * 13 * 96
            in_channels=96,
            out_channels=128,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：13 * 13 * 128
        self.resA_relu = nn.ReLU(inplace=True)

        # resB 1 个基因 预定义 2个【1 * 1 卷积】 和 1个【1 * 1 Linear】==> 3 个卷积
        self.resB_conv1_1 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=64,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 64
        self.resB_conv1_2 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=64,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 64
        self.resB_linear = nn.Conv2d(  # 输入 6 * 6 * 128
            in_channels=128,
            out_channels=448,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 448
        self.resB_relu = nn.ReLU(inplace=True)

        # redA 1 个基因 预定义 1个【3 * 3 MaxPool】 和 2个【3 * 3 卷积】 和 1个【1 * 1 卷积】==> 4 个卷积
        self.redA_maxp = nn.MaxPool2d(  # 输入：13 * 13 * 128
            kernel_size=3,
            stride=2,
        )  # 输出：6 * 6 * 128
        self.redA_conv3_192 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=192,
            kernel_size=3,
            bias=False,
            stride=2,
            padding=0
        )  # 输出：6 * 6 * 192
        self.redA_conv1 = nn.Conv2d(  # 输入：13 * 13 * 128
            in_channels=128,
            out_channels=96,
            kernel_size=3,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 192
        self.redA_conv3_128 = nn.Conv2d(  # 输入：13 * 13 * 96
            in_channels=96,
            out_channels=128,
            kernel_size=3,
            bias=False,
            stride=2,
            padding=0
        )  # 输出：6 * 6 * 128
        self.redA_relu = nn.ReLU(inplace=True)

        # redB 3 个基因 预定义 1个【3 * 3 MaxPool】和 1个【3 * 3 卷积】 和 3个【1 * 1 卷积】==> 5 个卷积
        self.redB_maxp = nn.MaxPool2d(  # 输入：6 * 6 * 448
            kernel_size=3,
            stride=2,
        )  # 输出：2 * 2 * 448
        self.redB_conv3 = nn.Conv2d(  # 输入：6 * 6 * 128
            in_channels=128,
            out_channels=192,
            kernel_size=3,
            bias=False,
            stride=2,
            padding=0
        )  # 输出：6 * 6 * 192
        self.redB_conv1_1 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 128
        self.redB_conv1_2 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 128
        self.redB_conv1_3 = nn.Conv2d(  # 输入：6 * 6 * 448
            in_channels=448,
            out_channels=128,
            kernel_size=1,
            bias=False,
            stride=1,
            padding='same'
        )  # 输出：6 * 6 * 128


        channel_list = [32, 64, 96, 128]
        sw_list = [3, 5]
        dw_list = [3, 5]
        stride_list = [1, 2]
        for s in stride_list:  # 两种 stride
            for i in channel_list:  # 四种 channel

                for j in sw_list:  # 构造 sw 操作 两种 命名：stride_channel_type(sw)
                    exec(f'''self.conv_sw_{str(s)}_{str(i)}_{str(j)} = nn.Sequential(
nn.Conv2d(
    in_channels={i},
    out_channels={i},
    kernel_size=(1, {j}),
    bias=False,
    stride={s},
    padding='same'
),
nn.Conv2d(
    in_channels={i},
    out_channels={i},
    kernel_size=({j}, 1),
    bias=False,
    stride={s},
    padding='same'))''')

                for j in dw_list:  # 构造 dw 操作
                    exec(f'''
self.conv_dw_{str(s)}_{str(i)}_{str(j)}= nn.Sequential(
    nn.Conv2d(
        in_channels={i},
        out_channels={i},
        kernel_size={j},
        bias=False,
        stride={s},
        padding='same',
        groups={i},
    ),
                        )''')

                # 构造 3*3 卷积
                exec(f'''
self.conv_3_{str(s)}_{str(i)} = nn.Sequential(
    nn.Conv2d(
        in_channels={i},
        out_channels={i},
        kernel_size=3,
        bias=False,
        stride={s},
        padding='same',
    ),
                    )''')

        self.bottom = nn.Sequential(  # redB输入bottom：2 * 2 * 896 bottom输出：1
            nn.AvgPool2d(
                kernel_size=2,
                stride=1
            ),
            nn.Dropout(p=0.2),
            nn.Softmax(dim=0)  # 通道上做 softmax
        )

    def forward(self, x):  # 输入 x： 29 * 29 * 1

        x = self.stem(x)  # stem 输出 x： 13 * 13 * 128

        return x

        # # resA 输入：13 * 13 * 128 输出：13 * 13 * 128
        # x_resA = None  # resA块的整体输出
        # x_resA_op_return1 = None  # 操作 1 的输出 直连 Linear
        # x_resA_op_return2 = None  # 操作 2 的输出 作为 基因3 的输入
        # x_resA_op_return3 = None  # 操作 3 的输出 直连 Linear
        # x_resA_conv1_1 = self.resA_conv1_1(x)  # 1*1 卷积 直连 Linear
        # x_resA_conv1_2 = self.resA_conv1_2(x)  # 1*1 卷积 作为 基因1 输入
        # x_resA_conv1_3 = self.resA_conv1_3(x)  # 1*1 卷积 作为 基因2 输入
        #
        # # redA 输入：13 * 13 * 128 输出：6 * 6 * 448
        # x_redA = None  # redA块的整体输出
        # x_redA_MaxPool_return = None
        # x_redA_conv3_192_return = None
        # x_redA_conv1_return = None
        # x_redA_conv3_96_return = None
        # x_redA_conv3_128_return = None
        #
        # # resB 输入：6 * 6 * 448 输出：6 * 6 * 448
        # x_resB_conv1_1 = None  # 1*1 卷积 直连 Linear
        # x_resB = None
        #
        # # redB 输入：6 * 6 * 448 输出：2 * 2 * 896
        # x_redB = None
        # x_redB_MaxPool_return = None
        # x_redB_conv3_return = None
        # x_redB_conv1_1 = None
        # x_redB_conv1_2 = None
        # x_redB_conv1_3 = None
        # x_redB_op_return_1 = None
        # x_redB_op_return_2 = None
        # x_redB_op_return_3 = None
        #
        #
        # for index, value in self.individual_list:  # [[block1],[block2],[block3],[block4]]
        #
        #     # resA 3 个基因 (index = 0 - 2) 输入：13 * 13 * 128 输出：13 * 13 * 128  relu 输出
        #     if index == 0:
        #         x_resA_op = self.decode_op(value, 1, 32)
        #         x_resA_op_return1 = x_resA_op(x_resA_conv1_2)  # 基因1 的输出
        #         continue
        #     if index == 1:
        #         x_resA_op = self.decode_op(value, 1, 32)
        #         x_resA_op_return2 = x_resA_op(x_resA_conv1_3)  # 基因2 的输出 ｜ 基因3 的输入
        #     if index == 2:
        #         x_resA_op = self.decode_op(value, 1, 32)
        #         x_resA_op_return3 = x_resA_op(x_resA_op_return2)  # 基因3 的输出
        #
        #         x_resA_out_2_linear = torch.cat((x_resA_conv1_1, x_resA_op_return1, x_resA_op_return3), 1)
        #         x_resA = self.resA_linear(x_resA_out_2_linear)
        #         x_resA += x  # 得到 resA 块 的输出
        #         x_resA = self.resA_relu(x_resA)  # resA 结束的relu
        #
        #     # redA 1 个基因 (index = 3 - 3) 输入：13 * 13 * 128 输出：6 * 6 * 448 concate
        #     if index == 3:
        #         x_redA_MaxPool_return = self.redA_maxp(x_resA)
        #         x_redA_conv3_192_return = self.redA_conv3_192(x_resA)
        #         x_redA_conv1_return = self.redA_conv1(x_resA)  # 作为 基因4 的输入
        #
        #         redA_op = self.decode_op(value, 1, 96)
        #         x_redA_conv3_96_return = redA_op(x_redA_conv1_return)
        #         x_redA_conv3_128_return = self.redA_conv3_128(x_redA_conv3_96_return)
        #
        #         x_redA = torch.cat((x_redA_MaxPool_return, x_redA_conv3_192_return, x_redA_conv3_128_return), 1)
        #         x_redA = self.redA_relu(x_redA)  # resB 开始的relu
        #
        #     # resB 1 个基因 (index = 4 - 4) 输入：6 * 6 * 448 输出：6 * 6 * 448
        #     if index == 4:
        #         x_resB_conv1_1 = self.resB_conv1_1(x_redA)  # 1*1 卷积 直连 Linear
        #         x_resB_conv1_2 = self.resB_conv1_2(x_redA)  # 1*1 卷积 作为 基因5 的输入
        #
        #         x_resB_op = self.decode_op(value, 1, 64)
        #         x_resB_op_return = x_resB_op(x_resB_conv1_2)
        #         x_resB_out_2_linear = torch.cat(x_resB_conv1_1, x_resB_op_return)
        #         x_resB = self.resB_linear(x_resB_out_2_linear)  # 输出 6 * 6 * 448
        #         x_resB += x_redA
        #         x_resB = self.resB_relu(x_resB)  # resB 结束的relu
        #
        #     # redB 3个基因 (index = 5 - 7) 输入：6 * 6 * 448 输出：2 * 2 * 896
        #     if index == 5:
        #
        #         # 初始化 block redB 的标准组件
        #         x_redB_MaxPool_return = self.redB_maxp(x_resB)
        #         x_redB_conv1_1 = self.redB_conv1_1(x_resB)  # 3*3*192 的输入
        #         x_redB_conv3_return = self.redB_conv3(x_redB_conv1_1)
        #         x_redB_conv1_2 = self.redB_conv1_2(x_resB)  # 3*3*128 的输入 基因6 的输入
        #         x_redB_conv1_3 = self.redB_conv1_3(x_resB)  # 3*3*128 的输入 基因7 的输入
        #
        #         x_redB_op = self.decode_op(value, 2, 128)
        #         x_redB_op_return_1 = x_redB_op(x_redB_conv1_2)  # 基因6 的输出
        #
        #     if index == 6:
        #         x_redB_op = self.decode_op(value, 1, 128)
        #         x_redB_op_return_2 = x_redB_op(x_resB_conv1_3)  # 基因7 的输出 ｜ 基因8 的输入
        #
        #     if index == 7:
        #         x_redB_op = self.decode_op(value, 2, 128)
        #         x_redB_op_return_3 = x_redB_op(x_redB_op_return_2)
        #
        #         x_redB = torch.cat((x_redB_MaxPool_return, x_redB_conv3_return, x_redB_op_return_1, x_redB_op_return_3), 1)
        #
        #         # bottom
        #         out = self.bottom(x_redB)
        #         return out




    def decode_op(self, code, stride, channel):
        component = None
        if code == 0:  # conv_sw_stride_channel_3
            exec(f'''component = self.conv_sw_{str(stride)}_{str(channel)}_{str(3)}''')
        if code == 1:  # conv_sw_stride_channel_5
            exec(f'''component = self.conv_sw_{str(stride)}_{str(channel)}_{str(5)}''')
        if code == 2:  # conv_dw_stride_channel_3
            exec(f'''component = self.conv_dw_{str(stride)}_{str(channel)}_{str(3)}''')
        if code == 3:  # conv_dw_stride_channel_5
            exec(f'''component = self.conv_dw_{str(stride)}_{str(channel)}_{str(5)}''')
        if code == 4:  # conv_3_stride_channel
            exec(f'''component = self.conv_3_{str(stride)}_{str(channel)}''')

        return component