# TAAC2020
1.403576	onlineAcc

## 待研究
+ embedding层是否需要在预训练word2vector基础上在进一步Fine-tuning，即各embedding层的trainable参数的设置
+ 是否该将全部信息在保留行内结构的情况下合成一个大的dict.txt进行处理
+ transformer的结构是采用单层多头还是多层少头
+ 将BiLSTM或者LSTM接在transformer前是否有提高（不看好，因为输入时序信息弱）
+ 按大的dict.txt处理时，将放入序列的行提前放入Conv1D、RNN、LSTM以压缩尺寸是否会有提高
+ 将BiLSTM或者LSTM接在transformer后是否有提高
