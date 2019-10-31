# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import string
import time
import math

# 输入：文件名
# 输出：二维矩阵，每一行代表一个训练样本里的两个句子以及相似度
def load_data(filename):
    file=open(filename,'r',encoding='utf-8')
    result=[]
    for line in file:
        # 取第4列到第6列
        # 分别是 相似度 句子1 句子2
        temp=[]
        li = line.split('\t')[4:7]
        temp.append(float(li[0]))
        for sentence in [li[1],li[2]]:
            for punc in string.punctuation:
                sentence=sentence.replace(punc,' ')
            sentence=sentence.replace('  ',' ')
            sentence=sentence.strip().split(' ')

            temp.append(sentence)
        result.append(temp)

    return result

# 构建词典
# 输入是train和dev的数据
def get_word_dict(train_data,dev_data):
    word_set=set()
    dict={}
    # cnt储存字的位置
    cnt=0
    for data in [train_data,dev_data]:
        for v in data:
            # 句子1和句子2
            for s in [v[1],v[2]]:
                for word in s:
                    if word not in word_set:
                        word_set.add(word)
                        dict[word]=cnt
                        cnt+=1
    return dict

# 获得表示成向量的词语
def get_word_vect(data,word_dict):
    result=[]
    # 获取词典长度，加1是为了存放不在词典中出现的单词
    dict_len=len(word_dict)+1
    word_set=set(word_dict.keys())

    # 每个词的初始向量
    vector = [0] * dict_len

    # 对每一行操作
    for v in data:
        temp=[]
        temp.append(v[0])
        for s in [v[1],v[2]]:
            # 每个句子表示为二维矩阵，行数是词的个数，列数是词表的大小
            sentence=[]
            for word in s:
                vector_copy=vector.copy()
                if word in word_set:
                    vector_copy[word_dict[word]]=1
                else:
                    vector_copy[-1]=1
                sentence.append([vector_copy])
            temp.append(sentence)
        result.append(temp)
        # print(temp)
    return result

class RNN(nn.Module):
    # 输入为：x的特征的维度，隐藏层的特征的维度，输出的维度
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化神经网络
        super(RNN, self).__init__()
        # 隐藏层的维度
        self.hidden_size = hidden_size
        # 输入：输入的input以及上一层的隐层状态，输出：本层隐层状态
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 本层的输出
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # dim=1说明在第1维上求概率。即不加log时，[0][0]+[0][1]+[0][2]...+[0][n]=1
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 沿着第1维拼接两个张量
        combined = torch.cat((input, hidden), 1)
        # 该层的隐状态
        hidden = self.i2h(combined)
        # 该层的输出
        output = self.i2o(combined)
        # 将输出进行映射
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size).to(device)


# 参数为 相似度，第一个句子,第二个句子，原始相似度得分 ,第一个rnn模型，第二个rnn模型，以及学习率
def train(sentence_tensor1,sentence_tensor2, original_score, rnn1, rnn2,learning_rate,criterion):
    hidden1 = rnn1.initHidden()

    hidden2 = rnn2.initHidden()
    # 置梯度为0
    rnn1.zero_grad()
    rnn2.zero_grad()

    # 两个模型的输出
    output1=0
    output2=0

    # 将句子里的每一个单词输入进模型
    for i in range(sentence_tensor1.size()[0]):
        # print(sentence_tensor1[i])
        output1, hidden1 = rnn1(sentence_tensor1[i], hidden1)
    for i in range(sentence_tensor2.size()[0]):
        output2, hidden2 = rnn2(sentence_tensor2[i], hidden2)

    # 计算两个output的余弦相似度
    cos_simi=torch.cosine_similarity(output1,output2)
    # 计算得分，区间为0-5
    predict_score=5*cos_simi
    loss=criterion(predict_score,original_score)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn1.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    for p in rnn2.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return predict_score, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    print('%dm %ds' % (m, s))


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Now using: "+str(device))
    # 开始时间
    prerocess_start = time.time()

    train_data_raw=load_data('data/sts-train.txt')
    dev_data_raw=load_data('data/sts-dev.txt')
    # test_data_raw=load_data('data/sts-test.txt')[:100]

    word_dict=get_word_dict(train_data_raw,dev_data_raw)
    dict_len=len(word_dict)+1
    print("The length of the word_list is "+str(dict_len))
    train_data=get_word_vect(train_data_raw,word_dict)
    dev_data=get_word_vect(dev_data_raw,word_dict)
    # test_data=get_word_vect(test_data_raw,word_dict)

    # print(train_data[:3])
    print("Done preprocessing data")
    timeSince(prerocess_start)
    #---------------------以上是数据处理部分----------------------------

    # 隐层的维度
    n_hidden = 128
    # 输出向量的维度
    n_output = 20
    # 损失函数
    criterion = nn.SmoothL1Loss()
    # 句子1的RNN模型
    rnn1 = RNN(dict_len, n_hidden, n_output).to(device)
    # 句子2的RNN模型
    rnn2 = RNN(dict_len, n_hidden, n_output).to(device)
    # 迭代次数
    n_iters = 10
    # 学习率
    learning_rate=0.5

    # print_every = 5000
    # plot_every = 1000

    # 一次迭代的总损失
    current_loss = 0
    # 最好的一次损失
    best_loss=float('inf')
    # 保存每次迭代的总损失
    all_losses = []
    # 记录预测的结果
    all_score=[]


    train_start_time=time.time()
    # 迭代的次数
    for iter in range(1, n_iters + 1):
        current_loss=0
        temp_all_score=[]
        # 遍历训练数据的每一行
        iter_start_time=time.time()
        for feature in train_data:
            original_score=torch.tensor(feature[0],dtype=torch.float).to(device)
            sentence_tensor1=torch.tensor(feature[1],dtype=torch.float).to(device)
            sentence_tensor2=torch.tensor(feature[2],dtype=torch.float).to(device)

            # 参数为 相似度，第一个句子,第二个句子，原始相似度得分 ,第一个rnn模型，第二个rnn模型，以及学习率
            predict_score,loss=train(sentence_tensor1, sentence_tensor2, original_score, rnn1, rnn2, learning_rate, criterion)
            temp_all_score.append(predict_score.data.item())
            current_loss += loss

        # 记录本次迭代总损失
        all_losses.append(current_loss)
        if current_loss<best_loss:
            best_loss=current_loss
            all_score=temp_all_score
            torch.save(rnn1.state_dict(), "rnn1_params.pkl")
            torch.save(rnn2.state_dict(), "rnn2_params.pkl")
        print(current_loss)

        timeSince(iter_start_time)

    print(all_score)
    timeSince(train_start_time)

