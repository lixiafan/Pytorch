import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.nn import CrossEntropyLoss

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root='../dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_data_len = len(train_data)
test_data_len = len(test_data)
# print("训练数据的长度为： {}".format(train_data_len)) # 50000
# print("测试数据的长度为： {}".format(test_data_len)) # 10000

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
model = Model()

# 定义损失函数
loss_fn = CrossEntropyLoss()

# 定义优化器
learning_rate = 0.01 # 1e-2
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_model")

for i in range(epoch):
    print("----------第 {} 轮训练开始------------".format(i+1))

    # 训练步骤开始
    # model.train()
    for data in train_dataloader:
        imgs, targets = data # 获取数据
        outputs = model(imgs) # 输入模型后得到预测输出
        loss = loss_fn(outputs,targets) # 计算预测输出和数据集中的真实标签的损失

        # 优化器优化模型
        optimizer.zero_grad() # 将优化器的梯度清零
        loss.backward() # loss反向传播
        optimizer.step() # 开始使用优化器进行优化

        total_train_step += 1 # 训练步加一
        if total_train_step % 100 == 0:
            print("训练次数: {},  Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每训练完一轮，在测试集上进行评估，判断模型的参数有没有训练好
    # 测试步骤开始
    # model.eval()
    total_test_loss = 0
    total_accuracy = 9
    with torch.no_grad(): # 不进行调优
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    test_accuracy = torch.true_divide(total_accuracy, test_data_len)
    print("整体测试集上的loss： {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(test_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    # 测试完之后保存模型的参数
    torch.save(model,"./weight/model_{}.pth".format(i))
    # torch.save(model.state_dict(), "model_{}".format(i)) # 官方推荐的网络模型的保存方式
    print("模型已保存！")

writer.close()
