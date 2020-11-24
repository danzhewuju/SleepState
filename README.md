# 项目说明
## 数据集
项目使用的数据集来自于：公开数据集
利用深度学习的框架，来完成睡眠的分类的任务；

## 模型
基于RCNN模型可以有效的完成睡眠分期的任务；
值得注意的是：基于传统的方法更的是从波形的特征的角度出发，但是这样往往会有大量的信息被遗漏；
RCNN模型被常用于nlp的文本分类的相关任务中，应用在脑电信号处理的过程中具有一定的创新；
论文链接：https://link.springer.com/chapter/10.1007/978-3-030-00847-5_42

## 运行环境
torch           1.6.0

torchvision     0.7.0

tqdm            4.50.2

mne             0.21.dev0

numpy           1.19.2

Python          3.7

pandas          1.1.3

其中MNE是有名的脑电信号的处理框架，具体使用见官网：https://mne.tools/
pip 安装方法：pip install -U https://api.github.com/repos/mne-tools/mne-python/zipball/master

## 代码框架
1. 将目标数据集解压到文件夹下面

2. 利用preprocess 来做数据预处理，会对原数据进行切分出训练数据集和验证数据集，生成的文件会生成到config文件夹下面；
 
    1. 进入 preprocess 文件夹下面
 
    运行
 
    2. Python ./Run.py -dp dataSplit  
    这个会自动从原始文件中提取特定的文件，并在dataset 里面创建6个文件夹 W, R, S1, S2, S3, S4
 并且将各个对应的数据切分为30s的片段；
 
    3. Python ./Run.py -dp createDataset  执行该命令可以将提取的数据在config目录下自动创建训练（train.csv）验证（val.csv） 测试（test.csv）、
 文件进行后面训练和测试；
  
   3. 利用模型进行训练；

       1. 进入model文件夹下面
    
        2. 训练：python ./Run.py -m train -bs 64 -lr 0.001 -ep 50 -gpu 0
    
        3. 测试：python ./Run.py -m test
    
        4. 测试结果会自动保存到./log/log.txt 里面

## 实验结果
Accuracy: 71.80%


