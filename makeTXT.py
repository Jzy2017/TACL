import os
import random
 
trainval_percent = 0.05
train_percent = 0.95
xmlfilepath = './chinamm2019uw_train/annotations'
txtsavepath = './chinamm2019uw_train/ImageSets'
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open('./chinamm2019uw_train/ImageSets/Main/trainval.txt', 'w')
ftest = open('./chinamm2019uw_train/ImageSets/Main/test.txt', 'w')
ftrain = open('./chinamm2019uw_train/ImageSets/Main/train.txt', 'w')
fval = open('./chinamm2019uw_train/ImageSets/Main/val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()