import numpy as np
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

#tensorboard --logdir="Inpainting"
#cd Documents/Git/MyInpainting2020
import csv
#LossのCSVからつくる

#python train_all.py --dataset=Inpainting_food --batchSize=10 --testBatchSize=15 --nEpochs=200 --cuda --threads=0

#Result_Array
writer = SummaryWriter(log_dir="Inpainting")

with open('LossForBoard//loss_log_result_old100-20-55.csv') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        #print(row) #rowは1行文の配列
        #row_float = float(row)
        if(i != 0):
            writer.add_scalar("Epoch",float(row[0]),i-1)
            writer.add_scalar("Train_Loss_G",float(row[1]),i-1)
            writer.add_scalar("Train_Loss_D",float(row[2]),i-1)
            writer.add_scalar("Train_Loss_Dg_R",float(row[3]),i-1)
            writer.add_scalar("Train_Loss_Dg_F",float(row[4]),i-1)
            writer.add_scalar("Train_Loss_Dl_R",float(row[5]),i-1)
            writer.add_scalar("Train_Loss_Dl_F",float(row[6]),i-1)
            writer.add_scalar("Train_Loss_De_R",float(row[7]),i-1)
            writer.add_scalar("Train_Loss_De_F",float(row[8]),i-1)
            #writer.add_scalar("Test_Loss_G",float(row[9]),i-1)
            #writer.add_scalar("Test_Loss_D",float(row[10]),i-1)
            #writer.add_scalar("Test_Loss_Dg_R",float(row[11]),i-1)
            #writer.add_scalar("Test_Loss_Dg_F",float(row[12]),i-1)
            #writer.add_scalar("Test_Loss_Dl_R",float(row[13]),i-1)
            #writer.add_scalar("Test_Loss_Dl_F",float(row[14]),i-1)
            #writer.add_scalar("Test_Loss_De_R",float(row[15]),i-1)
            #writer.add_scalar("Test_Loss_De_F",float(row[16]),i-1)

            #for j in range(len(row)):
                #writerに追加

        i += 1

writer.close()
print("end")
'''
# ログをとる対象を増やしてみる
x1 = np.random.randn(100)
y1 = x1.cumsum() 

x2 = np.random.randn(100)
y2 = x2.cumsum() 

writer = SummaryWriter(log_dir="many_logs")

for i in range(100):


# tagの書き方に注目！
for i in range(100):
    writer.add_scalar("X/x1", x1[i], i)
    writer.add_scalar("Y/y1", y1[i], i)
    writer.add_scalar("X/x2", x2[i], i)
    writer.add_scalar("Y/y2", y2[i], i)
writer.close()
'''