import numpy as np
from torch.utils.tensorboard import SummaryWriter

x = np.random.randn(100)
y = x.cumsum() # xの累積和

# log_dirでlogのディレクトリを指定
writer = SummaryWriter(log_dir="logs")

# xとyの値を記録していく
for i in range(100):
    writer.add_scalar("x", x[i], i)
    writer.add_scalar("y", y[i], i)

# writerを閉じる
writer.close()