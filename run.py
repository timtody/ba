from model import RNNModel
from configs import *
import sys

configs = [LoadB, LoadB_wodeb, LoadBL, LoadBT, LoadBLT, LoadBK]
model = RNNModel(configs[int(sys.argv[1])-1])
model.run()
