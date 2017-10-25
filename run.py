from model import RNNModel
from configs import *
import sys

configs = [LoadB, LoadBL, LoadBT, LoadBLT, noLoadBLT]
model = RNNModel(configs[int(sys.argv[1])-1])
model.run()
