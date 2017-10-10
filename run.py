from model import RNNModel
from configs import *
import sys

configs = [VanillaConfig, Exp1, Exp2, Exp3]
model = RNNModel(configs[int(sys.argv[1])-1])
