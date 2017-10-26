class BaseConfig:
    batch_size = 100
    time_steps = 4
    lateral = False
    top_down = False
    iterations = 12000
    kernel_size = 3
    sum = False
    eval_steps = 6
    save = False
    from_checkpoint = False


class BaseMNISTConfig(BaseConfig):
    dataset = "mnist"


class BaseDIGITConfig(BaseConfig):
    dataset = "digits"
    debris = False


class BConfig(BaseDIGITConfig):
    debris = False 
    save = True
    iterations = 2000
    time_steps = 4
    eval_steps = 5
    

class LoadBase(BConfig):
    from_checkpoint = False
#    checkpoint_path = "/home/taylor/ba/checkpoints/B.ckpt"
    

class LoadB(BConfig):
    time_steps = 1
    eval_steps = 1
    iterations = 25000
    debris = False

    
class LoadBL(LoadBase):
    time_steps = 3
    eval_steps = 3
    iterations = 25000
    debris = True
    lateral = True


class LoadBT(LoadBase):
    time_steps = 3
    eval_steps = 3
    iterations = 25000
    debris = True
    top_down = True


class LoadBLT(LoadBase):
    time_steps = 3
    eval_steps = 3
    iterations = 25000
    debris = True
    top_down = True
    lateral = True

class noLoadBLT(LoadBLT):
    from_checkpoint = False


class TestConf(BaseDIGITConfig):
    iterations = 10
