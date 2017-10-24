class BaseConfig:
    batch_size = 100
    time_steps = 4
    lateral = False
    top_down = False
    iterations = 10000
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


class ExpConfig1(BaseDIGITConfig):
    pass


class ExpConfig2(BaseDIGITConfig):
    debris = True


class ExpConfig3(ExpConfig2):
    lateral = True


class ExpConfig4(ExpConfig3):
    top_down = True


class ExpConfig5(ExpConfig4):
    sum = True


class ExpConfig6(ExpConfig4):
    time_steps = 6


class ExpConfigMNIST1(BaseMNISTConfig):
    pass


class ExpConfigMNIST2(BaseMNISTConfig):
    lateral = True


class ExpConfigMNIST3(ExpConfigMNIST2):
    top_down = True


class ExpConfig7(ExpConfig6):
    time_steps = 3
    sum = True


class ExpConfig8(ExpConfig6):
    time_steps = 6
    sum = True


class ExpConfig9(BaseDIGITConfig):
    debris = True
    lateral = True
    top_down = True
    kernel_size = 3
    sum = False

class VanillaConfig(BaseDIGITConfig):
    time_steps = 3


class BConfig(BaseDIGITConfig):
    debris = False 
    save = True
    

class LoadBase(BConfig):
    from_checkpoint = True
    checkpoint_path = "/home/taylor/ba/checkpoints/B.ckpt"
    

class LoadB(BConfig):
    time_steps = 1
    eval_steps = 1
    debris = True

    
class LoadBL(LoadBase):
    debris = True
    lateral = True


class LoadBT(LoadBase):
    debris = True
    top_down = True


class LoadBLT(LoadBase):
    debris = True
    top_down = True
    lateral = True