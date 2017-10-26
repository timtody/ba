class BaseConfig:
    batch_size = 100
    time_steps = 1
    lateral = False
    top_down = False
    iterations = 12000
    kernel_size = 3
    sum = False
    eval_steps = 1
    save = False
    from_checkpoint = False
    model = "base"
    learning_rate = 7e-2


class BaseMNISTConfig(BaseConfig):
    dataset = "mnist"


class BaseDIGITConfig(BaseConfig):
    dataset = "digits"
    debris = False


class BConfig(BaseDIGITConfig):
    debris = False 
    save = True
    iterations = 15000
    

class LoadBase(BConfig):
    from_checkpoint = False
    

class LoadB_wodeb(BConfig):
    time_steps = 1
    eval_steps = 1
    debris = False
    model = "saveB_nodebris"

class LoadB(BConfig):
    time_steps = 1
    eval_steps = 1
    debris = True
    model = "saveB"

    
class LoadBL(LoadBase):
    time_steps = 3
    eval_steps = 3
    debris = True
    lateral = True
    model = "saveBL"


class LoadBT(LoadBase):
    time_steps = 3
    eval_steps = 3
    debris = True
    top_down = True
    model = "saveBT"


class LoadBLT(LoadBase):
    time_steps = 3
    eval_steps = 3
    debris = True
    top_down = True
    lateral = True
    model = "saveBLT"


class LoadBK(LoadBase):
    debris = True
    model = "saveBK"
    kernel_size = 5

