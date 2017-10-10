import time, os
from jinja2 import Template
from matplotlib import pyplot as plt
class Log:
    def __init__(self, iterations, time_steps, **kwargs):
        self.error = []
        self.accuracy = []
        self.f1 = []
        self.fname = "log_%s_iterations_%s_time_steps_%s" % (iterations, time_steps, time.strftime("%I:%M%p"))
        self.string_builder = "%s iterations, %s time steps\n" % (iterations, time_steps)
        self.string_builder += "args --\n"
        for k, v in kwargs.items():
            self.string_builder += "{}: {}".format(str(k), str(v)) +"\n"
        self.string_builder += "-- args\nLOG:\n"     

    def log(self, line):
        print(line)
        self.string_builder += line + "\n"
    
    def save(self):
        if not os.path.exists("logs/"+self.fname):
            os.mkdir("logs/"+self.fname)
        with open("log/"+self.fname+"/out", "w+") as f:
            f.write(self.string_builder)
        if self.accuracy:
            self.render_accuracy()
        if self.error:
            self.render_error()

    def render_accuracy(self):
        if not os.path.exists(self.fname+"/plots"):
            os.mkdir(self.fname+"/plots")
        x = map(lambda x: x[0], self.accuracy)
        y = map(lambda x: x[1], self.accuracy)
        plt.plot(x, y)
        plt.savefig(self.fname + "/plots/accuracy.png")
    
    def render_error(self):
        if not os.path.exists(self.fname+"/plots"):
            os.mkdir(self.fname+"/plots")
        x = map(lambda x: x[0], self.error)
        y = map(lambda x: x[1], self.error)
        plt.plot(x, y)
        plt.savefig(self.fname + "/plots/error.png")

        

    
