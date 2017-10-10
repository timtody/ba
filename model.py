import tensorflow as tf
import digits, tempfile, time, sys, os
from log import Log
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
class RNNModel:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps
        self.lateral =  config.lateral
        self.top_down = config.top_down
        self.iterations = config.iterations
        self.kernel = config.kernel_size
        self.dataset_name = config.dataset
        self.debris = config.debris
        self.sum = config.sum

        if self.dataset_name == "mnist":
            self.dataset = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
        elif self.dataset_name == "digits":
            self.dataset = digits.Digits("../../DIGIT_data/light_debris{}.mat".\
            format("_with_debris" if self.debris else ""), split=.97)
        else:
            raise BaseException("dataset not specified")
        self.imsize = self.dataset.train.images[0].shape[0]
        self.image_dim1 = int(np.sqrt(self.imsize))
        self.x = tf.placeholder('float', [None, self.imsize])
        self.y_ = tf.placeholder('float', [None, 10])
        # bottom up weights
        self.conv_1 = {'weights': self.weight_variable([self.kernel, self.kernel, 1, 32], 9),
                    'biases': self.bias_variable([32])}
        self.conv_2 = {'weights': self.weight_variable([self.kernel, self.kernel, 32, 64], 9*32),
                    'biases': self.bias_variable([64])}
        self.read = {'weights': self.weight_variable([(self.image_dim1**2)//16, 10]),
                    'biases': self.bias_variable([10])}
        # lateral weights
        self.l1_lateral = self.weight_variable([self.kernel, self.kernel, 32, 32])
        self.l2_lateral = self.weight_variable([self.kernel, self.kernel, 64, 64])
        # top down weights
        self.td_filter = self.weight_variable([self.image_dim1//2, self.image_dim1//2, 32, 64])
        self.td_output_shape = tf.constant([self.image_dim1, self.image_dim1, 32])
        # dropout parmeter
        self.keep_prob = tf.placeholder(tf.float32)
        # prelu weights
        self.alphas_l1 = tf.variable(tf.zeros([32]))
        self.alphas_l2 = tf.variable(tf.zeros([64]))
        # build the layers
        self.build_graph()
        self.define_objective()
        # run the graph
        self.run()

    def define_objective(self):
        with tf.name_scope('loss'):
            if sum:
                self.cross_entropy = tf.reduce_sum(
                    [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.layers[layer]["yHat"]))
                    for layer in self.layers])
            if not sum:
                self.cross_entropy = tf.reduce_mean(
                    [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.layers[layer]["yHat"]))
                    for layer in self.layers])
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).\
                minimize(self.cross_entropy)
        with tf.name_scope('accuracy'):
                self.outputs = [self.layers[layer]["yHat"] for layer in self.layers]
                self.correct_predictions = [tf.equal(tf.argmax(yhat, 1), tf.argmax(self.y_, 1)) for yhat in self.outputs]
                self.accuracies = [tf.reduce_mean(tf.cast(e, tf.float32)) for e in self.correct_predictions]
        for i in range(self.time_steps):
            tf.summary.scalar('accuracy_t{}'.format(i), self.accuracies[i])
        
        # Merge all the summaries 
        self.merged = tf.summary.merge_all()
        graph_location = "/home/taylor/ba/tensorboard/"       
        self.train_writer = tf.summary.FileWriter(graph_location+ \
        "prelu__debris:{}_ds:{}_lt:{}_td:{}_time_depth:{}_sum:{}_ks:{}".format(self.debris, \
        self.dataset_name, self.lateral, self.top_down, self.time_steps, self.sum, self.kernel))

        self.train_writer.add_graph(tf.get_default_graph())

        self.test_writer = tf.summary.FileWriter(graph_location+ \
        "prelu__debris:{}_ds:{}_lt:{}_td:{}_time_depth:{}_sum:{}".format(self.debris, \
        self.dataset_name, self.lateral, self.top_down, self.time_steps, self.sum))

        self.test_writer.add_graph(tf.get_default_graph())


    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def deconv(self, x, filter, shape):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        output_shape = tf.stack([batch_size, shape[0], shape[1], shape[2]])
        return tf.nn.conv2d_transpose(x, filter, output_shape, [1, 2, 2, 1], padding='SAME')


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    def weight_variable(self, shape, n=0):
        if n != 0:
            stddev = tf.sqrt(2/n)
        else:
            stddev = 0.1
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def build_graph(self):
        with tf.name_scope('reshape'):
            # image transformation
            self.image = tf.reshape(self.x, [-1, self.image_dim1, self.image_dim1, 1])

        self.layers = {}
        for i in range(self.time_steps):
            self.layer = {}
            self.layer["conv_pre"] = self.conv2d(self.image, self.conv_1['weights']) + self.conv_1['biases'] + \
                                (self.conv2d(self.layers[i-1]["conv_pre"], self.l1_lateral) if i > 0 and self.lateral else 0)+ \
                                (self.deconv(self.layers[i-1]["conv2_pre"], \
                                self.td_filter, self.td_output_shape) if i > 0 and self.top_down else 0)
            self.layer["conv"] = tf.nn.lrn(self.prelu(self.layer["conv_pre"], self.alphas_l1), alpha=0.0001)
            self.layer["pool"] = self.max_pool_2x2(self.layer["conv"])
            self.layer["conv2_pre"] = self.conv2d(self.layer["pool"], self.conv_2['weights']) + self.conv_2['biases'] + \
                                (self.conv2d(self.layers[i-1]["conv2_pre"], self.l2_lateral) if i > 0 and self.lateral else 0)
            self.layer["conv2"] = tf.nn.lrn(self.prelu(self.layer["conv2_pre"], self.alphas_l2), alpha=0.0001)
            self.layer["pool2"] = self.max_pool_2x2(self.layer["conv2"])
            self.layer["global_max"] = tf.reduce_max(self.layer["pool2"], axis=[3], keep_dims=True)
            self.layer["flat"] = tf.reshape(self.layer["global_max"], [-1, self.image_dim1//4 * self.image_dim1//4])
            self.layer["yHat"] = tf.matmul(self.layer["flat"], self.read['weights']) + self.read['biases']
            self.variable_summaries(self.layer["yHat"])
            self.layers[i] = self.layer
    
    def prelu(self, input, alphas):
        return tf.nn.relu(input) + tf.multiply(alphas, (inputs - tf.abs(inputs))) * 0.5

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print_frequency = 50    
            for i in range(self.iterations):
                batch = self.dataset.train.next_batch(self.batch_size)
                if i % print_frequency == 0:
                    summary = sess.run([self.merged], feed_dict={
                        self.x: self.dataset.test.images[-2000:], self.y_: self.dataset.test.labels[-2000:], self.keep_prob: 1.0})
                    self.test_writer.add_summary(summary[0], i)
                
                _, summary = sess.run([self.train_step, self.merged], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
                self.train_writer.add_summary(summary, i)

            
        


