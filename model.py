import tensorflow as tf
import digits, tempfile, time, sys, os
from log import Log
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
class RNNModel:
    def __init__(self, config, test=False):
        self.from_checkpoint = config.from_checkpoint
        if self.from_checkpoint:
            self.checkpoint_path = config.checkpoint_path
        self.save = config.save
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps
        self.lateral =  config.lateral
        self.top_down = config.top_down
        self.iterations = config.iterations
        self.kernel = config.kernel_size
        self.dataset_name = config.dataset
        self.debris = config.debris
        self.sum = config.sum
        self.eval_steps = config.eval_steps
        self.test = test

        if self.dataset_name == "mnist":
            self.dataset = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
        elif self.dataset_name == "digits":
            if self.test:
                format_val = ["", "_with_debris"]
            elif self.debris:
                format_val = ["../../", "_with_debris"]
            else:
                format_val = ["../../", ""]
            self.dataset = digits.Digits("{}DIGIT_data/light_debris{}.mat".\
            format(*format_val), split=.93)
        else:
            raise BaseException("dataset not specified")
        self.imsize = self.dataset.train.images[0].shape[0]
        self.image_dim1 = int(np.sqrt(self.imsize))
        self.x = tf.placeholder('float', [None, self.imsize])
        self.y_ = tf.placeholder('float', [None, 10])
        # bottom up weights
        self.conv_1 = {'weights': self.weight_variable([self.kernel, self.kernel, 1, 32], 9, name="weights_conv_1"),
                    'biases': self.bias_variable([32], name="bias_conv_1")}
        self.conv_2 = {'weights': self.weight_variable([self.kernel, self.kernel, 32, 64], 9*32, name="weights_conv_2"),
                    'biases': self.bias_variable([64], name="bias_conv_2")}
        self.read = {'weights': self.weight_variable([(self.image_dim1**2)//16, 10], name="weights_read"),
                    'biases': self.bias_variable([10], name="bias_read")}
        # lateral weights
        self.l1_lateral = self.weight_variable([self.kernel, self.kernel, 32, 32], name="weights_lateral_1")     
        self.l2_lateral = self.weight_variable([self.kernel, self.kernel, 64, 64], name="weights_lateral_2")
        # monitoring
        if self.lateral:
            self.variable_summaries(self.l1_lateral, "l1_lateral_weights")
            self.variable_summaries(self.l2_lateral, "l2_lateral_weights")
        # top down weights
        self.td_filter = self.weight_variable([self.image_dim1//2, self.image_dim1//2, 32, 64], name="weights_top_down")
        self.td_output_shape = tf.constant([self.image_dim1, self.image_dim1, 32])
        if self.top_down:
            self.variable_summaries(self.td_filter, "top_down_weights")
        # dropout parmeter
        self.keep_prob = tf.placeholder(tf.float32)
        # prelu weights
        self.alphas_l1 = tf.Variable(tf.zeros([32]), name="weights_alphas_1")
        self.alphas_l2 = tf.Variable(tf.zeros([64]), name="weights_alphas_2")
        self.variable_summaries(self.alphas_l1, "l1_prelu_weights")
        self.variable_summaries(self.alphas_l2, "l2_prelu_weights")
        # build the layers
        self.build_graph()
        self.define_objective()

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries_{}'.format(name)):
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

    def weight_variable(self, shape, n=0, name=""):
        if n != 0:
            stddev = tf.sqrt(2/n)
        else:
            stddev = 0.1
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=""):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def define_objective(self):
        with tf.name_scope('loss'):
            if sum:
                self.cross_entropy = tf.reduce_sum(
                    [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.layers[layer]["yHat"]))
                    for layer in range(self.time_steps)])
            if not sum:
                self.cross_entropy = tf.reduce_mean(
                    [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.layers[layer]["yHat"]))
                    for layer in range(self.time_steps)])
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-2).\
                minimize(self.cross_entropy)
        with tf.name_scope('unrolled_acc'):
                self.outputs_ = [self.layers[layer]["yHat"] for layer in self.layers]
                self.correct_predictions_ = [tf.equal(tf.argmax(yhat, 1), tf.argmax(self.y_, 1)) for yhat in self.outputs_]
                self.unrolled_accuracies = [tf.reduce_mean(tf.cast(e, tf.float32)) for e in self.correct_predictions_]
        for i in range(self.eval_steps):
            if i > self.time_steps:
                tf.summary.scalar('accuracy_t{}'.format(i), self.unrolled_accuracies[i])
	
        with tf.name_scope('accuracy'):
            self.outputs = [self.layers[layer]["yHat"] for layer in range(self.time_steps)]
            self.correct_predictions = [tf.equal(tf.argmax(yhat, 1), tf.argmax(self.y_, 1)) for yhat in self.outputs]
            self.accuracies = [tf.reduce_mean(tf.cast(e, tf.float32)) for e in self.correct_predictions]
        for i in range(self.time_steps):
            tf.summary.scalar('accuracy_t{}'.format(i), self.accuracies[i]) 
	
        # Merge all the summaries 
        self.merged = tf.summary.merge_all()
        graph_location = "/home/taylor/ba/tensorboard/" if not self.test else "/home/julius/workspace/ba/tensorboard/"

        # recursive models
        if self.lateral and self.top_down:
            self.model = "BLT"  
        if self.lateral and self.top_down and not self.from_checkpoint:
            self.model = "nlBLT"
        elif self.lateral and not self.top_down:
            self.model = "BL"
        elif not self.lateral and self.top_down:
            self.model = "BT"
        # non recursive models
        elif not self.lateral and not self.top_down and self.kernel == 3:
            self.model = "B"
        elif not self.lateral and not self.top_down and self.kernel > 3:
            self.model = "BK"

        self.train_writer = tf.summary.FileWriter(graph_location+ \
        "nodeb{}_train".format(self.model))

        self.train_writer.add_graph(tf.get_default_graph())

        self.test_writer = tf.summary.FileWriter(graph_location+ \
        "nodeb{}_test".format(self.model))

        self.test_writer.add_graph(tf.get_default_graph())


    def build_graph(self):
        with tf.name_scope('reshape'):
            # image transformation
            self.image = tf.reshape(self.x, [-1, self.image_dim1, self.image_dim1, 1])

        self.layers = {}
        for i in range(self.eval_steps):
            self.layer = {}
            self.layer["conv_pre"] = self.conv2d(self.image, self.conv_1['weights']) + self.conv_1['biases'] + \
                                (self.conv2d(self.layers[i-1]["conv_pre"], self.l1_lateral) if i > 0 and self.lateral else 0)+ \
                                (self.deconv(self.layers[i-1]["conv2_pre"], \
                                self.td_filter, self.td_output_shape) if i > 0 and self.top_down else 0)
            self.layer["conv"] = tf.nn.lrn(self.prelu(self.layer["conv_pre"], self.alphas_l1), alpha=0.0001)
            self.variable_summaries(self.layer["conv"], "l1_conv_activation")
            self.layer["pool"] = self.max_pool_2x2(self.layer["conv"])
            self.layer["conv2_pre"] = self.conv2d(self.layer["pool"], self.conv_2['weights']) + self.conv_2['biases'] + \
                                (self.conv2d(self.layers[i-1]["conv2_pre"], self.l2_lateral) if i > 0 and self.lateral else 0)
            self.layer["conv2"] = tf.nn.lrn(self.prelu(self.layer["conv2_pre"], self.alphas_l2), alpha=0.0001)
            self.variable_summaries(self.layer["conv2"], "l2_conv_activation")
            self.layer["pool2"] = self.max_pool_2x2(self.layer["conv2"])
            self.layer["global_max"] = tf.reduce_max(self.layer["pool2"], axis=[3], keep_dims=True)
            self.layer["flat"] = tf.reshape(self.layer["global_max"], [-1, self.image_dim1//4 * self.image_dim1//4])
            self.layer["yHat"] = tf.matmul(self.layer["flat"], self.read['weights']) + self.read['biases']
            #self.variable_summaries(self.layer["yHat"])
            self.layers[i] = self.layer
    
    def prelu(self, input, alphas):
#	tf.nn.relu(input) - tf.nn.relu(-input) * alpha
      return tf.nn.relu(input) + tf.multiply(alphas, (input - tf.abs(input))) * 0.5

    def get_image_foreach_label(self):
        images = []
        labels = []
        i = 0
        while len(labels) < 10:
            label = self.dataset.test.labels[i]
            image = self.dataset.test.images[i]
            if label not in labels:
                labels.append(label)
                images.append(image)
            i += 1
        return images, labels

    def run(self):
        # gather relevant variables
        saver = tf.train.Saver(var_list=[self.conv_1["weights"], self.conv_1["biases"], \
        self.conv_2["weights"], self.conv_2["biases"], self.read["weights"], \
        self.read["biases"], self.l1_lateral, self.l2_lateral, self.td_filter, \
        self.alphas_l1, self.alphas_l2])

        with tf.Session() as sess:
            # todo: fetch unitialized variables, 
            sess.run(tf.global_variables_initializer())
            if self.from_checkpoint:
                saver.restore(sess, self.checkpoint_path)
            eval_test_frequency = 200
            for i in range(self.iterations):
                batch = self.dataset.train.next_batch(self.batch_size)
                if i % eval_test_frequency == 0:
                    summary = sess.run([self.merged], feed_dict={
                        self.x: self.dataset.test.images, self.y_: self.dataset.test.labels, self.keep_prob: 1.0})
                    self.test_writer.add_summary(summary[0], i)
                
                _, summary = sess.run([self.train_step, self.merged], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
                self.train_writer.add_summary(summary, i)
            
            # make pictures for tensorboard
            self.evaluate_feature_maps(sess, layer=1)
            self.evaluate_feature_maps(sess, layer=2)
            # store variables
            if self.save:
                saver.save(sess, "/home/taylor/ba/checkpoints/{}.ckpt".format(self.model))

    def evaluate_feature_maps(self, sess, layer=1):
        # evaluate pixel maps with fully trained network
        channels = 32 if layer == 1 else 64
        self.V = self.weight_variable([1, 32, 32, channels], name="V")
        sess.run(self.V.initializer)
        ix = 32 if layer == 1 else 16
        iy = 32 if layer == 1 else 16
        self.V = tf.slice(self.layers[self.time_steps-1]["conv{}".format("" if layer == 1 else "2")],(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
        self.V = tf.reshape(self.V,(iy,ix,channels))
        ix += 4 if layer == 1 else 2
        iy += 4 if layer == 1 else 2
        self.V = tf.image.resize_image_with_crop_or_pad(self.V, ix, iy)
        cy = 4 if layer == 1 else 8
        cx = 8
        self.V = tf.reshape(self.V,(iy,ix,cy,cx)) 
        self.V = tf.transpose(self.V,(2,0,3,1)) #cy,iy,cx,ix
        self.V = tf.reshape(self.V,(1,cy*iy,cx*ix,1))

        images, labels = self.get_image_foreach_label()
        for im, lb in zip(images, labels):
            im.shape = (1, 1024)  
            v = sess.run([self.V], feed_dict={self.x: im})
            summary_op = tf.summary.image("layer_{}_img_label_{}".format(layer, np.argmax(lb)), v[0])
            summ = sess.run(summary_op)
            self.test_writer.add_summary(summ, np.argmax(lb))

            
        


