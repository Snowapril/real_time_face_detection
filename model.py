import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import my_Model

class SimpleModel:
    def __init__(self, input_dim=96*96, hidden_dim=64, output_dim=30, name="simple",
                  optimizer=tf.train.AdadeltaOptimizer(), model_path="./model/simpleModel.ckpt", using_gpu=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = output_dim

        config = tf.ConfigProto()

        if using_gpu:
            config.gpu_options.per_process_gpu_memory_fraction = 0.7

        self.sess = tf.Session(config=config)

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim*input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, output_dim])
        #train or non-train
        self.mode = tf.placeholder(tf.bool)
        #for dropout layer
        self.keep_prob = tf.placeholder(tf.float32)

        self.network = self.set_up(name, using_gpu)

        #MSE Loss for regression
        self.loss = tf.losses.mean_squared_error(self.y, self.network)

        vars = tf.trainable_variables()
        self.optimizer = optimizer.minimize(self.loss, var_list=vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.model_path = model_path
        self.saver = tf.train.Saver()
        if os.path.exists("./model/checkpoint"):
            self.saver.restore(self.sess, self.model_path)
            print("Load Model Complete")

    def set_up(self, name="simpleconv", using_gpu=True):
        print("Set up {} Model".format(name))
        if using_gpu:
            with tf.device("/gpu:0"):
                return my_Model(self.x, self.input_dim, self.hidden_dim, self.ouput_dim, self.keep_prob)
        else:
            with tf.device("/cpu:0"):
                return my_Model(self.x, self.input_dim, self.hidden_dim, self.ouput_dim, self.keep_prob)


    def predict(self, input_x):
        if input_x.ndim == 1:
            input_x = input_x.reshape(1,-1)

        return self.sess.run(self.network, feed_dict={self.x:input_x, self.keep_prob:1.0, self.mode:False})

    def train(self, input_x, labels, valid_x, valid_labels, batch_size=50, epochs=30, save_graph=True, save_model=True):
        train_loss = []
        valid_loss = []
        step_per_epoch = int(input_x.shape[0]/batch_size)

        print("Start Training\nTrain Sample:{}".format(input_x.shape[0]))
        for epoch in range(epochs):
            train_idx = 0
            for step in range(step_per_epoch):
                feed_dict = {self.x:input_x[train_idx:train_idx+batch_size], self.y:labels[train_idx:train_idx+batch_size], self.keep_prob:0.8, self.mode:True}
                _, cost_train = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                train_idx += batch_size
                if step % 50 == 0:
                    print("Epoch:{:01d}, step:{:04d}, Loss:{:.5f}".format(epoch, step, cost_train))

                    feed_dict = {self.x:valid_x, self.y:valid_labels, self.keep_prob:1.0, self.mode:False}
                    cost_valid = self.sess.run([self.loss], feed_dict=feed_dict)

                    train_loss.append(cost_train)
                    valid_loss.append(cost_valid)

            if save_model == True:
                if(not os.path.exists("./model")):
                    os.makedirs("./model")

                save_path = self.saver.save(self.sess, self.model_path)
                print("Epochs:{:01d}, Model saved in file{}".format(epoch, save_path))

        print("Training Finished")

        x = np.arange(len(train_loss))

        #train loss graph
        fig = plt.figure()
        plt.plot(x, train_loss, label='train_loss')
        plt.plot(x, valid_loss, label="valid_loss", color="r")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='lower right')

        if save_graph == True:
            fig.savefig("./loss_graph.png")
            print("Save Loss Graph")

        print("Show Loss Graph")
        plt.show()


    def validation(self, input_x, labels, valid_x, valid_labels, lr_list, batch_size=50, epochs=30):
        validation_cost = []
        step_per_epoch = int(input_x.shape[0]/batch_size)

        print("Start Validation\nTrain Sample:{}, Validation Sample:{}".format(input_x.shape[0], valid_x.shape[0]))
        for lr in lr_list:
            self.learning_rate = lr
            for epoch in range(epochs):
                train_idx = 0
                for step in range(step_per_epoch):
                    feed_dict = {self.x:input_x[train_idx:train_idx+batch_size], self.y:labels[train_idx:train_idx+batch_size], self.keep_prob:1.0, self.mode:True}
                    _, cost_train = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    train_idx += batch_size

            valid_cost = self.sess.run(self.loss, feed_dict={self.x:valid_x, self.y:valid_labels, self.keep_prob:1.0, self.mode:False})
            validation_cost.append(valid_cost)
            print("validation loss:{}, learing_rate:{}".format(valid_cost, self.learning_rate))

            init = tf.global_variables_initializer()
            self.sess.run(init)

        loss_valid_min_idx = validation_cost.index(min(validation_cost))
        print("{} learning rate is best choice.".format(lr_list[loss_valid_min_idx]))

        print("Validation Finished")
