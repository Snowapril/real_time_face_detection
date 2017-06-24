import tensorflow as tf
from utils import *


class SimpleConvModel:
    def __init__(self, input_dim=96*96, hidden_dim=64, output_dim=30, name="simpleconv",stddev=0.01,
                  optimizer=tf.train.AdadeltaOptimizer(), model_path="./model/simpleConvModel.ckpt"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = output_dim

        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, output_dim])
        self.keep_prob = tf.placeholder(tf.float32)

        self.network = self.set_up(name, stddev)

        self.loss = tf.losses.mean_squared_error(self.y, self.network)

        vars = tf.trainable_variables()
        self.optimizer = optimizer.minimize(self.loss, var_list=vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.model_path = model_path
        self.saver = tf.train.Saver()

        if os.path.exists(self.model_path):
            self.saver.restore(self.sess, self.model_path)
            print("Load Model Complete")

    def set_up(self, name="simpleconv", stddev=0.01):
        print("Set up {} Model".format(name))
        batch_1 = tf.nn.batch_normalization(self.x, 0, 1, None, None, 10e-7)

        W1 = tf.get_variable(name+"_w1", shape=[self.input_dim, self.hidden_dim], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(self.input_dim)))
        b1 = tf.get_variable(name+"_b1", shape=[self.hidden_dim], initializer=tf.constant_initializer(0.))

        c1 = tf.matmul(batch_1, W1) + b1
        c1 = tf.nn.relu(c1)
        c1 = tf.nn.dropout(c1, self.keep_prob)

        W2 = tf.get_variable(name+"_w2", shape=[self.hidden_dim, self.hidden_dim*2], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(self.hidden_dim)))
        b2 = tf.get_variable(name+"_b2", shape=[self.hidden_dim*2], initializer=tf.constant_initializer(0.))

        c2 = tf.matmul(c1, W2) + b2
        c2 = tf.nn.relu(c2)
        c2 = tf.nn.dropout(c2, self.keep_prob)

        W3 = tf.get_variable(name+"_w3", shape=[self.hidden_dim*2, self.hidden_dim*4], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(self.hidden_dim*2)))
        b3 = tf.get_variable(name+"_b3", shape=[self.hidden_dim*4], initializer=tf.constant_initializer(0.))

        c3 = tf.matmul(c2, W3) + b3
        c3 = tf.nn.relu(c3)
        c3 = tf.nn.dropout(c3, self.keep_prob)

        W4 = tf.get_variable(name+"_w4", shape=[self.hidden_dim*4, self.hidden_dim*8], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(self.hidden_dim*4)))
        b4 = tf.get_variable(name+"_b4", shape=[self.hidden_dim*8], initializer=tf.constant_initializer(0.))

        c4 = tf.matmul(c3, W4) + b4
        c4 = tf.nn.relu(c4)
        c4 = tf.nn.dropout(c4, self.keep_prob)

        W5 = tf.get_variable(name+"_w5", shape=[self.hidden_dim*8, self.ouput_dim], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(self.hidden_dim*8)))
        b5 = tf.get_variable(name+"_b5", shape=[self.ouput_dim], initializer=tf.constant_initializer(0.))

        logits = tf.matmul(c4, W5) + b5

        return logits

    def predict(self, input_x):
        return self.sess.run(self.network, feed_dict={self.x:input_x, self.keep_prob:1.0})

    def predict_image(self, input_x):
        fig, ax = plt.subplots()
        logits = self.predict(input_x)
        ax.imshow(input_x.reshape(96,96), cmap="gray")
        fig = patch_pointing(fig, logits.reshape(-1))

        canvas = fig.canvas.manager
        canvas.canvas.figure = fig
        fig.set_canvas(canvas.canvas)

        plt.show()

    def get_accuracy(self, input_x, label):
        #cosine similarity
        logits = self.predict(input_x)
        if logits.shape != label.shape:
            print("logit shape and label shape must be equal")
            raise Exception

        similarity = np.mean(np.sum( logits * label, axis=1) / (np.linalg.norm(logits, axis=1) * (np.linalg.norm(label, axis=1))))
        return similarity

    def compare_image(self, input_x, label):
        logits = self.predict(input_x)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(input_x.reshape(96,96), cmap="gray")
        fig = patch_pointing(fig,label.reshape(-1))
        ax[1].imshow(input_x.reshape(96,96), cmap="gray")

        fig = patch_pointing(fig,logits.reshape(-1))
        canvas = fig.canvas.manager
        canvas.canvas.figure = fig
        fig.set_canvas(canvas.canvas)

        plt.show()

    def train(self,input_x, labels, batch_size=50, epochs=30, save_graph=True, save_model=True):
        train_loss = []

        step_per_epoch = int(input_x.shape[0]/batch_size)

        print("Start Training\nTrain Sample:{}".format(input_x.shape[0]))
        for epoch in range(epochs):
            train_idx = 0
            for step in range(step_per_epoch):
                feed_dict = {self.x:input_x[train_idx:train_idx+batch_size], self.y:labels[train_idx:train_idx+batch_size], self.keep_prob:1.0}
                _, cost_train = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                train_idx += batch_size
                if step % 50 == 0:
                    print("Epoch:{:01d}, step:{:04d}, Loss:{:.5f}".format(epoch, step, cost_train))
                    train_loss.append(cost_train)

            if save_model == True:
                if(not os.path.exists("./model")):
                    os.makedirs("./model")

                save_path = self.saver.save(self.sess, self.model_path)
                print("Epochs:{:01d}, Model saved in file{}".format(epoch, save_path))

        print("Training Finished")

        print("Show Loss Graph")
        x = np.arange(len(train_loss))

        #train loss, valid loss, valid acc(cosine similarity) graph
        fig = plt.figure()
        plt.plot(x, train_loss, label='train_loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='lower right')
        plt.show()

        if save_graph == True:
            fig.savefig("./loss_graph.png")
            print("Save Loss Graph")

    def validation(self, input_x, labels, valid_x, valid_labels, lr_list, batch_size=50, epochs=30):
        validation_cost = []
        step_per_epoch = int(input_x.shape[0]/batch_size)

        print("Start Validation\nTrain Sample:{}, Validation Sample:{}".format(input_x.shape[0], valid_x.shape[0]))
        for lr in lr_list:
            self.learning_rate = lr
            for epoch in range(epochs):
                train_idx = 0
                for step in range(step_per_epoch):
                    feed_dict = {self.x:input_x[train_idx:train_idx+batch_size], self.y:labels[train_idx:train_idx+batch_size], self.keep_prob:1.0}
                    _, cost_train = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    train_idx += batch_size

            valid_cost = self.sess.run(self.loss, feed_dict={self.x:valid_x, self.y:valid_labels, self.keep_prob:1.0})
            validation_cost.append(valid_cost)
            print("validation loss:{}, learing_rate:{}".format(valid_cost, self.learning_rate))

            init = tf.global_variables_initializer()
            self.sess.run(init)

        acc_valid_max_idx = validation_cost.index(max(validation_cost))
        print("{} learning rate is best choice.".format(lr_list[acc_valid_max_idx]))

        print("Validation Finished")
