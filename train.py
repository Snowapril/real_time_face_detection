import tensorflow as tf
from model import SimpleConvModel
from utils import *

tf.set_random_seed(777)

def main(FLAG):
    Model = SimpleConvModel(FLAG.input_dim, FLAG.hidden_dim, FLAG.output_dim, optimizer=tf.train.RMSPropOptimizer(FLAG.learning_rate))

    image, label = load_dataset()
    image, label = image_augmentation(image, label)
    label = label / 255.

    if FLAG.Mode == "validation":
        (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = split_data(image, label)
        lr_list = 10 ** np.random.uniform(-6, -2, 20)
        Model.validation(train_X, train_y, valid_X, valid_y, lr_list)
    else:
        (train_X, train_y), _, (test_X, test_y) = split_data(image, label, 0.9, 0.0, 0.1)
        Model.train(train_X, train_y, FLAG.batch_size, FLAG.Epoch, FLAG.save_graph, FLAG.save_model)
        print(Model.get_accuracy(test_X, test_y))
        Model.predict(image[123])

def arg_flags():
    flags = tf.app.flags

    flags.DEFINE_integer("Epoch", 15, "Total Epoch to train model")
    flags.DEFINE_integer("batch_size", 70, "batch size to train model")
    flags.DEFINE_integer("input_dim", 96*96, "input layer's dimension")
    flags.DEFINE_integer("hidden_dim", 64, "hidden layer's dimension")
    flags.DEFINE_integer("output_dim", 30, "output layer's dimension")
    flags.DEFINE_integer("Width", 96, "Input Image Width")
    flags.DEFINE_integer("Height", 96, "Input Image Height")
    flags.DEFINE_boolean("save_model", True, "if True, save the model's weights")
    flags.DEFINE_boolean("save_graph", True, "if True, save the model's loss graph")
    flags.DEFINE_float("learning_rate", 1e-6, "learning rate")
    flags.DEFINE_string("Mode", "train", "Model mode")
    return flags.FLAGS

if __name__ == "__main__":
    FLAG = arg_flags()
    main(FLAG=FLAG)
