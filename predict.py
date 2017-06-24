from model import SimpleConvModel
import tensorflow as tf
from utils import *
def main(FLAG):
    Model = SimpleConvModel(FLAG.input_dim, FLAG.hidden_dim, FLAG.output_dim, optimizer=tf.train.RMSPropOptimizer(FLAG.learning_rate))

    image, label = load_dataset()
    image, label = image_augmentation(image, label)
    label = label / 255.

    print(Model.predict(image[123].reshape(1,-1)))

def arg_flags():
    flags = tf.app.flags

    flags.DEFINE_integer("input_dim", 96*96, "input layer's dimension")
    flags.DEFINE_integer("hidden_dim", 64, "hidden layer's dimension")
    flags.DEFINE_integer("output_dim", 30, "output layer's dimension")
    flags.DEFINE_integer("Width", 96, "Input Image Width")
    flags.DEFINE_integer("Height", 96, "Input Image Height")
    flags.DEFINE_float("learning_rate", 1e-6, "learning rate")

    return flags.FLAGS

if __name__ == "__main__":
    FLAG = arg_flags()
    main(FLAG=FLAG)
