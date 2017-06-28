from model import SimpleModel
import tensorflow as tf
import sys
import cv2
from scipy.misc import imresize
from utils import draw_features_point_on_image
def main(FLAG):
    Model = SimpleModel(FLAG.input_dim, FLAG.hidden_dim, FLAG.output_dim, optimizer=tf.train.RMSPropOptimizer(FLAG.learning_rate), using_gpu=False)

    image_path = sys.argv[1]
    cascPath = "./haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(image_path)
    src_height, src_width, src_channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for x, y, w, h in faces:
        print("faceLocation : ({},{}), width={}, height={}".format(x,y,w,h))
        cropped_image = gray[x:x+w, y:y+h]
        resized_image = imresize(cropped_image, (FLAG.Width, FLAG.Height))
        resized_image = resized_image.flatten() / 255

        pred_feature = Model.predict(resized_image).flatten()
        pred_feature[::2] = pred_feature[::2] * w + x
        pred_feature[1::2] = pred_feature[1::2] * h + y

    result_img = draw_features_point_on_image(image, [pred_feature], src_width, src_height)
    print(pred_feature)
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow('Result', result_img)
    cv2.imwrite("./result_img.png", result_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def arg_flags():
    flags = tf.app.flags

    flags.DEFINE_integer("input_dim", 96, "input layer's dimension")
    flags.DEFINE_integer("hidden_dim", 64, "hidden layer's dimension")
    flags.DEFINE_integer("output_dim", 30, "output layer's dimension")
    flags.DEFINE_integer("Width", 96, "Input Image Width")
    flags.DEFINE_integer("Height", 96, "Input Image Height")
    flags.DEFINE_float("learning_rate", 1e-6, "learning rate")
    flags.DEFINE_string("output_path", "./result_image.png", "path where result image be saved")
    return flags.FLAGS

if __name__ == "__main__":
    FLAG = arg_flags()
    main(FLAG=FLAG)
