import cv2
import argparse
from model import SimpleModel
import tensorflow as tf
from scipy.misc import imresize
from utils import draw_features_point_on_image

Model  = SimpleModel(96, 64, 30, optimizer=tf.train.RMSPropOptimizer(1e-6), using_gpu=False)

def detect_features(image, faces, src_width, src_height, width=96, height=96):
    pred_features = []
    for x, y, w, h in faces:
        print("faceLocation : ({},{}), width={}, height={}".format(x,y,w,h))
        cropped_image = image[x:x+w, y:y+h]
        resized_image = imresize(cropped_image, (width, height))
        resized_image = resized_image.flatten() / 255

        pred_feature = Model.predict(resized_image).flatten()
        pred_feature[::2] = pred_feature[::2] * w + x
        pred_feature[1::2] = pred_feature[1::2] * h + y

        pred_features.append(pred_feature)

    return pred_features


def main(parser):
    capture = cv2.VideoCapture(parser.source)
    src_width, src_height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if(parser.record == True):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(parser.output_path,fourcc, 20.0, (src_width,src_height))

    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret, frame = capture.read()


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        pred_features = detect_features(gray, faces, src_width, src_height, parser.width, parser.height)
        result_img = draw_features_point_on_image(frame, pred_features, src_width, src_height)

        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        if (ret==True) and (parser.record == True):
            out.write(result_img)

        cv2.imshow('Video', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

    if parser.record == True:
        out.release()

    cv2.destroyAllWindows()

def arg_parse():
    arg = argparse.ArgumentParser()
    arg.add_argument("-width", default=96)
    arg.add_argument("-height", default=96)
    arg.add_argument("-source", default=0)
    arg.add_argument("-record", default=True, type=bool)
    arg.add_argument("-output_path", default="./output.avi", type=str)
    parser = arg.parse_args()

    return parser

if __name__ == "__main__":
    main(arg_parse())
