import cv2
import numpy as np
import sys
import os
from random import randint
from keras.models import load_model
import dlib
from imutils.face_utils import FaceAligner
from imutils import face_utils
from keras.preprocessing.image import img_to_array


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

'''
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

trackerType = "CSRT"

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

path_haarcascade_frontalface = resource_path("haarcascade_frontalface_default.xml")
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(path_haarcascade_frontalface)

# Initialise model
path_face_detector_pbtxt = resource_path("opencv_face_detector.pbtxt")
path_face_detector_uint8 = resource_path("opencv_face_detector_uint8.pb")
faceProto = path_face_detector_pbtxt
faceModel = path_face_detector_uint8

path_age_deploy_prototxt = resource_path("age_deploy.prototxt")
path_age_net_caffemodel = resource_path("age_net.caffemodel")
ageProto = path_age_deploy_prototxt
ageModel = path_age_net_caffemodel

path_gender_deploy_prototxt = resource_path("gender_deploy.prototxt")
path_gender_net_caffemodel = resource_path("gender_net.caffemodel")
genderProto = path_gender_deploy_prototxt
genderModel = path_gender_net_caffemodel

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

path_mini_xception = resource_path("_mini_XCEPTION.106-0.65.hdf5")
emotion_model_path = path_mini_xception

# Load model
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_classifier._make_predict_function()

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

DIM = (1920, 1080)
KC = np.array([[875.2490096818776, 0.0, 1035.9301696751131], [0.0, 889.3800099127299, 705.4702918706131], [0.0, 0.0, 1.0]])
DC = np.array([[-0.0439612734002022], [0.009055135064110726], [-0.0081999213710754], [-0.0010587278165177605]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

orientation_value_y = 0
orientation_value_x = 0

colors = []
bboxes = []

padding = 20

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)
'''

def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_opencv_dnn, bboxes


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def face_detection(gray):
    return faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# Convert to tuple because result of faceCascade.detectMultiScale doesn't have the good format.
def convert_tuple(faces):
    for face in faces:
        bboxes.append(tuple(face))
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))


def undistort_cam(frame):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(KC, DC, np.eye(3), KC, DIM, cv2.CV_16SC2)
    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    return frame


def alignement(frame, gray):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=512)

    rects = detector(gray, 2)
    for rect in rects:
        try:
            frame = fa.align(frame, gray, rect)
        finally:
            print('')
    return frame


def detector_face(frame):
    face_rects = detector(frame, 0)
    return face_rects


def shape(frame, face_rects):
    shape = predictor(frame, face_rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape


def draw(shape, frame, reprojectdst):
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    for start, end in line_pairs:
        try:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        finally:
            print("test")

    return frame


def draw_facebox(frame, faces):
    for face in faces:
        bboxes.append(tuple(face))
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

    for i, newbox in enumerate(bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    for i, newbox in enumerate(bboxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    return frame


def append_boxes_colors(face):
    bboxes.append(tuple(face))
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))


def reshape_face_frame(frame, face):
    face_frame = frame[max(0, int(face[1]) - padding):min(int(face[3]) + padding,
                                                          frame.shape[0] - 1),
                 max(0, int(face[0]) - padding):min(int(face[2]) + padding,
                                                    frame.shape[
                                                        1] - 1)]
    return face_frame


def roi_processing(roi):
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi


def prediction_gender(blob):
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender = genderList[gender_preds[0].argmax()]
    return gender, gender_preds


def prediction_age(blob):
    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = ageList[age_preds[0].argmax()]
    return age, age_preds


def prediction_emotion(roi):
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    emotion = emotions[preds.argmax()]
    return emotion, emotion_probability


def blob(face_frame):
    blob = cv2.dnn.blobFromImage(face_frame, 1.0, (227, 227),
                                 MODEL_MEAN_VALUES, swapRB=False)
    return blob
