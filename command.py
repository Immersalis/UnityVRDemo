import cv2
import time
import sys
sys.path.append('franceAsia')

import image_processing as improcessing
import log

flag_order = False
flag_stop = False
flag_end = False


def set_flag_order(b):
    global flag_order
    flag_order = b


def launch_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    return cap


def launch(nb_command):

    cap = launch_camera()

    while flag_order:

        print("HERE 0")
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = improcessing.face_detection(gray)

            if len(faces) != 0:

                improcessing.convert_tuple(faces)

                log.init_list(faces)

                oldfaces = faces
                nboldfaces = len(faces)
                print("HERE")
                while flag_order:
                    if flag_end:
                        quit()

                    ret, frame = cap.read()
                    cv2.waitKey(1)

                    if ret:
                        print("HERE 2")
                        improcessing.undistort_cam(frame)

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        frame = improcessing.alignement(frame, gray)

                        faces = improcessing.face_detection(frame)

                        face_rects = improcessing.detector(frame)

                        if len(face_rects) > 0:

                            print("HERE 3")
                            shape = improcessing.shape(frame, face_rects)

                            reprojectdst, euler_angle = improcessing.get_head_pose(shape)

                            frame = improcessing.draw(frame, shape, reprojectdst)

                            orientation_value_x = euler_angle[1, 0]
                            orientation_value_y = euler_angle[0, 0]

                        if (orientation_value_x > -30) and (orientation_value_x < 30) and (orientation_value_y > -30) and (orientation_value_y < 30):
                            print("ORIENTATION OK")

                            frame = improcessing.draw_facebox(frame, faces)

                            if nboldfaces == len(faces):
                                print("HERE 4")
                                for i, face in enumerate(faces):

                                    improcessing.append_boxes_colors(face)

                                    save_face_3 = face[3]
                                    save_face_2 = face[2]

                                    face[3] = face[1] + face[3]
                                    face[2] = face[0] + face[2]

                                    for j, oldface in enumerate(oldfaces):
                                        oldface[3] = oldface[1] + oldface[3]
                                        oldface[2] = oldface[0] + oldface[2]

                                        face_frame = improcessing.reshape_face_frame(frame, face)

                                        if face_frame.size != 0:
                                            (fX, fY, fW, fH) = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]

                                            roi = gray[fY:fY + fH, fX:fX + fW]

                                            if roi.size != 0:

                                                blob = improcessing.blob(face_frame)

                                                gender, gender_preds = improcessing.prediction_gender(blob)

                                                age, age_preds = improcessing.prediction_age(blob)

                                                emotion, emotion_probability = improcessing.prediction_emotion(blob)

                                                if gender_preds[0].max() > 0.95 and age_preds[0].max() > 0.95:
                                                    print(age, age_preds.max(), gender, gender_preds.max())

                                                    log.register_gender(gender, gender_preds, i)
                                                    log.register_age(age, age_preds, i)

                                                if emotion_probability > 0.50:
                                                    log.register_emotion(emotion, emotion_probability, i)

                                    # reset good value for face before store it in oldface
                                    face[3] = save_face_3
                                    face[2] = save_face_2

                                # store the actual frame to compare it to the old
                                oldfaces = faces
                                nboldfaces = len(faces)
                            else:
                                flag_found = False
                                t_end = time.time() + 15  # timer to wait for a valid face detection
                                while time.time() < t_end and not flag_found:
                                    if flag_end:
                                        quit()

                                    ret, frame = cap.read()
                                    cv2.waitKey(1)
                                    if ret:
                                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        faces = improcessing.face_detection(gray)

                                        if len(faces) == nboldfaces:
                                            flag_found = True
                log.write_data(faces)