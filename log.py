from datetime import datetime
import numpy as np
from pathlib import Path
import os

# list to store data during an order
value_gender = []
value_age = []
value_emotion = []
conf_gender = []
conf_age = []
conf_emotion = []

emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

command = 0
datetime_start = ""
name_csv = ""
type_end = ""


def write_data(faces):
    # empty list to store all data contains in a row
    data = np.empty((len(faces), 0)).tolist()

    for x in range(len(faces)):

        # Initialise list for mean.
        compt_gender = [0, 0]
        compt_age = [0, 0, 0, 0, 0, 0, 0, 0]
        compt_emotion = [0, 0, 0, 0, 0, 0, 0]

        # Add up each value weighted by the conf
        for i in range(len(value_gender[x])):
            if value_gender[x][i] == genderList[0]:
                compt_gender[0] += conf_gender[x][i]
            else:
                compt_gender[1] += conf_gender[x][i]
            for j in range(len(ageList)):
                if value_age[x][i] == ageList[j]:
                    compt_age[j] += conf_age[x][i]

        for i in range(len(value_emotion[x])):
            for j in range(len(emotions)):
                if value_emotion[x][i] == emotions[j]:
                    compt_emotion[j] += conf_emotion[x][i]

        if len(value_gender[x]) !=0:
            data[x].append(" " + str(genderList[0]) + ": " + str(compt_gender[0] / len(value_gender[x])))
            data[x].append(" " + str(genderList[1]) + ": " + str(compt_gender[1] / len(value_gender[x])))

        if len(value_age[x]) !=0:
            for i in range(len(ageList)):
                data[x].append(" " + str(ageList[i]) + ": " + str(compt_age[i] / len(value_age[x])))

        if len(value_emotion[x]) != 0:
            for i in range(len(emotions)):
                data[x].append(" " + str(emotions[i]) + ": " + str(compt_emotion[i] / len(value_emotion[x])))

    return data


def creation_csv_file():
    if os.path.exists("C:\\FranceAsia\\App\\Logs"):
        # create or update csv file, the name corresponding to the date of the actual day
        name_csv = "C:\\FranceAsia\\App\\Logs\\" + datetime.now().strftime("%d%m%Y") + "_age_gender_emotion" + '.log'

        # if file doesn't exist, create heading line
        my_file = Path(name_csv)
        if not my_file.is_file():
            with open(name_csv, mode='a') as csv_file:
                csv_file.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format("NumOrder", "StartTime", "CurrentTime", "TypeEnd", "NbPerson", "Male", "Female", "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)", "angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"))


def write_csv_file(data):
    global csv_file, type_end
    dataT = data

    for i in range(len(dataT)):

        data = str(dataT[i]).split(",")

        if os.path.exists("C:\\FranceAsia\\App\\Logs"):
            # Write results of every parameter
            with open(name_csv, mode='a') as csv_file:
                if len(data) == 10:
                    csv_file.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(command, datetime_start,
                                                                                               datetime.now().strftime(
                                                                                                   "%H:%M:%S"), type_end
                                                                                               , len(dataT),data[0][9:-1], data[1][11:-1],
                                                                                               data[2][10:-1], data[3][10:-1], data[4][11:-1],
                                                                                               data[5][12:-1],
                                                                                               data[6][12:-1], data[7][12:-1], data[8][12:-1],
                                                                                               data[9][13:-2]))
                elif len(data) == 17:
                    csv_file.write(
                        "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(command, datetime_start,datetime.now().strftime("%H:%M:%S")
                                                                                             , type_end, len(dataT), data[0][9:-1], data[1][11:-1],
                                                                                             data[2][10:-1], data[3][10:-1], data[4][11:-1], data[5][12:-1],
                                                                                             data[6][12:-1], data[7][12:-1], data[8][12:-1],
                                                                                             data[9][13:-1], data[10][10:-1], data[11][12:-1],
                                                                                             data[12][11:-1], data[13][10:-1], data[14][8:-1], data[15][14:-1],
                                                                                             data[16][12:-2]))
                else:
                    csv_file.write(
                        "{};{};{};{};{}\n".format(command, datetime_start, datetime.now().strftime("%H:%M:%S"), type_end, len(dataT)))


def init_list(faces):

    global value_gender, value_emotion, value_age
    global conf_emotion, conf_gender, conf_age
    # lists containing value and conf for every new client
    value_gender = np.empty((len(faces), 0)).tolist()
    value_age = np.empty((len(faces), 0)).tolist()
    value_emotion = np.empty((len(faces), 0)).tolist()
    conf_gender = np.empty((len(faces), 0)).tolist()
    conf_age = np.empty((len(faces), 0)).tolist()
    conf_emotion = np.empty((len(faces), 0)).tolist()


def register_age(age, age_preds, i):
    value_age[i].append(age)
    conf_age[i].append(age_preds[0].max())


def register_gender(gender, gender_preds, i):
    value_gender[i].append(gender)
    conf_gender[i].append(gender_preds[0].max())


def register_emotion(emotion, emotion_probability, i):
    value_emotion[i].append(emotion)
    conf_emotion[i].append(emotion_probability)
