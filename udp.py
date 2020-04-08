import os
from datetime import datetime
import socket
from tkinter import *
from functools import partial
from pathlib import Path
sys.path.append('franceAsia')
import main

UDP_PORT = 5001
UDP_IP = "127.0.0.1"


def choose_udp_config():
    root = Tk()

    Label(root, text="IP :").grid(row=0, sticky=W)
    text_ip = StringVar(root)
    ent_ip = Entry(root, textvariable=text_ip)
    ent_ip.grid(row=1, sticky=W)

    Label(root, text="PORT :").grid(row=2, sticky=W)
    text_port = StringVar(root)
    ent_port = Entry(root, textvariable=text_port)
    ent_port.grid(row=3, sticky=W)

    b = Button(root, text='Apply', command=partial(update_config, text_ip, text_port))
    b.grid(row=4, sticky=W)

    root.mainloop()


def update_config(text_ip, text_port):
    global UDP_IP, UDP_PORT
    re_ip = '^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    re_port = '^()([1-9]|[1-5]?[0-9]{2,4}|6[1-4][0-9]{3}|65[1-4][0-9]{2}|655[1-2][0-9]|6553[1-5])$'
    if re.search(re_ip, text_ip.get()) and re.search(re_port, text_port.get()):
        UDP_IP = text_ip.get()
        UDP_PORT = int(text_port.get())

    else:
        print("IP invalid or Port invalid")


if os.path.exists("C:\\FranceAsia\\App\\Logs"):
    f = open("C:\\FranceAsia\\App\\Logs\\terminal.log", "a")
    # sys.stdout = f
    # sys.stderr = f


def init_udp():
    global flag_order, flag_launch, command, type_end

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    while True:
        name_udp = "C:\\FranceAsia\\App\\Logs\\" + datetime.now().strftime("%d%m%Y") + "_udp" + '.log'
        if os.path.exists("C:\\FranceAsia\\App\\Logs"):
            # if file doesn't exist, create heading line
            my_file = Path(name_udp)
            if not my_file.is_file():
                with open(name_udp, mode='a') as csv_file_udp:
                    csv_file_udp.write("{};{}\n".format("Time", "Command"))

        data, addr = sock.recvfrom(1024)
        data = data.decode()

        if len(data) == 1:
            data = int(data)

            if data == 0:  # if no nb command
                print("IN UDP NO COMMAND")
                if os.path.exists("C:\\FranceAsia\\App\\Logs"):
                    with open(name_udp, mode='a') as csv_file_udp:
                        csv_file_udp.write("{};{}\n".format(datetime.now().strftime("%H:%M:%S"), "Start"))
                main.set_flag_launch(True)
                type_end = "None"

            else:  # end command
                print("IN UDP END COMMAND")
                flag_order = False
                if data == 1:
                    if os.path.exists("C:\\FranceAsia\\App\\Logs"):
                        with open(name_udp, mode='a') as csv_file_udp:
                            csv_file_udp.write("{};{}\n".format(datetime.now().strftime("%H:%M:%S"), "PAIEMENTOK"))
                    type_end = "OK"

                if data == 2:
                    if os.path.exists("C:\\FranceAsia\\App\\Logs"):
                        with open(name_udp, mode='a') as csv_file_udp:
                            csv_file_udp.write("{};{}\n".format(datetime.now().strftime("%H:%M:%S"), "CANCEL"))
                    type_end = "CANCEL"
                if data == 3:
                    if os.path.exists("C:\\FranceAsia\\App\\Logs"):
                        with open(name_udp, mode='a') as csv_file_udp:
                            csv_file_udp.write("{};{}\n".format(datetime.now().strftime("%H:%M:%S"), "TIMEOUT"))
                    type_end = "TIME-OUT"
                if data == 4:
                    global flag_end
                    flag_end = True
                    quit()
                    exit()

        else:  # part that handle nb command
            print("IN UDP NB COMMAND")
            data = str(data).split("/")
            if int(data[0]) == 0:
                if os.path.exists("C:\\FranceAsia\\App\\Logs"):
                    with open(name_udp, mode='a') as csv_file_udp:
                        csv_file_udp.write("{};{}\n".format(datetime.now().strftime("%H:%M:%S"), "START"))
                command = int(data[1])
                main.set_flag_launch(True)
                type_end = "None"
