from threading import Thread
import sys
sys.path.append('franceAsia')
import udp
import command

flag_launch = False
nb_command = 0


def set_nb_command(nb):
    global nb_command
    nb_command = nb


def set_flag_launch(b):
    global flag_launch
    flag_launch = b


# Permit user to choose
udp.choose_udp_config()

# Start udp server
thread_udp_init = Thread(target=udp.init_udp)
thread_udp_init.start()

# Infinite loop, start a new command when flag_launch is set to true
while True:

    if flag_launch:
        command.set_flag_order(True)
        print("BEFORE LAUNCH")
        command.launch(nb_command)
        flag_launch = False


