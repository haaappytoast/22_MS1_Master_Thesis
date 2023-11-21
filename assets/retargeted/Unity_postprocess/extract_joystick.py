import os
import numpy as np
import json
import torch

def from_txt_to_npy(path, txt_name, np_name):
    joystick_file = open(path + txt_name)
    joystick_json = json.load(joystick_file)

    print(joystick_json.keys())
    # parse motion info
    frameCount = joystick_json['frameCount']
    joystick = joystick_json['joystick']
    print(len(joystick))
    
    joystick_input = torch.zeros((len(joystick), 2))
    
    for i in range(len(joystick)):
        x, y = joystick[i]['x'], joystick[i]['y']
        joystick_input[i] = torch.tensor([x, y])

    torch.save(joystick_input, np_name)

if __name__ == "__main__":

    path = '/Unity_postprocess/joystick_input/'
    txt_name = 'joystick3.txt'
    np_name = os.getcwd() + path + txt_name[:-4]
    print(os.getcwd())
    from_txt_to_npy(os.getcwd() + path, txt_name, np_name)
    pass