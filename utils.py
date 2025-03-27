import os

def check_folder(example):
    cwd = os.getcwd()
    newpath = os.path.join(os.path.join(cwd,"images"),"images_"+example) 
    if not os.path.exists(newpath): #We create the a folder called gif
        os.makedirs(newpath)
    return newpath