import shutil
import os
import sys

directories = [x[0] for x in os.walk(".")][1:]
for d in directories:
    f = open(d+"/output.txt")
    l = f.readlines()[-1]
    finished = "finalized" in l

    if(not finished):
        print("Removing",d)
        shutil.rmtree(d)
