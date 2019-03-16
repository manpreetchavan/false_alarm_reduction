import pandas as pd
import glob
path=r'./training'

def all_false_files(path, all = False):
    allFiles=glob.glob(path + "\*.hea")
    allf=0
    falf=0
    word='#False'
    falselist=[]
    for file in allFiles:
        allf+=1
        f=open(file,'r')
        line=f.readlines()
        for x in line:
                if word in x:
                    falf+=1
                    fname=file
                    falselist.append(fname[-9:-4])
    if all : return falselist, allf, falf
    return falselist

print(all_false_files(path))