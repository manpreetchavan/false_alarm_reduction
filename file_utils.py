import pandas as pd
import glob
import numpy
from os import chdir, getcwd

path=r'./training'

def all_hea_files(path, alarm = 'False', allparams = False, generator = False):
    chdir(path) # Setting current working directory to path
    allFiles=glob.glob("*.hea")
    allf=0 # all file count
    falf=0 # false file count 
    falselist=[] # false file name  list(w/o extension)
    for f in allFiles:
        file_hea = open_hea_physio_ecg(f)
        allf += 1
        if file_hea['alarm'] == alarm:
            falf += 1
            falselist.append(file_hea['file'])
            # if generator: yield file_hea['file']
    if allparams: return falselist,allf, falf
    return falselist
    
def open_hea_physio_ecg(path):
    with open(path, 'r') as f :
        lines = f.readlines()
        res = {}
        l = lines[0].split()
        res['file_csv'] = l[0] + '.csv'
        res['file'] = l[0]
        res['columns'] = int(l[1])
        res['sampling_frequency'] = int(l[2])
        res['samples'] = int(l[3])
        res['data'] = {}
        for i in range(res['columns']):
            l = lines[1 + i].split()
            res['data'].update({l[-1] : l[:-1]})
        res['type'] = lines[-2].split()[0][1:] # Type of Arhythmis classified
        res['alarm'] = lines[-1].split()[0][1:] # Giving False or True
    return res

if __name__ == "__main__":
    df = all_hea_files("./training/", 'True')
    print(df)
    print(open_hea_physio_ecg( df[0] + '.hea'))
