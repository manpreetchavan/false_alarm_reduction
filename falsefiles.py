import pandas as pd
import glob
path=r'E:\TE\VI\false_alarm_reduction\training'
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
print('All header files : ',allf)
print('All false alarm files : ',falf)
print(falselist)