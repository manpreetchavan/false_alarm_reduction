import scipy.io as loader
import os
import pandas
import numpy
import heartpy
import matplotlib.pyplot as plt
# import c_labpl_qrs_detector as qrs_detector
import mne.preprocessing.ecg as preps
def get_files(di, ext = ".mat"):
    # return [f for f in os.listdir(".") if os.path.isfile(f)]
    files = []
    for r, d, f in os.walk(di):
        for fil in f:
            if '.mat' in fil:
                files.append(os.path.join(r, fil))
    return files

def to_csv(file_path):
    file_name = file_path.split("/")[-1]
    # print(file_name)
    if not file_name.endswith('.mat') :
        return 
        # raise NameError("Invalid file extension ") 
    mat = loader.loadmat(file_path)
    # mat = {k : v for k, v in mat.items() if k[0] != '_'}
    # data = pandas.DataFrame({k : pandas.Series(v[0]) for k, v in mat.iteritems()})
    # data.to_csv("C" + file_name + ".csv")
    fi = open("./csv/" + os.path.splitext(file_name)[0] + ".csv", "w")

    II = mat['val'][0]
    V = mat['val'][1]
    le = len(V)
    fi.write("TIME,II,V\n")
    for i in range(le):
        fi.write(str(i) + "," + str(II[i]) +  "," + str(V[i]) + "\n")
    fi.close()
# file_list = get_files("./training/")
# # print(file_list)
# for f in file_list:
#     to_csv(f)

# array = pandas.read_csv('./csv/a103l.csv')
# # ecg_arr = array[]
# arr = heartpy.raw_to_ecg(array.iloc[:500, 0])  

# # art = numpy.asarray(arr, 'np.float64')
# print(array)
# plt.plot(array)
# plt.show()
# qrs_detector.QRSDetectorOffline(ecg_data_path = "./raw.csv", verbose=True)

# to_csv("./training/a104s.mat")
