import matplotlib.pyplot as plt
import pandas
import scipy.signal as signal
from biosppy.signals import ecg
from sklearn.preprocessing import StandardScaler
# from scipy.fftpack import fft, fft2, fftshift
import numpy

def low_pass_filter(xn, cutoff, fs, order = 5):
    nyq = .5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    yn = signal.lfilter(b, a, xn)
    # yn = numpy.zeros(len(xn))
    return yn

def butter(cutoff, fs, order = 5):
    nyq = .5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def high_pass_filter(xn, cutoff, fs, order = 5):
    b, a = butter(cutoff, fs, order=order)
    return signal.filtfilt(b, a, xn)
def fft_(xn):
    return numpy.fft.fft(xn)
def fft_freq(xn):
    return numpy.fft.fftfreq(xn.shape[-1])

def diff(xn):
    le = len(xn) - 1
    yn = numpy.array([])
    for i in range(le):
        di = xn[i + 1] - xn[i]
        yn = numpy.append(yn, di)
    yn = numpy.append(yn, 0)
    print('diff : ', len(yn))
    return yn

def standardise(xn):
    scaler = StandardScaler().fit(xn)
    return scaler.transform(xn)

def get_qrs(xn, sampling_rate = 250):
    out = ecg.ecg(xn, sampling_rate=250., show=False)
    rpeaks, xnr = r_peaks(xn, 250)
    le = len(xn)
    thr = numpy.max(xn) * 0.70
    #get q's
    xnq, q = [], []
    for rin in rpeaks:
        ind = rin
        active = False
        while True:
            if xn[ind] < thr: active = True
            if active and xn[ind - 1] > xn[ind] : 
                xnq.append(xn[ind])
                q.append(ind)
                break
            ind -= 1
    #get r's
    xns, s = [], []
    for rin in rpeaks:
        ind = rin
        active = False
        while True:
            if xn[ind] < thr: active = True
            if active and xn[ind - 1] < xn[ind] : 
                xns.append(xn[ind - 1])
                s.append(ind - 1)
                break
            ind += 1
    #get qrs duration
    du_ts = []
    for jt in range(len(q)):    
        # in1 = q[jt]
        # while True:
        #     if xn[in1] > thr:
        #         break
        #     in1 -= 1
        # in2 = s[jt]
        # while True:
        #     if xn[in2] > thr:
        #         break
        #     in2 += 1
        # du_ts.append((in2 - in1) / 250)
        du_ts.append((s[jt] - q[jt]) / sampling_rate)
    return {'q_ts' : q, 'q' : xnq, 'r_ts' : rpeaks, 'r' : xnr, 's_ts' : s, 's' : xns, 'du_ts' : du_ts}


def get_heart_rate(xn, sampling_rate = 250, show = False):
    out = ecg.ecg(xn, sampling_rate=250., show=False)
    # print(out)
    return out['heart_rate_ts'] , out['heart_rate']
def r_peaks(xn, sampling_rate = 250, show = False):
    out = ecg.ecg(xn, sampling_rate=250., show=show)
    r_p = out['rpeaks']
    val = [xn[i] for i in r_p]
    return r_p, val
if __name__ == '__main__':
    dataset = pandas.read_csv('./csv/a386s.csv').values
    # print(dataset[:1000,0:1])
    xn = dataset[:10000,1]
    it = dataset[:10000, 0]
    plt.plot(it, xn, label = 'signal')
    yn = diff(xn)
    plt.plot(it, yn, label = 'diff')
    plt.legend()
    plt.show()

    # xn = low_pass_filter(xn, 20, 250)
    # xn = standardise(numpy.array(dataset[:1000, 0:1])).flat

   
    
    # print(xn)
    # out = ecg.ecg(xn, sampling_rate=250., show=True)
    # print(out['heart_rate'])
    # out = get_heart_rate(xn, 250, False)
    # ct = get_qrs(xn,250)
    # # print(out[1][5])
    # print(ct)
    # # print(get_heart_rate(xn, 250))
    # plt.plot(it, xn, label = 'ECG Signal')
    # plt.title("QRS  Plot")
    # plt.xlabel("Time in ms")
    
    # plt.plot(ct['q_ts'], ct['q'], 'bs', label = "Q peaks",  )
    
    # plt.plot(ct['r_ts'], ct['r'], 'r--' , label = "R peaks", color = "black", linewidth = 3.0 )
    
    # plt.plot(ct['s_ts'], ct['s'], 'g^', label= "S peaks")
    # plt.ylabel("ECG readings in mV")
    # # plt.legend("Yellow - R peaks\nOrange - Q peaks\nGreen - S")
    # plt.legend()
    # print(get_heart_rate(xn, 250, True))
    # # yn = low_pass_filter(xn, 12, 250, 6)
    # # plt.plot(it, yn, label= 'filtered signal')
    # # yn1 = high_pass_filter(yn, 12, 250, 6)
    # # plt.plot(it, yn1, label='High pass')
    # plt.show()
    # # # fft = fft_(xn)
    # # print(fft)
    # # freq = fft_freq(xn)
    # # print(freq)
    # # plt.plot(freq, fft.real,freq, fft.imag)
    # # plt.show()
    