import matplotlib.pyplot as plt
import pandas
import scipy.signal as signal
from biosppy.signals import ecg
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

def get_qrs(xn, sampling_rate = 250):
    out = ecg.ecg(xn, sampling_rate=250., show=False)
    rpeaks, xnr = r_peaks(xn, 250)
    le = len(xn)
    #get q's
    xnq, q = [], []
    for rin in rpeaks:
        ind = rin
        active = False
        while True:
            if xn[ind] < 0: active = True
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
            if xn[ind] < 0: active = True
            if active and xn[ind - 1] < xn[ind] : 
                xns.append(xn[ind - 1])
                s.append(ind - 1)
                break
            ind += 1
 
 
    return {'q_ts' : q, 'q' : xnq, 'r_ts' : rpeaks, 'r' : xnr, 's_ts' : s, 's' : xns}


def get_heart_rate(xn, sampling_rate = 250, show = False):
    out = ecg.ecg(xn, sampling_rate=250., show=show)
    # print(out)
    return out['heart_rate_ts'] , out['heart_rate']
def r_peaks(xn, sampling_rate = 250, show = False):
    out = ecg.ecg(xn, sampling_rate=250., show=show)
    r_p = out['rpeaks']
    val = [xn[i] for i in r_p]
    return r_p, val
if __name__ == '__main__':
    dataset = pandas.read_csv('./csv/a103l.csv').values
    print(dataset[:1250,1])
    xn = dataset[:1250, 1]
    it = dataset[:1250, 0]
    # out = ecg.ecg(xn, sampling_rate=250., show=True)
    # print(out['heart_rate'])
    out = get_heart_rate(xn, 250)
    ct = get_qrs(xn,250)
    print(out[1][5])
    # print(get_heart_rate(xn, 250))
    plt.plot(it, xn, label = 'signal')
    
    plt.plot(ct['q_ts'], ct['q'], label = 'signal')
    
    plt.plot(ct['r_ts'], ct['r'], label = 'signal')
    
    plt.plot(ct['s_ts'], ct['s'], label = 'signal')
    # yn = low_pass_filter(xn, 12, 250, 6)
    # plt.plot(it, yn, label= 'filtered signal')
    # yn1 = high_pass_filter(yn, 12, 250, 6)
    # plt.plot(it, yn1, label='High pass')
    plt.show()
    # # fft = fft_(xn)
    # print(fft)
    # freq = fft_freq(xn)
    # print(freq)
    # plt.plot(freq, fft.real,freq, fft.imag)
    # plt.show()
    