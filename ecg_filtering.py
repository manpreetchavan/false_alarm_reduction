import matplotlib.pyplot as plt
import pandas
import scipy.signal as signal
from biosppy.signals import ecg
from sklearn.preprocessing import StandardScaler
# from scipy.fftpack import fft, fft2, fftshift
import numpy

def notch_fi(data, low, high, fs, ripple, order = 3, ftype = 'butter', filt = True):
    nyq  = fs/2.0
    low  = low/nyq
    high = high/nyq
    b, a = signal.iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=ftype)
    filtered_data = signal.filtfilt(b, a, data) if filt else signal.lfilter(b, a, data)
    return filtered_data

def low_pass_filter(xn, cutoff, fs, order = 5, filt = True):
    nyq = .5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    yn = signal.filtfilt(b, a, xn) if filt else signal.lfilter(b, a, xn)
    # yn = numpy.zeros(len(xn))
    return yn

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, filt = True):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data) if filt else signal.lfilter(b, a, data)
    return y

def butter_pass(cutoff, fs, order = 5):
    nyq = .5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def high_pass_filter(xn, cutoff, fs, order = 5):
    b, a = butter_pass(cutoff, fs, order=order)
    return signal.filtfilt(b, a, xn)

def fft_(xn, ab = False):
    res = numpy.fft.fft(xn)
    if ab : return numpy.abs(res)
    return res

def fft_freq(xn, ab = False):
    res = numpy.fft.fftfreq(xn.shape[-1])
    if ab: return numpy.abs(res)
    return res

def diff(xn):
    le = len(xn) - 1
    yn = numpy.array([])
    for i in range(le):
        di = xn[i + 1] - xn[i]
        yn = numpy.append(yn, di)
    yn = numpy.append(yn, 0)
    print('diff : ', len(yn))
    return yn
def intrgl(xn, c = 0):
    le = len(xn) - 1
    yn = numpy.zeros(le + 1)
    for i in range(le):
        yn[i + 1] = yn[i] + xn[i]
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

def preprocess(xn):
    tn = xn
    # print(ec)

    # yn = butter_bandpass_filter(tn, 10, 50, 250, order= 4)
    # yn = notch_fi(tn, 30, 124, 250, 15)
    # yn = low_pass_filter(tn, 50, 250)
    # yn = high_pass_filter(yn, 0.67, 250)
    ec = ecg.ecg(xn, 250, False)
    yn = ec['filtered']
    plt.plot(it, yn, label = 'signal')
    # ect = butter_bandpass_filter(yn, 10, 40, 250)
    # yn = butter_bandpass_filter(yn, 0.67, 4, 250, order= 2 )
    # th = butter_bandpass_filter(yn, 3, 7, 250 )
    # # ffy = fft_(yn)
    # ff = [v * 250 for v in fft_freq(yn, True)]
    # print(ff)
    # print(ffy)
    # tn = diff(yn)
    # yn = intrgl(tn)
    # plt.plot(it, ect, label = 'ECG')
    pt = peaks_detection(xn, 250)
    p, q, r, s, t = pt['p_ts'], pt['q_ts'],  pt['r_ts'],  pt['s_ts'],  pt['t_ts']
    vp, vq, vr, vs, vt = [], [], [], [], []
    for i in range(len(r)):
        vp.append(yn[p[i]])
        vq.append(yn[q[i]])
        vr.append(yn[r[i]])
        vs.append(yn[s[i]])
        vt.append(yn[t[i]])

    # plt.plot(it, yn, label = '')
    plt.plot(p, vp, label = 'P')
    plt.plot(q, vq, label = 'q')
    plt.plot(r, vr, label = 'r')
    plt.plot(s, vs, label = 's')
    plt.plot(t, vt, label = 't')
    # plt.plot(it, th, label = 'T')
    plt.legend()
    plt.show()

def peaks_detection(xn, sampling_rate):
    rpeaks, rval = r_peaks(xn, sampling_rate)
    qrs = get_qrs(xn, sampling_rate)
    q = qrs['q_ts']
    r = qrs['r_ts']
    s = qrs['s_ts']
    du_ts = qrs['du_ts']
    p, t= [], []

    #detecting p
    for q_ind in q:
        maxin = q_ind
        for i in range(q_ind, q_ind - 64, -1):
            if xn[maxin] < xn[i + 1]: maxin = i
        p.append(maxin)

    #detecting t
    for s_ind in s:
        maxin = s_ind
        for i in range(s_ind, s_ind + 40):
            if xn[maxin] < xn[i]: maxin = i
        t.append(maxin)

    return {'p_ts' : p, 'q_ts' : q, 'r_ts' : rpeaks, 's_ts' : s, 't_ts' : t, 'qrs_ts' : du_ts}

def get_dur(xn, sampling_rate = 250):
    fe = peaks_detection(xn, sampling_rate)
    qrs = fe['qrs_ts']
    p, q, r, s, t = fe['p_ts'], fe['q_ts'], fe['r_ts'], fe['s_ts'], fe['t_ts'], 
    pr, qt, st = [], [], []
    for i in range(len(p)):
        pr.append((r[i] - p[i]) / sampling_rate)
        qt.append((t[i] - q[i]) / sampling_rate)
        st.append((t[i] - s[i]) / sampling_rate)

    return {'pr': pr, 'qt': qt, 'st': st, 'qrs': qrs}

    
        



if __name__ == '__main__':
    dataset = pandas.read_csv('./csv/a103l.csv').values
    # print(dataset[:1000,0:1])
    xn = dataset[:500,1]
    it = dataset[:500, 0]
    # preprocess(xn)
    print("HJK : ", get_dur(xn, 250))
    

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
    