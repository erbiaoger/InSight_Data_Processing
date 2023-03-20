import os
from scipy import signal, fft
import matplotlib.pyplot as plt
import obspy

def plot_all_trace(data):
    """
    To plot three trace E, N, Z of the station
    
    type: data: list
    """
    fig = plt.figure()
    for index, value in enumerate(data):
        fig.subplots(140 + index)
        value.plot(color='red',
                        size=(1600, 600))

    plt.show()
    
def specgram(data, fs):
    ## plt specgram
    Pxx, freqs, bins, im = plt.specgram(data, Fs=fs)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequence [Hz]')
    plt.show()
    
def plot_freq(data, delta):
    npts = len(data)
    xf = fft.fftfreq(npts, delta)
    yf = fft.fft(data)

    plt.plot(xf[:npts//2], abs(yf[0: npts//2]))
    plt.xlabel('Frequence [Hz]')
    plt.show()
    
def plot_time(my_xticks, data):
    plt.plot(my_xticks, data)
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.show()
    
    
class Solar():
    """
    Read solar data to a dictory
    
    usage:
        >>> solar = utils.Solar('LF', 'A').get_solar()
        
        <<< 
 {'S0173a': {'ACC': <obspy.core.stream.Stream object at 0x1514ed1c0>,
            'DISP': <obspy.core.stream.Stream object at 0x1514ed850>,
            'VEL': <obspy.core.stream.Stream object at 0x1514edee0>,
            'origin': <obspy.core.stream.Stream object at 0x151a3baf0>},
 'S0809a': {'ACC': <obspy.core.stream.Stream object at 0x15144a040>,
            'DISP': <obspy.core.stream.Stream object at 0x151482130>,
            'VEL': <obspy.core.stream.Stream object at 0x151482640>,
            'origin': <obspy.core.stream.Stream object at 0x1513770d0>},
 'S0820a': {'ACC': <obspy.core.stream.Stream object at 0x15147fa00>,
            'DISP': <obspy.core.stream.Stream object at 0x15137bf40>,
            'VEL': <obspy.core.stream.Stream object at 0x15137b970>,
            'origin': <obspy.core.stream.Stream object at 0x15147fd30>},
    
    """
    def __init__(self, solar_type, quality):
        self.solar_type = solar_type
        self.quality = quality
        

    def get_path(self):
        solar_type = self.solar_type
        quality = self.quality
        pathinSightDATA = os.path.join("/home/erbiaoger/MyProjects/InSight_Data_Processing/InSight-seismic-data-downloader/DATA/")
        file_type = {'BB': 'BROADBAND',
                    'LF': 'LOW_FREQUENCY',
                    '2.4': '2.4_HZ',
                    'HF': 'HIGH_FREQUENCY',
                    'VF': 'VERY_HIGH_FREQUENCY'}
        
        return os.path.join(pathinSightDATA, file_type[solar_type], quality)
        
    def get_solar(self):
        path = self.get_path()
        name_dic = {'raw': '.mseed',
                    'ACC': '_ACC.mseed',
                    'DISP': '_DISP.mseed', 
                    'VEL': '_VEL.mseed'}
        
        data = {}
        for folder in os.listdir(path):
            data[folder] = {}
            for key, value in name_dic.items():
                file_path = os.path.join(path, folder, folder + value)
                data[folder][key] = obspy.read(file_path)
                
        return data
