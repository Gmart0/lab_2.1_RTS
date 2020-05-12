import numpy as np
import matplotlib.pyplot as plt
from lab1.lab1_main import generate_signal
from time import perf_counter
from lab2.furier_transform import *
if __name__=="__main__":
    n=5 #n-число гармонік
    Wmax = 1200 #Wmax-гранична частота
    N=2048 #N-кількість дискретних відліків
    step=0.0004 #step - крок аргументу для дискретних відліків
    N_2 = N//2+1

    t, signal1 = generate_signal(n, Wmax, N, step)
    T=t[N-1] #період часу

    spectrum = np.fft.rfft(signal1)
    amplitudeSpectrum = np.abs(spectrum)

    #timeBefore = perf_counter()
    spectrum2=fourier_transform_table_faster(signal1)
    #timeDiff = perf_counter() - timeBefore
    #print("time elapsed(table fast): ", timeDiff)
    amplitudeSpectrum2 = np.abs(spectrum2)

    signalComplex = np.array(signal1, dtype= np.complex)
    print(signalComplex);
    fast_fourier_transform(signalComplex)
    amplitudeSpectrum3=np.abs(signalComplex[:N_2])

    freq1 = np.fft.rfftfreq(signal1.size, d=step)
    freq2 = np.linspace(1/T, 0.5/step, amplitudeSpectrum.size) #freq1 == freq2

    #######################################################
    #print("T = ", T, "(0.5/step) = ", 0.5/step)
    #print("fmin = ", 1/T, " fmax = ", 0.5 / step)

    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle('Згенерований сигнал')

    ax1.set_ylabel('x(t) - сигнал')
    ax1.set_xlabel('t')

    ax1.plot(t, signal1, color='blue', linewidth=0.5)
    ax1.grid(True)

    ax2.set_xlabel('частота  (f)')
    ax2.set_ylabel('амплітуда')
    ax2.plot(freq2, amplitudeSpectrum, color='green', linewidth=1)
    ax2.grid(True)

    fig2, ax3 = plt.subplots(1)
    fig2.suptitle("Перетвор Фур'є (швидке)")
    ax3.set_ylabel('амплітуда')
    ax3.set_xlabel('частота')
    ax3.plot(freq2, amplitudeSpectrum3,   color='green', linewidth=0.5)
    plt.show()

