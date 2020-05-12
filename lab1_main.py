import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import pi
from time import perf_counter

def math_expectation(arr):
    sum=0.0
    for elem in arr:
        sum+=elem
    return sum/len(arr)

def dispersion(arr):
    m=math_expectation(arr)
    sum=0.0
    for elem in arr:
        sum+=(elem-m)**2
    return sum/len(arr)


def generate_signal(n:int, Wmax:float, N:int, step:float = 1.0):
    Wmax2 = Wmax*2*pi #циклічна частота
    A, fi, w = 0.0, 0.0, 0.0
    maxArgument = step * (N-1)
    signal = np.zeros(N, dtype=np.float32)
    t = np.linspace(0, maxArgument, N, dtype=np.float32)
    for i in range(n):
        A = rnd.random()
        fi = rnd.uniform(0, 2 * pi)
        w = Wmax2 / n * (i + 1)

        signal += A * np.sin(w * t + fi)
    return t, signal

def correlation_function(arr1, arr2, M1=None, M2=None)->np.ndarray:
    if(M1==None):
        M1=sum(arr1)/len(arr1)
    if(M2==None):
        M2=sum(arr2)/len(arr2)

    N=len(arr1)
    correlation=np.empty(N)
    sum=0.0
    for tau in range(N-1):
        sum=0.0
        for i in range(N-tau):
            sum+=(arr1[i]-M1)*(arr2[i+tau]-M2)
        correlation[tau]=sum/(N-tau-1)

    correlation[N-1]=correlation[N-2]
    return correlation

def correlation_function_2(arr1, arr2, M1=None, M2=None)->np.ndarray:
    if(M1==None):
        M1=sum(arr1)/len(arr1)
    if(M2==None):
        M2=sum(arr2)/len(arr2)

    N1, N2 = len(arr1), len(arr2)

    if (N2//2<N1): print("Error: arrays length:  ", N1, " ",N2)

    correlation = np.empty(N)
    sum = 0.0

    for tau in range(N1):
        sum=0.0
        for i in range(N1):
            sum+=(arr1[i]-M1)*(arr2[i+tau]-M2)
        correlation[tau] = sum /(N1-1)

    return correlation

if __name__=="__main__":
    n=8 #n-число гармонік
    Wmax = 1200 #Wmax-гранична частота
    N=1024 #N-кількість дискретних відліків

    t, signal1 = generate_signal(n, Wmax, N, 0.0001)
    t, signal2 = generate_signal(n, Wmax, N, 0.0001)

    #M-мат. очікування, D - дисперсія
    # timeBeforeMx = process_time()
    # Mx1 = signal1.mean()
    # timAfterMx = process_time()
    # timeMx=timAfterMx-timeBeforeMx

    timeBeforeMx = perf_counter()
    Mx1 = signal1.mean()
    timAfterMx = perf_counter()
    timeMx=timAfterMx-timeBeforeMx

    timeBeforeDx = perf_counter()
    Dx1=signal1.var()
    timeAfterDx = perf_counter()
    timeDx=timeAfterDx-timeBeforeDx

    Mx2, Dx2 = signal2.mean(), signal2.var()

    Rxx = correlation_function(signal1, signal1, Mx1, Mx1)
    Rxy = correlation_function(signal1, signal2, Mx1, Mx2)

    print("Сигнал 1:")
    print("Математичне очікування Mx1 = ",Mx1)
    print("Дисперсія Dx1 = ", Dx1)

    print("Час обчислення Mx1 = ", timeMx, " c")
    print("Час обчислення Dx1 = ", timeDx, " c")

    print("Сигнал 2:")
    print("Математичне очікування Mx2 = ", Mx2)
    print("Дисперсія Dx2 = ", Dx2)

    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle('Випадкові сигнали')

    #ax1, ax2 = fig1.add_subplot(211)

    #ax1.set_xlabel('t')
    ax1.set_ylabel('x(t) - сигнал 1')
    ax1.plot(t, signal1, color='blue', linewidth=0.5)
    ax1.grid(True)

    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t) - сигнал 2')
    ax2.plot(t, signal2, color='blue', linewidth=0.5)
    ax2.grid(True)

    fig2, (ax3, ax4) = plt.subplots(2)
    fig2.suptitle('Кореляційні функції')

    ax3.set_ylabel('Rxx')
    ax3.plot(t, Rxx, color='red', linewidth=0.5)
    ax3.grid(True)

    ax4.set_xlabel('tau')
    ax4.set_ylabel('Rxy')
    ax4.plot(t, Rxy, color='red', linewidth=0.5)
    ax4.grid(True)

    #
    # fig2 = plt.figure(2)
    # fig2.suptitle('Кореляційна функція')
    # ax2 = fig2.add_subplot(111)
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('x(t) - сигнал')
    # ax2.plot(t, Rxx, color='blue', linewidth=0.5)
    # ax2.grid(True)

    plt.show()










