import numpy as np
from math import sin, cos, pi

#найпростіший алгоритм перетворення Фур'є
def slow_discrete_furier_transform(arr):
    N = len(arr)
    N2 = N//2+1
    result = [0.0+0.0j]*N2

    j=complex(0.0,1.0)

    for f in range(N2):
        for n in range(N):
            result[f]+=arr[n]*(cos(2*pi*f*n/N) + j*sin(2*pi*f*n/N))


    return result
#повільний табличний метод (підрахунок майже n^2 табличних значень)
#з врахуванням nf=fn(під головною діагоналю)
def discrete_transform_table(arr):
    N = len(arr)
    N2 = N // 2 + 1
    result = [0.0 + 0.0j] * N2

    j = complex(0.0, 1.0)
    a = 2 * pi / N

    table = np.empty((N2,N), dtype=np.complex)
    table[0,:]=1.0+0.0j
    table[:,0]=1.0+0.0j

    #підрахунок значень таблиці (під головною діагоналлю)
    for  f in range(1,N2):
        for n in range(1,f+1):
            temp =cos(a*f*n)+j*sin(a*f*n)
            table[f,n],table[n,f] = temp, temp #під і над діагоналлю - однакові значення

    #значення поза блоком, який перетинає діагональ
    for  f in range(1, N2):
        for n in range(N2,N):
            table[f,n] =cos(a*f*n)+j*sin(a*f*n)

    for f in range(0, N2):
        for n in range(0, N):
            result[f] += arr[n] * table[f,n]

    return result
#пришвидшений табличний метод(n табличних значень)
#з врахуванням періодичності sin, cos T=2*pi
def fourier_transform_table_faster(arr):
    N = len(arr)
    N2 = N // 2 + 1
    result = np.zeros(N2, dtype=np.complex)

    j = complex(0.0, 1.0)
    a = 2 * pi / N

    table= np.empty(N, dtype=np.complex)

    #підрахунок значень таблиці
    for i in range(N):
        table[i]=cos(a*i)+j*sin(a*i)

    for f in range(N2):
        for n in range(N):
            result[f] += arr[n] * table[f*n%N]

    return result

def fast_fourier_transform(a:np.ndarray):
    n=a.size
    if n==1: return
    n_2=n//2
    #a0, a1 - парні, непарні елементи масиву a відповідно
    a0 = np.empty(n_2, dtype=np.complex)
    a1 = np.empty(n_2, dtype=np.complex)
    j=0
    for i in range(0,n,2): #розділення
        a0[j]=a[i]
        a1[j]=a[i+1]
        j+=1
    #рекурсія
    fast_fourier_transform(a0)
    fast_fourier_transform(a1)

    ang = 2*pi/n
    w = 1.0+0.0j #WpN
    wn = cos(ang)+sin(ang)*1.0j #W1N

    for i  in range(0,n_2): #об'єднання
        a[i]=a0[i] + w * a1[i] #операція "метелик"
        a[i + n_2] = a0[i] - w * a1[i]
        w *= wn







