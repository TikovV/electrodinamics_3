import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, maxSize_metr, dx, y_min, y_max, probePos, sourcePos):
        plt.ion()
        self.probePos = probePos
        self.sourcePos = sourcePos
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_metr, dx), [0]*int(maxSize_metr/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')
        self.ax.plot(sourcePos*dx, 0, 'ok')
        self.ax.set_xlim(0, maxSize_metr)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.ax.grid()

    def updateData(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Probe:
    def __init__(self, probePos, maxTime, dt):
        self.maxTime = maxTime
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.maxTime)
        
    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

def showProbeSignal(probe):
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, probe.maxTime * probe.dt, probe.dt), probe.E)
    ax.set_xlabel('t, c')
    ax.set_ylabel('Ez, В/м')
    ax.set_xlim(0, probe.maxTime * probe.dt)
    ax.grid()
    plt.show(block=False)
    plt.pause(20)

def showProbeSpectrum(probe):
    spectrum = np.abs(fft(probe.E))
    spectrum = fftshift(spectrum)
    df = 1/(probe.maxTime*probe.dt)
    freq = np.arange(-probe.maxTime*df /2, probe.maxTime*df/2, df)
    fig, ax = plt.subplots()
    ax.plot(freq, spectrum/max(spectrum))
    ax.set_xlabel('f, Гц')
    ax.set_ylabel('|S|/|Smax|')
    ax.set_xlim(0, 200e6)
    ax.grid()
    plt.show(block=False)
    plt.pause(20)


W0 = 120*np.pi
Sc = 1
eps = 8.5

#время моделирования в отчетах
maxTime = 2100

#размер в метрах
maxSize_metr = 9

#шаг по пространству
dx = maxSize_metr/1200

#размер в отчетах
maxSize_disk = int(maxSize_metr/dx)
probePos = int(maxSize_metr/4/dx) # датчик
sourcePos = int(maxSize_metr/2/dx) # источник



#шаг по времени
dt = dx*np.sqrt(eps)*Sc/3e8

probe = Probe(probePos, maxTime, dt)
display = FieldDisplay(maxSize_metr, dx, -2, 2, probePos, sourcePos)
Ez = np.zeros(maxSize_disk)
Hy = np.zeros(maxSize_disk-1)

A_max = 1e2
F_max = 150e6
w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)/dt
d_g = w_g * np.sqrt(np.log(A_max))

#Расчет коэффициентов для граничных условий
Sc_L= Sc/np.sqrt(eps)
koefABC_L = (Sc_L - 1) / (Sc_L + 1)
Sc_R = Sc/np.sqrt(eps)
koefABC_R = (Sc_R - 1) / (Sc_R + 1)

Ez_old_R = Ez[-2]
Ez_old_L = Ez[1]


for q in range(1, maxTime):
     Hy = Hy +(Ez[1:]-Ez[:-1])*Sc/W0
     Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1])*Sc*W0/eps
     Ez[sourcePos] += np.exp(-((q - d_g) / w_g) ** 2)
     Ez[0] = Ez_old_L + koefABC_L * (Ez[1] - Ez[0])
     Ez[-1] = Ez_old_R + koefABC_R * (Ez[-2] - Ez[-1])
     Ez_old_R = Ez[-2]
     Ez_old_L = Ez[1]
     probe.addData(Ez)
     if q % 20 == 0:
          display.updateData(Ez)


showProbeSpectrum(probe)
showProbeSignal(probe)
