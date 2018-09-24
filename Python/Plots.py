import numpy as np
import matplotlib.pyplot as plt

def polarPlot(title, frequencies, angles, magnitudes):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    ax.set_title(title)
    for f, aMagnitude in zip(frequencies, magnitudes):
        ax.plot(angles, aMagnitude, label=str(f) + " Hz")
    ax.legend()
    ax.set_rgrids([30, 60, 90, 120, 150], horizontalalignment="left")
    ax.set_rlabel_position(100.0)  # get radial labels away from plotted line
    ax.grid(True)
    plt.show()
    
def plotPolarMap(title, frequencies, angles, magnitudes, stepSize = 6):
    X, Y = np.meshgrid(frequencies, angles)

    minMagnitude = np.floor(np.min(magnitudes) / stepSize) * stepSize
    maxMagnitude = np.ceil(np.max(magnitudes) / stepSize) * stepSize
    levels = (maxMagnitude - minMagnitude) // stepSize

    fig, ax = plt.subplots(figsize = (18, 10))
    aLevel = np.linspace(minMagnitude, maxMagnitude, levels + 1)
    CS = ax.contourf(X, Y, magnitudes, aLevel, cmap = plt.cm.seismic)
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xscale("log") 
    ax.set_ylabel('Angle [rad]')

    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('Sound Magnitude [dB]')
    
def plotComplexImpedance(title, frequencies, impedance):
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xscale('log') 
    ax.set_ylabel('Mechanical Impedance')
    ax.plot(aFrequency, np.real(aNumericalZm), label='Real')
    ax.plot(aFrequency, np.imag(aNumericalZm), label='Imaginary')
    ax.legend()

