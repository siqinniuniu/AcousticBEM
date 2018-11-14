# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import numpy as np
from AcousticProperties import *


class BoundaryCondition(object):
    def __init__(self, size):
        self.alpha = np.empty(size, dtype = np.complex64)
        self.beta  = np.empty(size, dtype = np.complex64)
        self.f     = np.empty(size, dtype = np.complex64)

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "alpha = {}, ".format(self.alpha)
        result += "beta = {}, ".format(self.beta)
        result += "f = {})".format(self.f)
        return result

        
class BoundaryIncidence(object):
    def __init__(self, size):
        self.phi = np.empty(size, dtype = np.complex64)
        self.v   = np.empty(size, dtype = np.complex64)

        
class BoundarySolution(object):
    def __init__(self, parent, boundaryCondition, k, aPhi, aV):
        self.parent = parent
        self.boundaryCondidtion = boundaryCondition
        self.k      = k
        self.aPhi   = aPhi
        self.aV     = aV

    def __repr__(self):
        result  = self.__class__.__name__ +  "("
        result += "parent = " + repr(self.parent) + ", "
        result += "boundaryCondition = " + repr(self.boundaryCondition) + ", "
        result += "k = " + repr(self.k) + ", "
        result += "aPhi = " + repr(self.aPhi) + ", "
        result += "aV = " + repr(self.aV) + ")"
        return result

    def __str__(self):
        res =  "Density of medium:      {} kg/m^3\n".format(self.parent.density)
        res += "Speed of sound:         {} m/s\n".format(self.parent.c)
        res += "Wavenumber (Frequency): {} ({} Hz)\n\n".format(self.k, wavenumberToFrequency(self.k))
        res += "index          Potential                   Pressure                    Velocity              Intensity\n\n"
        for i in range(self.aPhi.size):
            pressure = soundPressure(self.k, self.aPhi[i], c=self.parent.c, density=self.parent.density)
            intensity = AcousticIntensity(pressure, self.aV[i])
            res += "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e}\n".format( \
                    i+1, self.aPhi[i].real, self.aPhi[i].imag, pressure.real, pressure.imag,                      \
                    self.aV[i].real, self.aV[i].imag, intensity)
        return res

    def pressure(self, namedPartition = None):
        if namedPartition is None:
            return soundPressure(self.k, self.aPhi, c=self.parent.c, density=self.parent.density)
        else:
            range = self.parent.oGeometry.namedPartition[namedPartition]
            return soundPressure(self.k, self.aPhi[range[0]: range[1]],
                                 c=self.parent.c, density=self.parent.density)

    def pressureDecibell(self, namedPartition = None):
        return SoundMagnitude(self.pressure(namedPartition))
    
    def radiationRatio(self):
        power  = 0.0
        bpower = 0.0
        for i in range(self.aPhi.size):
            pressure = soundPressure(self.k, self.aPhi[i], c=self.parent.c, density=self.parent.density)
            power   += AcousticIntensity(pressure, self.aV[i])
            bpower  += (self.parent.density * self.parent.c * np.abs(self.aV[i])**2)
        return 2 * power / bpower

    def mechanicalImpedance(self, namedPartition = None):
        Zm = 0.0
        if namedPartition is None:
            aV = self.aV
        else:
            partition = self.parent.oGeometry.namedPartition[namedPartition]
            aV = self.aV[partition[0]:partition[1]]
        for p, a, v in zip(self.pressure(namedPartition), self.parent.elementArea(namedPartition), aV):
            Zm += p * a / v
        return Zm

class InteriorBoundarySolution(BoundarySolution):
    def solveSamples(self, aIncidentPhi, aPoints):
        return SampleSolution(self, self.parent.solveSamples(self, aIncidentPhi, aPoints, 'interior'))
    
class ExteriorBoundarySolution(BoundarySolution):
    def solveSamples(self, aIncidentPhi, aPoints):
        return SampleSolution(self, self.parent.solveSamples(self, aIncidentPhi, aPoints, 'exterior'))

class RayleighBoundarySolution(BoundarySolution):
    def solveSamples(self, aSamplePoints):
        return self.parent.solveSamples(self, aSamplePoints)

class RayleighCavityBoundarySolution(BoundarySolution):
    def solveCavity(self, aCavitySamples):
        """Solve for point internal to the cavity."""
        return self.parent.solveInterior(self, aCavitySamples)

    def solveSamples(self, aSamplePoints):
        """Solve for points in half space (exterior)."""
        return self.parent.solveExterior(self, aSamplePoints)
            
class SampleSolution(object):
    def __init__(self, boundarySolution, aPhi):
        self.boundarySolution = boundarySolution
        self.aPhi    = aPhi

    def __repr__(self):
        result = "SampleSolution("
        result += "boundarySolution = " + repr(self.parent) + ", "
        result += "aPhi = " + repr(self.aPhi) + ")"
        return result

    def __str__(self):
        result = "index          Potential                    Pressure               Magnitude         Phase\n\n"
        for i in range(self.aPhi.size):
            pressure = soundPressure(self.boundarySolution.k, self.aPhi[i],
                                     c=self.boundarySolution.parent.c,
                                     density=self.boundarySolution.parent.density)
            magnitude = SoundMagnitude(pressure)
            phase = SignalPhase(pressure)
            result += "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}\n".format( \
                i+1, self.aPhi[i].real, self.aPhi[i].imag, pressure.real, pressure.imag, magnitude, phase)
            
        return result
