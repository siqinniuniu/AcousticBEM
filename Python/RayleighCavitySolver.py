# ---------------------------------------------------------------------------
# Copyright (C) 2018 Frank Jargstorff
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
from Solver import *

class RayleighCavitySolver(Solver):
    def __init__(self, aVertex, aElement, nOpenElements, c = 344.0, density = 1.205):
        super(RayleighCavitySolver, self).__init__(aVertex, aElement, c, density)
        self.nOpenElements = nOpenElements

    def numberOfInterfaceElements(self):
        return self.nOpenElements
    
    def numberOfElements(self):
        """The number of elements forming the cavity."""
        return self.aElement.shape[0] - self.nOpenElements

    def totalNumberOfElements(self):
        return self.aElement.shape[0]

    def solveBoundary(self, k, boundaryCondition):
        M = self.computeBoundaryMatrix(k,
                                       boundaryCondition.alpha,
                                       boundaryCondition.beta)
        numberOfElements = self.totalNumberOfElements()
        b = np.zeros(2*numberOfElements, dtype=np.complex64)
        b[numberOfElements + self.nOpenElements: 2*numberOfElements] = boundaryCondition.f
        x = np.linalg.solve(M, b)
        
        return RayleighCavityBoundarySolution(self, boundaryCondition, k,
                                              x[0:numberOfElements],
                                              x[numberOfElements:2*numberOfElements])


        
