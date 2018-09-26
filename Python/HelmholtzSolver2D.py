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
from HelmholtzSolver import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals2D_C import *
else:
    from HelmholtzIntegrals2D import *        
        
from Geometry import *
from AcousticProperties import *
import numpy as np


class HelmholtzSolver2D(HelmholtzSolver):
    def __init__(self, oChain, c = 344.0, density = 1.205):
        super(HelmholtzSolver2D, self).__init__(oChain, c, density)
        self.aCenters = self.oGeometry.centers()
        # lenght of the boundary elements (for the 3d shapes this is replaced by aArea
        self.aLength = self.oGeometry.lengths()

    def computeBoundaryMatrices(self, k, mu, orientation):
        A = np.empty((self.numberOfElements(), self.numberOfElements()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        aCenter = self.oGeometry.centers()
        aNormal = -self.oGeometry.normals()
        
        for i in range(self.numberOfElements()):
            center = aCenter[i]
            centerNormal = aNormal[i]
            for j in range(self.numberOfElements()):
                qa, qb = self.oGeometry.edgeVertices(j)

                elementL  = ComputeL(k, center, qa, qb, i==j)
                elementM  = ComputeM(k, center, qa, qb, i==j)
                elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
                
                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i,i] -= 0.5 * mu
                B[i,i] += 0.5
            elif orientation == 'exterior':
                A[i,i] += 0.5 * mu
                B[i,i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
                
        return A, B

    def computeBoundaryMatricesInterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'interior')
    
    def computeBoundaryMatricesExterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'exterior')

    def solveSamples(self, solution, aIncidentPhi, aSamples, orientation):
        assert aIncidentPhi.shape == aSamples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aIncidentPhi.size):
            p  = aSamples[i]
            sum = aIncidentPhi[i]
            for j in range(solution.aPhi.size):
                qa, qb = self.oGeometry.edgeVertices(j)

                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)
                if orientation == 'interior':
                    sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult

class InteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        return super(InteriorHelmholtzSolver2D, self).solveBoundary('interior', k, boundaryCondition, boundaryIncidence, mu)

class ExteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        return super(ExteriorHelmholtzSolver2D, self).solveBoundary('exterior', k, boundaryCondition, boundaryIncidence, mu)

