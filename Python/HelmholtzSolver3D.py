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
from Geometry import *
from numpy.linalg import norm

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        

    
class HelmholtzSolver3D(HelmholtzSolver):
    def __init__(self, *args, **kwargs):
        super(HelmholtzSolver3D, self).__init__(*args, **kwargs)
        self.aCenters = (self.aVertex[self.aElement[:, 0]] +\
                         self.aVertex[self.aElement[:, 1]] +\
                         self.aVertex[self.aElement[:, 2]]) / 3.0
        # area of the boundary alements
        self.aArea = np.empty(self.aElement.shape[0], dtype=np.float32)
        self.aNormals = np.empty((self.aElement.shape[0], 3), dtype=np.float32)
        for i in range(self.aArea.size):
            a = self.aVertex[self.aElement[i, 0], :]
            b = self.aVertex[self.aElement[i, 1], :]
            c = self.aVertex[self.aElement[i, 2], :]
            ab = b - a
            ac = c - a
            vNormal = np.cross(ab, ac)
            nNorm = norm(vNormal)
            self.aNormals[i] = vNormal / nNorm
            self.aArea[i] = 0.5 * nNorm
                                  
    def computeBoundaryMatrices(self, k, mu, orientation):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            p = self.aCenters[i]
            centerNormal = self.aNormals[i]
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(k, p, qa, qb, qc, i==j)
                elementM  = ComputeM(k, p, qa, qb, qc, i==j)
                elementMt = ComputeMt(k, p, centerNormal, qa, qb, qc, i==j)
                elementN  = ComputeN(k, p, centerNormal, qa, qb, qc, i==j)

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
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                elementM  = ComputeM(solution.k, p, qa, qb, qc, False)
                if orientation == 'interior':
                    sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult

    
class InteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        return super(InteriorHelmholtzSolver3D, self).solveBoundary('interior', k, boundaryCondition, boundaryIncidence, mu)

class ExteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        return super(ExteriorHelmholtzSolver3D, self).solveBoundary('exterior', k, boundaryCondition, boundaryIncidence, mu)

    
