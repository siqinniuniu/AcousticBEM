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
from BoundaryData import *
from Geometry import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        


class RayleighSolver(Solver):

    def solveBoundary(self, k, boundaryCondition):
        assert boundaryCondition.f.size == self.numberOfElements()
        M = self.computeBoundaryMatrix(k, boundaryCondition.alpha, boundaryCondition.beta)
        n = self.numberOfElements()
        b = np.zeros(2 * n, dtype=complex)
        b[n: 2*n] = boundaryCondition.f
        x = np.linalg.solve(M, b)
        
        return RayleighBoundarySolution(self, boundaryCondition, k, 
                                        x[0:self.numberOfElements()],
                                        x[self.numberOfElements():2*self.numberOfElements()])

    
class RayleighSolver3D(RayleighSolver):
    def __init__(self,  oMesh, c = 344.0, density = 1.205):
        super(RayleighSolver3D, self).__init__(oMesh, c, density)
        self.aCenters = self.oGeometry.centers()
        self.aArea = None
                                  
    def elementArea(self, namedPartition = None):
        return self.oGeometry.areas(namedPartition)

    def computeBoundaryMatrix(self, k, alpha, beta):
        n = self.numberOfElements()
        M = np.zeros((2*n, 2*n), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(n):
            p = self.aCenters[i]
            for j in range(n):
                qa, qb, qc = self.oGeometry.triangleVertices(j)
                elementL  = ComputeL(k, p, qa, qb, qc, i==j)
                M[i, j + n] =  2 * elementL

        # Fill in the bottom half of the "big matrix".
        M[0:n, 0:n]       = np.eye(n, dtype=np.float32)
        M[n: 2*n, 0:n]    = np.diag(alpha)
        M[n: 2*n, n: 2*n] = np.diag(beta)
        
        return M

    def solveSamples(self, solution, aSamples):
        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aSamples.shape[0]):
            p  = aSamples[i]
            sum = 0.0 + 0.0j
            for j in range(solution.aPhi.size):
                qa, qb, qc = self.oGeometry.triangleVertices(j)
                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                sum -= 2.0 * elementL * solution.aV[j]
            aResult[i] = sum
        return aResult

