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
from RayleighCavitySolver import *
from BoundaryData import *
from Geometry import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        

class RayleighCavitySolver3D(RayleighCavitySolver):
    def __init__(self, oMesh, c = 344.0, density = 1.205):
        super(RayleighCavitySolver3D, self).__init__(oMesh, c, density)
        self.aCenters = self.oGeometry.centers()
        
    def computeBoundaryMatrix(self, k, alpha, beta):
        m = self.nOpenElements
        n = self.totalNumberOfElements() - m
        M = np.zeros((2*(m+n), 2*(m+n)), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(m+n):
            p = self.aCenters[i]
            for j in range(m+n):
                qa, qb, qc = self.oGeometry.triangleVertices(j)

                elementM  = ComputeM(k, p, qa, qb, qc, i==j)
                elementL  = ComputeL(k, p, qa, qb, qc, i==j)

                M[i, j]         = -elementM
                M[i, j + m + n] =  elementL

            M[i, i] -= 0.5 # subtract half a "identity matrix" from the M-factor submatrix

        # Fill in the bottom half of the "big matrix".
        M[m+n:2*m+n, 0:m]               = np.eye(m, dtype=np.float32)
        M[2*m+n:2*(m+n), m:m+n]         = np.diag(alpha)
        M[m+n:2*m+n, m+n:2*m+n]         = 2.0 * M[0:m, m+n:2*m+n]
        M[2*m+n:2*(m+n), 2*m+n:2*(m+n)] = np.diag(beta)
        return M

    def solveInterior(self, solution, aSamples):
        aPhi = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aSamples.shape[0]):
            p = aSamples[i,:]
            sum = 0.0
            for j in range(solution.aPhi.size):
                qa, qb, qc = self.oGeometry.triangleVertices(j)
                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                elementM  = ComputeM(solution.k, p, qa, qb, qc, False)
                sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aPhi[i] = sum

        return SampleSolution(solution, aPhi)

    def solveExterior(self, solution, aSamples):
        aPhi = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aSamples.shape[0]):
            p = aSamples[i,:]
            sum = 0.0
            for j in range(self.nOpenElements):
                qa, qb, qc = self.oGeometry.triangleVertices(j)
                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                sum += -2.0 * elementL * solution.aV[j]
            aPhi[i] = sum

        return SampleSolution(solution, aPhi)
