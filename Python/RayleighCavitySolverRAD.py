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
    from HelmholtzIntegralsRAD_C import *
else:
    from HelmholtzIntegralsRAD import *        


class RayleighCavitySolverRAD(RayleighCavitySolver):
    def __init__(self, oChain, c = 344.0, density = 1.205):
        super(RayleighCavitySolverRAD, self).__init__(oChain, c, density)
        self.aCenters = self.oGeometry.centers()
        self.aLength = self.oGeometry.lengths()
        self.aNormals = self.oGeometry.normals()
        self.aArea = None

    def elementArea(self, namedPartition = None):
        return self.oGeometry.areas(namedPartition)
        
    def cavityNormals(self):
        cavityStart = self.oGeometry.namedPartition['cavity'][0]
        cavityEnd   = self.oGeometry.namedPartition['cavity'][1]
        return self.aNormals[cavityStart:cavityEnd, :]

    def computeBoundaryMatrix(self, k, alpha, beta):
        m = self.nOpenElements
        n = self.totalNumberOfElements() - m
        M = np.zeros((2*(m+n), 2*(m+n)), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(m+n):
            p = self.aCenters[i]
            for j in range(m+n):
                qa, qb = self.oGeometry.edgeVertices(j)

                elementM  = ComputeM(k, p, qa, qb, i==j)
                elementL  = ComputeL(k, p, qa, qb, i==j)

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
                qa, qb = self.oGeometry.edgeVertices(j)
                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)
                sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aPhi[i] = sum

        return SampleSolution(solution, aPhi)

    def solveExterior(self, solution, aSamples):
        aPhi = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aSamples.shape[0]):
            p = aSamples[i,:]
            sum = 0.0
            for j in range(self.nOpenElements):
                qa, qb = self.oGeometry.edgeVertices(j)
                elementL  = ComputeL(solution.k, p, qa, qb, False)
                sum += -2.0 * elementL * solution.aV[j]
            aPhi[i] = sum

        return SampleSolution(solution, aPhi)
