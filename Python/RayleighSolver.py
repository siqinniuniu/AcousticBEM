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
from BoundaryData import *
from Geometry import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        


class RayleighSolver(object):
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct RayleighSolver without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct RayleighSolver without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density

    def __repr__(self):
        result = "RayleighSolover("
        result += "  aVertex = " + repr(self.aVertex) + ", "
        result += "  aElement = " + repr(self.aElement) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  density = " + repr(self.density) + ")"
        return result

    @classmethod
    def SolveLinearEquation(cls, Ai, Bi, ci, alpha, beta, f):
        A = np.copy(Ai)
        B = np.copy(Bi)
        c = np.copy(ci)

        x = np.empty(c.size, dtype=np.complex)
        y = np.empty(c.size, dtype=np.complex)

        normA = np.linalg.norm(A, np.inf)
        normB = np.linalg.norm(B, np.inf)
        gamma = normB / normA
        swapXY = np.empty(c.size, dtype=bool)
        for i in range(c.size):
            if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
                swapXY[i] = False
            else:
                swapXY[i] = True

        for i in range(c.size):
            if swapXY[i]:
                for j in range(alpha.size):
                    c[j] += f[i] * B[j,i] / beta[i]
                    B[j, i] = -alpha[i] * B[j, i] / beta[i]
            else:
                for j in range(alpha.size):
                    c[j] -= f[i] * A[j, i] / alpha[i]
                    A[j, i] = -beta[i] * A[j, i] / alpha[i]

        A -= B
        y = np.linalg.solve(A, c)

        for i in range(c.size):
            if swapXY[i]:
                x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
            else:
                x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

        for i in range(c.size):
            if swapXY[i]:
                temp = x[i]
                x[i] = y[i]
                y[i] = temp

        return x, y

    def solveBoundary(self, k, boundaryCondition):
        assert boundaryCondition.f.size == self.aElement.shape[0]
        M = self.computeBoundaryMatrix(k)
        I = np.identity(self.aElement.shape[0], dtype=complex)
        c = np.zeros(self.aElement.shape[0], dtype=complex)
        phi, v = self.SolveLinearEquation(I, M, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)

        return BoundarySolution(self, k, phi, v)

    
class RayleighSolver3D(RayleighSolver):
    def __init__(self, *args, **kwargs):
        super(RayleighSolver3D, self).__init__(*args, **kwargs)
        self.aCenters = (self.aVertex[self.aElement[:, 0]] +\
                         self.aVertex[self.aElement[:, 1]] +\
                         self.aVertex[self.aElement[:, 2]]) / 3.0
        # area of the boundary alements
        self.aArea = np.empty(self.aElement.shape[0], dtype=np.float32)
        for i in range(self.aArea.size):
            a = self.aVertex[self.aElement[i, 0], :]
            b = self.aVertex[self.aElement[i, 1], :]
            c = self.aVertex[self.aElement[i, 2], :]
            self.aArea[i] = 0.5 * np.linalg.norm(np.cross(b-a, c-a))
                                  
    def computeBoundaryMatrix(self, k):
        # create NxN-Matrix where N is number of boundary elements
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)

        for i in range(self.aElement.shape[0]):
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(k, self.aCenters[i], qa, qb, qc, i==j)
                A[i, j] = -2.0 * elementL

        return A

    def solveSamples(self, solution, aSamples):
        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aSamples.shape[0]):
            p  = aSamples[i]
            sum = 0.0 + 0.0j
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                sum -= 2.0 * elementL * solution.aV[j]
            aResult[i] = sum
        return aResult

