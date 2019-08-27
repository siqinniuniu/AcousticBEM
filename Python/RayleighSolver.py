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

    def solve_boundary(self, k, boundary_condition):
        assert boundary_condition.f.size == self.len()
        M = self.compute_boundary_matrix(k, boundary_condition.alpha, boundary_condition.beta)
        n = self.len()
        b = np.zeros(2 * n, dtype=complex)
        b[n: 2*n] = boundary_condition.f
        x = np.linalg.solve(M, b)
        
        return RayleighBoundarySolution(self, boundary_condition, k,
                                        x[0:self.len()],
                                        x[self.len():2 * self.len()])

    
class RayleighSolver3D(RayleighSolver):
    def __init__(self, mesh, c=344.0, density=1.205):
        super(RayleighSolver3D, self).__init__(mesh, c, density)
        self.centers = self.geometry.centers()
        self.areas = None
                                  
    def element_area(self, named_partition=None):
        return self.geometry.areas(named_partition)

    def compute_boundary_matrix(self, k, alpha, beta):
        n = self.len()
        M = np.zeros((2*n, 2*n), dtype=np.complex64)

        # Compute the top half of the "big matrix".
        for i in range(n):
            p = self.centers[i]
            for j in range(n):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = compute_l(k, p, qa, qb, qc, i == j)
                M[i, j + n] = 2 * element_l

        # Fill in the bottom half of the "big matrix".
        M[0:n,       0:n] = np.eye(n, dtype=np.float32)
        M[n: 2*n,    0:n] = np.diag(alpha)
        M[n: 2*n, n: 2*n] = np.diag(beta)
        
        return M

    def solve_samples(self, solution, samples):
        results = np.empty(samples.shape[0], dtype=complex)

        for i in range(samples.shape[0]):
            p = samples[i]
            sum = 0.0 + 0.0j
            for j in range(solution.phis.size):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = compute_l(solution.k, p, qa, qb, qc, False)
                sum -= 2.0 * element_l * solution.velocities[j]
            results[i] = sum
        return results
