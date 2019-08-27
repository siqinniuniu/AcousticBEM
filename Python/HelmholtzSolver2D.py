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
    def __init__(self, chain, c=344.0, density=1.205):
        super(HelmholtzSolver2D, self).__init__(chain, c, density)
        self.centers = self.geometry.centers()
        # lenght of the boundary elements (for the 3d shapes this is replaced by aArea
        self.lengths = self.geometry.lengths()

    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        centers = self.geometry.centers()
        normals = -self.geometry.normals()
        
        for i in range(self.len()):
            center = centers[i]
            normal = normals[i]
            for j in range(self.len()):
                qa, qb = self.geometry.edge_vertices(j)

                elementL  = compute_l(k, center, qa, qb, i == j)
                elementM  = compute_m(k, center, qa, qb, i == j)
                elementMt = compute_mt(k, center, normal, qa, qb, i == j)
                elementN  = compute_n(k, center, normal, qa, qb, i == j)
                
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

    def compute_boundary_matrices_interior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'interior')
    
    def compute_boundary_matrices_exterior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'exterior')

    def solve_samples(self, solution, aIncidentPhi, aSamples, orientation):
        assert aIncidentPhi.shape == aSamples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aIncidentPhi.size):
            p  = aSamples[i]
            sum = aIncidentPhi[i]
            for j in range(solution.phis.size):
                qa, qb = self.geometry.edge_vertices(j)

                elementL  = compute_l(solution.k, p, qa, qb, False)
                elementM  = compute_m(solution.k, p, qa, qb, False)
                if orientation == 'interior':
                    sum += elementL * solution.velocities[j] - elementM * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.velocities[j] - elementM * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult


class InteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(InteriorHelmholtzSolver2D, self).solve_boundary('interior', k, boundary_condition, boundary_incidence, mu)


class ExteriorHelmholtzSolver2D(HelmholtzSolver2D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(ExteriorHelmholtzSolver2D, self).solve_boundary('exterior', k, boundary_condition, boundary_incidence, mu)
