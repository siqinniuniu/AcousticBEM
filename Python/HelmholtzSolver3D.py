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

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        

    
class HelmholtzSolver3D(HelmholtzSolver):
    def __init__(self, geometry, c=344.0, density=1.205):
        super(HelmholtzSolver3D, self).__init__(geometry, c, density)
        self.centers = self.geometry.centers()
        self.areas = self.geometry.areas()
        self.normals = self.geometry.normals()

    # noinspection PyPep8Naming
    def compute_boundary_matrices(self, k, mu, orientation):
        A = np.empty((self.len(), self.len()), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.len()):
            p = self.centers[i]
            normal = self.normals[i]
            for j in range(self.len()):
                qa, qb, qc = self.geometry.triangle_vertices(j)

                element_l = compute_l(k, p, qa, qb, qc, i == j)
                element_m = compute_m(k, p, qa, qb, qc, i == j)
                element_mt = compute_mt(k, p, normal, qa, qb, qc, i == j)
                element_n = compute_n(k, p, normal, qa, qb, qc, i == j)

                A[i, j] = element_l + mu * element_mt
                B[i, j] = element_m + mu * element_n

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i, i] -= 0.5 * mu
                B[i, i] += 0.5
            elif orientation == 'exterior':
                A[i, i] += 0.5 * mu
                B[i, i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)

        return A, B

    def compute_boundary_matrices_interior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'interior')
    
    def compute_boundary_matrices_exterior(self, k, mu):
        return self.compute_boundary_matrices(k, mu, 'exterior')
    
    def solve_samples(self, solution, incident_phis, samples, orientation):
        assert incident_phis.shape == samples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        results = np.empty(samples.shape[0], dtype=complex)

        for i in range(incident_phis.size):
            p = samples[i]
            sum = incident_phis[i]
            for j in range(solution.phis.size):
                qa, qb, qc = self.geometry.triangle_vertices(j)
                element_l = compute_l(solution.k, p, qa, qb, qc, False)
                element_m = compute_m(solution.k, p, qa, qb, qc, False)
                if orientation == 'interior':
                    sum += element_l * solution.velocities[j] - element_m * solution.phis[j]
                elif orientation == 'exterior':
                    sum -= element_l * solution.velocities[j] - element_m * solution.phis[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            results[i] = sum
        return results

    
class InteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(InteriorHelmholtzSolver3D,
                     self).solve_boundary('interior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)


class ExteriorHelmholtzSolver3D(HelmholtzSolver3D):
    def solve_boundary(self, k, boundary_condition, boundary_incidence, mu = None):
        return super(ExteriorHelmholtzSolver3D,
                     self).solve_boundary('exterior', k,
                                          boundary_condition, boundary_incidence,
                                          mu)
