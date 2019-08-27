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
    def __init__(self, geometry, c=344.0, density=1.205):
        super(RayleighCavitySolver, self).__init__(geometry, c, density)
        self.open_elements = self.geometry.named_partition['interface'][1]

    def number_of_interface_elements(self):
        return self.open_elements
    
    def len(self):
        """The number of elements forming the cavity."""
        return self.total_number_of_elements() - self.open_elements

    def total_number_of_elements(self):
        return super(RayleighCavitySolver, self).len()

    def solve_boundary(self, k, boundary_condition):
        M = self.compute_boundary_matrix(k,
                                         boundary_condition.alpha,
                                         boundary_condition.beta)
        number_of_elements = self.total_number_of_elements()
        b = np.zeros(2*number_of_elements, dtype=np.complex64)
        b[number_of_elements + self.open_elements: 2 * number_of_elements] = boundary_condition.f
        x = np.linalg.solve(M, b)
        
        return RayleighCavityBoundarySolution(self, boundary_condition, k,
                                              x[0:number_of_elements],
                                              x[number_of_elements:2*number_of_elements])
