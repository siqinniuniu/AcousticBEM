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
from BoundaryData import *

class Solver(object):
    def __init__(self, oGeometry, c = 344.0, density = 1.205):
        self.oGeometry = oGeometry
        self.c        = c
        self.density  = density

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "  oGeometry = " + repr(self.oGeometry) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  density = " + repr(self.density) + ")"
        return result

    def numberOfElements(self):
        return self.oGeometry.numberOfElements()

    def dirichletBoundaryCondition(self):
        """Returns a boundary contidition with alpha the 1-function and f and beta 0-functions."""
        boundaryCondition = BoundaryCondition(self.numberOfElements())
        boundaryCondition.alpha.fill(1.0)
        boundaryCondition.beta.fill(0.0)
        boundaryCondition.f.fill(1.0)
        return boundaryCondition

    def neumannBoundaryCondition(self):
        """Returns a boundary contidition with f and alpha 0-functions and beta the 1-function."""
        boundaryCondition = BoundaryCondition(self.numberOfElements())
        boundaryCondition.alpha.fill(0.0)
        boundaryCondition.beta.fill(1.0)
        boundaryCondition.f.fill(0.0)
        return boundaryCondition

