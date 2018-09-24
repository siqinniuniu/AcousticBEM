from BoundaryData import *

class Solver(object):
    def __init__(self, aVertex, aElement, c = 344.0, density = 1.205):
        self.aVertex  = aVertex
        self.aElement = aElement
        self.c        = c
        self.density  = density

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "  aVertex = " + repr(self.aVertex) + ", "
        result += "  aElement = " + repr(self.aElement) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  density = " + repr(self.density) + ")"
        return result

    def numberOfElements(self):
        return self.aElement.shape[0]

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

