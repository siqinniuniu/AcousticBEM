import numpy as np
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semiCircle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *

class HalfSpaceHorn(object):
    def __init__(self, name, maxElementSize = 0.01):
        self.name = name
        self.maxElementSize = maxElementSize

    def plotConeSection(self):
        self.chain() # make sure polygonal chain is available
        aX = np.empty(self.oGeometry.aEdge.shape[0] + 1, dtype=np.float32)
        aY = np.empty(self.oGeometry.aEdge.shape[0] + 1, dtype=np.float32)
        aX[0] = self.oGeometry.aVertex[self.oGeometry.aEdge[0, 0], 1]
        aY[0] = self.oGeometry.aVertex[self.oGeometry.aEdge[0, 0], 0]
        for i in range(self.oGeometry.aEdge.shape[0]):
            aX[i+1] = self.oGeometry.aVertex[self.oGeometry.aEdge[i, 1], 1]
            aY[i+1] = self.oGeometry.aVertex[self.oGeometry.aEdge[i, 1], 0]
    
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.plot(aX, aY)
        ax.set_aspect('equal', 'datalim')
        
    def makePolarPlot(self, aFrequency):
        aSamples, _, aAngle = semiCircle(5, 180, "yz")
        aSamples = aSamples[:, 1:3] # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())
        boundaryCondition = solver.neumannBoundaryCondition()
        driverPartition = np.asarray(solver.oGeometry.namedPartition['driver'])
        driverPartition = driverPartition - solver.oGeometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driverPartition[0], driverPartition[1]):
            boundaryCondition.f[i] = 1.0

        aK = frequencyToWavenumber(aFrequency)
        aMagnitudes = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.magnitude)
                                         (k, solver, boundaryCondition, aSamples) for k in aK)
        '''
        aMagnitudes = []
        for k in aK:
            aMagnitudes.append(HalfSpaceHorn.magnitude(k, solver, boundaryCondition, aSamples))
        '''
        polarPlot(self.name + " into Half-Space", aFrequency, aAngle, aMagnitudes)
        
    def makePolarMap(self):
        aSamples, _, aAngle = semiCircle(5, 180, "yz")
        aSamples = aSamples[:, 1:3] # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())

        nBoundaryElements = solver.numberOfElements()
        boundaryCondition = solver.neumannBoundaryCondition()
        driverPartition = np.asarray(solver.oGeometry.namedPartition['driver'])
        driverPartition = driverPartition - solver.oGeometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driverPartition[0], driverPartition[1]):
            boundaryCondition.f[i] = 1.0

        aFrequency = np.logspace(np.log10(self.frequencyRange[0]), np.log10(self.frequencyRange[1]), self.frequencySamples)
        aWavenumber = frequencyToWavenumber(aFrequency)

        aMagnitudes = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.magnitude)
                                         (k, solver, boundaryCondition, aSamples) for k in aWavenumber)
        aMagnitudes = np.asarray(aMagnitudes).transpose()

        plotPolarMap(self.name + " - Directivity Map",
                     aFrequency, aAngle, aMagnitudes, 3)

    def makeImpedancePlot(self):
        solver = RayleighCavitySolverRAD(self.chain())

        nBoundaryElements = solver.numberOfElements()
        boundaryCondition = solver.neumannBoundaryCondition()
        driverPartition = np.asarray(solver.oGeometry.namedPartition['driver'])
        driverPartition = driverPartition - solver.oGeometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driverPartition[0], driverPartition[1]):
            boundaryCondition.f[i] = 1.0
            
        aFrequency = np.logspace(np.log10(self.frequencyRange[0]), np.log10(self.frequencyRange[1]), self.frequencySamples)
        aK = frequencyToWavenumber(aFrequency)
        
        aZm = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.mechanicalImpedance)
                                 (k, solver, boundaryCondition) for k in aK)
        aZm = -1.0 * np.asarray(aZm) # -1 because of normals point outside of cavity

        plotMechanicalImpedance(self.name + " in Infinite Baffle - Mechanical Impedance", 
                        aFrequency, aZm)
        
    @staticmethod
    def magnitude(k, solver, boundaryCondition, aSamples):
        solution = solver.solveBoundary(k, boundaryCondition)
        sampleSolution = solver.solveExterior(solution, aSamples)
        return SoundMagnitude(soundPressure(k, sampleSolution.aPhi))

    @staticmethod
    def mechanicalImpedance(k, solver, boundaryCondition):
        solution = solver.solveBoundary(k, boundaryCondition)
        return solution.mechanicalImpedance('driver')
    
