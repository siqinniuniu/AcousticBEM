import numpy as np
from scipy.optimize import brentq
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semiCircle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *

class CompoundExponentialHorn(object):
    def __init__(self, name, throatRadius, cutoffFrequency, angle, length, maxElementSize = 0.01):
        self.name = name
        self.throatRadius   = throatRadius
        self.cutoffK        = frequencyToWavenumber(cutoffFrequency)
        self.angle          = np.pi * angle / 180.0 # to radians
        self.length         = length
        self.shapeFactor    = 1.0 # for now basic exponential geometry
        self.maxElementSize = maxElementSize
        self.oGeometry      = None
        
        self.frequencySamples = 600
        self.frequencyRange   = (16, 16000)

    def firstTransitionRadius(self):
        return 0.95 * np.sin(self.angle) / self.cutoffK

    def firstTransitionZ(self):
        root = brentq(lambda z: self.exponentialRadius(z) - self.firstTransitionRadius(),
                      -self.length, 0.0)
        return root

    def secondTransitionZ(self):
        return -self.length / 3.0

    def secondTransitionRadius(self):
        derivative = np.tan(self.angle)
        length = self.secondTransitionZ() - self.firstTransitionZ()
        offset = length * derivative
        return self.firstTransitionRadius() + offset

    def mouthRadius(self):
        derivative = np.tan(2.0 * self.angle)
        offset = derivative * self.length / 3.0
        return self.secondTransitionRadius() + offset
    
    def exponentialRadius(self, z):
        z += self.length # shift z to the left by length, horn start at z = -length
        K0 = self.cutoffK
        T  = self.shapeFactor
        return self.throatRadius * (np.cosh(z*K0) + T * np.sinh(z*K0))

    def exponentialSamplesZ(self):
        nSamples = np.ceil((self.length - self.firstTransitionZ()) * np.sqrt(2.0) / self.maxElementSize)
        return np.linspace(self.firstTransitionZ(), -self.length, nSamples)

    def edgesOfPhysicalGroup(self, groupTag):
        aTags = []
        # extract horn elements
        elements = gmsh.model.getEntitiesForPhysicalGroup(1, groupTag)
        nElements = 0
        for e in elements:
            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(1, e)
            assert len(elementTypes) == 1 and elementTypes[0] == 1 # only line segments
            assert len(elementTags) == 1
            assert len(nodeTags) == 1 and len(nodeTags[0]) == len(elementTags[0]) * 2
            aTags.extend(nodeTags[0]) # extract line segment node tags
            nElements += len(elementTags[0])
        return aTags, nElements
            
    def chain(self, maxElementSize = None):
        """Get a polygonal chain representing the horn geometry."""
        # check if cached polygonal chain is still good
        if not (maxElementSize is None or maxElementSize == self.maxElementSize):
            self.maxElementSize = maxElementSize
            self.oGeometry = None
        if self.oGeometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Compound Conical Exponential Horn")

            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.maxElementSize, 1)
            gmsh.model.geo.addPoint(0.0, self.mouthRadius(), 0.0, self.maxElementSize, 2)
            gmsh.model.geo.addPoint(0.0, self.secondTransitionRadius(), self.secondTransitionZ(), self.maxElementSize, 3)
        
            pointTag = 4
            for x in self.exponentialSamplesZ():
                gmsh.model.geo.addPoint(0.0, self.exponentialRadius(x), x, self.maxElementSize, pointTag)
                pointTag += 1
            gmsh.model.geo.addPoint(0.0, 0.0, -self.length, self.maxElementSize, pointTag)

            for i in range(1, pointTag):
                gmsh.model.geo.addLine(i, i+1, i)
            
            gmsh.model.addPhysicalGroup(1, [1], 1)
            # gmsh.model.setPhysicalName(1, 1, "Interface")
            gmsh.model.addPhysicalGroup(1, list(range(2, pointTag-1)), 2)
            # gmsh.model.setPhysicalName(1, 2, "Horn")
            gmsh.model.addPhysicalGroup(1, [pointTag-1], 3)
            # gmsh.model.setPhysicalName(1, 3, "Driver")

            gmsh.model.geo.synchronize()
       
            gmsh.model.mesh.generate(1)
            nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes(1, -1, True)
            aVertex = np.asarray(coord, dtype=np.float32).reshape(len(nodeTags), 3)
            # build reordering dictionary
            nodeTagToIdx = dict()
            for i, t in enumerate(nodeTags):
                nodeTagToIdx[t] = i
            aNodeTags = []

            # extract "open elements" or "Interface" first
            nodeTags, nInterfaceElements = self.edgesOfPhysicalGroup(1)
            aNodeTags.extend(nodeTags)
            # extract horn elements
            nodeTags, nHornElements = self.edgesOfPhysicalGroup(2)
            aNodeTags.extend(nodeTags)
            # extract driver elements
            nodeTags, nDriverElements = self.edgesOfPhysicalGroup(3)
            aNodeTags.extend(nodeTags)

            # relabel node tags with index into vertex array
            aTempNodeTags = np.empty(len(aNodeTags), dtype=np.int32)
            for i, t in enumerate(aNodeTags):
                aTempNodeTags[i] = nodeTagToIdx[t]
            aSegment = aTempNodeTags.reshape(len(aNodeTags) // 2, 2)
            gmsh.finalize()
            
            self.oGeometry = Chain(aVertex.shape[0], aSegment.shape[0])
            self.oGeometry.aVertex = aVertex[:, 1:3]
            self.oGeometry.aEdge = aSegment
            self.oGeometry.namedPartition['interface'] = (0, nInterfaceElements)
            self.oGeometry.namedPartition['horn'] = (nInterfaceElements, nInterfaceElements + nHornElements)
            self.oGeometry.namedPartition['driver'] = (nInterfaceElements + nHornElements,
                                                       nInterfaceElements + nHornElements + nDriverElements)
            
        return self.oGeometry
    
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
        aMagnitudes = Parallel(n_jobs=4)(delayed(CompoundExponentialHorn.magnitude)
                                         (k, solver, boundaryCondition, aSamples) for k in aK)
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

        aMagnitudes = Parallel(n_jobs=4)(delayed(CompoundExponentialHorn.magnitude)
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
        
        aZm = Parallel(n_jobs=4)(delayed(CompoundExponentialHorn.mechanicalImpedance)
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
