import numpy as np
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semiCircle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *

class ConicalHorn(object):
    def __init__(self, name, throatRadius, mouthRadius, length, maxElementSize = 0.01):
        self.name = name
        self.throatRadius   = throatRadius
        self.mouthRadius    = mouthRadius
        self.length         = length
        self.maxElementSize = maxElementSize
        self.oGeometry      = None
        
        self.frequencySamples = 600
        self.frequencyRange   = (16, 16000)

        
    def chain(self, maxElementSize = None):
        """Get a polygonal chain representing the speaker membrane."""
        # check if cached polygonal chain is still good
        if not (maxElementSize is None or maxElementSize == self.maxElementSize):
            self.maxElementSize = maxElementSize
            self.oGeometry = None
        if self.oGeometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Cone Speaker")
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0,                        self.maxElementSize, 1)
            gmsh.model.geo.addPoint(0.0, self.mouthRadius, 0.0,           self.maxElementSize, 2)
            gmsh.model.geo.addPoint(0.0, self.throatRadius, -self.length, self.maxElementSize, 3)
            gmsh.model.geo.addPoint(0.0, 0.0, -self.length,               self.maxElementSize, 4)
        
            gmsh.model.geo.addLine(1, 2, 1)
            gmsh.model.addPhysicalGroup(1, [1], 1)
            gmsh.model.setPhysicalName(1, 1, "Interface")
            
            gmsh.model.geo.addLine(2, 3, 2)
            gmsh.model.addPhysicalGroup(1, [2], 2)
            gmsh.model.setPhysicalName(1, 2, "Horn")

            gmsh.model.geo.addLine(3, 4, 3)
            gmsh.model.addPhysicalGroup(1, [3], 3)
            gmsh.model.setPhysicalName(1, 2, "Driver")
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
            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(1, 1)
            assert len(elementTypes) == 1 and elementTypes[0] == 1 # only line segments
            assert len(elementTags) == 1
            assert len(nodeTags) == 1 and len(nodeTags[0]) == len(elementTags[0]) * 2
            aNodeTags.extend(nodeTags[0]) # extract line segment node tags
            nInterfaceElements = len(elementTags[0])
            
            # extract horn elements
            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(1, 2)
            assert len(elementTypes) == 1 and elementTypes[0] == 1 # only line segments
            assert len(elementTags) == 1
            assert len(nodeTags) == 1 and len(nodeTags[0]) == len(elementTags[0]) * 2
            aNodeTags.extend(nodeTags[0]) # extract line segment node tags
            nHornElements = len(elementTags[0])
            
            # extract driver elements
            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(1, 3)
            assert len(elementTypes) == 1 and elementTypes[0] == 1 # only line segments
            assert len(elementTags) == 1
            assert len(nodeTags) == 1 and len(nodeTags[0]) == len(elementTags[0]) * 2
            aNodeTags.extend(nodeTags[0]) # extract line segment node tags
            nDriverElements = len(elementTags[0])
            
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
        '''
        aMagnitudes = Parallel(n_jobs=4)(delayed(ConicalHorn.magnitude)
                                         (k, solver, boundaryCondition, aSamples) for k in aK)
        '''
        aMagnitudes = []
        for k in aK:
            aMagnitudes.append(ConicalHorn.magnitude(k, solver, boundaryCondition, aSamples))
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

        aMagnitudes = Parallel(n_jobs=4)(delayed(ConicalHorn.magnitude)
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
        
        aZm = Parallel(n_jobs=4)(delayed(ConicalHorn.mechanicalImpedance)
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
