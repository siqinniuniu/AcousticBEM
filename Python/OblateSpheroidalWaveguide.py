import numpy as np
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semiCircle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *

class OblateSpheroidalWaveguide(object):
    def __init__(self, name, throatRadius, angle, length, flareRadius = None, maxElementSize = 0.01):
        self.name = name
        self.throatRadius   = throatRadius
        self.angle          = np.pi * angle / 180.0 # half the included angle; convert to rad
        self.length         = length
        if not flareRadius is None:
            assert flareRadius > 0.0, "flareRadius must be greater than 0."
        self.flareRadius    = flareRadius
        self.maxElementSize = maxElementSize
        self.oGeometry      = None

        self.c                = 344.0 # speed of sound in air
        self.frequencySamples = 600
        self.frequencyRange   = (16, 16000)

    def flareTransitionNormal(self):
        derivative = self.oblateRadiusPrime(self.length)
        normal = np.array([derivative, -1], dtype=np.float32)
        normal = normal / np.linalg.norm(normal)
        return normal

    def cutoffFrequency(self):
        return 0.2 * self.c / np.pi * np.sin(self.angle) / self.throatRadius    
        
    def oblateRadius(self, z):
        return np.sqrt(self.throatRadius**2 + np.tan(self.angle)**2 * z**2)

    def oblateRadiusPrime(self, z):
        outer = 0.5 / np.sqrt(self.throatRadius**2 + np.tan(self.angle)**2 * z**2)
        inner = 2.0 * np.tan(self.angle)**2 * z
        return outer * inner
    
    def oblateSamplesZ(self, oblateOffset = 0.0):
        oblateLength = self.length
        nSamples = np.ceil(oblateLength * np.sqrt(2.0) / self.maxElementSize)
        samples = np.linspace(-self.length - oblateOffset, -oblateOffset, nSamples, endpoint = False)
        return np.flip(samples)

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
            gmsh.model.add("Oblate Spheroidal Horn")
            pointTag = 1
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.maxElementSize, 1)
            pointTag +=1
            if self.flareRadius is None:
                gmsh.model.geo.addPoint(0.0, self.oblateRadius(self.length), 0.0, self.maxElementSize, pointTag)
                oblateStartTag = pointTag
                pointTag += 1
                segmentTag = 1
                gmsh.model.geo.addLine(1, 2, segmentTag)
                segmentTag += 1
                flareOffset = 0.0
            else:
                flareVector = self.flareRadius * self.flareTransitionNormal()
                flareOffset = self.flareRadius - flareVector[0]
                flareHeight = -flareVector[1]
                # add the three points of the radial flare (topEdge, center, connecting point)
                flareEdgeY = self.oblateRadius(self.length) + flareHeight
                p1 = pointTag
                gmsh.model.geo.addPoint(0.0, flareEdgeY, 0.0, self.maxElementSize, p1)
                pointTag += 1
                p2 = pointTag
                gmsh.model.geo.addPoint(0.0, flareEdgeY, -self.flareRadius, self.maxElementSize, p2)
                pointTag += 1
                p3 = pointTag
                gmsh.model.geo.addPoint(0.0, self.oblateRadius(self.length), -flareOffset, self.maxElementSize, p3)
                oblateStartTag = pointTag
                pointTag += 1
                # add interface line
                segmentTag = 1
                gmsh.model.geo.addLine(1, 2, segmentTag)
                segmentTag += 1
                # add the flare
                gmsh.model.geo.addCircleArc(p1, p2, p3, segmentTag)
                segmentTag += 1
                
            oblateOffset = self.length + flareOffset
            for x in self.oblateSamplesZ(flareOffset):
                gmsh.model.geo.addPoint(0.0, self.oblateRadius(x + oblateOffset), x, self.maxElementSize, pointTag)
                pointTag += 1
            gmsh.model.geo.addPoint(0.0, 0.0, -oblateOffset, self.maxElementSize, pointTag)

            for i in range(oblateStartTag, pointTag):
                gmsh.model.geo.addLine(i, i+1, segmentTag)
                segmentTag += 1
            
            gmsh.model.addPhysicalGroup(1, [1], 1)
            # gmsh.model.setPhysicalName(1, 1, "Interface")
            gmsh.model.addPhysicalGroup(1, list(range(2, segmentTag - 1)), 2)
            # gmsh.model.setPhysicalName(1, 2, "Horn")
            gmsh.model.addPhysicalGroup(1, [segmentTag - 1], 3)
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
        aMagnitudes = Parallel(n_jobs=4)(delayed(OblateSpheroidalWaveguide.magnitude)
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

        aMagnitudes = Parallel(n_jobs=4)(delayed(OblateSpheroidalWaveguide.magnitude)
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
        
        aZm = Parallel(n_jobs=4)(delayed(OblateSpheroidalWaveguide.mechanicalImpedance)
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
