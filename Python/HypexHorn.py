from HalfSpaceHorn import *

class HypexHorn(HalfSpaceHorn):
    def __init__(self, name, throatRadius, cutoffFrequency, length, shapeFactor = 1.0, maxElementSize = 0.01):
        super(HypexHorn, self).__init__(name, maxElementSize)

        self.throatRadius   = throatRadius
        self.cutoffK        = frequencyToWavenumber(cutoffFrequency)
        self.length         = length
        self.shapeFactor    = shapeFactor
        self.oGeometry      = None
        
        self.frequencySamples = 600
        self.frequencyRange   = (16, 16000)

    def radius(self, z):
        z += self.length # shift z to the left by length, horn start at z = -length
        K0 = self.cutoffK
        T  = self.shapeFactor
        return self.throatRadius * (np.cosh(z*K0) + T * np.sinh(z*K0))

    def samplesZ(self):
        nSamples = np.ceil(self.length * np.sqrt(2.0) / self.maxElementSize)
        return np.linspace(0.0, -self.length, nSamples)

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
            gmsh.model.add("Cone Speaker")
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.maxElementSize, 1)
            pointTag = 2
            for x in self.samplesZ():
                gmsh.model.geo.addPoint(0.0, self.radius(x), x, self.maxElementSize, pointTag)
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
    
