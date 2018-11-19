from HalfSpaceHorn import *

class ConicalHorn(HalfSpaceHorn):
    def __init__(self, name, throatRadius, mouthRadius, length, maxElementSize = 0.01):
        super(ConicalHorn, self).__init__(name, maxElementSize)

        self.throatRadius   = throatRadius
        self.mouthRadius    = mouthRadius
        self.length         = length
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
    
