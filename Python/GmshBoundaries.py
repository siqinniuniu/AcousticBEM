import gmsh
import numpy as np


def disk(r = 0.1, zOffset = 1.0, name = "Disk", maxElementSize = 0.01):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("disk")
    gmsh.model.geo.addPoint(0.0,  0.0, zOffset, maxElementSize, 1)
    gmsh.model.geo.addPoint(  r,  0.0, zOffset, maxElementSize, 2)
    gmsh.model.geo.addPoint(0.0,    r, zOffset, maxElementSize, 3)
    gmsh.model.geo.addPoint( -r,  0.0, zOffset, maxElementSize, 4)
    gmsh.model.geo.addPoint(0.0,   -r, zOffset, maxElementSize, 5)

    gmsh.model.geo.addCircleArc(2, 1, 3)
    gmsh.model.geo.addCircleArc(3, 1, 4)
    gmsh.model.geo.addCircleArc(4, 1, 5)
    gmsh.model.geo.addCircleArc(5, 1, 2)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, name)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.renumberNodes() # force continuous node tags
    gmsh.model.mesh.renumberElements()
    gmsh.write(name + ".msh")
    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes(2, 1, True)
    # reshape -> (len, 3)
    coord = np.asarray(coord, dtype=np.float32).reshape(len(nodeTags), 3)
    aVertex = np.empty_like(coord)
    minNodeTag = np.min(nodeTags)
    for i, t in enumerate(nodeTags):
        aVertex[t - minNodeTag, :] = coord[i, :]
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2, 1)
    assert len(elementTypes) == 1 and elementTypes[0] == 2
    assert len(elementTags) == 1
    assert len(nodeTags) == 1 and len(nodeTags[0] == len(elementTags) * 3)
    aTriangle = np.asarray(nodeTags[0], dtype=np.int32).reshape(len(elementTags[0]), 3)
    aTriangle = aTriangle - np.min(aTriangle) # shift node tags to [0,...,#tags-1] to match node array indices
    gmsh.finalize()

    return aVertex, aTriangle


def woofersSLA(maxElementSize = 0.01):
    r = 0.0635 # Effective membrane radius of Eminence Alpha 6a (computed from Sd as per spec sheet)
    zOffset = 0.0
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("disk")
    gmsh.model.geo.addPoint(0.0,  0.0, zOffset, maxElementSize, 1)
    gmsh.model.geo.addPoint(  r,  0.0, zOffset, maxElementSize, 2)
    gmsh.model.geo.addPoint(0.0,    r, zOffset, maxElementSize, 3)
    gmsh.model.geo.addPoint( -r,  0.0, zOffset, maxElementSize, 4)
    gmsh.model.geo.addPoint(0.0,   -r, zOffset, maxElementSize, 5)

    gmsh.model.geo.addCircleArc(2, 1, 3)
    gmsh.model.geo.addCircleArc(3, 1, 4)
    gmsh.model.geo.addCircleArc(4, 1, 5)
    gmsh.model.geo.addCircleArc(5, 1, 2)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.translate([(2, 1)], 0.0, r + 0.005, 0.0)
    w2 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w2, 0.0, 2.0 * r + 0.01, 0.0)
    w3 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w3, 0.0, -(2* r + 0.01), 0.0)
    w4 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w4, 0.0, -(4.0 * r + 0.02), 0.0)
    gmsh.model.addPhysicalGroup(2, [1, w2[0][1], w3[0][1], w4[0][1]], 1)
    gmsh.model.setPhysicalName(2, 1, "Woofers")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("WoofersSLA.msh")
    # Renumber triangle nodeTags array with indices into aVertex
    nodeTags, coord, parametricCoord = gmsh.model.mesh.getNodes(2, -1, True)
    aVertex = np.asarray(coord, dtype=np.float32).reshape(len(nodeTags), 3)
    nodeTagToIdx = dict()
    for i, t in enumerate(nodeTags):
        nodeTagToIdx[t] = i
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(2, -1)
    assert len(elementTypes) == 1 and elementTypes[0] == 2
    assert len(elementTags) == 1
    assert len(nodeTags) == 1 and len(nodeTags[0] == len(elementTags) * 3)
    nodeTags = nodeTags[0]
    aTempNodeTags = np.empty(len(nodeTags), dtype=np.int32)
    for i, t in enumerate(nodeTags):
        aTempNodeTags[i] = nodeTagToIdx[t]
    nodeTags = np.asarray(nodeTags, dtype=np.int32).reshape(len(elementTags[0]), 3)
    aTriangle = aTempNodeTags.reshape(len(elementTags[0]), 3)
    gmsh.finalize()

    return aVertex, aTriangle
