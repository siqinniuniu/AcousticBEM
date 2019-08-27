import gmsh
import numpy as np
from Mesh import Mesh


def get_triangle_mesh(file_name=None):
    gmsh.model.mesh.generate(2)
    if file_name is not None:
        gmsh.write(file_name + ".msh")
    # Renumber triangle node_tags array with indices into aVertex
    node_tags, coord, parametric_coord = gmsh.model.mesh.getNodes(2, -1, True)
    vertices = np.asarray(coord, dtype=np.float32).reshape(len(node_tags), 3)
    node_tag_to_idx = dict()
    for i, t in enumerate(node_tags):
        node_tag_to_idx[t] = i
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2, -1)
    assert len(element_types) == 1 and element_types[0] == 2
    assert len(element_tags) == 1
    assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 3
    node_tags = node_tags[0]
    temp_node_tags = np.empty(len(node_tags), dtype=np.int32)
    for i, t in enumerate(node_tags):
        temp_node_tags[i] = node_tag_to_idx[t]
    triangles = temp_node_tags.reshape(len(element_tags[0]), 3)
    mesh = Mesh(0, 0)
    mesh.vertices = vertices
    mesh.triangles = triangles
    return mesh


def disk(r=0.1, z_offset=1.0, name="Disk", max_element_size=0.01):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("disk")
    gmsh.model.geo.addPoint(0.0, 0.0, z_offset, max_element_size, 1)
    gmsh.model.geo.addPoint(r, 0.0, z_offset, max_element_size, 2)
    gmsh.model.geo.addPoint(0.0, r, z_offset, max_element_size, 3)
    gmsh.model.geo.addPoint(-r, 0.0, z_offset, max_element_size, 4)
    gmsh.model.geo.addPoint(0.0, -r, z_offset, max_element_size, 5)

    gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    gmsh.model.geo.addCircleArc(3, 1, 4, 2)
    gmsh.model.geo.addCircleArc(4, 1, 5, 3)
    gmsh.model.geo.addCircleArc(5, 1, 2, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, name)
    gmsh.model.geo.synchronize()

    mesh = get_triangle_mesh(name)
    gmsh.finalize()

    return mesh


def woofers_sla(max_element_size=0.01):
    # Effective membrane radius of Eminence Alpha 6a (computed from Sd as per
    # spec sheet).
    r = 0.0635
    z_offset = 0.0
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("disk")
    gmsh.model.geo.addPoint(0.0, 0.0, z_offset, max_element_size, 1)
    gmsh.model.geo.addPoint(r,   0.0, z_offset, max_element_size, 2)
    gmsh.model.geo.addPoint(0.0,   r, z_offset, max_element_size, 3)
    gmsh.model.geo.addPoint(-r,  0.0, z_offset, max_element_size, 4)
    gmsh.model.geo.addPoint(0.0,  -r, z_offset, max_element_size, 5)

    gmsh.model.geo.addCircleArc(2, 1, 3, 1)
    gmsh.model.geo.addCircleArc(3, 1, 4, 2)
    gmsh.model.geo.addCircleArc(4, 1, 5, 3)
    gmsh.model.geo.addCircleArc(5, 1, 2, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.translate([(2, 1)], 0.0, r + 0.005, 0.0)
    w2 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w2, 0.0, 2.0 * r + 0.01, 0.0)
    w3 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w3, 0.0, -(2.0 * r + 0.01), 0.0)
    w4 = gmsh.model.geo.copy([(2, 1)])
    gmsh.model.geo.translate(w4, 0.0, -(4.0 * r + 0.02), 0.0)
    gmsh.model.addPhysicalGroup(2, [1, w2[0][1], w3[0][1], w4[0][1]], 1)
    gmsh.model.setPhysicalName(2, 1, "Woofers")
    gmsh.model.geo.synchronize()

    mesh = get_triangle_mesh("WoofersSLA")
    gmsh.finalize()

    return mesh


def speaker_box(width, height, depth, radius, max_element_size=0.01):
    center = (width / 2.0, height / 2.0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # --------------------------------------------------------------------------
    # points
    # --------------------------------------------------------------------------
    # speaker box
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, max_element_size, 1)
    gmsh.model.geo.addPoint(0.0, height, 0.0, max_element_size, 2)
    gmsh.model.geo.addPoint(width, height, 0.0, max_element_size, 3)
    gmsh.model.geo.addPoint(width, 0.0, 0.0, max_element_size, 4)

    gmsh.model.geo.addPoint(0.0, 0.0, depth, max_element_size, 5)
    gmsh.model.geo.addPoint(0.0, height, depth, max_element_size, 6)
    gmsh.model.geo.addPoint(width, height, depth, max_element_size, 7)
    gmsh.model.geo.addPoint(width, 0.0, depth, max_element_size, 8)

    # speaker membrane
    gmsh.model.geo.addPoint(*center, 0.0, max_element_size, 9)
    gmsh.model.geo.addPoint(center[0] + radius, center[1], 0.0,
                            max_element_size, 10)
    gmsh.model.geo.addPoint(center[0], center[1] + radius, 0.0,
                            max_element_size, 11)
    gmsh.model.geo.addPoint(center[0] - radius, center[1], 0.0,
                            max_element_size, 12)
    gmsh.model.geo.addPoint(center[0], center[1] - radius, 0.0,
                            max_element_size, 13)

    # --------------------------------------------------------------------------
    # lines
    # --------------------------------------------------------------------------
    # box
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addLine(1, 5, 5)
    gmsh.model.geo.addLine(6, 5, 6)
    gmsh.model.geo.addLine(6, 2, 7)
    gmsh.model.geo.addLine(3, 7, 8)
    gmsh.model.geo.addLine(8, 7, 9)
    gmsh.model.geo.addLine(8, 4, 10)
    gmsh.model.geo.addLine(5, 8, 11)
    gmsh.model.geo.addLine(7, 6, 12)

    # speaker hole loop
    gmsh.model.geo.addCircleArc(10, 9, 11, 13)
    gmsh.model.geo.addCircleArc(11, 9, 12, 14)
    gmsh.model.geo.addCircleArc(12, 9, 13, 15)
    gmsh.model.geo.addCircleArc(13, 9, 10, 16)

    # --------------------------------------------------------------------------
    # loops
    # --------------------------------------------------------------------------
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)  # baffle
    gmsh.model.geo.addCurveLoop([13, 14, 15, 16], 2)  # speaker hole
    gmsh.model.geo.addCurveLoop([-16, -15, -14, -13], 3)  # speaker/piston
    gmsh.model.geo.addCurveLoop([-1, 5, -6, 7], 4)  # left
    gmsh.model.geo.addCurveLoop([-3, 8, -9, 10], 5)  # right
    gmsh.model.geo.addCurveLoop([6, 11, 9, 12], 6)  # back
    gmsh.model.geo.addCurveLoop([-2, -7, -12, -8], 7)  # top
    gmsh.model.geo.addCurveLoop([-4, -10, -11, -5], 8)  # bottom

    # --------------------------------------------------------------------------
    # plane surfaces
    # --------------------------------------------------------------------------
    gmsh.model.geo.addPlaneSurface([1, 2], 1)  # baffle
    gmsh.model.geo.addPlaneSurface([4], 2)  # left
    gmsh.model.geo.addPlaneSurface([5], 3)  # right
    gmsh.model.geo.addPlaneSurface([6], 4)  # back
    gmsh.model.geo.addPlaneSurface([7], 5)  # top
    gmsh.model.geo.addPlaneSurface([8], 6)  # bottom
    gmsh.model.addPhysicalGroup(2, [1, 2, 3, 4, 5, 6], 1)
    gmsh.model.setPhysicalName(2, 1, "Box")
    gmsh.model.geo.addPlaneSurface([3], 7)
    gmsh.model.addPhysicalGroup(2, [7], 2)
    gmsh.model.setPhysicalName(2, 2, "Membrane")
    gmsh.model.geo.synchronize()

    mesh = get_triangle_mesh("SpeakerBox")
    gmsh.finalize()

    return mesh
