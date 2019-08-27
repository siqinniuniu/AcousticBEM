from HalfSpaceHorn import *

class OblateSpheroidalWaveguide(HalfSpaceHorn):
    def __init__(self, name, throat_radius, angle, length, flare_radius=None,
                 max_element_size=0.01):
        super(OblateSpheroidalWaveguide, self).__init__(name, max_element_size)
        self.throat_radius = throat_radius
        self.angle = np.pi * angle / 180.0  # half the included angle; convert to rad
        self.length = length
        if flare_radius is not None:
            assert flare_radius > 0.0, "flareRadius must be greater than 0."
        self.flare_radius = flare_radius
        self.geometry = None

        self.c = 344.0  # speed of sound in air
        self.frequency_samples = 600
        self.frequency_range = (16, 16000)

    def flare_transition_normal(self):
        derivative = self.oblate_radius_prime(self.length)
        normal = np.array([derivative, -1], dtype=np.float32)
        normal = normal / np.linalg.norm(normal)
        return normal

    def cutoff_frequency(self):
        return 0.2 * self.c / np.pi * np.sin(self.angle) / self.throat_radius
        
    def oblate_radius(self, z):
        return np.sqrt(self.throat_radius ** 2 + np.tan(self.angle) ** 2 * z ** 2)

    def oblate_radius_prime(self, z):
        outer = 0.5 / np.sqrt(self.throat_radius ** 2 + np.tan(self.angle) ** 2 * z ** 2)
        inner = 2.0 * np.tan(self.angle)**2 * z
        return outer * inner
    
    def oblate_samples_z(self, oblate_offset=0.0):
        oblateLength = self.length
        nSamples = np.ceil(oblateLength * np.sqrt(2.0) / self.max_element_size)
        samples = np.linspace(-self.length - oblate_offset, -oblate_offset, nSamples, endpoint = False)
        return np.flip(samples)

    def edges_of_physical_group(self, group_tag):
        tags = []
        # extract horn elements
        elements = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
        num_elements = 0
        for e in elements:
            elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(1, e)
            assert len(elementTypes) == 1 and elementTypes[0] == 1 # only line segments
            assert len(elementTags) == 1
            assert len(nodeTags) == 1 and len(nodeTags[0]) == len(elementTags[0]) * 2
            tags.extend(nodeTags[0]) # extract line segment node tags
            num_elements += len(elementTags[0])
        return tags, num_elements
            
    def chain(self, max_element_size=None):
        """Get a polygonal chain representing the horn geometry."""
        # check if cached polygonal chain is still good
        if not (max_element_size is None or max_element_size == self.max_element_size):
            self.max_element_size = max_element_size
            self.geometry = None
        if self.geometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Oblate Spheroidal Horn")
            point_tag = 1
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.max_element_size, 1)
            point_tag +=1
            if self.flare_radius is None:
                gmsh.model.geo.addPoint(0.0, self.oblate_radius(self.length),
                                        0.0, self.max_element_size, point_tag)
                oblate_start_tag = point_tag
                point_tag += 1
                segment_tag = 1
                gmsh.model.geo.addLine(1, 2, segment_tag)
                segment_tag += 1
                flare_offset = 0.0
            else:
                flare_vector = self.flare_radius * self.flare_transition_normal()
                flare_offset = self.flare_radius - flare_vector[0]
                flare_height = -flare_vector[1]
                # add the three points of the radial flare (topEdge, center, connecting point)
                flare_edge_y = self.oblate_radius(self.length) + flare_height
                p1 = point_tag
                gmsh.model.geo.addPoint(0.0, flare_edge_y,
                                        0.0, self.max_element_size, p1)
                point_tag += 1
                p2 = point_tag
                gmsh.model.geo.addPoint(0.0, flare_edge_y, -self.flare_radius,
                                        self.max_element_size, p2)
                point_tag += 1
                p3 = point_tag
                gmsh.model.geo.addPoint(0.0, self.oblate_radius(self.length),
                                        -flare_offset, self.max_element_size, p3)
                oblate_start_tag = point_tag
                point_tag += 1
                # add interface line
                segment_tag = 1
                gmsh.model.geo.addLine(1, 2, segment_tag)
                segment_tag += 1
                # add the flare
                gmsh.model.geo.addCircleArc(p1, p2, p3, segment_tag)
                segment_tag += 1
                
            oblate_offset = self.length + flare_offset
            for x in self.oblate_samples_z(flare_offset):
                gmsh.model.geo.addPoint(0.0, self.oblate_radius(x + oblate_offset), x, self.max_element_size, point_tag)
                point_tag += 1
            gmsh.model.geo.addPoint(0.0, 0.0, -oblate_offset, self.max_element_size, point_tag)

            for i in range(oblate_start_tag, point_tag):
                gmsh.model.geo.addLine(i, i+1, segment_tag)
                segment_tag += 1
            
            gmsh.model.addPhysicalGroup(1, [1], 1)
            # gmsh.model.setPhysicalName(1, 1, "Interface")
            gmsh.model.addPhysicalGroup(1, list(range(2, segment_tag - 1)), 2)
            # gmsh.model.setPhysicalName(1, 2, "Horn")
            gmsh.model.addPhysicalGroup(1, [segment_tag - 1], 3)
            # gmsh.model.setPhysicalName(1, 3, "Driver")

            gmsh.model.geo.synchronize()
       
            gmsh.model.mesh.generate(1)
            node_tags, coord, parametric_coord = gmsh.model.mesh.getNodes(1, -1, True)
            aVertex = np.asarray(coord, dtype=np.float32).reshape(len(node_tags), 3)
            # build reordering dictionary
            node_tag_to_idx = dict()
            for i, t in enumerate(node_tags):
                node_tag_to_idx[t] = i
            a_node_tags = []

            # extract "open elements" or "Interface" first
            node_tags, interface_elements = self.edges_of_physical_group(1)
            a_node_tags.extend(node_tags)
            # extract horn elements
            node_tags, horn_elements = self.edges_of_physical_group(2)
            a_node_tags.extend(node_tags)
            # extract driver elements
            node_tags, driver_elements = self.edges_of_physical_group(3)
            a_node_tags.extend(node_tags)

            # relabel node tags with index into vertex array
            temp_node_tags = np.empty(len(a_node_tags), dtype=np.int32)
            for i, t in enumerate(a_node_tags):
                temp_node_tags[i] = node_tag_to_idx[t]
            segments = temp_node_tags.reshape(len(a_node_tags) // 2, 2)
            gmsh.finalize()
            
            self.geometry = Chain(aVertex.shape[0], segments.shape[0])
            self.geometry.vertices = aVertex[:, 1:3]
            self.geometry.edges = segments
            self.geometry.named_partition['interface'] = (0, interface_elements)
            self.geometry.named_partition['horn'] = (interface_elements, interface_elements + horn_elements)
            self.geometry.named_partition['driver'] = (interface_elements + horn_elements,
                                                       interface_elements + horn_elements + driver_elements)
            
        return self.geometry
