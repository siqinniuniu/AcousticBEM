from HalfSpaceHorn import *


class TractrixHorn(HalfSpaceHorn):
    def __init__(self, name, throat_radius, cutoff_frequency,
                 max_element_size=0.01):
        super(TractrixHorn, self).__init__(name, max_element_size)

        self.throat_radius = throat_radius
        self.mouth_radius = 1.0 / frequency_to_wavenumber(cutoff_frequency)
        self.geometry = None
        
        self.frequency_samples = 600
        self.frequency_range   = (16, 16000)

    def z(self, r):
        mr = self.mouth_radius
        root_term = np.sqrt(mr ** 2 - r ** 2)
        log_term = mr * np.log((mr + root_term) / r)
        return root_term - log_term

    def samples_y(self):
        radius_spread = self.mouth_radius - self.throat_radius
        samples = np.ceil(radius_spread * 2.0 / self.max_element_size)
        return np.linspace(self.mouth_radius, self.throat_radius, samples)

    def edges_of_physical_group(self, group_tag):
        tags = []
        # extract horn elements
        elements = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
        num_elements = 0
        for e in elements:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, e)
            assert len(element_types) == 1 and element_types[0] == 1  # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            tags.extend(node_tags[0])  # extract line segment node tags
            num_elements += len(element_tags[0])
        return tags, num_elements
            
    def chain(self, max_element_size = None):
        """Get a polygonal chain representing the horn geometry."""
        # check if cached polygonal chain is still good
        if not (max_element_size is None or max_element_size == self.max_element_size):
            self.max_element_size = max_element_size
            self.geometry = None
        if self.geometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Cone Speaker")
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.max_element_size, 1)
            point_tag = 2
            samples_y = self.samples_y()
            for r in samples_y:
                gmsh.model.geo.addPoint(0.0, r, self.z(r), self.max_element_size, point_tag)
                point_tag += 1
            print("samples_y[-1] = {}".format(samples_y[-1]))
            gmsh.model.geo.addPoint(0.0, 0.0, self.z(samples_y[-1]), self.max_element_size, point_tag)

            for i in range(1, point_tag):
                gmsh.model.geo.addLine(i, i+1, i)
            
            gmsh.model.addPhysicalGroup(1, [1], 1)
            # gmsh.model.setPhysicalName(1, 1, "Interface")
            gmsh.model.addPhysicalGroup(1, list(range(2, point_tag-1)), 2)
            # gmsh.model.setPhysicalName(1, 2, "Horn")
            gmsh.model.addPhysicalGroup(1, [point_tag-1], 3)
            # gmsh.model.setPhysicalName(1, 3, "Driver")

            gmsh.model.geo.synchronize()
       
            gmsh.model.mesh.generate(1)
            node_tags, coord, parametric_coord = gmsh.model.mesh.getNodes(1, -1, True)
            vertices = np.asarray(coord, dtype=np.float32).reshape(len(node_tags), 3)
            # build reordering dictionary
            node_tag_to_idx = dict()
            for i, t in enumerate(node_tags):
                node_tag_to_idx[t] = i
            node_tags = []

            # extract "open elements" or "Interface" first
            node_tags, interface_elements = self.edges_of_physical_group(1)
            node_tags.extend(node_tags)
            # extract horn elements
            node_tags, horn_elements = self.edges_of_physical_group(2)
            node_tags.extend(node_tags)
            # extract driver elements
            node_tags, driver_elements = self.edges_of_physical_group(3)
            node_tags.extend(node_tags)

            # relabel node tags with index into vertex array
            temp_node_tags = np.empty(len(node_tags), dtype=np.int32)
            for i, t in enumerate(node_tags):
                temp_node_tags[i] = node_tag_to_idx[t]
            segments = temp_node_tags.reshape(len(node_tags) // 2, 2)
            gmsh.finalize()
            
            self.geometry = Chain(vertices.shape[0], segments.shape[0])
            self.geometry.vertices = vertices[:, 1:3]
            self.geometry.edges = segments
            self.geometry.named_partition['interface'] = (0, interface_elements)
            self.geometry.named_partition['horn'] = (interface_elements, interface_elements + horn_elements)
            self.geometry.named_partition['driver'] = (interface_elements + horn_elements,
                                                       interface_elements + horn_elements + driver_elements)
            
        return self.geometry
