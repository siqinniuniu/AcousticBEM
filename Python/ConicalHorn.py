from HalfSpaceHorn import *


class ConicalHorn(HalfSpaceHorn):
    def __init__(self, name, throat_radius, mouth_radius, length, max_element_size=0.01):
        super(ConicalHorn, self).__init__(name, max_element_size)

        self.throat_radius = throat_radius
        self.mouth_radius = mouth_radius
        self.length = length
        self.geometry = None
        
        self.frequency_samples = 600
        self.frequency_range = (16, 16000)

    def chain(self, max_element_size=None):
        """Get a polygonal chain representing the speaker membrane."""
        # check if cached polygonal chain is still good
        if not (max_element_size is None or max_element_size == self.max_element_size):
            self.max_element_size = max_element_size
            self.geometry = None
        if self.geometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Cone Speaker")
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.max_element_size, 1)
            gmsh.model.geo.addPoint(0.0, self.mouth_radius, 0.0, self.max_element_size, 2)
            gmsh.model.geo.addPoint(0.0, self.throat_radius, -self.length, self.max_element_size, 3)
            gmsh.model.geo.addPoint(0.0, 0.0, -self.length, self.max_element_size, 4)
        
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
            node_tags, coord, parametric_coord = gmsh.model.mesh.getNodes(1, -1, True)
            vertices = np.asarray(coord, dtype=np.float32).reshape(len(node_tags), 3)
            # build reordering dictionary
            node_tag_to_idx = dict()
            for i, t in enumerate(node_tags):
                node_tag_to_idx[t] = i
            node_tag_list = []
            # extract "open elements" or "Interface" first
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 1)
            assert len(element_types) == 1 and element_types[0] == 1 # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tag_list.extend(node_tags[0]) # extract line segment node tags
            interface_elements = len(element_tags[0])
            
            # extract horn elements
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 2)
            assert len(element_types) == 1 and element_types[0] == 1 # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tag_list.extend(node_tags[0]) # extract line segment node tags
            horn_elements = len(element_tags[0])
            
            # extract driver elements
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 3)
            assert len(element_types) == 1 and element_types[0] == 1 # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tag_list.extend(node_tags[0]) # extract line segment node tags
            driver_elements = len(element_tags[0])
            
            # relabel node tags with index into vertex array
            temp_node_tags = np.empty(len(node_tag_list), dtype=np.int32)
            for i, t in enumerate(node_tag_list):
                temp_node_tags[i] = node_tag_to_idx[t]
            segments = temp_node_tags.reshape(len(node_tag_list) // 2, 2)
            gmsh.finalize()
            
            self.geometry = Chain(vertices.shape[0], segments.shape[0])
            self.geometry.vertices = vertices[:, 1:3]
            self.geometry.edges = segments
            self.geometry.named_partition['interface'] = (0, interface_elements)
            self.geometry.named_partition['horn'] = (interface_elements,
                                                     interface_elements + horn_elements)
            self.geometry.named_partition['driver'] = (interface_elements + horn_elements,
                                                       interface_elements + horn_elements
                                                       + driver_elements)
            
        return self.geometry
