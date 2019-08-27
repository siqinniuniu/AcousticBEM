from HalfSpaceHorn import *
from scipy.optimize import brentq


class CompoundExponentialHorn(HalfSpaceHorn):
    def __init__(self, name, throat_radius, cutoff_frequency, angle, length, max_element_size=0.01):
        super(CompoundExponentialHorn, self).__init__(name, max_element_size)

        self.throat_radius = throat_radius
        self.cutoff_k = frequency_to_wavenumber(cutoff_frequency)
        self.angle = np.pi * angle / 180.0  # to radians
        self.length = length
        self.shape_factor = 1.0  # for now basic exponential geometry
        self.geometry = None
        
        self.frequency_samples = 600
        self.frequency_range = (16, 16000)

    def first_transition_radius(self):
        return 0.95 * np.sin(self.angle) / self.cutoff_k

    def first_transition_z(self):
        root, info = brentq(lambda z: self.exponential_radius(z) - self.first_transition_radius(),
                            -self.length, 0.0)
        assert info.converged
        return root

    def second_transition_radius(self):
        derivative = np.tan(self.angle)
        length = self.second_transition_z() - self.first_transition_z()
        offset = length * derivative
        return self.first_transition_radius() + offset

    def second_transition_z(self):
        return -self.length / 3.0

    def mouth_radius(self):
        derivative = np.tan(2.0 * self.angle)
        offset = derivative * self.length / 3.0
        return self.second_transition_radius() + offset
    
    def exponential_radius(self, z):
        z += self.length  # shift z to the left by length, horn start at z = -length
        K0 = self.cutoff_k
        T = self.shape_factor
        return self.throat_radius * (np.cosh(z * K0) + T * np.sinh(z * K0))

    def exponential_samples_z(self):
        nSamples = np.ceil((self.length - self.first_transition_z())
                           * np.sqrt(2.0) / self.maxElementSize)
        return np.linspace(self.first_transition_z(), -self.length, nSamples)

    def edges_of_physical_group(self, group_tag):
        tags = []
        # extract horn elements
        elements = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
        num_elements = 0
        for e in elements:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, e)
            assert len(element_types) == 1 and element_types[0] == 1 # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            tags.extend(node_tags[0]) # extract line segment node tags
            num_elements += len(element_tags[0])
        return tags, num_elements
            
    def chain(self, max_element_size=None):
        """Get a polygonal chain representing the horn geometry."""
        # check if cached polygonal chain is still good
        if max_element_size is not None or max_element_size == self.maxElementSize:
            self.maxElementSize = max_element_size
            self.geometry = None
        if self.geometry is None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Compound Conical Exponential Horn")

            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.maxElementSize, 1)
            gmsh.model.geo.addPoint(0.0, self.mouth_radius(),
                                    0.0, self.maxElementSize, 2)
            gmsh.model.geo.addPoint(0.0,
                                    self.second_transition_radius(),
                                    self.second_transition_z(),
                                    self.maxElementSize, 3)
        
            pointTag = 4
            for x in self.exponential_samples_z():
                gmsh.model.geo.addPoint(0.0, self.exponential_radius(x), x,
                                        self.maxElementSize, pointTag)
                pointTag += 1
            gmsh.model.geo.addPoint(0.0, 0.0, -self.length,
                                    self.maxElementSize, pointTag)

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
            self.geometry.named_partition['horn'] = (interface_elements,
                                                     interface_elements + horn_elements)
            self.geometry.named_partition['driver'] = (interface_elements + horn_elements,
                                                       interface_elements + horn_elements
                                                       + driver_elements)
            
        return self.geometry
    
