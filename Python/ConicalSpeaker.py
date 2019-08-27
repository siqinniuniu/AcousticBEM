import numpy as np
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semi_circle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *

class ConicalSpeaker(object):
    def __init__(self, name, cone_radius, cone_angle, cap_radius, cap_height,
                 max_element_size=0.01):
        self.name = name
        self.cone_radius = cone_radius
        self.cone_angle  = cone_angle * np.pi / 180.0
        self.cap_radius  = cap_radius
        self.cap_height  = cap_height
        self.max_element_size = max_element_size
        self.vertices = None
        self.segments = None
        self.num_open_elements = -1
        self.geometry = None
        
    def cavity_depth(self):
        return (self.cone_radius - self.cap_radius) * np.tan(self.cone_angle)
    
    def cap_sphere_radius(self):
        return (self.cap_radius ** 2 + self.cap_height ** 2) / (2.0 * self.cap_height)
    
    def cone_edge(self):
        return [0.0, self.cone_radius, 0.0]
    
    def cap_edge(self):
        return [0.0, self.cap_radius, -self.cavity_depth()]
    
    def cap_center(self):
        return [0.0, 0.0, self.cap_height - self.cavity_depth()]
    
    def cap_sphere_center(self):
        result = self.cap_center()
        result[2] = result[2] - self.cap_sphere_radius()
        return result
    
    def chain(self, max_element_size=None):
        """Get a polygonal chain representing the speaker membrane."""
        # check if cached polygonal chain is still good
        if not (max_element_size is None or max_element_size == self.max_element_size):
            self.max_element_size = max_element_size
            self.vertices = None
            self.segments = None
            self.num_open_elements = -1
        if self.vertices is None or self.segments is None or self.num_open_elements == -1:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("Cone Speaker")
            gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.max_element_size, 1)
            gmsh.model.geo.addPoint(*self.cone_edge(), self.max_element_size, 2)
            gmsh.model.geo.addPoint(*self.cap_edge(), self.max_element_size, 3)
            gmsh.model.geo.addPoint(*self.cap_sphere_center(), self.max_element_size, 4)
            gmsh.model.geo.addPoint(*self.cap_center(), self.max_element_size, 5)
        
            gmsh.model.geo.addLine(1, 2, 1)
            gmsh.model.addPhysicalGroup(1, [1], 1)
            gmsh.model.setPhysicalName(1, 1, "Interface")
            
            gmsh.model.geo.addLine(2, 3, 2)
            gmsh.model.geo.addCircleArc(3, 4, 5, 3)
        
            gmsh.model.addPhysicalGroup(1, [2, 3], 2)
            gmsh.model.setPhysicalName(1, 2, "Speaker Cone")
            gmsh.model.geo.synchronize()
        
            gmsh.model.mesh.generate(1)
            node_tags, coord, parametric_coord = gmsh.model.mesh.getNodes(1, -1, True)
            self.vertices = np.asarray(coord, dtype=np.float32).reshape(len(node_tags), 3)
            # build reordering dictionary
            node_tag_to_idx = dict()
            for i, t in enumerate(node_tags):
                node_tag_to_idx[t] = i
            node_tags = []
            # extract "open elements" or "Interface" first
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 1)
            assert len(element_types) == 1 and element_types[0] == 1  # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tags.extend(node_tags[0])  # extract line segment node tags
            self.num_open_elements = len(element_tags[0])
            
            # extract speaker elements
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 2)
            assert len(element_types) == 1 and element_types[0] == 1  # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tags.extend(node_tags[0])  # extract line segment node tags
            
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(1, 3)
            assert len(element_types) == 1 and element_types[0] == 1  # only line segments
            assert len(element_tags) == 1
            assert len(node_tags) == 1 and len(node_tags[0]) == len(element_tags[0]) * 2
            node_tags.extend(node_tags[0])  # extract line segment node tags
            
            # relabel node tags with index into vertex array
            temp_node_tags = np.empty(len(node_tags), dtype=np.int32)
            for i, t in enumerate(node_tags):
                temp_node_tags[i] = node_tag_to_idx[t]
            self.segments = temp_node_tags.reshape(len(node_tags) // 2, 2)
            gmsh.finalize()
            
            self.geometry = Chain(self.vertices.shape[0], self.segments.shape[0])
            self.geometry.vertices = self.vertices[:, 1:3]
            self.geometry.edges = self.segments
            self.geometry.named_partition['interface'] = (0, self.num_open_elements)
            self.geometry.named_partition['cavity'] = (self.num_open_elements,
                                                       self.segments.shape[0])
        return self.geometry
    
    def plot_cone_section(self):
        self.chain()  # make sure polygonal chain is available
        x_coords = np.empty(self.segments.shape[0] + 1, dtype=np.float32)
        y_coords = np.empty(self.segments.shape[0] + 1, dtype=np.float32)
        x_coords[0] = self.vertices[self.segments[0, 0], 0]
        y_coords[0] = self.vertices[self.segments[0, 0], 1]
        for i in range(self.segments.shape[0]):
            x_coords[i+1] = self.vertices[self.segments[i, 1], 2]
            y_coords[i+1] = self.vertices[self.segments[i, 1], 1]
    
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.plot(x_coords, y_coords)
        ax.set_aspect('equal', 'datalim')
        
    def make_polar_plot(self, frequencies):
        samples, _, angles = semi_circle(5, 180, "yz")
        samples = samples[:, 1:3] # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())
        boundary_elements = solver.len()
        boundary_condition = solver.neumann_boundary_condition()
        velocity = np.array([1.0, 0.0], dtype=np.float32)
        normals = solver.cavity_normals()
        for i in range(boundary_elements):
            boundary_condition.f[i] = np.dot(velocity, normals[i, :])

        k_values = frequency_to_wavenumber(frequencies)
        magnitudes = Parallel(n_jobs=4)(delayed(ConicalSpeaker.magnitude)
                                         (k, solver, boundary_condition, samples) for k in k_values)
        polar_plot(self.name + " into Half-Space", frequencies, angles, magnitudes)
        
    def make_polar_map(self):
        samples, _, angles = semi_circle(5, 180, "yz")
        samples = samples[:, 1:3] # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())
        boundary_elements = solver.len()
        boundary_condition = solver.neumann_boundary_condition()
        velocity = np.array([1.0, 0.0], dtype=np.float32)
        normals = solver.cavity_normals()
        for i in range(boundary_elements):
            boundary_condition.f[i] = np.dot(velocity, normals[i, :])

        frequency_samples = 600
        frequencies = np.logspace(np.log10(20), np.log10(5000), frequency_samples)
        wavenumbers = frequency_to_wavenumber(frequencies)

        magnitudes = Parallel(n_jobs=4)(delayed(ConicalSpeaker.magnitude)
                                         (k, solver, boundary_condition, samples) for k in wavenumbers)
        magnitudes = np.asarray(magnitudes).transpose()

        plot_polar_map(self.name + " - Directivity Map",
                       frequencies, angles, magnitudes, 3)

    def make_impedance_plot(self):
        solver = RayleighCavitySolverRAD(self.chain())

        boundary_elements = solver.len()
        boundary_condition = solver.neumann_boundary_condition()
        velocity = np.array([1.0, 0.0], dtype=np.float32)
        normals = solver.cavity_normals()
        for i in range(boundary_elements):
            boundary_condition.f[i] = np.dot(velocity, normals[i, :])
            
        frequency_samples = 600
        frequencies = np.logspace(np.log10(20), np.log10(5000), frequency_samples)
        wavenumbers = frequency_to_wavenumber(frequencies)
        
        mechanical_impedances = Parallel(n_jobs=4)(delayed(ConicalSpeaker.mechanical_impedance)
                                 (k, solver, boundary_condition) for k in wavenumbers)
        # -1 because of normals point outside of cavity
        mechanical_impedances = -1.0 * np.asarray(mechanical_impedances)

        plot_mechanical_impedance(self.name + " in Infinite Baffle - Mechanical Impedance",
                                  frequencies, mechanical_impedances)
        
    @staticmethod
    def magnitude(k, solver, boundary_condition, samples):
        solution = solver.solve_boundary(k, boundary_condition)
        sample_solution = solver.solve_exterior(solution, samples)
        return sound_magnitude(sound_pressure(k, sample_solution.aPhi))

    @staticmethod
    def mechanical_impedance(k, solver, boundary_condition):
        solution = solver.solve_boundary(k, boundary_condition)
        return solution.mechanical_impedance('cavity')
