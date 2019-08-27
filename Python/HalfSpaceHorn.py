import numpy as np
import gmsh
from joblib import Parallel, delayed

from GmshBoundaries import *
from SamplingPatterns import semi_circle
from RayleighCavitySolverRAD import *
from Mesh import Chain

from Plots import *


class HalfSpaceHorn(object):
    def __init__(self, name, max_element_size=0.01):
        self.name = name
        self.max_element_size = max_element_size

    def plot_cone_section(self):
        self.chain()  # make sure polygonal chain is available
        x_values = np.empty(self.oGeometry.aEdge.shape[0] + 1, dtype=np.float32)
        y_values = np.empty(self.oGeometry.aEdge.shape[0] + 1, dtype=np.float32)
        x_values[0] = self.oGeometry.aVertex[self.oGeometry.aEdge[0, 0], 1]
        y_values[0] = self.oGeometry.aVertex[self.oGeometry.aEdge[0, 0], 0]
        for i in range(self.oGeometry.aEdge.shape[0]):
            x_values[i+1] = self.oGeometry.aVertex[self.oGeometry.aEdge[i, 1], 1]
            y_values[i+1] = self.oGeometry.aVertex[self.oGeometry.aEdge[i, 1], 0]
    
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.plot(x_values, y_values)
        ax.set_aspect('equal', 'datalim')
        
    def make_polar_plot(self, frequencies):
        samples, _, angles = semi_circle(5, 180, "yz")
        samples = samples[:, 1:3]  # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())
        boundary_condition = solver.neumann_boundary_condition()
        driver_partition = np.asarray(solver.geometry.namedPartition['driver'])
        driver_partition = driver_partition - solver.geometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driver_partition[0], driver_partition[1]):
            boundary_condition.f[i] = 1.0

        wavenumbers = frequency_to_wavenumber(frequencies)
        magnitudes = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.magnitude)
                                         (k, solver, boundary_condition,
                                          samples) for k in wavenumbers)
        polar_plot(self.name + " into Half-Space", frequencies, angles, magnitudes)
        
    def make_polar_map(self):
        samples, _, angles = semi_circle(5, 180, "yz")
        samples = samples[:, 1:3] # project down to 2D
        solver = RayleighCavitySolverRAD(self.chain())

        boundary_elements = solver.len()
        boundary_condition = solver.neumann_boundary_condition()
        driver_partition = np.asarray(solver.geometry.namedPartition['driver'])
        driver_partition = driver_partition - solver.geometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driver_partition[0], driver_partition[1]):
            boundary_condition.f[i] = 1.0

        frequencies = np.logspace(np.log10(self.frequencyRange[0]),
                                  np.log10(self.frequencyRange[1]),
                                  self.frequencySamples)
        wavenumbers = frequency_to_wavenumber(frequencies)

        magnitutes = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.magnitude)
                                         (k, solver, boundary_condition, samples) for k in wavenumbers)
        magnitutes = np.asarray(magnitutes).transpose()

        plot_polar_map(self.name + " - Directivity Map",
                       frequencies, angles, magnitutes, 3)

    def make_impedance_plot(self):
        solver = RayleighCavitySolverRAD(self.chain())

        boundary_elements = solver.len()
        boundary_condition = solver.neumann_boundary_condition()
        driver_partition = np.asarray(solver.geometry.namedPartition['driver'])
        driver_partition = driver_partition - solver.geometry.namedPartition['interface'][1]
        # set only driver to velocity 1, horn walls are v = 0.
        for i in range(driver_partition[0], driver_partition[1]):
            boundary_condition.f[i] = 1.0
            
        frequencies = np.logspace(np.log10(self.frequencyRange[0]),
                                  np.log10(self.frequencyRange[1]),
                                  self.frequencySamples)
        wavenumbers = frequency_to_wavenumber(frequencies)
        
        mechanical_impedances = Parallel(n_jobs=4)(delayed(HalfSpaceHorn.mechanica_iImpedance)
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
    def mechanica_iImpedance(k, solver, boundary_condition):
        solution = solver.solve_boundary(k, boundary_condition)
        return solution.mechanical_impedance('driver')
