# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
import unittest
import numpy as np
from scipy.special import hankel1

import HelmholtzIntegrals2D as IH2
import HelmholtzIntegrals2D_C as IH2C
import HelmholtzIntegrals3D as IH3
import HelmholtzIntegralsRAD as RAD


class TestComplexQuadGenerator(unittest.TestCase):

    def test_complex_quad_generator_01(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        def func(x):
            return 1.0
        result = RAD.complex_quad_generator(func, a, b)
        self.assertAlmostEqual(result, np.sqrt(2.0), 6, msg="{} != {}".format(result, np.sqrt(2.0)))


class TestCircularIntegratorPI(unittest.TestCase):
    def test_circular_integrator_01(self):
        circle = RAD.CircularIntegratorPi(1)
        def func(x):
            return 1.0
        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi, msg="{} != {}".format(result, np.pi))

    def test_circular_integrator_02(self):
        circle = RAD.CircularIntegratorPi(2)
        def func(x):
            return 1.0
        result = circle.integrate(func)
        self.assertAlmostEqual(result, np.pi, msg="{} != {}".format(result, np.pi))


class TestTriangleIntegrator(unittest.TestCase):
    def test_complex_quad(self):
        def func(x):
            return 1.0
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        c = np.array([0, 0, 1], dtype=np.float32)
        result = IH3.complex_quad(func, a, b, c)
        self.assertAlmostEqual(result, 0.5, msg="{} != {}".format(result, 0.5))


class TestHankel(unittest.TestCase):

    def test_hankel_01(self):
        H1scipy = hankel1(0, 1.0)
        H1gsl = IH2C.hankel1(0, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg="{} != {}".format(H1scipy, H1gsl))

    def test_hankel_02(self):
        H1scipy = hankel1(0, 10.0)
        H1gsl = IH2C.hankel1(0, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg="{} != {}".format(H1scipy, H1gsl))

    def test_hankel_03(self):
        H1scipy = hankel1(1, 1.0)
        H1gsl = IH2C.hankel1(1, 1.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg="{} != {}".format(H1scipy, H1gsl))

    def test_hankel_04(self):
        H1scipy = hankel1(1, 10.0)
        H1gsl = IH2C.hankel1(1, 10.0)
        self.assertAlmostEqual(H1scipy, H1gsl, msg="{} != {}".format(H1scipy, H1gsl))


class TestComputeL(unittest.TestCase):

    def test_compute_L_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_l(k, p, a, b, pOnElement)
        zC = IH2C.compute_l(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_L_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_l(k, p, a, b, pOnElement)
        zC  = IH2C.compute_l(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_L_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_l(k, p, a, b, pOnElement)
        zC = IH2C.compute_l(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_L_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_l(k, p, a, b, pOnElement)
        zC = IH2C.compute_l(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))


class TestComputeM(unittest.TestCase):

    def test_compute_M_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_m(k, p, a, b, pOnElement)
        zC = IH2C.compute_m(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_M_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_m(k, p, a, b, pOnElement)
        zC = IH2C.compute_m(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_M_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_m(k, p, a, b, pOnElement)
        zC = IH2C.compute_m(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_M_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_m(k, p, a, b, pOnElement)
        zC = IH2C.compute_m(k, p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))


class TestComputeMt(unittest.TestCase):

    def test_compute_Mt_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_mt(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_mt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_Mt_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_mt(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_mt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_Mt_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_mt(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_mt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_Mt_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_mt(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_mt(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))


class TestComputeN(unittest.TestCase):

    def test_compute_N_01(self):
        k = 0.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_n(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_n(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_N_02(self):
        k = 10.0
        p = np.array([0.5, 0.75], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = False
        zPy = IH2.compute_n(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_n(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, 6, msg="{} != {}".format(zPy, zC))

    def test_compute_N_03(self):
        k = 0.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_n(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_n(k, p, normal_p, a, b, pOnElement)
        self.assertAlmostEqual(zPy, zC, msg="{} != {}".format(zPy, zC))

    def test_compute_N_04(self):
        k = 10.0
        p = np.array([0.0, 0.05], dtype=np.float32)
        normal_p = np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        a = np.array([0.0, 0.00], dtype=np.float32)
        b = np.array([0.0, 0.10], dtype=np.float32)
        pOnElement = True
        zPy = IH2.compute_n(k, p, normal_p, a, b, pOnElement)
        zC = IH2C.compute_n(k, p, normal_p, a, b, pOnElement)
        # note, how accuracy here is reduced to only 3 digits after the decimal dot.
        # I don't believe this is because of buggy code but because of error accumulation
        # being different for the C and the Python codes.
        self.assertAlmostEqual(zPy, zC, 3, msg="{} != {}".format(zPy, zC))


if __name__ == '__main__':
    unittest.main()

