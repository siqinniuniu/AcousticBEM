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
from NativeInterface import *
import numpy as np


def compute_l(k, p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    helmholtz.ComputeL_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def compute_m(k, p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    helmholtz.ComputeM_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def compute_mt(k, p, vec_p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    helmholtz.ComputeMt_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)


def compute_n(k, p, vec_p, qa, qb, p_on_element):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(p_on_element)
    helmholtz.ComputeN_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)
