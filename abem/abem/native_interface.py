from ctypes import *
import os

libraryPath = os.path.join(os.path.dirname(__file__), '../..', 'C', 'libhelmholtz.so')

# "/home/fjargsto/AcousticBEM/C/libhelmholtz.so"

helmholtz = CDLL(libraryPath)

helmholtz.Hankel1.argtypes = [c_int, c_float, c_void_p]


class Complex(Structure):
    _fields_ = [('re', c_float), ('im', c_float)]

class Float2(Structure):
    _fields_ = [('x', c_float), ('y', c_float)]

class Float3(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('z', c_float)]
