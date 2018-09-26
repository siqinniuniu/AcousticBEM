# ---------------------------------------------------------------------------
# Copyright (C) 2018 Frank Jargstorff
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
import numpy as np

class Chain(object):
    def __init__(self, nVertices = 0, nEdges = 0):
         self.aVertex = np.empty((nVertices, 2), dtype=np.float32)
         self.aEdge = np.empty((nEdges, 2), dtype=np.int32)
         # named partitions are duples of start and end indices into the triangle array
         self.namedPartition = {}
         self.aCenter = None
         self.aLength = None
         self.aArea   = None
         self.aNormal = None

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "aVertex = {}, ".format(self.aVertex)
        result += "aEdge = {}, ".format(self.aEdge)
        result += "namedPartition = {})".format(self.namedPartition)
        return result

    def edgeVertices(self, iEdge):
        return self.aVertex[self.aEdge[iEdge, 0]], \
               self.aVertex[self.aEdge[iEdge, 1]]

    def numberOfElements(self):
        return self.aEdge.shape[0]

    def centers(self):
        if self.aCenter is None:
            self.aCenter = (self.aVertex[self.aEdge[:, 0]] +\
                            self.aVertex[self.aEdge[:, 1]]) / 2.0
        return self.aCenter

    def computeLengthsAndNormals(self):
        # length of the boundary alements
        self.aLength = np.empty(self.aEdge.shape[0], dtype=np.float32)
        self.aNormal = np.empty((self.aEdge.shape[0], 2), dtype=np.float32)
        for i in range(self.aLength.size):
            a = self.aVertex[self.aEdge[i, 0], :]
            b = self.aVertex[self.aEdge[i, 1], :]
            ab = b - a
            vNormal = np.empty_like(ab)
            vNormal[0] = ab[1]
            vNormal[1] = -ab[0]
            nNorm = np.linalg.norm(vNormal)
            self.aNormal[i] = vNormal / nNorm
            self.aLength[i] = nNorm

    def lengths(self):
        if self.aLength is None:
            self.computeLengthsAndNormals()
        return self.aLength

    def normals(self):
        if self.aNormal is None:
            self.computeLengthsAndNormals()
        return self.aNormal

    def areas(self):
        """The areas of the surfaces created by rotating an edge around the x-axis."""
        if self.aArea is None:
            self.aArea = np.empty(self.aEdge.shape[0], dtype=np.float32)
            for i in range(self.aArea.size):
                a, b = self.edgeVertices(i)
                self.aArea[i] = np.pi * (a[0] + b[0]) * np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        return self.aArea

    
class Mesh(object):
    def __init__(self, nVertices = 0, nTriangles = 0):
         self.aVertex = np.empty((nVertices, 3), dtype=np.float32)
         self.aTriangle = np.empty((nTriangles, 3), dtype=np.int32)
         # named partitions are duples of start and end indices into the triangle array
         self.namedPartition = {}
         self.aCenters = None
         self.aArea = None
         self.aNormal = None

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "aVertex = {}, ".format(self.aVertex)
        result += "aTriangle = {}, ".format(self.aTriangle)
        result += "namedPartition = {})".format(self.namedPartition)
        return result

    def triangleVertices(self, iTriangle):
        return self.aVertex[self.aTriangle[iTriangle, 0]], \
               self.aVertex[self.aTriangle[iTriangle, 1]], \
               self.aVertex[self.aTriangle[iTriangle, 2]]

    def numberOfElements(self):
        return self.aTriangle.shape[0]

    def centers(self):
        if self.aCenters is None:
            self.aCenters = (self.aVertex[self.aTriangle[:, 0]] +\
                             self.aVertex[self.aTriangle[:, 1]] +\
                             self.aVertex[self.aTriangle[:, 2]]) / 3.0
        return self.aCenters

    def computeAreasAndNormals(self):
        # area of the boundary alements
        self.aArea = np.empty(self.numberOfElements(), dtype=np.float32)
        self.aNormal = np.empty((self.numberOfElements(), 3), dtype=np.float32)
        for i in range(self.aArea.size):
            a = self.aVertex[self.aTriangle[i, 0], :]
            b = self.aVertex[self.aTriangle[i, 1], :]
            c = self.aVertex[self.aTriangle[i, 2], :]
            ab = b - a
            ac = c - a
            vNormal = np.cross(ab, ac)
            nNorm = np.linalg.norm(vNormal)
            self.aNormal[i] = vNormal / nNorm
            self.aArea[i] = 0.5 * nNorm

    def areas(self):
        if self.aArea is None:
            self.computeAreasAndNormals()
        return self.aArea

    def normals(self):
        if self.aNormal is None:
            self.computeAreasAndNormals()
        return self.aNormal
