
#    This script is part of skeletonizer (http://www.github.com/schlegelp/skeletonizer).
#    Copyright (C) 2018 Philipp Schlegel
#    Modified from https://github.com/aalavandhaann/Py_BL_MeshSkeletonization
#    by #0K Srinivasan Ramachandran.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

import numpy as np
from .utilities import buildKDTree, getMeshVPos

#context - The blender scene context
#mesh - Mesh representing the contracted shape
#radius - value to be used for the KDTree search

def farthest_sampling_by_sphere(mesh, radius):
    n = len(mesh.data.vertices)
    kdtree = buildKDTree(mesh)
    spls = np.zeros((0, 3))
    corresp = np.zeros((n, 1))
    mindst = np.zeros((n, 1))
    mindst[:] = np.nan
    pts = getMeshVPos(mesh)
    for k in range(n):
        if(corresp[k] != 0.0):
            continue

        vco = mesh.data.vertices[k].co
        mindst[k] = np.inf

        while (not np.all(corresp != 0.0)):
            maxIdx = np.argmax(mindst)

            if(mindst[maxIdx] == 0.0):
                break

            valuesInRange = kdtree.find_range(vco, radius)
            if(len(valuesInRange)):
                valuesInRange = np.array(valuesInRange)
                nIdxs, nDsts = valuesInRange[:,1].flatten().tolist() , valuesInRange[:,2].flatten().tolist()
                if(np.all(corresp[nIdxs] != 0.0)):
                    mindst[maxIdx] = 0.0
                    continue

                spls = np.append(spls, [pts[maxIdx,:]], axis=0)
                for i in range(len(nIdxs)):
                    if(mindst[nIdxs[i]] > nDsts[i] or np.isnan(mindst[nIdxs[i]])):
                        mindst[nIdxs[i]] = nDsts[i]
                        corresp[nIdxs[i]] = spls.shape[0]


    return spls, corresp





