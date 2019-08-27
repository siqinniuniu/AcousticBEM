import numpy as np

def semi_circle(r, samples, plane):
    distance = 0.5
    aAngle = np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=samples)
    aVertex = np.empty((samples, 3), dtype=np.float32)
    if plane == "xz":
        aVertex[:,0] = np.sin(aAngle) * r
        aVertex[:,1] = np.zeros(samples)
        aVertex[:,2] = np.cos(aAngle) * r
    elif plane == "yz":
        aVertex[:,0] = np.zeros(samples)
        aVertex[:,1] = np.sin(aAngle) * r
        aVertex[:,2] = np.cos(aAngle) * r
    else:
        assert False, 'plane must be "xz" or "yz"'
        
    aElement = np.empty((samples - 1, 2), dtype=np.int32)
    assert samples > 1, 'samples (= {}) must be > 1'.format(samples)
    aElement[:, 0] = range(samples - 1)
    aElement[:, 1] = range(1, samples)
    
    return aVertex, aElement, aAngle
