import torch as t
import numpy as np



a = [  [ [1,2,3,4],[3,2,1,4],[1,2,5,4] ], [ [1,2,3,4],[3,2,1,4],[1,2,5,4] ] ]


npA = np.array(a)

print(npA.ndim)


npTensor = t.from_numpy(npA)
print(npTensor.shape)
print(npTensor.dim())

tensor = t.tensor(a)

print(tensor.shape)
print(tensor.dim())
