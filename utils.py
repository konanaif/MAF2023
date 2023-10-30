import numpy as np


## Categorical vector mapper
# returns map
def mapping(class_vector):
    # Flatten
    class_vector = class_vector.ravel()

    cls2val = np.unique(class_vector)
    val2cls = dict(zip(cls2val, range(len(cls2val))))

    converted_vector = [val2cls[v] for v in class_vector]

    return cls2val, val2cls, converted_vector