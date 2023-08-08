import numpy as np
INPUT_SHAPE = (26, 3, 384, 1280)
def load_data():
    for _ in range(20):
        yield {"images": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}  # Still totally real data