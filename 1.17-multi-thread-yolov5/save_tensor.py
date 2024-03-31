import numpy as np


def save_tensor(tensor, file):
    with open(file, "wb") as f:
        typeid = 0
        if tensor.dtype == np.float32:
            typeid = 0
        elif tensor.dtype == np.float16:
            typeid = 1
        elif tensor.dtype == np.int32:
            typeid = 2
        elif tensor.dtype == np.uint8:
            typeid = 3

        # 自定义文件头， 0xFCCFE2E2 是标记, ndim维度, typeid
        head = np.array([0xFCCFE2E2, tensor.ndim, typeid], dtype=np.uint32).tobytes()
        f.write(head)
        f.write(np.array(tensor.shape, dtype=np.uint32).tobytes())
        f.write(tensor.tobytes())

if __name__ == "__main__":
    # data = np.arange(100, dtype=np.float32).reshape(10, 10, 1)
    data = np.arange(100, dtype=np.uint8).reshape(10, 10, 1)
    save_tensor(data, "workspace/data.tensor")
