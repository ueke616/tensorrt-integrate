import numpy as np

def load_tensor(file):
    
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
        
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


if __name__ == "__main__":
    input   = load_tensor("workspace/input.tensor")
    output  = load_tensor("workspace/output.tensor")
    # print(input.shape, output.shape)
    
    image = input * 255
    image = image.transpose(0, 2, 3, 1)[0].astype(np.uint8)[..., ::-1]
    
    import cv2
    cv2.imwrite("image.jpg", image)
