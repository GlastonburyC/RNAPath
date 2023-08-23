import math
import h5py
import PIL
    

def slide_to_scaled_pil_image(slide, SCALE_FACTOR=32):
    """
    Convert a WSI slide to a scaled-down PIL image.
    Args:
        slide: An OpenSlide object.
    Returns:
        Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, (large_w, large_h, new_w, new_h)


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    try:
        file.close()
    except RuntimeError:
        print(output_path)
    return output_path



def get_coords_h5(filename):
    with h5py.File(filename, "r") as f:
        # get first object name/key; may or may NOT be a group
        coords_group_key = list(f.keys())[0]
        coords = f[coords_group_key][()]  # returns as a numpy array
        return coords



