import numpy as np
import cv2
# from .image import NcorrImage # Import if NcorrImage type hint is needed

def is_proper_image_format(img_array):
    """
    Checks if the NumPy array has a format suitable for Ncorr.
    (uint8, uint16, float64), 2D (grayscale) or 3D (color, assumed BGR for OpenCV).
    Ncorr internally converts to float64 grayscale (0-1).
    """
    if not isinstance(img_array, np.ndarray):
        return False
    
    valid_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
    if img_array.dtype not in valid_dtypes:
        return False
        
    if img_array.ndim == 2: # Grayscale
        return True
    if img_array.ndim == 3 and img_array.shape[2] == 3: # Color (BGR or RGB)
        return True
        
    return False

def load_images_from_paths(image_paths):
    """
    Loads multiple images from a list of file paths.

    Args:
        image_paths (list of str): List of file paths to images.

    Returns:
        list of NcorrImage: List of NcorrImage objects.
    """
    from .image import NcorrImage # Local import to avoid circular dependency at module load time
    images = []
    for path in image_paths:
        try:
            images.append(NcorrImage(source=path))
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
    return images

def load_saved_ncorr_image(saved_img_struct):
    """
    Loads an NcorrImage from a saved structure (placeholder).
    The structure would typically come from a loaded .mat file or similar.
    """
    # from .image import NcorrImage # Local import
    # This depends heavily on the format of saved_img_struct
    # For example:
    # if saved_img_struct['type'] == 'file' or saved_img_struct['type'] == 'lazy':
    #     return NcorrImage(source=saved_img_struct['path'], name=saved_img_struct['name'])
    # elif saved_img_struct['type'] == 'load':
    #     return NcorrImage(source=saved_img_struct['gs_data'], name=saved_img_struct['name'])
    print("load_saved_ncorr_image is a placeholder and needs implementation based on save format.")
    return None


def is_real_in_bounds(value, low, high):
    """Checks if x is a real number between and including low and high."""
    return isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value) and low <= value <= high

def is_int_in_bounds(value, low, high):
    """Tests if x is an integer between and including bounds low and high."""
    return isinstance(value, (int, np.integer)) and low <= value <= high


def form_region_constraint_func(ncorr_roi_object, region_index):
    """
    Creates a constraint function for a specific region within an NcorrROI object.
    The returned function, given a point (x, y), will return the closest valid point
    within that specific region.

    Args:
        ncorr_roi_object (NcorrROI): The NcorrROI object.
        region_index (int): The index of the region within the ROI.

    Returns:
        function: A constraint function `func(pos_xy)` where pos_xy is [x, y].
                  Returns [constrained_x, constrained_y].
    """
    if not (0 <= region_index < len(ncorr_roi_object.regions)):
        raise ValueError("Invalid region_index.")

    region_data = ncorr_roi_object.regions[region_index]
    if region_data['totalpoints'] == 0:
        # If region is empty, constraint can't really work.
        # Return a function that doesn't change the point or raises error.
        def empty_constraint(pos_xy):
            # print(f"Warning: Region {region_index} is empty. Constraint cannot be applied.")
            return pos_xy 
        return empty_constraint

    # Extract data for closure - ensure data is NumPy for efficiency if used heavily
    # Convert nodelist to a more usable format if it's a flat C-style array from C++ bindings
    height_nl = region_data['height_nodelist']
    width_nl = region_data['width_nodelist']

    # The nodelist from CppVecStructRegion is a flat std::vector<int>
    # The python conversion _cpp_region_to_python makes it a numpy array.
    # If it's flat, needs reshaping. If it's already HxW, good.
    # Assuming it's already shaped [height_nodelist, max_noderange_width] or similar
    # from _cpp_region_to_python. If flat, it would be
    # nodelist_flat = region_data['nodelist']
    # nodelist_c_order = nodelist_flat.reshape((height_nl, width_nl)) # if data was copied row-major
    # nodelist_f_order = nodelist_flat.reshape((height_nl, width_nl), order='F') # if data was copied col-major
    # Based on C++ (value[j+k*height_nodelist]), it's column-major.

    # Using the reshaped numpy array directly from _cpp_region_to_python
    nodelist = region_data['nodelist'] # This is already HxW from the conversion
    noderange = region_data['noderange'] # This is 1D array of length H
    
    leftbound = region_data['leftbound']
    rightbound = region_data['rightbound']
    # upperbound_reg = region_data['upperbound'] # Min y value in region
    # lowerbound_reg = region_data['lowerbound'] # Max y value in region

    def constraint_func(pos_xy):
        # pos_xy is [x, y] using 0-based indexing from image coordinates
        # The original MATLAB code uses 1-based for input `pos` to constraintfcn,
        # then converts to 0-based. Here, assume Python side uses 0-based throughout.
        x_in, y_in = int(round(pos_xy[0])), int(round(pos_xy[1]))
        
        constrained_x, constrained_y = x_in, y_in

        # 1. Constrain X
        # Find overall X bounds of the region in image coordinates
        # region_xs = np.arange(leftbound, rightbound + 1)
        # valid_cols_mask = noderange > 0
        # if not np.any(valid_cols_mask): return pos_xy # Empty region after considering noderange

        # min_x_region = region_xs[valid_cols_mask][0]
        # max_x_region = region_xs[valid_cols_mask][-1]
        
        # The `leftbound` and `rightbound` in region_data are already the min/max X for the region.
        # However, we need to find the actual min/max X that has *any* nodes.
        
        actual_min_x = -1
        actual_max_x = -1
        
        for i_col_rel in range(height_nl): # height_nl is number of columns in region's local coord
            if noderange[i_col_rel] > 0:
                if actual_min_x == -1:
                    actual_min_x = leftbound + i_col_rel
                actual_max_x = leftbound + i_col_rel
        
        if actual_min_x == -1 : # Region is effectively empty
            return pos_xy


        if x_in < actual_min_x:
            constrained_x = actual_min_x
        elif x_in > actual_max_x:
            constrained_x = actual_max_x
        else: # x_in is within [actual_min_x, actual_max_x]
            # Check if this column (constrained_x) actually has nodes
            # If not, find nearest column that does
            col_idx_rel = constrained_x - leftbound
            if not (0 <= col_idx_rel < height_nl and noderange[col_idx_rel] > 0):
                # Find nearest valid column
                dist_left, dist_right = np.inf, np.inf
                nearest_left_col, nearest_right_col = -1, -1

                # Search left
                for i_search_rel in range(col_idx_rel - 1, -1, -1):
                    if noderange[i_search_rel] > 0:
                        nearest_left_col = leftbound + i_search_rel
                        dist_left = constrained_x - nearest_left_col
                        break
                # Search right
                for i_search_rel in range(col_idx_rel + 1, height_nl):
                    if noderange[i_search_rel] > 0:
                        nearest_right_col = leftbound + i_search_rel
                        dist_right = nearest_right_col - constrained_x
                        break
                
                if nearest_left_col != -1 and nearest_right_col != -1:
                    constrained_x = nearest_left_col if dist_left <= dist_right else nearest_right_col
                elif nearest_left_col != -1:
                    constrained_x = nearest_left_col
                elif nearest_right_col != -1:
                    constrained_x = nearest_right_col
                else: # Should not happen if actual_min/max_x were found
                    return pos_xy 
        
        # 2. Constrain Y for the chosen constrained_x
        col_idx_rel_final = constrained_x - leftbound
        if not (0 <= col_idx_rel_final < height_nl and noderange[col_idx_rel_final] > 0) :
             return [constrained_x, y_in] # Fallback if constrained_x is still invalid somehow

        min_dist_y = np.inf
        final_y = y_in

        col_nodelist = nodelist[col_idx_rel_final, :int(noderange[col_idx_rel_final])].reshape(-1, 2) # Get pairs

        for y_start, y_end in col_nodelist:
            if y_in < y_start:
                dist = y_start - y_in
                if dist < min_dist_y:
                    min_dist_y = dist
                    final_y = y_start
            elif y_in > y_end:
                dist = y_in - y_end
                if dist < min_dist_y:
                    min_dist_y = dist
                    final_y = y_end
            else: # y_in is within [y_start, y_end]
                final_y = y_in
                min_dist_y = 0
                break # Found exact placement

        constrained_y = final_y
        return [constrained_x, constrained_y]

    return constraint_func