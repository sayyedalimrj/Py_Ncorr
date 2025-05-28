import numpy as np
# Assuming your C++ bindings are importable like this:
from ncorr_app._ext import _ncorr_cpp_core, _ncorr_cpp_algs 
# Import NcorrImage if needed for type hints or operations
# from .image import NcorrImage

class NcorrROI:
    """
    Python equivalent of ncorr_class_roi.m.
    Handles Region of Interest (ROI) data, including mask, region properties,
    reduction, and conversion to C++ compatible types.
    """

    def __init__(self, image_shape=None, mask_array=None):
        """
        Initializes NcorrROI.

        Args:
            image_shape (tuple, optional): Shape (height, width) of the image this ROI belongs to.
                                           Required if mask_array is not provided initially.
            mask_array (np.ndarray, optional): A boolean NumPy array representing the mask.
        """
        self.type = ''  # 'load', 'draw', 'region', 'boundary', 'reduced'
        self.mask = None
        self.regions = []  # List of dicts, each dict from CppVecStructRegion or CppNcorrClassRegion
        self.boundary_data = {'add': [], 'sub': []} # {'add': [np.array_Nx2], 'sub': [[np.array_Mx2], ...]}
        self.draw_objects_data = [] # List of Python dicts {'pos_imroi': np.array, 'type': str, 'addorsub': str}
        self._image_shape_internal = image_shape

        if mask_array is not None:
            if image_shape is not None and mask_array.shape != image_shape:
                raise ValueError("Provided mask_array shape does not match image_shape.")
            self.set_roi_from_mask(mask_array)
            self._image_shape_internal = mask_array.shape
        elif image_shape is not None:
            self.mask = np.zeros(image_shape, dtype=bool)
        # else, it's an empty ROI, to be defined later

    def _numpy_to_cpp_logical_array(self, np_array):
        if np_array is None:
            return _ncorr_cpp_core.CppClassLogicalArray() # Return empty one
        h, w = np_array.shape
        cpp_array = _ncorr_cpp_core.CppClassLogicalArray()
        cpp_array.alloc(h, w)
        cpp_array.set_value_numpy(np_array.astype(bool))
        return cpp_array

    def _numpy_to_cpp_double_array(self, np_array):
        if np_array is None:
            return _ncorr_cpp_core.CppClassDoubleArray()
        
        # Handle 1D or 2D array cases for pos_imroi
        if np_array.ndim == 1: # E.g. for rect [x,y,w,h]
             h, w = 1, np_array.shape[0]
             np_array_2d = np_array.reshape(h,w)
        elif np_array.ndim == 2:
             h, w = np_array.shape
             np_array_2d = np_array
        else:
            raise ValueError("pos_imroi must be 1D or 2D numpy array")

        cpp_array = _ncorr_cpp_core.CppClassDoubleArray()
        cpp_array.alloc(h, w)
        cpp_array.set_value_numpy(np_array_2d.astype(np.float64))
        return cpp_array
        
    def _cpp_region_to_python(self, cpp_region_obj):
        """Converts a single CppVecStructRegion/CppNcorrClassRegion to a Python dict."""
        # Assuming CppVecStructRegion and CppNcorrClassRegion have similar accessible attributes
        # after binding (nodelist, noderange are std::vector<int> or CppClassIntegerArray)
        
        py_region = {
            'nodelist': np.array(cpp_region_obj.nodelist).reshape((cpp_region_obj.height_nodelist, cpp_region_obj.width_nodelist), order='F') if hasattr(cpp_region_obj, 'width_nodelist') and cpp_region_obj.width_nodelist > 0 else np.array(cpp_region_obj.nodelist), # Convert std::vector<int>
            'noderange': np.array(cpp_region_obj.noderange), # Convert std::vector<int>
            'height_nodelist': cpp_region_obj.height_nodelist if hasattr(cpp_region_obj, 'height_nodelist') else 0,
            'width_nodelist': cpp_region_obj.width_nodelist if hasattr(cpp_region_obj, 'width_nodelist') else 0,
            'leftbound': cpp_region_obj.leftbound,
            'rightbound': cpp_region_obj.rightbound,
            'upperbound': cpp_region_obj.upperbound,
            'lowerbound': cpp_region_obj.lowerbound,
            'totalpoints': cpp_region_obj.totalpoints
        }
        # If nodelist/noderange were bound as CppClassIntegerArray, access their values:
        if isinstance(cpp_region_obj.nodelist, _ncorr_cpp_core.CppClassIntegerArray):
             if cpp_region_obj.nodelist.width > 0 and cpp_region_obj.nodelist.height > 0:
                py_region['nodelist'] = np.array(cpp_region_obj.nodelist, copy=False) # Relies on buffer protocol
             else:
                py_region['nodelist'] = np.array([])
        if isinstance(cpp_region_obj.noderange, _ncorr_cpp_core.CppClassIntegerArray):
            if cpp_region_obj.noderange.width > 0 and cpp_region_obj.noderange.height > 0:
                py_region['noderange'] = np.array(cpp_region_obj.noderange, copy=False)
            else:
                py_region['noderange'] = np.array([])


        return py_region

    def _cpp_regions_to_python_list(self, cpp_regions_list):
        """Converts a list of CppVecStructRegion/CppNcorrClassRegion to a list of Python dicts."""
        return [self._cpp_region_to_python(cpp_reg) for cpp_reg in cpp_regions_list]

    def set_roi_from_drawings(self, draw_objects_data, image_shape, cutoff=2000):
        """
        Sets the ROI based on a list of drawing objects (rectangles, ellipses, polygons).

        Args:
            draw_objects_data (list): List of dicts, e.g., 
                                      {'pos_imroi': np.array, 'type': 'rect'|'ellipse'|'poly', 'addorsub': 'add'|'sub'}
            image_shape (tuple): (height, width) of the image.
            cutoff (int, optional): Cutoff for small regions. Defaults to 2000.
        """
        self.type = 'draw'
        self.draw_objects_data = draw_objects_data # Store Python version
        self._image_shape_internal = image_shape
        
        cpp_draw_objects = []
        for py_dobj in draw_objects_data:
            cpp_dobj = _ncorr_cpp_algs.PyDrawObject() # Use the bound C++ compatible struct
            cpp_dobj.pos_imroi = self._numpy_to_cpp_double_array(py_dobj['pos_imroi'])
            cpp_dobj.type = py_dobj['type']
            cpp_dobj.addorsub = py_dobj['addorsub']
            cpp_draw_objects.append(cpp_dobj)

        cpp_mask = _ncorr_cpp_core.CppClassLogicalArray()
        cpp_mask.alloc(image_shape[0], image_shape[1]) # form_mask expects mask to be pre-allocated
        
        _ncorr_cpp_algs.form_mask(cpp_draw_objects, cpp_mask)
        
        # Convert CppClassLogicalArray back to NumPy array to use with set_roi_from_mask
        # This relies on CppClassLogicalArray having a buffer protocol or a get_value_numpy method
        numpy_mask = np.array(cpp_mask, copy=True) # copy=True to be safe
        self.set_roi_from_mask(numpy_mask, cutoff)


    def set_roi_from_mask(self, mask_array, cutoff=2000):
        """
        Sets the ROI from a boolean NumPy mask array.

        Args:
            mask_array (np.ndarray): Boolean array where True indicates ROI.
            cutoff (int, optional): Cutoff for small regions. Defaults to 2000.
        """
        if not (isinstance(mask_array, np.ndarray) and mask_array.dtype == bool):
            raise TypeError("mask_array must be a boolean NumPy array.")
        
        self.mask = mask_array.copy()
        self._image_shape_internal = self.mask.shape
        self.type = 'load' # Or 'computed', 'mask_provided'
        
        cpp_mask = self._numpy_to_cpp_logical_array(self.mask)
        
        # form_regions returns tuple: (list_of_cpp_vec_struct_region, removed_flag)
        cpp_regions_list, removed = _ncorr_cpp_algs.form_regions(cpp_mask, cutoff, False)
        self.regions = self._cpp_regions_to_python_list(cpp_regions_list)

        # Boundary formation (simplified adaptation)
        # The original ncorr_class_roi.m forms boundaries for 'load' and 'draw' types.
        # This involves complex logic with form_boundary and handling sub-regions (holes).
        # For a basic port, this might be deferred or simplified.
        # Here's a very basic placeholder for the structure:
        self.boundary_data = {'add': [], 'sub': []}
        # for region_idx, py_region_data in enumerate(self.regions):
        #     if py_region_data['totalpoints'] > 0:
        #         # Need a starting point for form_boundary
        #         # This logic is non-trivial to replicate perfectly without deep dive into MATLAB
        #         # point_init_cpp = ... create CppClassIntegerArray for a top-left point of the region
        #         # mask_for_this_region_cpp = ... create CppClassLogicalArray for only this region
        #         # add_boundary_cpp, _ = _ncorr_cpp_algs.form_boundary(point_init_cpp, 0, mask_for_this_region_cpp)
        #         # self.boundary_data['add'].append(np.array(add_boundary_cpp, copy=True))
        #         # ... logic for sub-boundaries (holes) ...
        #         pass # Placeholder for complex boundary formation
        if removed:
            # If regions were removed, the main mask might need updating to reflect only kept regions
            self.mask = np.zeros(self._image_shape_internal, dtype=bool)
            for region_data in self.regions:
                if region_data['totalpoints'] > 0:
                    for j_col in range(region_data['height_nodelist']): # Iterating over columns in region's own coordinate system
                        x_abs = j_col + region_data['leftbound']
                        noderange_val = int(region_data['noderange'][j_col])
                        for k_pair_idx in range(0, noderange_val, 2):
                            y_start_abs = int(region_data['nodelist'].reshape(region_data['height_nodelist'], -1, order='F')[j_col, k_pair_idx])
                            y_end_abs = int(region_data['nodelist'].reshape(region_data['height_nodelist'], -1, order='F')[j_col, k_pair_idx + 1])
                            if x_abs >=0 and x_abs < self.mask.shape[1]:
                                y_min_clip = max(0, y_start_abs)
                                y_max_clip = min(self.mask.shape[0]-1, y_end_abs)
                                if y_min_clip <= y_max_clip:
                                    self.mask[y_min_clip : y_max_clip+1, x_abs] = True


    def get_mask(self):
        """Returns the boolean mask NumPy array."""
        return self.mask

    def get_regions_data(self):
        """Returns the list of region data (Python dictionaries)."""
        return self.regions

    def get_full_regions_count(self):
        """Returns the count of non-empty regions."""
        if not self.regions:
            return 0
        return sum(1 for r in self.regions if r.get('totalpoints', 0) > 0)

    def reduce(self, spacing):
        """
        Reduces the ROI based on a spacing factor.

        Args:
            spacing (int): Spacing factor.

        Returns:
            NcorrROI: A new, reduced NcorrROI object.
        """
        if self.mask is None:
            raise ValueError("ROI mask has not been set.")
        if spacing == 0:
            return self # Or a deep copy: copy.deepcopy(self)

        new_image_shape = (self.mask.shape[0] // (spacing + 1), self.mask.shape[1] // (spacing + 1))
        reduced_roi = NcorrROI(image_shape=new_image_shape)
        
        reduced_regions_py = []
        for region_data in self.regions:
            if region_data['totalpoints'] == 0:
                reduced_regions_py.append(self._cpp_region_to_python(_ncorr_cpp_core.CppVecStructRegion())) # empty
                continue

            left_bound_new = int(np.ceil(region_data['leftbound'] / (spacing + 1)))
            right_bound_new = int(np.floor(region_data['rightbound'] / (spacing + 1)))
            
            if right_bound_new < left_bound_new : # region becomes empty or invalid
                reduced_regions_py.append(self._cpp_region_to_python(_ncorr_cpp_core.CppVecStructRegion()))
                continue

            height_nodelist_new = right_bound_new - left_bound_new + 1
            # Max noderange width for the original region to size nodelist_new
            max_noderange_orig_width = region_data['nodelist'].shape[1] if region_data['nodelist'].ndim ==2 else 2 # Default to 2 if 1D

            nodelist_new = np.full((height_nodelist_new, max_noderange_orig_width), -1, dtype=np.int32)
            noderange_new = np.zeros(height_nodelist_new, dtype=np.int32)
            total_points_new = 0
            
            upper_bound_new = new_image_shape[0] 
            lower_bound_new = 0

            original_nodelist_reshaped = region_data['nodelist'].reshape(region_data['height_nodelist'],-1, order='F')

            for j_orig_col_idx in range(region_data['height_nodelist']): # Iterate original region's columns
                x_orig = j_orig_col_idx + region_data['leftbound']
                if x_orig % (spacing + 1) != 0: # Consider only points that fall on the new grid
                    continue
                
                x_new_col_idx = x_orig // (spacing + 1) - left_bound_new
                if not (0 <= x_new_col_idx < height_nodelist_new):
                    continue

                current_noderange_count = 0
                for k_pair_idx in range(0, int(region_data['noderange'][j_orig_col_idx]), 2):
                    y_start_orig = original_nodelist_reshaped[j_orig_col_idx, k_pair_idx]
                    y_end_orig = original_nodelist_reshaped[j_orig_col_idx, k_pair_idx+1]

                    y_start_new = int(np.ceil(y_start_orig / (spacing + 1)))
                    y_end_new = int(np.floor(y_end_orig / (spacing + 1)))

                    if y_end_new >= y_start_new:
                        if current_noderange_count + 1 < nodelist_new.shape[1]:
                            nodelist_new[x_new_col_idx, current_noderange_count] = y_start_new
                            nodelist_new[x_new_col_idx, current_noderange_count + 1] = y_end_new
                            current_noderange_count += 2
                            total_points_new += (y_end_new - y_start_new + 1)
                            upper_bound_new = min(upper_bound_new, y_start_new)
                            lower_bound_new = max(lower_bound_new, y_end_new)
                noderange_new[x_new_col_idx] = current_noderange_count
            
            py_reg = {
                'nodelist': nodelist_new[:,:np.max(noderange_new)] if np.max(noderange_new)>0 else np.array([]), # Trim unused columns
                'noderange': noderange_new,
                'height_nodelist': height_nodelist_new,
                'width_nodelist': np.max(noderange_new) if np.max(noderange_new)>0 else 0,
                'leftbound': left_bound_new,
                'rightbound': right_bound_new,
                'upperbound': upper_bound_new if total_points_new > 0 else 0,
                'lowerbound': lower_bound_new if total_points_new > 0 else 0,
                'totalpoints': total_points_new
            }
            reduced_regions_py.append(py_reg)

        reduced_roi.regions = reduced_regions_py
        # Re-generate mask from reduced regions
        reduced_roi.mask = np.zeros(new_image_shape, dtype=bool)
        for region_data in reduced_roi.regions:
            if region_data['totalpoints'] > 0:
                nodelist_reshaped = region_data['nodelist'].reshape(region_data['height_nodelist'],-1, order='F')
                for j_col in range(region_data['height_nodelist']):
                    x_abs = j_col + region_data['leftbound']
                    noderange_val = int(region_data['noderange'][j_col])
                    for k_pair_idx in range(0, noderange_val, 2):
                        y_start_abs = int(nodelist_reshaped[j_col, k_pair_idx])
                        y_end_abs = int(nodelist_reshaped[j_col, k_pair_idx + 1])
                        if x_abs >=0 and x_abs < reduced_roi.mask.shape[1]:
                            y_min_clip = max(0, y_start_abs)
                            y_max_clip = min(reduced_roi.mask.shape[0]-1, y_end_abs)
                            if y_min_clip <= y_max_clip:
                                reduced_roi.mask[y_min_clip:y_max_clip+1, x_abs] = True
        reduced_roi.type = 'reduced'
        return reduced_roi

    def get_union(self, other_mask_array, spacing):
        """
        Computes the union of this ROI (after reduction) with another mask.
        """
        if self.mask is None:
            raise ValueError("ROI mask not set.")
        
        reduced_self = self.reduce(spacing)
        cpp_self_roi_regions = []
        for py_reg in reduced_self.regions:
            if py_reg['totalpoints'] > 0:
                cpp_reg = _ncorr_cpp_core.CppNcorrClassRegion()
                # ... (populate cpp_reg from py_reg - tedious, involves CppClassIntegerArray for nodelist/range)
                # This part requires careful conversion
                # For simplicity, let's assume to_cpp_roi handles creating a list of CppNcorrClassRegion directly
                # or that form_union can take a list of Python dicts and convert internally (less likely)
                # For now, this part of get_union is complex to fully implement without CppNcorrClassRegion binding details from py_ncorr_core
                pass # Placeholder for populating cpp_self_roi_regions
        
        # Fallback: if cpp_self_roi_regions cannot be easily made, form_union might need to operate on reduced_self.to_cpp_roi().region
        # For this example, let's use the mask of the reduced ROI for form_union's first arg (if form_union can take a CppNcorrClassRoi instead of list of regions)

        cpp_other_mask = self._numpy_to_cpp_logical_array(other_mask_array)
        
        # Assuming form_union expects a list of CppNcorrClassRegion from the first argument's ROI
        # Need to correctly convert reduced_self.regions to std::vector<CppNcorrClassRegion>
        # This is simplified here:
        cpp_reduced_self_regions = [r for r in reduced_self.to_cpp_roi().region if r.totalpoints > 0]


        cpp_unioned_regions_list = _ncorr_cpp_algs.form_union(cpp_reduced_self_regions, cpp_other_mask)
        
        unioned_mask_shape = other_mask_array.shape
        final_unioned_roi = NcorrROI(image_shape=unioned_mask_shape)
        final_unioned_roi.regions = self._cpp_regions_to_python_list(cpp_unioned_regions_list)
        
        # Reconstruct the mask for the unioned ROI
        final_unioned_roi.mask = np.zeros(unioned_mask_shape, dtype=bool)
        for region_data in final_unioned_roi.regions:
            if region_data['totalpoints'] > 0:
                nodelist_reshaped = region_data['nodelist'].reshape(region_data['height_nodelist'],-1, order='F')
                for j_col in range(region_data['height_nodelist']):
                    x_abs = j_col + region_data['leftbound']
                    noderange_val = int(region_data['noderange'][j_col])
                    for k_pair_idx in range(0, noderange_val, 2):
                        y_start_abs = int(nodelist_reshaped[j_col, k_pair_idx])
                        y_end_abs = int(nodelist_reshaped[j_col, k_pair_idx+1])
                        if x_abs >=0 and x_abs < final_unioned_roi.mask.shape[1]:
                             y_min_clip = max(0, y_start_abs)
                             y_max_clip = min(final_unioned_roi.mask.shape[0]-1, y_end_abs)
                             if y_min_clip <= y_max_clip:
                                final_unioned_roi.mask[y_min_clip:y_max_clip+1, x_abs] = True
        final_unioned_roi.type = 'union'
        return final_unioned_roi


    def update_roi(self, plot_u_np, plot_v_np, roi_plot_obj, new_image_shape, spacing, radius_dic, border_interp_cpp=20):
        """
        Updates the ROI based on displacement fields. Highly complex.
        Placeholder for now.
        """
        # 1. Extrapolate displacement plots (plot_u_np, plot_v_np) using roi_plot_obj.
        #    - This might involve calling a Python version of ncorr_alg_extrapdata or a bound C++ version.
        #    - Let's assume extrap_data_u_list = _ncorr_cpp_algs.extrap_data(numpy_to_cpp_double_array(plot_u_np), roi_plot_obj.to_cpp_roi(), border_interp_cpp)
        #      (returns list of CppClassDoubleArray, one per region in roi_plot_obj)

        # 2. Form B-spline coefficients for each extrapolated plot.
        #    - bcoef_u_list = [self._form_bcoef_py(np.array(ext_u_cpp, copy=True)) for ext_u_cpp in extrap_data_u_list]

        # 3. Iterate through self.boundary_data:
        #    updated_boundary_data = {'add': [], 'sub': []}
        #    for add_boundary_np in self.boundary_data['add']:
        #        scaled_boundary_np = add_boundary_np / (spacing + 1.0)
        #        # For each point in scaled_boundary_np, find which region of roi_plot_obj it falls into
        #        # Then use the corresponding bcoef_u and bcoef_v to interpolate displacements u_disp, v_disp
        #        # new_boundary_points_np = add_boundary_np + np.column_stack((u_disp_interp, v_disp_interp))
        #        # Filter points outside new_image_shape +/- radius_dic
        #        # updated_boundary_data['add'].append(filtered_new_boundary_points_np)
        #    # ... similar logic for 'sub' boundaries ...
        
        # 4. Create new NcorrROI instance:
        #    updated_roi = NcorrROI(image_shape=new_image_shape)
        #    updated_roi.boundary_data = updated_boundary_data
        #    # Then call form_mask and form_regions using this boundary_data (would need set_roi_from_boundary)
        #    # This part is complex as set_roi_from_boundary wasn't fully detailed.

        print("NcorrROI.update_roi is a complex method and is currently a placeholder.")
        # For now, return a new empty ROI or a copy of the current one.
        return NcorrROI(image_shape=new_image_shape)


    def to_cpp_roi(self):
        """
        Converts this NcorrROI to a CppNcorrClassRoi for C++ bindings.
        """
        cpp_roi = _ncorr_cpp_core.CppNcorrClassRoi()
        if self.mask is not None:
            cpp_roi.mask = self._numpy_to_cpp_logical_array(self.mask)
        
        cpp_regions_list = []
        for py_reg_data in self.regions:
            if py_reg_data.get('totalpoints', 0) > 0:
                cpp_reg = _ncorr_cpp_core.CppNcorrClassRegion()
                cpp_reg.leftbound = py_reg_data['leftbound']
                cpp_reg.rightbound = py_reg_data['rightbound']
                cpp_reg.upperbound = py_reg_data['upperbound']
                cpp_reg.lowerbound = py_reg_data['lowerbound']
                cpp_reg.totalpoints = py_reg_data['totalpoints']
                
                # nodelist and noderange (assuming they are NumPy arrays from _cpp_region_to_python)
                nl_np = py_reg_data['nodelist']
                nr_np = py_reg_data['noderange']

                if nl_np.size > 0:
                    h_nl, w_nl = nl_np.shape if nl_np.ndim == 2 else (nl_np.shape[0], 1 if nl_np.ndim ==1 else 0)
                    cpp_reg.nodelist.alloc(h_nl, w_nl)
                    cpp_reg.nodelist.set_value_numpy(nl_np.astype(np.int32))
                
                if nr_np.size > 0:
                    h_nr = nr_np.shape[0]
                    cpp_reg.noderange.alloc(h_nr, 1)
                    cpp_reg.noderange.set_value_numpy(nr_np.astype(np.int32).reshape(h_nr,1))
                
                cpp_regions_list.append(cpp_reg)
        cpp_roi.region = cpp_regions_list # Relies on pybind11 to convert Python list of CppNcorrClassRegion to std::vector

        # cirroi is not typically set from Python side for to_cpp_roi, but filled by C++ set_cirroi
        return cpp_roi

    def __repr__(self):
        return f"<NcorrROI type='{self.type}' num_regions={len(self.regions)} shape={self.mask.shape if self.mask is not None else None}>"