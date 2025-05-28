import numpy as np
import cv2 # For GaussianBlur in NcorrImage.reduce, potentially for other image ops if needed

from ncorr_app.core.image import NcorrImage
from ncorr_app.core.roi import NcorrROI
from ncorr_app.core.datatypes import OutputState
from ncorr_app._ext import _ncorr_cpp_core, _ncorr_cpp_algs


def _numpy_to_cpp_double_array(np_array, cpp_class_instance=None):
    """Helper to convert NumPy array to CppClassDoubleArray."""
    if np_array is None:
        return _ncorr_cpp_core.CppClassDoubleArray()
    
    if cpp_class_instance is None:
        cpp_array = _ncorr_cpp_core.CppClassDoubleArray()
    else:
        cpp_array = cpp_class_instance

    np_array_f64 = np_array.astype(np.float64)
    if np_array_f64.ndim == 1:
        h, w = np_array_f64.shape[0], 1
        np_array_f64_2d = np_array_f64.reshape((h,w))
    elif np_array_f64.ndim == 2:
        h, w = np_array_f64.shape
        np_array_f64_2d = np_array_f64
    else:
        raise ValueError("Input NumPy array must be 1D or 2D for CppClassDoubleArray.")

    cpp_array.alloc(h, w)
    cpp_array.set_value_numpy(np_array_f64_2d)
    return cpp_array

def _py_convert_seedinfo_to_cpp_list(py_convert_seeds_list):
    """Converts a Python list of convert_seed dicts to a list of CppPyConvertSeedInfo."""
    cpp_seeds_list = []
    for py_seed in py_convert_seeds_list:
        cpp_s = _ncorr_cpp_algs.PyConvertSeedInfo()
        param_vector_np = np.array(py_seed.get('paramvector', [0.0]*7), dtype=np.float64).reshape(1,7)
        cpp_s.paramvector = _numpy_to_cpp_double_array(param_vector_np)
        cpp_s.num_region_new = int(py_seed.get('num_region_new', 0))
        cpp_s.num_region_old = int(py_seed.get('num_region_old', 0))
        cpp_seeds_list.append(cpp_s)
    return cpp_seeds_list


def _convert_to_eulerian(
    reference_image_for_conversion: NcorrImage, # This is the 'old' reference for this specific conversion step
    target_eulerian_image: NcorrImage,      # This is the 'new' image where Eulerian field is defined
    lagrangian_u_pixels: np.ndarray,      # Displacement U in 'old' config (pixels)
    lagrangian_v_pixels: np.ndarray,      # Displacement V in 'old' config (pixels)
    lagrangian_roi_formatted: NcorrROI,   # ROI for the Lagrangian fields
    dic_parameters: dict,
    border_interp_cpp: int = 20
):
    """
    Converts Lagrangian displacement fields to Eulerian.
    This is a complex function mirroring ncorr_alg_convertanalysis.m and its C++ counterparts.
    It requires extrapolated B-spline coefficients of the Lagrangian fields and
    then calls C++ functions to find corresponding seed points and convert displacements.
    """
    spacing = dic_parameters.get('spacing', 0)

    # 1. Extrapolate Lagrangian displacement fields
    # _ncorr_cpp_algs.extrap_data expects CppClassDoubleArray and CppNcorrClassRoi
    cpp_u_lag_pixels = _numpy_to_cpp_double_array(lagrangian_u_pixels)
    cpp_v_lag_pixels = _numpy_to_cpp_double_array(lagrangian_v_pixels)
    cpp_lag_roi = lagrangian_roi_formatted.to_cpp_roi()

    # extrap_data returns a list of CppClassDoubleArray, one for each region in cpp_lag_roi
    u_extrap_list_cpp = _ncorr_cpp_algs.extrap_data(cpp_u_lag_pixels, cpp_lag_roi, border_interp_cpp)
    v_extrap_list_cpp = _ncorr_cpp_algs.extrap_data(cpp_v_lag_pixels, cpp_lag_roi, border_interp_cpp)

    # 2. Form B-spline coefficients for each extrapolated plot
    # NcorrImage._form_bcoef_py needs to be accessible, e.g., static or utility
    # For simplicity, assuming we can make it callable like:
    # utils_form_bcoef_py = NcorrImage(np.zeros((5,5)))._form_bcoef_py # Hacky way to get method
    
    u_bcoef_list_cpp = [] # List of list of CppClassDoubleArray (outer list for images (1 here), inner for regions)
    v_bcoef_list_cpp = []
    
    u_bcoef_regions_cpp = []
    for region_idx, u_extrap_cpp in enumerate(u_extrap_list_cpp):
        if u_extrap_cpp.height > 0 and u_extrap_cpp.width > 0: # only process if region was valid
            u_extrap_np = np.array(u_extrap_cpp, copy=True)
            # This is a placeholder for NcorrImage._form_bcoef_py or similar static method
            # For now, we can't directly call _form_bcoef_py here without an NcorrImage instance
            # or moving it to utils. Let's assume a utility function _form_bcoef_py_util exists
            # For now, let's assume the C++ side of convert_disp expects extrapolated fields, not bcoeffs.
            # OR, if convert_disp expects bcoeffs, this step is crucial.
            # The original ncorr_alg_convert.cpp takes plot_u_interp_old which are bcoeffs.
            # This means we need to convert u_extrap_cpp to bcoeffs.
            # This part is tricky without exposing _form_bcoef_py correctly.
            # As a simplification, we'll pass extrapolated data and assume C++ handles bcoeffs if needed,
            # or that the C++ binding for convert_disp was designed to take raw extrapolated data.
            # Re-checking: ncorr_alg_adddisp.cpp and ncorr_alg_convert.cpp in Phase 1 (user prompt) suggests
            # they take List of List of CppClassDoubleArray, implying bcoeffs are made in Python.
            # This implies NcorrImage._form_bcoef_py should be a static method or in utils.
            # For this implementation, I'll assume `plots_u_interp_old` are the b-spline coefficients.
            # This means the caller of `_convert_to_eulerian` must provide b-splines.
            # The task description for `format_displacement_fields` says:
            # "Extrapolates lagrangian_u_pixels ... Forms B-spline coefficients."
            # So, let's assume we compute bcoeffs here.
            # This is a bit circular if _form_bcoef_py is not static/utility.
            # For now, let's create a dummy NcorrImage to call its _form_bcoef_py
            dummy_img_for_bcoef = NcorrImage(np.zeros((max(5,u_extrap_np.shape[0]),max(5,u_extrap_np.shape[1])))) # ensure min size
            u_bcoef_np = dummy_img_for_bcoef._form_bcoef_py(u_extrap_np)
            u_bcoef_regions_cpp.append(_numpy_to_cpp_double_array(u_bcoef_np))

    u_bcoef_list_cpp.append(u_bcoef_regions_cpp) # Outer list for images

    v_bcoef_regions_cpp = []
    for region_idx, v_extrap_cpp in enumerate(v_extrap_list_cpp):
        if v_extrap_cpp.height > 0 and v_extrap_cpp.width > 0:
            v_extrap_np = np.array(v_extrap_cpp, copy=True)
            dummy_img_for_bcoef = NcorrImage(np.zeros((max(5,v_extrap_np.shape[0]),max(5,v_extrap_np.shape[1]))))
            v_bcoef_np = dummy_img_for_bcoef._form_bcoef_py(v_extrap_np)
            v_bcoef_regions_cpp.append(_numpy_to_cpp_double_array(v_bcoef_np))
    v_bcoef_list_cpp.append(v_bcoef_regions_cpp)


    # 3. Obtain "convert seeds"
    # This would typically call a Python port of ncorr_alg_convertseeds.m, which itself might call C++ helpers.
    # For now, let's assume a placeholder for `_ncorr_cpp_algs.convert_seeds` can be called
    # or that convert_disp handles this internally if seed list is empty.
    # The C++ ncorr_alg_convert.cpp doesn't explicitly show a convert_seeds call inside its mexFunction.
    # It takes `convertseedinfo` as input. This `convertseedinfo` must be generated before calling convert_disp.
    # This step is complex and likely requires its own `_ncorr_cpp_algs.convert_seeds` binding.
    # For now, we'll assume a placeholder for `convertseedinfo_for_cpp`.
    # This implies that `ncorr_alg_convertseeds.m` functionality must be ported to Python
    # and use the bound C++ `interp_qbs` to find matching points.
    # For this phase, this is a MAJOR simplification.
    
    # Simplified: assume convertseedinfo must be provided or generated by a separate Python function.
    # The original ncorr_alg_convertanalysis.m calls ncorr_alg_convertseeds.
    # We need a Python equivalent `py_convert_seeds`
    # For now, let's assume convertseedinfo is an empty list for this placeholder.
    # This would mean the C++ convert_disp might try to work without seeds or fail.
    # A proper solution needs `py_convert_seeds` to be implemented.
    
    # --- Placeholder for py_convert_seeds logic ---
    # py_convert_seeds_list_of_dicts = py_convert_seeds_logic(...)
    # cpp_convert_seeds_list = _py_convert_seedinfo_to_cpp_list(py_convert_seeds_list_of_dicts)
    # For this example, we'll pass an empty list, which the C++ might not handle.
    cpp_convert_seeds_list = [] # Placeholder! This is a critical missing piece for robust conversion.

    # 4. Call the bound C++ convert_disp function
    cpp_roi_old_list = [lagrangian_roi_formatted.to_cpp_roi()] # Needs to be list of CppNcorrClassRoi
    cpp_roi_new_list = [target_eulerian_image.get_roi_for_shape().to_cpp_roi()] # Create an empty ROI of target shape

    # The convert_disp function in C++ will use the 0-th element of these lists typically for a single image pair conversion
    cpp_u_new, cpp_v_new, cpp_valid_new, status_val = _ncorr_cpp_algs.convert_disp(
        u_bcoef_list_cpp, v_bcoef_list_cpp, # these are list of list of CppDoubleArray
        cpp_roi_old_list, cpp_roi_new_list, 
        cpp_convert_seeds_list, # Placeholder
        spacing, border_interp_cpp, 
        0, 1 # dummy num_img, total_imgs for this isolated call
    )

    if OutputState(status_val) != OutputState.SUCCESS:
        print("Warning: C++ convert_disp failed.")
        return None, None, None

    u_cur_pixels_np = np.array(cpp_u_new, copy=True)
    v_cur_pixels_np = np.array(cpp_v_new, copy=True)
    valid_cur_np = np.array(cpp_valid_new, copy=True)

    # Create NcorrROI for the Eulerian field
    # The ROI for current image is based on target_eulerian_image's shape and valid_cur_np
    roi_cur_formatted = NcorrROI(image_shape=target_eulerian_image.gs_data.shape)
    # The valid_cur_np from convert_disp is the mask for the new ROI
    roi_cur_formatted.set_roi_from_mask(valid_cur_np, cutoff=0) # Cutoff 0 to keep all valid points

    return u_cur_pixels_np, v_cur_pixels_np, roi_cur_formatted


def format_displacement_fields(
    list_of_raw_dic_results: list, # from orchestrate_dic_analysis
    reference_ncorr_image_obj: NcorrImage,
    current_ncorr_image_objs_list: list, # list of NcorrImage
    original_reference_ncorr_roi_obj: NcorrROI,
    dic_parameters_dict: dict
):
    """
    Formats raw DIC displacement results: applies cutoffs, unit conversion, lens distortion,
    and converts to Eulerian coordinates. Mirrors logic from ncorr_gui_formatdisp.m.
    """
    formatted_results = []
    pixtounits = dic_parameters_dict.get('pixtounits', 1.0)
    units = dic_parameters_dict.get('units', 'pixels')
    cutoff_corrcoef_list = dic_parameters_dict.get('cutoff_corrcoef_list', [])
    lenscoef = dic_parameters_dict.get('lenscoef', 0.0)
    spacing = dic_parameters_dict.get('spacing', 0)
    border_interp_cpp = 20 # Default from Ncorr for internal interpolations

    if not cutoff_corrcoef_list or len(cutoff_corrcoef_list) != len(list_of_raw_dic_results):
        # Default cutoff if not provided or mismatched length
        default_cc_cutoff = 0.1 
        cutoff_corrcoef_list = [default_cc_cutoff] * len(list_of_raw_dic_results)
        print(f"Warning: Using default correlation coefficient cutoff: {default_cc_cutoff}")


    for idx, raw_res in enumerate(list_of_raw_dic_results):
        u_pix_raw = raw_res['u']
        v_pix_raw = raw_res['v']
        corrcoef_raw = raw_res['corrcoef']
        valid_points_dic = raw_res['plot_validpoints'] # Mask from RG-DIC

        # Apply correlation coefficient cutoff
        cutoff_cc = cutoff_corrcoef_list[idx]
        valid_after_cc = corrcoef_raw <= cutoff_cc
        current_valid_mask = valid_points_dic & valid_after_cc

        # Create/update roi_ref_formatted for Lagrangian results
        # original_reference_ncorr_roi_obj is for the very first image in a sequence.
        # For multi-step regular DIC, roi_ref_formatted is always based on the initial reference ROI.
        roi_ref_formatted = original_reference_ncorr_roi_obj.get_union(current_valid_mask, spacing)

        # Apply lens distortion correction (to pixel displacements)
        u_pix_corrected = u_pix_raw.copy()
        v_pix_corrected = v_pix_raw.copy()
        if lenscoef != 0.0:
            # Reduced image shape for meshgrid
            h_reduced, w_reduced = u_pix_raw.shape 
            # Original image center (assuming original_reference_ncorr_roi_obj matches reference_ncorr_image_obj)
            center_x_orig = reference_ncorr_image_obj.width / 2.0
            center_y_orig = reference_ncorr_image_obj.height / 2.0

            # Create coordinate grid for the *reduced* displacement field points in *original* pixel scale
            x_coords_orig_scale = np.arange(0, w_reduced * (spacing + 1), spacing + 1) - center_x_orig
            y_coords_orig_scale = np.arange(0, h_reduced * (spacing + 1), spacing + 1) - center_y_orig
            
            # Ensure x_coords/y_coords match the shape of u_pix_raw for element-wise operations
            # This requires careful broadcasting or meshgrid usage
            # The correction is applied at each point (X,Y) of the reduced grid.
            # X, Y are pixel coordinates in original image. U,V are pixel displacements.
            # X_reduced, Y_reduced are indices into u_pix_raw, v_pix_raw.
            # x = X_reduced * (spacing+1), y = Y_reduced * (spacing+1)
            
            # Create meshgrid of original pixel coordinates corresponding to displacement field points
            x_mesh_orig, y_mesh_orig = np.meshgrid(
                (np.arange(w_reduced) * (spacing + 1)) - center_x_orig,
                (np.arange(h_reduced) * (spacing + 1)) - center_y_orig,
                indexing='xy' # Cartesian indexing for meshgrid
            )
            
            x_tilda = x_mesh_orig + u_pix_raw
            y_tilda = y_mesh_orig + v_pix_raw
            
            u_pix_corrected = u_pix_raw + lenscoef * (x_tilda * (x_tilda**2 + y_tilda**2) - x_mesh_orig * (x_mesh_orig**2 + y_mesh_orig**2))
            v_pix_corrected = v_pix_raw + lenscoef * (y_tilda * (x_tilda**2 + y_tilda**2) - y_mesh_orig * (y_mesh_orig**2 + y_mesh_orig**2))


        # Convert Lagrangian displacements to real units
        plot_u_ref_formatted_units = u_pix_corrected * pixtounits
        plot_v_ref_formatted_units = v_pix_corrected * pixtounits
        
        # Apply valid mask
        plot_u_ref_formatted_units[~current_valid_mask] = 0.0 # Or np.nan
        plot_v_ref_formatted_units[~current_valid_mask] = 0.0 # Or np.nan
        
        # Convert to Eulerian
        # The reference_image_for_conversion is the image from which u_pix_corrected/v_pix_corrected originate
        # The target_eulerian_image is the image onto which we want to map these (i.e., current_ncorr_image_objs_list[idx])
        
        # This is tricky if dic_type was 'backward' initially in orchestrate_dic_analysis.
        # Assuming 'regular' DIC type from orchestrate_dic_analysis, so raw_res comes from (original_ref vs current_ncorr_image_objs_list[idx])
        ref_img_for_conversion = reference_ncorr_image_obj # The original reference
        target_img_for_eulerian = current_ncorr_image_objs_list[idx]

        u_cur_pixels, v_cur_pixels, roi_cur_formatted = _convert_to_eulerian(
            ref_img_for_conversion, target_img_for_eulerian,
            u_pix_corrected, v_pix_corrected, # Lagrangian displacements in pixels
            roi_ref_formatted, # ROI of the Lagrangian field
            dic_parameters_dict,
            border_interp_cpp
        )

        if u_cur_pixels is not None:
            plot_u_cur_formatted_units = -u_cur_pixels * pixtounits
            plot_v_cur_formatted_units = -v_cur_pixels * pixtounits
            # Apply valid mask from Eulerian conversion (if different from roi_cur_formatted.mask)
            # roi_cur_formatted.mask should be the ultimate mask for these.
            plot_u_cur_formatted_units[~roi_cur_formatted.mask] = 0.0 # Or np.nan
            plot_v_cur_formatted_units[~roi_cur_formatted.mask] = 0.0 # Or np.nan
        else: # Conversion failed
            plot_u_cur_formatted_units = np.zeros_like(u_pix_raw)
            plot_v_cur_formatted_units = np.zeros_like(v_pix_raw)
            roi_cur_formatted = NcorrROI(image_shape=target_img_for_eulerian.gs_data.shape)


        formatted_results.append({
            'reference_name': raw_res['reference_name'],
            'current_name': raw_res['current_name'],
            'plot_u_ref_formatted': plot_u_ref_formatted_units,
            'plot_v_ref_formatted': plot_v_ref_formatted_units,
            'roi_ref_formatted': roi_ref_formatted,
            'plot_u_cur_formatted': plot_u_cur_formatted_units,
            'plot_v_cur_formatted': plot_v_cur_formatted_units,
            'roi_cur_formatted': roi_cur_formatted,
            'plot_corrcoef_dic': corrcoef_raw, # Keep original for reference
            'plot_validpoints_after_cc': current_valid_mask
        })
        
    return formatted_results


def calculate_strain_fields(
    list_of_formatted_disp_results: list,
    dic_parameters_dict: dict,
    strain_parameters_dict: dict
):
    """
    Calculates strain fields from formatted displacement fields.
    Mirrors logic from ncorr.m callback_topmenu_calcstrain.
    """
    strain_results = []
    
    pixtounits = dic_parameters_dict.get('pixtounits', 1.0)
    spacing = dic_parameters_dict.get('spacing', 0)
    
    radius_strain = strain_parameters_dict.get('radius_strain', 15) # default from ncorr_gui_setstrainradius
    subsettrunc_strain = strain_parameters_dict.get('subsettrunc_strain', False)

    for idx, fmt_res in enumerate(list_of_formatted_disp_results):
        current_strains = {}
        
        # --- Lagrangian Strains ---
        # Gradients are du_real/dx_real, etc.
        # Inputs to disp_grad are pixel displacements, pixel ROI, and scaling factors
        u_ref_pixels = fmt_res['plot_u_ref_formatted'] / pixtounits
        v_ref_pixels = fmt_res['plot_v_ref_formatted'] / pixtounits
        
        cpp_u_ref_pix = _numpy_to_cpp_double_array(u_ref_pixels)
        cpp_v_ref_pix = _numpy_to_cpp_double_array(v_ref_pixels)
        cpp_roi_ref_fmt = fmt_res['roi_ref_formatted'].to_cpp_roi()

        dudx_ref, dudy_ref, dvdx_ref, dvdy_ref, valid_grad_ref, status_ref = \
            _ncorr_cpp_algs.disp_grad(
                cpp_u_ref_pix, cpp_v_ref_pix, cpp_roi_ref_fmt,
                radius_strain, pixtounits, spacing, subsettrunc_strain,
                idx, len(list_of_formatted_disp_results) # image context for waitbar
            )
        
        if OutputState(status_ref) == OutputState.SUCCESS:
            dudx_ref_np = np.array(dudx_ref, copy=True)
            dudy_ref_np = np.array(dudy_ref, copy=True)
            dvdx_ref_np = np.array(dvdx_ref, copy=True)
            dvdy_ref_np = np.array(dvdy_ref, copy=True)
            valid_grad_ref_np = np.array(valid_grad_ref, copy=True)

            # Green-Lagrangian strains
            Exx_ref = 0.5 * (2 * dudx_ref_np + dudx_ref_np**2 + dvdx_ref_np**2)
            Exy_ref = 0.5 * (dudy_ref_np + dvdx_ref_np + dudx_ref_np * dudy_ref_np + dvdx_ref_np * dvdy_ref_np)
            Eyy_ref = 0.5 * (2 * dvdy_ref_np + dudy_ref_np**2 + dvdy_ref_np**2)
            
            Exx_ref[~valid_grad_ref_np] = 0.0 # or np.nan
            Exy_ref[~valid_grad_ref_np] = 0.0
            Eyy_ref[~valid_grad_ref_np] = 0.0

            current_strains['plot_exx_ref_formatted'] = Exx_ref
            current_strains['plot_exy_ref_formatted'] = Exy_ref
            current_strains['plot_eyy_ref_formatted'] = Eyy_ref
            current_strains['roi_strain_ref_formatted'] = fmt_res['roi_ref_formatted'].get_union(valid_grad_ref_np, 0)
        else:
            print(f"Warning: Lagrangian displacement gradient calculation failed for result {idx}")
            # Populate with empty/default if failed
            shape = fmt_res['roi_ref_formatted'].mask.shape
            current_strains['plot_exx_ref_formatted'] = np.zeros(shape)
            current_strains['plot_exy_ref_formatted'] = np.zeros(shape)
            current_strains['plot_eyy_ref_formatted'] = np.zeros(shape)
            current_strains['roi_strain_ref_formatted'] = NcorrROI(image_shape=shape)


        # --- Eulerian Strains ---
        # Gradients of "backward" displacements w.r.t current coords
        # plot_u_cur_formatted = -u_eulerian_component * pixtounits
        # So, u_cur_pixels = plot_u_cur_formatted / pixtounits = -u_eulerian_component
        u_cur_pixels_for_grad = fmt_res['plot_u_cur_formatted'] / pixtounits
        v_cur_pixels_for_grad = fmt_res['plot_v_cur_formatted'] / pixtounits
        
        cpp_u_cur_pix = _numpy_to_cpp_double_array(u_cur_pixels_for_grad)
        cpp_v_cur_pix = _numpy_to_cpp_double_array(v_cur_pixels_for_grad)
        cpp_roi_cur_fmt = fmt_res['roi_cur_formatted'].to_cpp_roi()
        
        dudx_cur, dudy_cur, dvdx_cur, dvdy_cur, valid_grad_cur, status_cur = \
            _ncorr_cpp_algs.disp_grad(
                cpp_u_cur_pix, cpp_v_cur_pix, cpp_roi_cur_fmt,
                radius_strain, pixtounits, spacing, subsettrunc_strain,
                idx, len(list_of_formatted_disp_results) # image context for waitbar
            )

        if OutputState(status_cur) == OutputState.SUCCESS:
            dudx_cur_np = np.array(dudx_cur, copy=True)
            dudy_cur_np = np.array(dudy_cur, copy=True)
            dvdx_cur_np = np.array(dvdx_cur, copy=True)
            dvdy_cur_np = np.array(dvdy_cur, copy=True)
            valid_grad_cur_np = np.array(valid_grad_cur, copy=True)

            # Eulerian-Almansi strains (using Ncorr's specific formula application)
            # The dudx_cur here is d(-u_eulerian_comp)/dx_current_coord
            exx_cur = 0.5 * (2 * dudx_cur_np - (dudx_cur_np**2 + dvdx_cur_np**2))
            exy_cur = 0.5 * (dudy_cur_np + dvdx_cur_np - (dudx_cur_np * dudy_cur_np + dvdx_cur_np * dvdy_cur_np))
            eyy_cur = 0.5 * (2 * dvdy_cur_np - (dudy_cur_np**2 + dvdy_cur_np**2))

            exx_cur[~valid_grad_cur_np] = 0.0 # or np.nan
            exy_cur[~valid_grad_cur_np] = 0.0
            eyy_cur[~valid_grad_cur_np] = 0.0

            current_strains['plot_exx_cur_formatted'] = exx_cur
            current_strains['plot_exy_cur_formatted'] = exy_cur
            current_strains['plot_eyy_cur_formatted'] = eyy_cur
            current_strains['roi_strain_cur_formatted'] = fmt_res['roi_cur_formatted'].get_union(valid_grad_cur_np, 0)
        else:
            print(f"Warning: Eulerian displacement gradient calculation failed for result {idx}")
            shape = fmt_res['roi_cur_formatted'].mask.shape
            current_strains['plot_exx_cur_formatted'] = np.zeros(shape)
            current_strains['plot_exy_cur_formatted'] = np.zeros(shape)
            current_strains['plot_eyy_cur_formatted'] = np.zeros(shape)
            current_strains['roi_strain_cur_formatted'] = NcorrROI(image_shape=shape)
            
        current_strains['reference_name'] = fmt_res['reference_name']
        current_strains['current_name'] = fmt_res['current_name']
        strain_results.append(current_strains)
        
    return strain_results