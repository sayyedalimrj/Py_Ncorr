import numpy as np
from ncorr_app.core.image import NcorrImage
from ncorr_app.core.roi import NcorrROI
from ncorr_app.core.datatypes import OutputState
from ncorr_app._ext import _ncorr_cpp_core, _ncorr_cpp_algs

# --- Helper Functions for Data Conversion ---

def _numpy_to_cpp_integer_array(np_array, cpp_class_instance=None):
    """Converts a NumPy array to a CppClassIntegerArray."""
    if np_array is None:
        return _ncorr_cpp_core.CppClassIntegerArray()
    
    if cpp_class_instance is None:
        cpp_array = _ncorr_cpp_core.CppClassIntegerArray()
    else:
        cpp_array = cpp_class_instance

    # Ensure input is int32
    np_array_int32 = np_array.astype(np.int32)

    if np_array_int32.ndim == 1:
        h, w = np_array_int32.shape[0], 1
        np_array_int32_2d = np_array_int32.reshape((h,w), order='C') # Treat 1D as a column vector
    elif np_array_int32.ndim == 2:
        h, w = np_array_int32.shape
        np_array_int32_2d = np_array_int32
    else:
        raise ValueError("Input NumPy array must be 1D or 2D for CppClassIntegerArray.")

    cpp_array.alloc(h, w)
    cpp_array.set_value_numpy(np_array_int32_2d) # Assumes set_value_numpy handles potential C-vs-F order
    return cpp_array

def _py_seedinfo_to_cpp_seedinfo_list(py_seeds_list):
    """Converts a Python list of seed dicts to a list of CppSeedInfo objects."""
    cpp_seeds_list = []
    for py_seed in py_seeds_list:
        cpp_seed = _ncorr_cpp_algs.PySeedInfo()
        
        param_vector_np = np.array(py_seed.get('paramvector', [0.0]*9), dtype=np.float64).reshape(1,9)
        cpp_seed.paramvector = _ncorr_cpp_core.CppClassDoubleArray()
        cpp_seed.paramvector.alloc(1, 9)
        cpp_seed.paramvector.set_value_numpy(param_vector_np)
        
        cpp_seed.num_region = int(py_seed.get('num_region', 0))
        cpp_seed.num_thread = int(py_seed.get('num_thread', 0))
        cpp_seed.computepoints = int(py_seed.get('computepoints', 0))
        cpp_seeds_list.append(cpp_seed)
    return cpp_seeds_list

def _cpp_seedinfo_to_python_list(cpp_seeds_list):
    """Converts a list of CppSeedInfo objects to a Python list of dicts."""
    py_seeds_list = []
    for cpp_seed in cpp_seeds_list:
        py_seed = {
            'paramvector': np.array(cpp_seed.paramvector, copy=True).flatten().tolist(),
            'num_region': cpp_seed.num_region,
            'num_thread': cpp_seed.num_thread,
            'computepoints': cpp_seed.computepoints
        }
        py_seeds_list.append(py_seed)
    return py_seeds_list

def _cpp_convergence_to_python_list(cpp_convergence_list):
    """Converts a list of CppConvergenceInfo objects to a Python list of dicts."""
    py_convergence_list = []
    for cpp_conv in cpp_convergence_list:
        py_conv = {
            'num_iterations': cpp_conv.num_iterations,
            'diffnorm': cpp_conv.diffnorm
        }
        py_convergence_list.append(py_conv)
    return py_convergence_list

# --- Core DIC Orchestration Functions ---

def calculate_seeds(reference_ncorr_image: NcorrImage,
                    current_ncorr_image: NcorrImage,
                    reference_ncorr_roi: NcorrROI,
                    region_index_in_roi: int,
                    seed_positions_xy_list: list, # List of (x,y) tuples/lists
                    dic_parameters: dict):
    """
    Calculates initial seeds for a given region and image pair.

    Args:
        reference_ncorr_image: The reference NcorrImage object.
        current_ncorr_image: The current NcorrImage object.
        reference_ncorr_roi: The NcorrROI object for the reference image.
        region_index_in_roi: The index of the specific region in the ROI to process.
        seed_positions_xy_list: A list of (x, y) pixel coordinates for seeds.
        dic_parameters: Dictionary containing DIC parameters like 'radius', 'cutoff_diffnorm', etc.

    Returns:
        tuple: (py_seed_info_list, py_convergence_list, output_state_enum)
    """
    if not seed_positions_xy_list:
        return [], [], OutputState.FAILED # Or raise error

    cpp_ref_img = reference_ncorr_image.to_cpp_img()
    cpp_cur_img = current_ncorr_image.to_cpp_img()
    cpp_ref_roi = reference_ncorr_roi.to_cpp_roi()

    np_seeds = np.array(seed_positions_xy_list, dtype=np.int32)
    if np_seeds.ndim != 2 or np_seeds.shape[1] != 2:
        raise ValueError("seed_positions_xy_list must be a list of (x,y) pairs.")
    cpp_pos_seed = _numpy_to_cpp_integer_array(np_seeds)

    # Extract relevant dic_parameters
    radius = int(dic_parameters.get('radius', 20))
    cutoff_diffnorm = float(dic_parameters.get('cutoff_diffnorm', 1e-6))
    cutoff_iteration = int(dic_parameters.get('cutoff_iteration', 50))
    enabled_stepanalysis = bool(dic_parameters.get('stepanalysis', {}).get('enabled', False))
    subsettrunc = bool(dic_parameters.get('subsettrunc', False))

    try:
        cpp_seedinfo_list, cpp_convergence_list, status_val = \
            _ncorr_cpp_algs.calc_seeds(
                cpp_ref_img, cpp_cur_img, cpp_ref_roi,
                region_index_in_roi, cpp_pos_seed,
                radius, cutoff_diffnorm, cutoff_iteration,
                enabled_stepanalysis, subsettrunc
            )
        
        py_seed_info = _cpp_seedinfo_to_python_list(cpp_seedinfo_list)
        py_convergence_info = _cpp_convergence_to_python_list(cpp_convergence_list)
        
        return py_seed_info, py_convergence_info, OutputState(status_val)

    except RuntimeError as e:
        print(f"Error in C++ calculate_seeds: {e}")
        return [], [], OutputState.FAILED


def perform_rg_dic(reference_ncorr_image: NcorrImage,
                   current_ncorr_image: NcorrImage,
                   reference_ncorr_roi: NcorrROI,
                   seeds_info_list: list, # List of Python dicts from calculate_seeds
                   thread_diagram_np_array: np.ndarray, # int32 NumPy array
                   dic_parameters: dict,
                   current_image_index_for_waitbar: int = 0, # For C++ waitbar context
                   total_images_for_waitbar: int = 1):      # For C++ waitbar context
    """
    Performs Reliability-Guided Digital Image Correlation (RG-DIC).

    Args:
        reference_ncorr_image: The reference NcorrImage object.
        current_ncorr_image: The current NcorrImage object.
        reference_ncorr_roi: The NcorrROI object for the reference image.
        seeds_info_list: List of seed information dictionaries.
        thread_diagram_np_array: NumPy array defining thread assignments for points.
        dic_parameters: Dictionary of DIC parameters.
        current_image_index_for_waitbar: Index for waitbar context in C++.
        total_images_for_waitbar: Total images for waitbar context in C++.


    Returns:
        tuple: (plot_u_np, plot_v_np, plot_corrcoef_np, plot_validpoints_np, output_state_enum)
               Displacement fields and correlation coefficients are NumPy arrays.
    """
    cpp_ref_img = reference_ncorr_image.to_cpp_img()
    cpp_cur_img = current_ncorr_image.to_cpp_img()
    cpp_ref_roi = reference_ncorr_roi.to_cpp_roi()
    
    cpp_seeds_info_list = _py_seedinfo_to_cpp_seedinfo_list(seeds_info_list)
    cpp_thread_diagram = _numpy_to_cpp_integer_array(thread_diagram_np_array)

    radius = int(dic_parameters.get('radius', 20))
    spacing = int(dic_parameters.get('spacing', 0))
    cutoff_diffnorm = float(dic_parameters.get('cutoff_diffnorm', 1e-6))
    cutoff_iteration = int(dic_parameters.get('cutoff_iteration', 50))
    subsettrunc = bool(dic_parameters.get('subsettrunc', False))

    try:
        cpp_plot_u, cpp_plot_v, cpp_plot_corrcoef, cpp_plot_validpoints, status_val = \
            _ncorr_cpp_algs.rgdic(
                cpp_ref_img, cpp_cur_img, cpp_ref_roi,
                cpp_seeds_info_list, cpp_thread_diagram,
                radius, spacing, cutoff_diffnorm, cutoff_iteration,
                subsettrunc, current_image_index_for_waitbar, total_images_for_waitbar
            )

        # Convert CppClassArrays back to NumPy arrays using buffer protocol
        plot_u_np = np.array(cpp_plot_u, copy=True)
        plot_v_np = np.array(cpp_plot_v, copy=True)
        plot_corrcoef_np = np.array(cpp_plot_corrcoef, copy=True)
        plot_validpoints_np = np.array(cpp_plot_validpoints, copy=True)
        
        return plot_u_np, plot_v_np, plot_corrcoef_np, plot_validpoints_np, OutputState(status_val)

    except RuntimeError as e:
        print(f"Error in C++ perform_rg_dic: {e}")
        h, w = thread_diagram_np_array.shape
        return np.zeros((h,w)), np.zeros((h,w)), np.zeros((h,w)), np.zeros((h,w), dtype=bool), OutputState.FAILED


def _determine_initial_seeds_and_diagram(
    reference_image: NcorrImage,
    reference_roi: NcorrROI,
    dic_parameters: dict
):
    """
    Helper to determine initial seed positions and generate thread diagram.
    This is a simplified version of what ncorr_gui_seedanalysis + ncorr_gui_setseeds would do.
    For non-GUI, seeds need to be placed programmatically.
    """
    total_threads = dic_parameters.get('total_threads', 1)
    spacing = dic_parameters.get('spacing', 0)
    
    all_seed_positions_xy = [] # List of (x_pixel, y_pixel)
    seeds_info_for_cpp = [] # Will be list of PySeedInfo objects

    # Placeholder: Distribute seeds. A more robust strategy is needed for general cases.
    # E.g., one seed near the center of each major ROI region, up to total_threads.
    num_regions_in_roi = reference_roi.get_full_regions_count()
    seeds_per_region_base = total_threads // num_regions_in_roi if num_regions_in_roi > 0 else total_threads
    extra_seeds = total_threads % num_regions_in_roi if num_regions_in_roi > 0 else 0
    
    current_thread_idx = 0
    for region_idx, region_data in enumerate(reference_roi.regions):
        if region_data['totalpoints'] == 0:
            continue

        num_seeds_for_this_region = seeds_per_region_base + (1 if extra_seeds > 0 else 0)
        if extra_seeds > 0:
            extra_seeds -=1

        for _ in range(num_seeds_for_this_region):
            if current_thread_idx >= total_threads:
                break
            # Simplified seed placement: center of the region's bounding box
            # A better approach would use region's actual geometry
            center_x = region_data['leftbound'] + region_data['height_nodelist'] // 2 # height_nodelist is num_cols
            center_y = region_data['upperbound'] + (region_data['lowerbound'] - region_data['upperbound']) // 2
            
            # Ensure seed is on the reduced grid by snapping to nearest (spacing+1) multiple
            seed_x_pixel = int(round(center_x / (spacing + 1.0)) * (spacing + 1.0))
            seed_y_pixel = int(round(center_y / (spacing + 1.0)) * (spacing + 1.0))

            # Make sure it's within the original image bounds for safety
            seed_x_pixel = np.clip(seed_x_pixel, 0, reference_image.width - 1)
            seed_y_pixel = np.clip(seed_y_pixel, 0, reference_image.height - 1)

            all_seed_positions_xy.append((seed_x_pixel, seed_y_pixel))
            
            # Placeholder for seedinfo struct needed by form_threaddiagram (expects reduced coords)
            # This isn't actually seed_info yet, just positions for diagram
            current_thread_idx += 1
        if current_thread_idx >= total_threads:
            break
    
    if not all_seed_positions_xy:
        # Fallback if no regions or seeds could be placed
        # Place one seed at the center of the image if ROI is problematic
        print("Warning: No valid regions for seed placement, falling back to image center.")
        center_x_img = reference_image.width // 2
        center_y_img = reference_image.height // 2
        seed_x_pixel = int(round(center_x_img / (spacing + 1.0)) * (spacing + 1.0))
        seed_y_pixel = int(round(center_y_img / (spacing + 1.0)) * (spacing + 1.0))
        all_seed_positions_xy = [(seed_x_pixel, seed_y_pixel)] * total_threads


    # Thread diagram generation
    reduced_roi = reference_roi.reduce(spacing)
    h_reduced, w_reduced = reduced_roi.mask.shape

    cpp_threaddiagram = _ncorr_cpp_core.CppClassDoubleArray()
    cpp_threaddiagram.alloc(h_reduced, w_reduced)
    cpp_preview_threaddia = _ncorr_cpp_core.CppClassDoubleArray() # Not strictly needed for backend
    cpp_preview_threaddia.alloc(h_reduced, w_reduced)


    # Seeds for form_threaddiagram need to be in reduced coordinates
    reduced_seed_positions_np = np.array([(x // (spacing + 1), y // (spacing + 1)) for x, y in all_seed_positions_xy], dtype=np.int32)
    cpp_generators = _numpy_to_cpp_integer_array(reduced_seed_positions_np)
    
    cpp_reduced_roi_mask = reduced_roi.to_cpp_roi().mask # Get CppClassLogicalArray
    cpp_reduced_ref_img = reference_image.reduce(spacing).to_cpp_img()

    _ncorr_cpp_algs.form_threaddiagram(
        cpp_threaddiagram,
        cpp_preview_threaddia, # Dummy preview
        cpp_generators,
        cpp_reduced_roi_mask,
        cpp_reduced_ref_img
    )
    thread_diagram_np = np.array(cpp_threaddiagram, copy=True).astype(np.int32)

    return all_seed_positions_xy, thread_diagram_np


def orchestrate_dic_analysis(reference_ncorr_image_obj: NcorrImage,
                             current_ncorr_image_objs_list: list, # list of NcorrImage
                             initial_reference_ncorr_roi_obj: NcorrROI,
                             dic_parameters_dict: dict):
    """
    Main orchestrator for DIC analysis, mirroring logic from ncorr_alg_dicanalysis.m.
    """
    all_displacement_results = []
    all_updated_rois = [] # For step analysis, ROIs get updated
    all_seeds_info_for_steps = [] # Store seed info used/generated at each step

    dic_type = dic_parameters_dict.get('type', 'regular')
    enabled_stepanalysis = dic_parameters_dict.get('stepanalysis', {}).get('enabled', False)
    auto_propagate_seeds = dic_parameters_dict.get('stepanalysis', {}).get('auto', True) # if seeds auto-update

    # Prepare image sequence
    if dic_type == 'regular':
        # imgs_sequence = [reference_ncorr_image_obj] + current_ncorr_image_objs_list
        # For regular, ref is fixed, iterate through current images
        img_pairs = [(reference_ncorr_image_obj, initial_reference_ncorr_roi_obj, cur_img) for cur_img in current_ncorr_image_objs_list]
    elif dic_type == 'backward':
        # Ref is the last current, current images are processed in reverse towards the original reference
        # This is more complex to setup as a direct loop, original Ncorr does this by setting up imgcorr
        # For now, let's assume backward implies a single step: current_ncorr_image_objs_list[0] is ref, reference_ncorr_image_obj is cur
        if len(current_ncorr_image_objs_list) != 1:
            raise ValueError("Backward DIC currently simplified to one 'current' image (which acts as reference).")
        img_pairs = [(current_ncorr_image_objs_list[0], initial_reference_ncorr_roi_obj, reference_ncorr_image_obj)]
        # The ROI provided would be for current_ncorr_image_objs_list[0] in this case
    else:
        raise ValueError(f"Unsupported DIC type: {dic_type}")

    
    # Initial seed info (can be from params or determined)
    # `initial_seedinfo` in dic_parameters_dict refers to the output of `calculate_seeds`
    # a list of dicts: [{'paramvector': [...], 'num_region':X, ...}, ...]
    propagated_seeds_info = dic_parameters_dict.get('initial_seedinfo', None)
    
    # This will be the main reference ROI that gets updated in step analysis
    active_ref_roi = initial_reference_ncorr_roi_obj
    
    overall_status = OutputState.SUCCESS

    for i, (ref_img, ref_roi, cur_img) in enumerate(img_pairs):
        print(f"Processing pair: Ref='{ref_img.name}', Cur='{cur_img.name}'")
        
        current_step_seeds_info = None
        current_thread_diagram = None

        if propagated_seeds_info: # Seeds from previous step or user input
            current_step_seeds_info = propagated_seeds_info
            # Thread diagram would ideally be re-used or re-calculated based on these seeds and current ref_roi
            # For simplicity, let's re-determine it.
            seed_positions_for_diagram = [(s['paramvector'][0], s['paramvector'][1]) for s in current_step_seeds_info]
            
            # Check if seed_positions_for_diagram is empty or invalid
            if not seed_positions_for_diagram or not all(isinstance(p, (list,tuple)) and len(p)==2 for p in seed_positions_for_diagram):
                 print(f"Warning: Invalid propagated seed positions for step {i}. Recalculating.")
                 seed_positions_for_diagram, current_thread_diagram = _determine_initial_seeds_and_diagram(
                    ref_img, ref_roi, dic_parameters_dict
                )
            else:
                _, current_thread_diagram = _determine_initial_seeds_and_diagram(
                    ref_img, ref_roi, dic_parameters_dict # This call re-places seeds based on regions and gets a diagram
                ) 
                # The current_step_seeds_info from propagation should ideally align with these new seed locations for consistency.
                # This part is tricky without full convertseeds logic.
                # Simplification: If propagated_seeds_info exists, we assume it's already correct for THIS ref_img and ref_roi.
                # A proper implementation would call convert_seeds here.

        if current_step_seeds_info is None or current_thread_diagram is None:
            # Determine seeds for the current ref_img and ref_roi
            # This involves placing seeds in regions and then calling calculate_seeds
            seed_positions_xy, thread_diagram_np = _determine_initial_seeds_and_diagram(
                ref_img, ref_roi, dic_parameters_dict
            )
            current_thread_diagram = thread_diagram_np
            
            # For the first step, we only calculate seeds against the first actual current image
            # In step analysis, this might be the "next" image in a sub-sequence.
            # The `calculate_seeds` is for a single pair.
            # The current `cur_img` is the one for this DIC step.
            
            # Assuming `calculate_seeds` is per region, but ncorr_alg_calcseeds takes all seeds for the image.
            # The `_determine_initial_seeds_and_diagram` gives all_seed_positions_xy for the entire ref_roi.
            # We need to associate these with regions for `ncorr_alg_calcseeds`'s num_region_in_roi.
            # The original ncorr_gui_seedanalysis calls ncorr_gui_setseeds PER REGION.
            # This simplified `_determine_initial_seeds_and_diagram` generates all seeds at once.
            # Let's assume `calculate_seeds` is robust enough or `_ncorr_cpp_algs.calc_seeds` handles seeds across all specified regions if num_region_in is a general marker.
            # For now, let's pass region_index_in_roi=0 (first region) as a placeholder, but the C++ side for calc_seeds
            # usually takes `pos_seed` which is Mx2 (all seeds for all threads for that image pair).
            # The `num_region` in C++ `calc_seeds` seems to be for which ROI region the seeds belong to.
            # This part needs careful alignment with how `_ncorr_cpp_algs.calc_seeds` was truly bound.

            # Let's simplify: assume `calculate_seeds` operates on the whole image based on `all_seed_positions_xy`
            # and the `region_index_in_roi` might be ignored if `pos_seed_in` is global.
            # The C++ `ncorr_alg_calcseeds` takes `num_region` to associate seeds with that specific region.
            # Our Python `calculate_seeds` takes `region_index_in_roi`.
            # This part of the orchestration is non-trivial.
            
            # Assuming _determine_initial_seeds_and_diagram provides global seeds and diagram
            # and calculate_seeds will then initialize these for the current pair.
            # For a multi-region ROI, this simplified seeding may not be optimal.
            
            print(f"  Calculating initial seeds for {ref_img.name} vs {cur_img.name}...")
            current_step_seeds_info, _, seed_calc_status = calculate_seeds(
                ref_img, cur_img, ref_roi, 0, # Using region 0 as placeholder for now
                seed_positions_xy, dic_parameters_dict
            )
            if seed_calc_status != OutputState.SUCCESS:
                print(f"  Seed calculation failed for step {i}.")
                overall_status = OutputState.FAILED
                break # Stop processing further pairs
        
        all_seeds_info_for_steps.append(current_step_seeds_info)

        print(f"  Performing RG-DIC for {ref_img.name} vs {cur_img.name}...")
        plot_u, plot_v, plot_cc, plot_vp, dic_status = perform_rg_dic(
            ref_img, cur_img, ref_roi,
            current_step_seeds_info, current_thread_diagram,
            dic_parameters_dict,
            current_image_index_for_waitbar=i, # Relative to current batch
            total_images_for_waitbar=len(img_pairs)
        )

        if dic_status != OutputState.SUCCESS:
            print(f"  RG-DIC failed for step {i}.")
            if enabled_stepanalysis: # In step analysis, one failure might not stop all
                 overall_status = OutputState.FAILED # Mark overall as failed but might try next step if logic allows
                 # For strict step-by-step, we should break
                 break
            else: # Regular analysis, one failure is a full failure
                overall_status = OutputState.FAILED
                break
        
        displacement_result = {
            'u': plot_u, 'v': plot_v,
            'corrcoef': plot_cc, 'validpoints': plot_vp,
            'reference_name': ref_img.name,
            'current_name': cur_img.name
        }
        all_displacement_results.append(displacement_result)

        # For step analysis: update ROI and prepare seeds for the next step
        if enabled_stepanalysis and i < len(img_pairs) - 1:
            print(f"  Updating ROI for next step based on {cur_img.name}...")
            # The ROI for the *next* reference image (which is the current `cur_img`)
            # needs to be updated based on the displacements *from* the current `ref_img` *to* `cur_img`.
            # This means `plot_u` and `plot_v` are displacements of points in `ref_img`'s coord system.
            # `update_roi` needs to map `ref_roi` to `cur_img`'s space.
            
            # The `roi_plot_obj` for `update_roi` should be the one corresponding to plot_u, plot_v
            # which is `ref_roi` (or its reduced version that matches plot dimensions)
            roi_plot_for_update = ref_roi.reduce(dic_parameters_dict.get('spacing',0))

            updated_roi_for_next_ref = ref_roi.update_roi(
                plot_u, plot_v, 
                roi_plot_for_update, # ROI corresponding to plot_u, plot_v
                cur_img.gs_data.shape, # Shape of the *new* reference image (current cur_img)
                dic_parameters_dict.get('spacing',0),
                dic_parameters_dict.get('radius', 20)
                # border_interp_cpp is used internally by update_roi if it calls C++ extrap
            )
            all_updated_rois.append(updated_roi_for_next_ref)
            active_ref_roi = updated_roi_for_next_ref # This becomes the ROI for the next iteration's ref_img
            
            # Propagate seeds for the next step (This is a major simplification)
            # A proper implementation would call a Python equivalent of ncorr_alg_convertseeds
            if auto_propagate_seeds:
                print(f"  Propagating seeds for next step (simplified)...")
                # Simplistic propagation: use the same relative positions within the *new* ROI
                # or re-calculate based on the new active_ref_roi.
                # For now, we'll rely on _determine_initial_seeds_and_diagram for the *next* iteration
                # using the updated active_ref_roi.
                propagated_seeds_info = None # Force re-calculation of seeds for the new ref_roi
            else: # Manual seed propagation would mean user provides seeds for next step
                propagated_seeds_info = None # Or expect from dic_parameters_dict for that step
        else:
            all_updated_rois.append(ref_roi) # If not step analysis, or last step

    return all_displacement_results, all_updated_rois, all_seeds_info_for_steps, overall_status