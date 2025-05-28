import os
import json
import numpy as np
from celery import Celery
from celery.utils.log import get_task_logger

# Assuming ncorr_app is in PYTHONPATH or installed
from ncorr_app.core.image import NcorrImage
from ncorr_app.core.roi import NcorrROI
from ncorr_app.core.datatypes import OutputState
from ncorr_app.algorithms.dic_proc import orchestrate_dic_analysis
from ncorr_app.algorithms.post_proc import format_displacement_fields, calculate_strain_fields

# Configure Celery
# Replace with your actual broker URL (e.g., Redis, RabbitMQ)
# For local development with Redis: CELERY_BROKER_URL = 'redis://localhost:6379/0'
# The result backend is also set to Redis to store task states and results metadata.
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'ncorr_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Optional: Route tasks to a specific queue
    # task_routes={'webapp.tasks.run_full_dic_pipeline_task': {'queue': 'dic_processing'}},
)

logger = get_task_logger(__name__)

# Define a base directory for storing results
RESULTS_BASE_DIR = os.path.join(os.path.dirname(__file__), 'task_results')
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

def serialize_results(results_dict, task_id):
    """Serializes NumPy arrays and other data to JSON-friendly formats."""
    serialized = {}
    for key, value in results_dict.items():
        if isinstance(value, np.ndarray):
            # For large arrays, consider saving to .npy and storing path
            # For simplicity here, convert to list for JSON if small, else error or save path
            if value.size > 10000: # Example threshold
                 # Create a subdirectory for this task's files
                task_results_dir = os.path.join(RESULTS_BASE_DIR, task_id)
                os.makedirs(task_results_dir, exist_ok=True)
                filename = f"{key}.npy"
                filepath = os.path.join(task_results_dir, filename)
                np.save(filepath, value)
                serialized[key] = {"type": "numpy_file", "path": filepath}
            else:
                serialized[key] = {"type": "numpy_array", "data": value.tolist()}
        elif isinstance(value, (NcorrImage, NcorrROI)):
            # These objects might be too complex for direct JSON.
            # Store relevant serializable data or a path to a pickled object.
            # For now, just a placeholder.
            serialized[key] = {"type": str(type(value)), "name": getattr(value, 'name', 'N/A')}
        elif isinstance(value, OutputState):
            serialized[key] = {"type": "OutputState", "name": value.name, "value": value.value}
        else:
            try: # Check if directly JSON serializable
                json.dumps(value)
                serialized[key] = value
            except TypeError:
                serialized[key] = {"type": str(type(value)), "error": "Not directly JSON serializable"}
    return serialized

@celery_app.task(bind=True)
def run_full_dic_pipeline_task(self, image_paths_dict, roi_data_dict, dic_params_dict, strain_params_dict):
    """
    Celery task to run the full Ncorr DIC pipeline.
    """
    task_id = self.request.id
    logger.info(f"Starting DIC pipeline for task {task_id}")
    self.update_state(state='PROGRESS', meta={'status': 'Loading images and ROI...'})

    try:
        # 1. Load reference and current images
        ref_img_path = image_paths_dict.get('reference')
        cur_img_paths = image_paths_dict.get('currents', [])
        if not ref_img_path or not cur_img_paths:
            raise ValueError("Reference image path and current image paths are required.")

        reference_image = NcorrImage(source=ref_img_path)
        current_images = [NcorrImage(source=p) for p in cur_img_paths]
        logger.info(f"Task {task_id}: Images loaded.")

        # 2. Create/set initial reference ROI
        # Assuming roi_data_dict contains necessary info for NcorrROI constructor
        # e.g., {'type': 'draw', 'draw_objects': [...], 'image_shape': reference_image.gs_data.shape}
        # or {'type': 'mask_file', 'path': 'path/to/mask.png'}
        
        initial_roi = NcorrROI(image_shape=reference_image.gs_data.shape) # Start with empty ROI of correct shape
        roi_type = roi_data_dict.get('type', 'mask_array_if_present')
        
        if roi_type == 'draw' and 'draw_objects' in roi_data_dict:
            initial_roi.set_roi_from_drawings(
                roi_data_dict['draw_objects'],
                reference_image.gs_data.shape,
                cutoff=dic_params_dict.get('roi_cutoff', 2000)
            )
        elif roi_type == 'mask_file' and 'path' in roi_data_dict:
            mask_img = cv2.imread(roi_data_dict['path'], cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"ROI mask file not found: {roi_data_dict['path']}")
            mask_array = mask_img > 128 # Example threshold to boolean
            initial_roi.set_roi_from_mask(mask_array, cutoff=dic_params_dict.get('roi_cutoff', 2000))
        elif 'mask_array' in roi_data_dict: # Direct mask array (e.g. from an upload)
             mask_array_np = np.array(roi_data_dict['mask_array'], dtype=bool)
             initial_roi.set_roi_from_mask(mask_array_np, cutoff=dic_params_dict.get('roi_cutoff', 2000))
        else:
            logger.warning(f"Task {task_id}: No detailed ROI definition provided, using full image or default empty ROI.")
            # If no specific ROI, Ncorr usually allows DIC on the whole image if no ROI is set.
            # The NcorrROI constructor with just image_shape creates a blank (all False) mask.
            # For DIC to run, a valid ROI from C++ (form_regions) is needed.
            # If no ROI is truly set, C++ side might fail.
            # Best to ensure a valid ROI (e.g., whole image if nothing else).
            # For now, it uses a blank one which will likely fail in form_regions unless logic is adapted.
            # A common default could be a full ROI:
            # initial_roi.set_roi_from_mask(np.ones(reference_image.gs_data.shape, dtype=bool))

        logger.info(f"Task {task_id}: Initial ROI prepared.")
        self.update_state(state='PROGRESS', meta={'status': 'Running DIC analysis...'})

        # 3. Orchestrate DIC Analysis
        raw_disp_results, final_rois, final_seeds, dic_status = \
            orchestrate_dic_analysis(reference_image, current_images, initial_roi, dic_params_dict)

        if dic_status != OutputState.SUCCESS:
            logger.error(f"Task {task_id}: DIC orchestration failed with status {dic_status.name}")
            raise Exception(f"DIC orchestration failed: {dic_status.name}")
        logger.info(f"Task {task_id}: DIC analysis completed.")
        self.update_state(state='PROGRESS', meta={'status': 'Formatting displacements...'})

        # 4. Format Displacement Fields
        # Ensure original_reference_ncorr_roi_obj is the one used for the first step of DIC.
        formatted_disp_results = format_displacement_fields(
            raw_disp_results, reference_image, current_images, initial_roi, dic_params_dict
        )
        logger.info(f"Task {task_id}: Displacements formatted.")
        self.update_state(state='PROGRESS', meta={'status': 'Calculating strains...'})

        # 5. Calculate Strain Fields
        strain_results_list = calculate_strain_fields(
            formatted_disp_results, dic_parameters_dict, strain_params_dict
        )
        logger.info(f"Task {task_id}: Strains calculated.")
        self.update_state(state='PROGRESS', meta={'status': 'Saving results...'})

        # 6. Store final results
        final_output = {
            "formatted_displacements": formatted_disp_results,
            "strain_fields": strain_results_list,
            # "final_rois": [roi.to_serializable_dict() for roi in final_rois], # NcorrROI needs to_serializable_dict
            # "final_seeds": final_seeds, # Needs serialization
        }
        
        # Serialize and save results to a JSON file specific to the task
        task_results_dir = os.path.join(RESULTS_BASE_DIR, task_id)
        os.makedirs(task_results_dir, exist_ok=True)
        results_filepath = os.path.join(task_results_dir, "results.json")
        
        # For complex objects like NcorrROI, a custom serialization is needed.
        # We'll use the simplified serialize_results for now.
        serializable_final_output = {}
        for i, disp_res in enumerate(final_output["formatted_displacements"]):
            serializable_final_output[f"disp_pair_{i}"] = serialize_results(disp_res, task_id)
        for i, strain_res in enumerate(final_output["strain_fields"]):
            serializable_final_output[f"strain_pair_{i}"] = serialize_results(strain_res, task_id)


        with open(results_filepath, 'w') as f:
            json.dump(serializable_final_output, f, indent=4)
        
        logger.info(f"Task {task_id}: Results saved to {results_filepath}")
        return {'status': 'COMPLETED', 'results_file': results_filepath, 'data': serializable_final_output}

    except Exception as e:
        logger.error(f"Task {task_id}: Exception caught: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'status': str(e), 'exc_type': type(e).__name__})
        # Consider re-raising to mark task as failed with traceback
        # raise
        return {'status': 'FAILURE', 'error_message': str(e)}