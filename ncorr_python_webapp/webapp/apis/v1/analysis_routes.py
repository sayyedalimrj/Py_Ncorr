from flask import request, jsonify
from flask_restful import Resource, Api
from webapp.tasks import run_full_dic_pipeline_task, celery_app
import os
import json # For loading results from file

RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task_results')


class DicAnalysis(Resource):
    def post(self):
        """
        Starts a new DIC analysis task.
        Expects JSON payload with:
        - reference_image_path: str
        - current_image_paths: list[str]
        - roi_definition: dict (e.g., {'type': 'draw', 'draw_objects': [...]})
        - dic_parameters: dict
        - strain_parameters: dict
        """
        if not request.is_json:
            return {"message": "Request must be JSON"}, 400
        
        data = request.get_json()

        ref_image_path = data.get('reference_image_path')
        cur_image_paths = data.get('current_image_paths')
        roi_def = data.get('roi_definition')
        dic_params = data.get('dic_parameters')
        strain_params = data.get('strain_parameters')

        if not all([ref_image_path, cur_image_paths, roi_def, dic_params, strain_params]):
            return {"message": "Missing required parameters"}, 400

        # Basic path validation (more robust validation needed for production)
        # For now, assuming paths are accessible by the Celery worker.
        # In a real app, you'd handle file uploads and secure path management.

        image_paths_dict = {"reference": ref_image_path, "currents": cur_image_paths}

        task = run_full_dic_pipeline_task.delay(
            image_paths_dict,
            roi_def,
            dic_params,
            strain_params
        )
        
        return {'task_id': task.id, 'status_url': f'/api/v1/analysis/status/{task.id}'}, 202


class DicTaskStatus(Resource):
    def get(self, task_id):
        """
        Gets the status of a DIC analysis task.
        """
        task = run_full_dic_pipeline_task.AsyncResult(task_id, app=celery_app)
        
        response = {
            'task_id': task_id,
            'status': task.state,
        }
        if task.state == 'PENDING':
            response['message'] = 'Task is waiting to be processed.'
        elif task.state == 'PROGRESS':
            response['message'] = 'Task is currently in progress.'
            response['progress_info'] = task.info # Custom progress metadata
        elif task.state == 'SUCCESS':
            response['message'] = 'Task completed successfully.'
            response['result_url'] = f'/api/v1/analysis/results/{task_id}'
            # response['result_summary'] = task.info # Could contain path to results file
        elif task.state == 'FAILURE':
            response['message'] = 'Task failed.'
            response['error'] = str(task.info) # Contains exception info
        else:
            response['message'] = 'Status unknown or task not found.'
            
        return response, 200


class DicTaskResult(Resource):
    def get(self, task_id):
        """
        Retrieves the results of a completed DIC analysis task.
        """
        task = run_full_dic_pipeline_task.AsyncResult(task_id, app=celery_app)

        if task.state == 'SUCCESS':
            # Results are stored in files by the task
            task_results_dir = os.path.join(RESULTS_BASE_DIR, task_id)
            results_filepath = os.path.join(task_results_dir, "results.json")
            
            if os.path.exists(results_filepath):
                try:
                    with open(results_filepath, 'r') as f:
                        results_data = json.load(f)
                    return {'task_id': task_id, 'status': 'SUCCESS', 'results': results_data}, 200
                except Exception as e:
                    return {'task_id': task_id, 'status': 'FAILURE', 'message': f'Error reading results file: {str(e)}'}, 500
            else:
                # If task.info contains the direct result (not file path)
                if isinstance(task.info, dict) and 'results_file' not in task.info:
                     return {'task_id': task_id, 'status': 'SUCCESS', 'results': task.info.get('data', task.info)}, 200
                return {'task_id': task_id, 'status': 'FAILURE', 'message': 'Results file not found, but task reported success.'}, 404
        elif task.state == 'FAILURE':
            return {'task_id': task_id, 'status': 'FAILURE', 'message': 'Task failed.', 'error': str(task.info)}, 200 # Or 500
        elif task.state == 'PENDING' or task.state == 'PROGRESS':
            return {'task_id': task_id, 'status': task.state, 'message': 'Task is not yet complete.'}, 202
        else:
            return {'task_id': task_id, 'status': 'NOT_FOUND', 'message': 'Task not found or status unknown.'}, 404

def initialize_analysis_routes(api_v1):
    api_v1.add_resource(DicAnalysis, '/analysis')
    api_v1.add_resource(DicTaskStatus, '/analysis/status/<string:task_id>')
    api_v1.add_resource(DicTaskResult, '/analysis/results/<string:task_id>')