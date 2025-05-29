"""
REST API endpoints for launching and tracking Ncorr analyses.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
    url_for,
)
from werkzeug.exceptions import BadRequest, NotFound

from webapp.tasks.dic_tasks import run_full_dic_pipeline_task
from webapp.config import cfg

bp_analysis = Blueprint("v1_analysis", __name__, url_prefix="/api/v1/analysis")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _validate_payload(payload: dict):
    required = {
        "reference_image_path_on_server",
        "current_image_paths_on_server_list",
        "roi_definition",
        "dic_parameters",
        "strain_parameters",
    }
    missing = required - payload.keys()
    if missing:
        raise BadRequest(f"Missing keys: {', '.join(missing)}")


# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────
@bp_analysis.route("", methods=["POST"])
def launch_analysis():
    data = request.get_json(silent=True) or {}
    _validate_payload(data)

    ref_path = data["reference_image_path_on_server"]
    cur_paths = data["current_image_paths_on_server_list"]
    roi_def = data["roi_definition"]
    dic_par = data["dic_parameters"]
    strain_par = data["strain_parameters"]

    # Basic path sanity checks
    paths_to_check = [ref_path, *cur_paths]
    for p in paths_to_check:
        if not Path(p).is_file():
            raise BadRequest(f"Image not found on server: {p}")

    results_dir = cfg.RESULTS_BASE_DIR / str(uuid.uuid4())
    results_dir.mkdir(parents=True, exist_ok=True)

    task = run_full_dic_pipeline_task.apply_async(
        args=[
            {"reference": ref_path, "current": cur_paths},
            roi_def,
            dic_par,
            strain_par,
            str(results_dir),
        ]
    )

    return (
        jsonify(
            {
                "task_id": task.id,
                "status_url": url_for(
                    "v1_analysis.analysis_status", task_id=task.id, _external=True
                ),
            }
        ),
        202,
    )


@bp_analysis.route("/status/<task_id>", methods=["GET"])
def analysis_status(task_id):
    task = run_full_dic_pipeline_task.AsyncResult(task_id)
    info = task.info if task.info else {}
    resp = {
        "task_id": task_id,
        "state": task.state,
        "meta": info,
    }
    if task.state == "SUCCESS":
        resp["result_url"] = url_for(
            "v1_analysis.analysis_results", task_id=task_id, _external=True
        )
    return jsonify(resp)


@bp_analysis.route("/results/<task_id>", methods=["GET"])
def analysis_results(task_id):
    task = run_full_dic_pipeline_task.AsyncResult(task_id)
    if task.state != "SUCCESS":
        raise NotFound("Results not ready or task failed.")
    return jsonify(task.result or {})
