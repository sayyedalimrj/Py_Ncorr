"""
Celery background execution for long-running Ncorr pipelines.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from celery import Celery, states

from ncorr_app.core import NcorrImage, NcorrROI, utils
from ncorr_app.algorithms import (
    orchestrate_dic_analysis,
    format_displacement_fields,
    calculate_strain_fields,
)
from webapp.config import cfg

# ────────────────────────────────────────────────────────────────────────────
# Celery initialisation
# ────────────────────────────────────────────────────────────────────────────
celery_app = Celery(
    "ncorr_web_tasks",
    broker=cfg.CELERY_BROKER_URL,
    backend=cfg.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
)


# ────────────────────────────────────────────────────────────────────────────
# Helper – persist NumPy array -> .npy and return relative path
# ────────────────────────────────────────────────────────────────────────────
def _save_array(arr: np.ndarray, folder: Path, name: str) -> str:
    path = folder / f"{name}.npy"
    np.save(path, arr)
    return path.name


# ────────────────────────────────────────────────────────────────────────────
# The main pipeline task
# ────────────────────────────────────────────────────────────────────────────
@celery_app.task(bind=True, name="run_full_dic_pipeline_task")
def run_full_dic_pipeline_task(  # noqa: C901  (complexity acceptable here)
    self,
    image_paths_dict: dict,      # {"reference": "...", "current": ["...", "..."]}
    roi_data_dict: dict,
    dic_params_dict: dict,
    strain_params_dict: dict,
    results_base_dir: str | None = None,
):
    """
    Execute the end-to-end DIC + strain pipeline.

    Return value (stored in backend & later served by Flask):
        {
            "results_dir": "<relative path>",
            "outputs": { ... filenames ... },
            "parameters": { ... }
        }
    """

    # ── 0.  output folder ────────────────────────────────────────────────
    out_dir = Path(results_base_dir or cfg.RESULTS_BASE_DIR) / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Update progress helper
    def _progress(stage: str, cur: int, tot: int):
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "current": cur, "total": tot},
        )

    # ── 1.  load images ──────────────────────────────────────────────────
    ref_path = image_paths_dict["reference"]
    cur_paths: Sequence[str] = image_paths_dict.get("current", [])
    imgs_ref = NcorrImage(ref_path)
    imgs_cur = [NcorrImage(p) for p in cur_paths]

    _progress("images-loaded", 1, 5)

    # ── 2.  ROI construction ────────────────────────────────────────────
    if roi_data_dict["type"] == "drawings":
        roi = NcorrROI(imgs_ref.get_gs().shape)
        roi.set_roi_from_drawings(
            roi_data_dict["data"],
            imgs_ref.get_gs().shape,
        )
    elif roi_data_dict["type"] == "mask_file":
        roi = NcorrROI()
        roi.set_roi_from_mask_file(
            roi_data_dict["path"],
            imgs_ref.get_gs().shape,
        )
    else:
        raise ValueError("Unsupported ROI definition")

    _progress("roi-built", 2, 5)

    # ── 3.  DIC core solve ──────────────────────────────────────────────
    dic_raw, rois_end, seeds = orchestrate_dic_analysis(
        imgs_ref,
        imgs_cur,
        roi,
        dic_params_dict,
    )
    _progress("dic-finished", 3, 5)

    # ── 4.  post-processing (displacements) ─────────────────────────────
    disp_formatted = format_displacement_fields(
        dic_raw,
        imgs_ref,
        imgs_cur,
        roi,
        dic_params_dict,
    )
    _progress("disp-formatted", 4, 5)

    # ── 5.  strain fields ───────────────────────────────────────────────
    strain_results = calculate_strain_fields(
        disp_formatted,
        dic_params_dict,
        strain_params_dict,
    )
    _progress("strain-done", 5, 5)

    # ── 6.  persist everything to disk  ─────────────────────────────────
    outputs = {}

    for i, d in enumerate(disp_formatted):
        tag = f"img_{i:03d}"
        outputs[_save_array(d["u_ref_unit"], out_dir, f"u_ref_{tag}")] = "u_ref_unit"
        outputs[_save_array(d["v_ref_unit"], out_dir, f"v_ref_{tag}")] = "v_ref_unit"
        outputs[_save_array(d["u_cur_unit"], out_dir, f"u_cur_{tag}")] = "u_cur_unit"
        outputs[_save_array(d["v_cur_unit"], out_dir, f"v_cur_{tag}")] = "v_cur_unit"
        outputs[_save_array(d["corrcoef"],   out_dir, f"corrcoef_{tag}")] = "corrcoef"

    for i, s in enumerate(strain_results):
        tag = f"img_{i:03d}"
        outputs[_save_array(s["Exx"], out_dir, f"Exx_{tag}")] = "Exx"
        outputs[_save_array(s["Exy"], out_dir, f"Exy_{tag}")] = "Exy"
        outputs[_save_array(s["Eyy"], out_dir, f"Eyy_{tag}")] = "Eyy"
        outputs[_save_array(s["exx"], out_dir, f"exx_{tag}")] = "exx"
        outputs[_save_array(s["exy"], out_dir, f"exy_{tag}")] = "exy"
        outputs[_save_array(s["eyy"], out_dir, f"eyy_{tag}")] = "eyy"

    # Save metadata / parameters
    (out_dir / "parameters.json").write_text(
        json.dumps(
            dict(dic_parameters=dic_params_dict,
                 strain_parameters=strain_params_dict,
                 seedinfo=seeds),
            indent=2),
        encoding="utf-8",
    )

    result_payload = {
        "results_dir": str(out_dir.relative_to(cfg.RESULTS_BASE_DIR)),
        "outputs": outputs,
    }
    return result_payload
