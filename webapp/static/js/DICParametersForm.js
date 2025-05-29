import React, { useState } from "react";
import axios from "axios";

/**
 *  DICParametersForm
 *  -----------------
 *  Collects ALL user-tuneable parameters, then POSTs to /api/v1/analysis.
 *
 *  Props
 *  -----
 *    reference      : { serverPath }        – from ImageUploader
 *    currents       : [ { serverPath } ]    – "
 *    roiShapes      : [ { … } ]             – from ROIEditor
 *    onTaskCreated(taskId)                  – callback on success
 */
export default function DICParametersForm({
  reference,
  currents,
  roiShapes,
  onTaskCreated,
}) {
  // --- DIC parameters ----------
  const [subsetRadius, setSubsetRadius] = useState(15);
  const [subsetSpacing, setSubsetSpacing] = useState(1);
  const [algorithmType, setAlgorithmType] = useState("regular");
  const [cutoffNorm, setCutoffNorm] = useState(0.001);
  const [cutoffIter, setCutoffIter] = useState(200);
  const [totalThreads, setTotalThreads] = useState(
    navigator.hardwareConcurrency || 4,
  );
  const [subsetTrunc, setSubsetTrunc] = useState(false);

  // --- Step-analysis sub-group --
  const [stepEnabled, setStepEnabled] = useState(false);
  const [stepType, setStepType] = useState("seed");
  const [autoSeed, setAutoSeed] = useState(true);
  const [leapStepN, setLeapStepN] = useState(1);

  // --- Strain -------------------
  const [strainRadius, setStrainRadius] = useState(3);
  const [strainTrunc, setStrainTrunc] = useState(false);

  // --- Post-proc ----------------
  const [pixToUnits, setPixToUnits] = useState(1.0);
  const [unitsName, setUnitsName] = useState("px");
  const [ccCutoff, setCcCutoff] = useState(0.2);
  const [lensCoeff, setLensCoeff] = useState(0.0);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const launchAnalysis = async () => {
    if (!reference || currents.length === 0 || roiShapes.length === 0) {
      setMsg("Need reference, current images, and ROI first.");
      return;
    }

    setBusy(true);
    setMsg("Launching analysis…");

    try {
      const payload = {
        reference_image_path_on_server: reference.serverPath,
        current_image_paths_on_server_list: currents.map((c) => c.serverPath),
        roi_definition: { type: "drawings", data: roiShapes },
        dic_parameters: {
          radius: subsetRadius,
          spacing: subsetSpacing,
          algorithm: algorithmType,
          cutoff_diffnorm: cutoffNorm,
          cutoff_iter: cutoffIter,
          total_threads: totalThreads,
          subset_trunc: subsetTrunc,
          stepanalysis: {
            enabled: stepEnabled,
            type: stepType,
            auto: autoSeed,
            leap_step: leapStepN,
          },
          pixtounits: pixToUnits,
          cutoff_corrcoef_list: [ccCutoff],
          lenscoeff: lensCoeff,
        },
        strain_parameters: {
          radius_strain: strainRadius,
          subsettrunc_strain: strainTrunc,
        },
      };

      const res = await axios.post("/api/v1/analysis", payload);
      onTaskCreated(res.data.task_id);
      setMsg(`Task created: ${res.data.task_id}`);
    } catch (err) {
      setMsg(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="param-form">
      <h2>DIC / Strain Parameters</h2>

      {/* --- layout is terse to save space here --- */}
      <div className="grid two-col">
        <label>
          Subset Radius
          <input
            type="number"
            value={subsetRadius}
            onChange={(e) => setSubsetRadius(+e.target.value)}
          />
        </label>
        <label>
          Subset Spacing
          <input
            type="number"
            value={subsetSpacing}
            onChange={(e) => setSubsetSpacing(+e.target.value)}
          />
        </label>
        <label>
          Algorithm
          <select
            value={algorithmType}
            onChange={(e) => setAlgorithmType(e.target.value)}
          >
            <option value="regular">Regular / Forward</option>
            <option value="backward">Backward</option>
          </select>
        </label>
        <label>
          Total Threads
          <input
            type="number"
            value={totalThreads}
            onChange={(e) => setTotalThreads(+e.target.value)}
          />
        </label>
        <label>
          Cutoff – Norm
          <input
            type="number"
            step="0.0001"
            value={cutoffNorm}
            onChange={(e) => setCutoffNorm(+e.target.value)}
          />
        </label>
        <label>
          Cutoff – Iter #
          <input
            type="number"
            value={cutoffIter}
            onChange={(e) => setCutoffIter(+e.target.value)}
          />
        </label>
        <label>
          Subset Trunc.
          <input
            type="checkbox"
            checked={subsetTrunc}
            onChange={(e) => setSubsetTrunc(e.target.checked)}
          />
        </label>
      </div>

      <h3>Step Analysis</h3>
      <label>
        Enable
        <input
          type="checkbox"
          checked={stepEnabled}
          onChange={(e) => setStepEnabled(e.target.checked)}
        />
      </label>
      {stepEnabled && (
        <>
          <label>
            Type
            <select
              value={stepType}
              onChange={(e) => setStepType(e.target.value)}
            >
              <option value="seed">Seed Propagation</option>
              <option value="leapfrog">Leapfrog</option>
            </select>
          </label>
          <label>
            Auto Seed
            <input
              type="checkbox"
              checked={autoSeed}
              onChange={(e) => setAutoSeed(e.target.checked)}
            />
          </label>
          {stepType === "leapfrog" && (
            <label>
              Leap Step #
              <input
                type="number"
                value={leapStepN}
                onChange={(e) => setLeapStepN(+e.target.value)}
              />
            </label>
          )}
        </>
      )}

      <h3>Strain</h3>
      <label>
        Radius
        <input
          type="number"
          value={strainRadius}
          onChange={(e) => setStrainRadius(+e.target.value)}
        />
      </label>
      <label>
        Subset Trunc.
        <input
          type="checkbox"
          checked={strainTrunc}
          onChange={(e) => setStrainTrunc(e.target.checked)}
        />
      </label>

      <h3>Post-Processing</h3>
      <div className="grid two-col">
        <label>
          Units / Pixel
          <input
            type="number"
            step="0.0001"
            value={pixToUnits}
            onChange={(e) => setPixToUnits(+e.target.value)}
          />
        </label>
        <label>
          Units Name
          <input
            type="text"
            value={unitsName}
            onChange={(e) => setUnitsName(e.target.value)}
          />
        </label>
        <label>
          CC Cutoff
          <input
            type="number"
            step="0.001"
            value={ccCutoff}
            onChange={(e) => setCcCutoff(+e.target.value)}
          />
        </label>
        <label>
          Lens Coeff
          <input
            type="number"
            step="0.000001"
            value={lensCoeff}
            onChange={(e) => setLensCoeff(+e.target.value)}
          />
        </label>
      </div>

      <button disabled={busy} onClick={launchAnalysis}>
        {busy ? "Starting…" : "Start Analysis"}
      </button>
      <p>{msg}</p>
    </div>
  );
}
