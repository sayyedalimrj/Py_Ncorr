import React, { useState } from "react";
import ImageUploader from "./ImageUploader";
import ROIEditor from "./ROIEditor";
import DICParametersForm from "./DICParametersForm";
import AnalysisStatusDisplay from "./AnalysisStatusDisplay";
import ResultsViewer from "./ResultsViewer";

export default function NcorrApp() {
  // ------------------- global state
  const [reference, setReference] = useState(null);
  const [currents, setCurrents]   = useState([]);
  const [roiShapes, setRoiShapes] = useState([]);
  const [taskId, setTaskId]       = useState(null);
  const [taskDone, setTaskDone]   = useState(false);

  // ------------------- callback chain
  const handleUpload = ({ reference: ref, currents: cur }) => {
    if (ref) setReference(ref);
    if (cur) setCurrents(cur);
  };
  const handleRoiConfirm = (shapesJson) => setRoiShapes(shapesJson);
  const handleTaskCreated = (id) => setTaskId(id);
  const handleFinished    = (state) => setTaskDone(state === "SUCCESS");

  // ------------------- UI flow
  if (!reference) {
    return (
      <>
        <h1>Ncorr – Upload Images</h1>
        <ImageUploader onUploadComplete={handleUpload} />
      </>
    );
  }

  if (roiShapes.length === 0) {
    return (
      <>
        <h1>Define ROI</h1>
        <ROIEditor imgUrl={reference.preview} onConfirm={handleRoiConfirm} />
      </>
    );
  }

  if (!taskId) {
    return (
      <>
        <h1>DIC Parameters</h1>
        <DICParametersForm
          reference={reference}
          currents={currents}
          roiShapes={roiShapes}
          onTaskCreated={handleTaskCreated}
        />
      </>
    );
  }

  if (!taskDone) {
    return (
      <AnalysisStatusDisplay taskId={taskId} onFinished={handleFinished} />
    );
  }

  return (
    <>
      <h1>Results</h1>
      <ResultsViewer taskId={taskId} refImage={reference.preview} />
    </>
  );
}
