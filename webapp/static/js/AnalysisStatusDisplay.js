import React, { useEffect, useState } from "react";
import axios from "axios";

/**
 * Periodically polls /status/<taskId>.
 *
 * Props
 * -----
 *   taskId       : string
 *   onFinished() : callback when SUCCESS (or FAILURE)
 */
export default function AnalysisStatusDisplay({ taskId, onFinished }) {
  const [state, setState] = useState("PENDING");
  const [meta, setMeta] = useState({});

  useEffect(() => {
    const intv = setInterval(() => {
      axios
        .get(`/api/v1/analysis/status/${taskId}`)
        .then((res) => {
          setState(res.data.state);
          setMeta(res.data.meta || {});
          if (["SUCCESS", "FAILURE"].includes(res.data.state)) {
            clearInterval(intv);
            onFinished(res.data.state);
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(intv);
  }, [taskId, onFinished]);

  return (
    <div className="status-box">
      <h2>Task Status: {taskId}</h2>
      <p>
        {state}
        {state === "PROGRESS" && meta.total && (
          <>
            {" "}
            – {meta.stage} ({meta.current}/{meta.total})
          </>
        )}
      </p>
    </div>
  );
}
