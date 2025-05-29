import React, { useState } from "react";

/**
 * Display component shows: reference (large) + one current image at a time.
 *
 * Props
 * -----
 * reference : { preview, serverPath, name }
 * currents  : array of same objects
 */
export default function ImageDisplay({ reference, currents }) {
  const [idx, setIdx] = useState(0);

  if (!reference) return <p>No reference image chosen.</p>;
  if (currents.length === 0)
    return <p>Please upload at least one current image.</p>;

  const cur = currents[idx];

  return (
    <div className="image-display">
      <div className="ref-pane">
        <h3>Reference</h3>
        <img src={reference.preview} alt="reference" className="main-img" />
      </div>
      <div className="cur-pane">
        <h3>
          Current&nbsp;
          {idx + 1}/{currents.length}
        </h3>
        <img src={cur.preview} alt={`current-${idx}`} className="main-img" />
        <div className="nav-controls">
          <button
            onClick={() => setIdx((idx - 1 + currents.length) % currents.length)}
          >
            ◀
          </button>
          <button onClick={() => setIdx((idx + 1) % currents.length)}>▶</button>
        </div>
      </div>
    </div>
  );
}
