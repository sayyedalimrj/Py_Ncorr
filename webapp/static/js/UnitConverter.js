import React, { useRef, useState } from "react";
import useImage from "use-image";

/**
 * UnitConverter
 * -------------
 * User draws a line → enters real-world length → returns { pix2unit, units }.
 *
 * Props
 * -----
 *   imgUrl         : string blob / URL
 *   onDone({scale, units})     – callback
 */
export default function UnitConverter({ imgUrl, onDone }) {
  const [image] = useImage(imgUrl);
  const [pt1, setPt1] = useState(null);
  const [pt2, setPt2] = useState(null);
  const [realLen, setRealLen] = useState("");
  const [units, setUnits] = useState("mm");

  const canRef = useRef();

  const handleClick = (e) => {
    const rect = canRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (!pt1) setPt1({ x, y });
    else if (!pt2) setPt2({ x, y });
    else {
      setPt1({ x, y });
      setPt2(null);
    }
  };

  const pixDist =
    pt1 && pt2 ? Math.hypot(pt2.x - pt1.x, pt2.y - pt1.y) : null;

  const confirm = () => {
    if (!pixDist || !parseFloat(realLen)) return;
    onDone({ scale: parseFloat(realLen) / pixDist, units });
  };

  return (
    <div>
      <h3>Calibration</h3>
      <p>Click two points that span a known length.</p>
      <div style={{ position: "relative", display: "inline-block" }}>
        {image && (
          <img src={imgUrl} alt="cal" onClick={handleClick} draggable={false} />
        )}
        <canvas
          ref={canRef}
          width={image?.width || 0}
          height={image?.height || 0}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            pointerEvents: "none",
          }}
        />
      </div>

      {pixDist && (
        <p>
          Pixel length: <strong>{pixDist.toFixed(2)}</strong>
        </p>
      )}
      <label>
        Real-world length
        <input
          type="number"
          value={realLen}
          onChange={(e) => setRealLen(e.target.value)}
        />
      </label>
      <label>
        Units
        <input value={units} onChange={(e) => setUnits(e.target.value)} />
      </label>
      <button disabled={!pixDist || !realLen} onClick={confirm}>
        Set Scale
      </button>
    </div>
  );
}
