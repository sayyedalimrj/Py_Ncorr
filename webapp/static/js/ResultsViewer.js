import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
  Fragment,
} from "react";
import axios from "axios";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { hsvToRgb } from "color-math";
import "./ResultsViewer.css";         // optional styles

/**
 * ResultsViewer
 * -------------
 *  • Zoom + pan (react-zoom-pan-pinch)
 *  • Live colour-map editor
 *  • Tooltip data-cursor
 *  • Toggle scale-bar / axes / min-max markers
 *
 * Props
 * -----
 *   taskId    : Celery id (must be SUCCESS)
 *   refImage  : <img src> (blob URL)
 */
export default function ResultsViewer({ taskId, refImage }) {
  /* ------------------------------------------------------------------ state */

  const [manifest, setManifest] = useState(null);
  const [fieldName, setFieldName] = useState("u_ref_unit"); // default plot
  const [imgIndex, setImgIndex] = useState(0);

  const [array2D, setArray2D] = useState(null);             // current field data
  const [range, setRange] = useState({ min: -5, max: 5 });

  const [showAxes, setShowAxes] = useState(true);
  const [showScale, setShowScale] = useState(true);
  const [showMinMax, setShowMinMax] = useState(false);

  const [cursorInfo, setCursorInfo] = useState(null);        // {x,y,val}

  /* ------------------------------------------------------------------ refs */

  const canRef = useRef();       // heatmap canvas
  const minPt  = useRef(null);   // [x,y] of global min
  const maxPt  = useRef(null);   // [x,y] of global max

  /* ------------------------------------------------------------------ helpers */

  const jet = (t) => {
    // HSV gradient blue→red (t in [0..1])
    const { r, g, b } = hsvToRgb({ h: (240 - 240 * t) % 360, s: 1, v: 1 });
    return [r * 255, g * 255, b * 255];
  };

  const redrawCanvas = useCallback(() => {
    if (!array2D) return;
    const h = array2D.length;
    const w = array2D[0].length;
    const ctx = canRef.current.getContext("2d", { willReadFrequently: false });

    canRef.current.width = w;
    canRef.current.height = h;

    const imgData = ctx.createImageData(w, h);
    let gMin = +Infinity,
      gMax = -Infinity,
      gMinXY,
      gMaxXY;

    array2D.forEach((row, y) =>
      row.forEach((val, x) => {
        if (val < gMin) (gMin = val), (gMinXY = [x, y]);
        if (val > gMax) (gMax = val), (gMaxXY = [x, y]);
        const norm = Math.min(
          1,
          Math.max(0, (val - range.min) / (range.max - range.min)),
        );
        const [r, g, b] = jet(norm);
        const idx = (y * w + x) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 180;                          // alpha
      }),
    );

    ctx.putImageData(imgData, 0, 0);

    // min/max markers
    minPt.current = gMinXY;
    maxPt.current = gMaxXY;
    if (showMinMax) {
      ctx.fillStyle = "cyan";
      ctx.fillRect(gMinXY[0] - 2, gMinXY[1] - 2, 5, 5);
      ctx.fillStyle = "magenta";
      ctx.fillRect(gMaxXY[0] - 2, gMaxXY[1] - 2, 5, 5);
    }

    // axes overlay
    if (showAxes) {
      ctx.strokeStyle = "rgba(255,255,255,0.6)";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(w, 0);
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.stroke();
    }

    // scale-bar
    if (showScale) {
      const barH = 10;
      const barW = 150;
      const x0 = 10;
      const y0 = h - barH - 10;
      for (let i = 0; i < barW; i++) {
        const t = i / barW;
        const [r, g, b] = jet(t);
        ctx.strokeStyle = `rgb(${r},${g},${b})`;
        ctx.beginPath();
        ctx.moveTo(x0 + i, y0);
        ctx.lineTo(x0 + i, y0 + barH);
        ctx.stroke();
      }
      ctx.fillStyle = "#fff";
      ctx.font = "10px sans-serif";
      ctx.fillText(range.min.toFixed(2), x0, y0 - 2);
      ctx.fillText(range.max.toFixed(2), x0 + barW - 30, y0 - 2);
    }
  }, [array2D, range, showAxes, showScale, showMinMax]);

  /* ------------------------------------------------------------------ fetch manifest & array */

  useEffect(() => {
    axios
      .get(`/api/v1/analysis/results/${taskId}`)
      .then((res) => setManifest(res.data))
      .catch((e) => console.error(e));
  }, [taskId]);

  // fetch 2-D array whenever selection changes
  useEffect(() => {
    if (!manifest) return;
    const matchKey = Object.keys(manifest.outputs).find((k) =>
      k.includes(fieldName) && k.includes(`img_${imgIndex.toString().padStart(3, "0")}`),
    );
    if (!matchKey) return;
    const url = `/static/results/${manifest.results_dir}/${matchKey}`;
    axios.get(url).then((res) => {
      const arr = res.data; // assume backend converts .npy → JSON 2-D
      setArray2D(arr);

      // auto-adjust colour range
      const flat = arr.flat();
      setRange({ min: Math.min(...flat), max: Math.max(...flat) });
    });
  }, [manifest, fieldName, imgIndex]);

  // redraw whenever deps change
  useEffect(() => redrawCanvas(), [redrawCanvas]);

  /* ------------------------------------------------------------------ tooltip  */

  const handleMouseMove = (e, transformState) => {
    if (!array2D) return;
    // translate screen → image coords
    const bounding = canRef.current.getBoundingClientRect();
    const xScreen = e.clientX - bounding.left;
    const yScreen = e.clientY - bounding.top;

    // compensate zoom/pan from react-zoom-pan-pinch:
    const { state } = transformState; // scale, positionX, positionY
    const x = Math.floor((xScreen - state.positionX) / state.scale);
    const y = Math.floor((yScreen - state.positionY) / state.scale);

    if (x >= 0 && y >= 0 && y < array2D.length && x < array2D[0].length) {
      setCursorInfo({ x, y, val: array2D[y][x] });
    } else setCursorInfo(null);
  };

  /* ------------------------------------------------------------------ render */

  if (!array2D)
    return <p style={{ padding: 20 }}>Loading result array …</p>;

  return (
    <div className="results-wrapper">
      {/* ---------------- side panel */}
      <aside className="results-controls">
        <h3>Plot Options</h3>
        <label>
          Field
          <select value={fieldName} onChange={(e) => setFieldName(e.target.value)}>
            {["u_ref_unit", "v_ref_unit", "corrcoef", "Exx", "Exy", "Eyy"].map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        <label>
          Image #
          <input
            type="number"
            min={0}
            max={manifest ? Object.keys(manifest.outputs).length - 1 : 0}
            value={imgIndex}
            onChange={(e) => setImgIndex(+e.target.value)}
          />
        </label>

        <h4>Colour Range</h4>
        <input
          type="number"
          value={range.min}
          onChange={(e) => setRange((r) => ({ ...r, min: +e.target.value }))}
        />
        <input
          type="number"
          value={range.max}
          onChange={(e) => setRange((r) => ({ ...r, max: +e.target.value }))}
        />

        <h4>Display</h4>
        <label>
          <input
            type="checkbox"
            checked={showAxes}
            onChange={(e) => setShowAxes(e.target.checked)}
          />
          Axes
        </label>
        <label>
          <input
            type="checkbox"
            checked={showScale}
            onChange={(e) => setShowScale(e.target.checked)}
          />
          Scale Bar
        </label>
        <label>
          <input
            type="checkbox"
            checked={showMinMax}
            onChange={(e) => setShowMinMax(e.target.checked)}
          />
          Min/Max Markers
        </label>
      </aside>

      {/* ---------------- canvas area with zoom / pan */}
      <div className="zoom-wrapper">
        <TransformWrapper
          wheel={{ step: 50 }}
          doubleClick={{ disabled: true }}
          minScale={0.2}
          maxScale={20}
        >
          {(zoomState) => (
            <Fragment>
              <TransformComponent
                wrapperClass="img-wrapper"
                contentClass="img-content"
              >
                <img src={refImage} alt="ref" draggable={false} />
                <canvas
                  ref={canRef}
                  className="heatmap-layer"
                  draggable={false}
                  onMouseMove={(e) => handleMouseMove(e, zoomState)}
                />
              </TransformComponent>
            </Fragment>
          )}
        </TransformWrapper>

        {/* tooltip */}
        {cursorInfo && (
          <div
            className="tooltip"
            style={{
              left: cursorInfo.x + 15,
              top: cursorInfo.y + 15,
            }}
          >
            ({cursorInfo.x}, {cursorInfo.y}) : {cursorInfo.val.toFixed(3)}
          </div>
        )}
      </div>
    </div>
  );
}
