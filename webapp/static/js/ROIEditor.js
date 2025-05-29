import React, { useState, useRef, useEffect, useCallback } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Ellipse, Line, Transformer } from "react-konva";
import useImage from "use-image";

/**
 * ROIEditor
 * ---------
 * Interactive drawing surface for ROI creation (Add/Sub shapes).
 *
 * Props
 * -----
 *  imgUrl : string                  – URL / blob of the *reference* image
 *  onConfirm(roiShapes) : function  – invoked with [{type, addorsub, pos_imroi}, …]
 *
 * Public shape JSON output  (matches back-end expectation):
 *  {
 *    type      : "rect" | "ellipse" | "poly",
 *    addorsub  : "add"  | "sub",
 *    pos_imroi : [x1,y1,x2,y2]  OR [ [x,y], [x,y], … ]  (polygon)
 *  }
 */
export default function ROIEditor({ imgUrl, onConfirm }) {
  // ------------------------------- internal drawing state
  const [image]              = useImage(imgUrl);
  const [tool, setTool]      = useState(null);        // 'rect-add', 'rect-sub', …
  const [drawing, setDrawing] = useState(false);
  const [shapes, setShapes]  = useState([]);          // committed shapes
  const [tempShape, setTemp] = useState(null);        // shape being drawn
  const [selectedId, setSelectedId] = useState(null);

  const stageRef = useRef();
  const trRef    = useRef();

  // ------------------------------- helpers
  const isAdd = tool && tool.endsWith("-add");
  const shapeColor = (add) => (add ? "rgba(0,255,0,0.35)" : "rgba(255,0,0,0.35)");

  const startPos = useRef({ x: 0, y: 0 }); // initial mouse down point

  const handleMouseDown = (e) => {
    if (!tool) return;
    const { x, y } = e.target.getStage().getPointerPosition();
    startPos.current = { x, y };
    setDrawing(true);

    if (tool.startsWith("poly")) {
      // polygon: accumulate points
      if (!tempShape) {
        setTemp({ type: "poly", addorsub: isAdd ? "add" : "sub", points: [x, y] });
      } else {
        // double-click closes polygon
        const pts = [...tempShape.points, x, y];
        if (pts.length > 4 && Math.hypot(pts[0] - x, pts[1] - y) < 6) {
          commitTempShape({ ...tempShape, points: pts.slice(0, -2) });
          setTemp(null);
          setDrawing(false);
        } else {
          setTemp({ ...tempShape, points: pts });
        }
      }
    } else {
      // rect / ellipse
      setTemp({ type: tool.split("-")[0], addorsub: isAdd ? "add" : "sub", x, y, width: 0, height: 0 });
    }
  };

  const handleMouseMove = (e) => {
    if (!drawing || !tempShape || tempShape.type === "poly") return;
    const { x, y } = e.target.getStage().getPointerPosition();
    const { x: sx, y: sy } = startPos.current;
    setTemp({ ...tempShape, x: Math.min(sx, x), y: Math.min(sy, y), width: Math.abs(x - sx), height: Math.abs(y - sy) });
  };

  const handleMouseUp = () => {
    if (!drawing) return;
    setDrawing(false);
    if (tempShape && tempShape.type !== "poly") {
      // avoid accidental clicks that produce 1×1 shapes
      if (tempShape.width > 3 && tempShape.height > 3) commitTempShape(tempShape);
      setTemp(null);
    }
  };

  const commitTempShape = (shape) => setShapes((prev) => [...prev, { ...shape, id: Date.now().toString() }]);

  // -------------------------------- selection/transform
  useEffect(() => {
    if (trRef.current && selectedId) {
      const node = stageRef.current.findOne(`#${selectedId}`);
      node && trRef.current.nodes([node]);
    } else {
      trRef.current?.nodes([]);
    }
  }, [selectedId, shapes]);

  const onDeleteSelected = () =>
    setShapes((prev) => prev.filter((s) => s.id !== selectedId));

  // -------------------------------- export JSON
  const confirmROI = () => {
    const out = shapes.map((s) => {
      if (s.type === "rect" || s.type === "ellipse") {
        return {
          type: s.type,
          addorsub: s.addorsub,
          pos_imroi: [s.x, s.y, s.x + s.width, s.y + s.height],
        };
      }
      return { type: "poly", addorsub: s.addorsub, pos_imroi: chunkArray(s.points, 2) };
    });
    onConfirm(out);
  };

  // -------------------------------- render helpers
  const chunkArray = (arr, n) => {
    const res = [];
    for (let i = 0; i < arr.length; i += n) res.push(arr.slice(i, i + n));
    return res;
  };

  const ToolButton = ({ mode, label }) => (
    <button
      className={`tool-btn ${tool === mode ? "active" : ""}`}
      onClick={() => setTool(mode)}
    >
      {label}
    </button>
  );

  // -------------------------------- JSX
  return (
    <section className="roi-editor">
      <div className="toolbar">
        <ToolButton mode="rect-add" label="Add Rect" />
        <ToolButton mode="rect-sub" label="Sub Rect" />
        <ToolButton mode="ellipse-add" label="Add Ellipse" />
        <ToolButton mode="ellipse-sub" label="Sub Ellipse" />
        <ToolButton mode="poly-add" label="Add Poly" />
        <ToolButton mode="poly-sub" label="Sub Poly" />
        <button disabled={!selectedId} onClick={onDeleteSelected}>
          Delete Selected
        </button>
        <button onClick={confirmROI}>Confirm ROI</button>
      </div>

      <Stage
        width={image?.width || 800}
        height={image?.height || 600}
        ref={stageRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <Layer>
          {/* background image */}
          {image && <KonvaImage image={image} />}

          {/* committed shapes */}
          {shapes.map((s) =>
            s.type === "rect" ? (
              <Rect
                key={s.id}
                id={s.id}
                x={s.x}
                y={s.y}
                width={s.width}
                height={s.height}
                fill={shapeColor(s.addorsub === "add")}
                draggable
                onClick={() => setSelectedId(s.id)}
                onTap={() => setSelectedId(s.id)}
                onDragEnd={(e) => {
                  const { x, y } = e.target.position();
                  updateShape(s.id, { x, y });
                }}
                onTransformEnd={(e) => {
                  const node = e.target;
                  updateShape(s.id, {
                    x: node.x(),
                    y: node.y(),
                    width: node.width() * node.scaleX(),
                    height: node.height() * node.scaleY(),
                  });
                  node.scaleX(1);
                  node.scaleY(1);
                }}
              />
            ) : s.type === "ellipse" ? (
              <Ellipse
                key={s.id}
                id={s.id}
                x={s.x}
                y={s.y}
                radiusX={s.width / 2}
                radiusY={s.height / 2}
                fill={shapeColor(s.addorsub === "add")}
                draggable
                offsetX={-s.width / 2}
                offsetY={-s.height / 2}
                onClick={() => setSelectedId(s.id)}
                onTap={() => setSelectedId(s.id)}
                onDragEnd={(e) => {
                  const { x, y } = e.target.position();
                  updateShape(s.id, { x, y });
                }}
                onTransformEnd={(e) => {
                  const node = e.target;
                  updateShape(s.id, {
                    radiusX: node.radiusX() * node.scaleX(),
                    radiusY: node.radiusY() * node.scaleY(),
                  });
                  node.scaleX(1);
                  node.scaleY(1);
                }}
              />
            ) : (
              <Line
                key={s.id}
                id={s.id}
                points={s.points}
                closed
                fill={shapeColor(s.addorsub === "add")}
                stroke="black"
                strokeWidth={1}
                draggable
                onClick={() => setSelectedId(s.id)}
                onTap={() => setSelectedId(s.id)}
                onDragEnd={(e) => {
                  const { x, y } = e.target.position();
                  const newPts = s.points.map((p, idx) =>
                    idx % 2 === 0 ? p + x : p + y
                  );
                  updateShape(s.id, { points: newPts });
                }}
              />
            )
          )}

          {/* temp drawing feedback */}
          {tempShape &&
            tempShape.type === "rect" && (
              <Rect
                x={tempShape.x}
                y={tempShape.y}
                width={tempShape.width}
                height={tempShape.height}
                fill={shapeColor(isAdd)}
              />
            )}
          {tempShape &&
            tempShape.type === "ellipse" && (
              <Ellipse
                x={tempShape.x}
                y={tempShape.y}
                radiusX={tempShape.width / 2}
                radiusY={tempShape.height / 2}
                offsetX={-tempShape.width / 2}
                offsetY={-tempShape.height / 2}
                fill={shapeColor(isAdd)}
              />
            )}
          {tempShape &&
            tempShape.type === "poly" && (
              <Line points={tempShape.points} stroke="cyan" strokeWidth={2} />
            )}

          {/* transformer */}
          <Transformer
            ref={trRef}
            rotateEnabled={false}
            enabledAnchors={["top-left", "top-right", "bottom-left", "bottom-right"]}
          />
        </Layer>
      </Stage>
    </section>
  );

  // -------------------- helper to update shape by id
  function updateShape(id, attrs) {
    setShapes((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...attrs } : s))
    );
  }
}
