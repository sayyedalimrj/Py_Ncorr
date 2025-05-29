import React, { useState, useRef } from "react";

/**
 * ImageUploader
 * -------------
 * • One “Reference image” file input (single file).
 * • One “Current images” file input (multi-select or drag-drop).
 * • Sends images to backend immediately; on success stores { preview, serverPath }.
 *
 * Props
 * -----
 * onUploadComplete({ reference, currents })   // callback -> parent
 */
export default function ImageUploader({ onUploadComplete }) {
  const [reference, setReference] = useState(null);   // { preview, serverPath, name }
  const [currents, setCurrents]   = useState([]);     // array of same objects

  // refs for the hidden file inputs
  const refInputRef = useRef();
  const curInputRef = useRef();

  const handleFiles = async (files, isReference) => {
    const fileArr = Array.from(files);
    const uploaded = [];

    for (const file of fileArr) {
      const preview = URL.createObjectURL(file);
      const serverPath = await uploadToServer(file);
      uploaded.push({ name: file.name, preview, serverPath });
    }

    if (isReference) {
      setReference(uploaded[0]);
      onUploadComplete({ reference: uploaded[0], currents });
    } else {
      const newCurrents = [...currents, ...uploaded];
      setCurrents(newCurrents);
      onUploadComplete({ reference, currents: newCurrents });
    }
  };

  const uploadToServer = async (file) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/v1/upload_image", { method: "POST", body: fd });
    if (!res.ok) throw new Error("Upload failed");
    const data = await res.json();
    return data.server_path; // adjust if backend uses another key
  };

  /* Drop-zone helpers */
  const preventDefaults = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const makeDropHandlers = (isReference = false) => ({
    onDragOver: preventDefaults,
    onDragEnter: preventDefaults,
    onDragLeave: preventDefaults,
    onDrop: (e) => {
      preventDefaults(e);
      handleFiles(e.dataTransfer.files, isReference);
    },
  });

  return (
    <div className="uploader">
      {/* Reference Image */}
      <section
        className="dropzone reference-drop"
        {...makeDropHandlers(true)}
        onClick={() => refInputRef.current.click()}
      >
        <input
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          ref={refInputRef}
          onChange={(e) => handleFiles(e.target.files, true)}
        />
        {reference ? (
          <img src={reference.preview} alt="reference" className="thumb" />
        ) : (
          <p>Click or drop reference image here</p>
        )}
      </section>

      {/* Current Images */}
      <section
        className="dropzone current-drop"
        {...makeDropHandlers(false)}
        onClick={() => curInputRef.current.click()}
      >
        <input
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          ref={curInputRef}
          onChange={(e) => handleFiles(e.target.files, false)}
        />
        {currents.length === 0 ? (
          <p>Click or drop current images here (multi)</p>
        ) : (
          <div className="thumb-strip">
            {currents.map((img) => (
              <img
                key={img.serverPath}
                src={img.preview}
                alt={img.name}
                className="thumb small"
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
