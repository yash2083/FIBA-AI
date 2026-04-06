/**
 * FIBA AI — Frontend Logic (Clean)
 * ==================================
 * Core: file upload, API, SSE progress, result rendering.
 * No RAG, no clip extraction. Fast and focused.
 */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dropZone = $("#drop-zone"), videoInput = $("#video-input");
  const browseLink = $("#browse-link"), uploadIcon = $("#upload-icon");
  const uploadText = $("#upload-text"), filePreview = $("#file-preview");
  const fileName = $("#file-name"), fileSize = $("#file-size");
  const fileClear = $("#file-clear"), queryInput = $("#query-input");
  const processBtn = $("#process-btn"), exampleChips = $$(".example-chip");

  const uploadSection = $("#upload-section"), progressSection = $("#progress-section");
  const resultsSection = $("#results-section"), errorSection = $("#error-section");
  const progressBar = $("#progress-bar"), progressPct = $("#progress-pct");
  const progressMsg = $("#progress-msg");

  const lightbox = $("#lightbox"), lightboxImg = $("#lightbox-img");
  const lightboxClose = $("#lightbox-close");
  const newAnalysisBtn = $("#new-analysis-btn"), errorRetryBtn = $("#error-retry-btn");

  let selectedFile = null;

  // ─── File ───────────────────────────────────────────────

  function selectFile(file) {
    if (!file || !file.type.startsWith("video/")) { alert("Please select a video file."); return; }
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    filePreview.hidden = false;
    uploadIcon.style.display = "none";
    uploadText.style.display = "none";
    $(".upload-hint").style.display = "none";
    dropZone.classList.add("has-file");
    updateBtn();
  }

  function clearFile() {
    selectedFile = null; videoInput.value = "";
    filePreview.hidden = true;
    uploadIcon.style.display = ""; uploadText.style.display = "";
    $(".upload-hint").style.display = "";
    dropZone.classList.remove("has-file");
    updateBtn();
  }

  function updateBtn() { processBtn.disabled = !(selectedFile && queryInput.value.trim()); }
  function formatBytes(b) { return b < 1024 ? b+" B" : b < 1048576 ? (b/1024).toFixed(1)+" KB" : (b/1048576).toFixed(1)+" MB"; }

  browseLink.addEventListener("click", e => { e.preventDefault(); videoInput.click(); });
  videoInput.addEventListener("change", () => { if (videoInput.files[0]) selectFile(videoInput.files[0]); });
  fileClear.addEventListener("click", e => { e.stopPropagation(); clearFile(); });
  dropZone.addEventListener("click", e => { if (!e.target.closest(".file-preview,.file-clear") && !selectedFile) videoInput.click(); });
  ["dragenter","dragover"].forEach(t => dropZone.addEventListener(t, e => { e.preventDefault(); dropZone.classList.add("drag-over"); }));
  ["dragleave","drop"].forEach(t => dropZone.addEventListener(t, e => { e.preventDefault(); dropZone.classList.remove("drag-over"); }));
  dropZone.addEventListener("drop", e => { if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]); });
  queryInput.addEventListener("input", updateBtn);
  queryInput.addEventListener("keydown", e => { if (e.key === "Enter" && !processBtn.disabled) startProcessing(); });
  exampleChips.forEach(c => c.addEventListener("click", () => { queryInput.value = c.dataset.query; updateBtn(); queryInput.focus(); }));

  // ─── Processing ─────────────────────────────────────────

  processBtn.addEventListener("click", startProcessing);

  async function startProcessing() {
    if (!selectedFile || !queryInput.value.trim()) return;
    showSection("progress");
    const fd = new FormData();
    fd.append("video", selectedFile);
    fd.append("query", queryInput.value.trim());
    try {
      const resp = await fetch("/api/process", { method: "POST", body: fd });
      if (!resp.ok) throw new Error((await resp.json()).error || "Upload failed");
      const data = await resp.json();
      pollProgress(data.job_id);
    } catch (err) { showError(err.message); }
  }

  function pollProgress(jobId) {
    const es = new EventSource(`/api/stream/${jobId}`);
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.error) { es.close(); showError(d.error); return; }
        updateProgress(d.progress, d.message);
        if (d.done) { es.close(); fetchResult(jobId); }
      } catch(e) {}
    };
    es.onerror = () => { es.close(); pollFallback(jobId); };
  }

  async function pollFallback(jobId) {
    const iv = setInterval(async () => {
      try {
        const d = (await (await fetch(`/api/status/${jobId}`)).json());
        updateProgress(d.progress, d.message);
        if (d.done) { clearInterval(iv); d.error ? showError(d.error) : d.result ? renderResults(d.result) : showError("No result"); }
      } catch(e) { clearInterval(iv); showError("Connection lost"); }
    }, 800);
  }

  async function fetchResult(jobId) {
    try {
      const d = (await (await fetch(`/api/status/${jobId}`)).json());
      d.error ? showError(d.error) : d.result ? renderResults(d.result) : showError("No result");
    } catch(e) { showError("Failed to fetch results"); }
  }

  // ─── Progress ───────────────────────────────────────────

  function updateProgress(pct, msg) {
    progressBar.style.width = pct + "%";
    progressPct.textContent = pct + "%";
    progressMsg.textContent = msg || "";
    [{ id:"stage-parse",min:0,max:15 },{ id:"stage-detect",min:15,max:45 },
     { id:"stage-track",min:45,max:72 },{ id:"stage-infer",min:72,max:85 },
     { id:"stage-render",min:85,max:100 }].forEach(s => {
      const el = $(`#${s.id}`);
      el.className = pct >= s.max ? "stage done" : pct >= s.min ? "stage active" : "stage";
    });
  }

  // ─── Results ────────────────────────────────────────────

  function renderResults(r) {
    showSection("results");

    // Banner
    const banner = $("#result-banner");
    if (r.action_detected) {
      banner.className = "result-banner detected";
      $("#banner-icon").textContent = "✅";
      $("#banner-title").textContent = "Action Detected";
      $("#banner-title").style.color = "var(--success)";
    } else {
      banner.className = "result-banner not-detected";
      $("#banner-icon").textContent = "❌";
      $("#banner-title").textContent = "Not Detected";
      $("#banner-title").style.color = "var(--error)";
    }
    $("#banner-subtitle").textContent = `"${r.query_info.raw}" → ${r.action_label} (${r.action_category})`;

    // Confidence ring
    const pct = Math.round(r.confidence * 100);
    $("#confidence-value").textContent = pct + "%";
    const circ = 2 * Math.PI * 35;
    const circle = $("#confidence-circle");
    circle.style.strokeDasharray = circ;
    requestAnimationFrame(() => {
      circle.style.transition = "stroke-dashoffset 1s ease";
      circle.style.strokeDashoffset = circ - r.confidence * circ;
    });
    circle.style.stroke = r.action_detected ? "var(--success)" : "var(--error)";
    $("#confidence-value").style.color = r.action_detected ? "var(--success)" : "var(--error)";

    // Description
    const desc = $("#description-text");
    desc.innerHTML = (r.action_description || "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Evidence
    $("#evidence-text").textContent = r.evidence || "No evidence.";

    // Key frames
    const kfGrid = $("#keyframes-grid");
    kfGrid.innerHTML = "";
    (r.key_frames || []).forEach((b64, i) => {
      const item = document.createElement("div");
      item.className = "keyframe-item";
      item.innerHTML = `<img src="data:image/jpeg;base64,${b64}" alt="Key frame ${i+1}" loading="lazy"/><div class="keyframe-label">Key Frame ${i+1}</div>`;
      item.addEventListener("click", () => openLightbox(b64));
      kfGrid.appendChild(item);
    });

    // Trajectory
    if (r.trajectory) {
      $("#trajectory-img").src = `data:image/jpeg;base64,${r.trajectory}`;
      $("#trajectory-card").hidden = false;
    } else { $("#trajectory-card").hidden = true; }

    // Motion stats
    renderStats($("#stats-grid"), r.motion_summary, [
      { label:"Rotation", key:"rotation_deg", unit:"°" },
      { label:"Displacement", key:"displacement_px", unit:"px" },
      { label:"Contact Events", key:"contact_events", unit:"" },
      { label:"Area Change", key:"area_change_ratio", unit:"×" },
      { label:"State Change", key:"state_change", unit:"" },
      { label:"Vertical Motion", key:"vertical_motion", unit:"" },
      { label:"Motion Speed", key:"motion_speed_px_per_frame", unit:"px/f" },
      { label:"Contact Freq", key:"contact_frequency", unit:"" },
      { label:"Approach Score", key:"approach_score", unit:"" },
      { label:"Grasp Change", key:"grasp_change", unit:"" },
      { label:"Area Growth", key:"area_growth_trend", unit:"" },
    ]);

    // Edge deployment stats
    renderEdgeStats(r.edge_stats);

    // Query info
    const qGrid = $("#query-detail-grid");
    qGrid.innerHTML = "";
    [{ l:"Query",v:r.query_info.raw },{ l:"Verb",v:r.query_info.verb },
     { l:"Category",v:r.query_info.category },{ l:"Object",v:r.query_info.object },
     { l:"Tool",v:r.query_info.tool||"—" }].forEach(f => {
      const d = document.createElement("div");
      d.className = "query-detail-item";
      d.innerHTML = `<div class="query-detail-label">${f.l}</div><div class="query-detail-value">${f.v}</div>`;
      qGrid.appendChild(d);
    });

    // Meta
    const meta = [];
    if (r.total_frames) meta.push(`${r.total_frames} frames`);
    if (r.fps) meta.push(`${r.fps.toFixed(1)} FPS`);
    if (r.processing_time_s) meta.push(`processed in ${r.processing_time_s}s`);
    $("#result-meta").textContent = meta.join(" · ");

    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderStats(grid, data, defs) {
    grid.innerHTML = "";
    if (!data) return;
    defs.forEach(d => {
      const v = data[d.key];
      if (v == null) return;
      const el = document.createElement("div");
      el.className = "stat-item";
      el.innerHTML = `<div class="stat-label">${d.label}</div><div class="stat-value">${typeof v==="number"?v.toFixed(1):v}<span class="stat-unit">${d.unit}</span></div>`;
      grid.appendChild(el);
    });
  }

  function renderEdgeStats(stats) {
    const badgesEl = $("#edge-badges");
    const grid = $("#edge-stats-grid");
    badgesEl.innerHTML = "";
    grid.innerHTML = "";
    if (!stats) return;

    // Badges
    const badges = [
      { label: "Edge Ready", active: stats.edge_ready, icon: "📱" },
      { label: "Zero-Shot", active: stats.zero_shot, icon: "🎯" },
      { label: "No Cloud", active: true, icon: "🔒" },
      { label: "Explainable", active: true, icon: "💡" },
    ];
    badges.forEach(b => {
      const el = document.createElement("span");
      el.className = "edge-badge" + (b.active ? " active" : "");
      el.textContent = `${b.icon} ${b.label}`;
      badgesEl.appendChild(el);
    });

    // Stats
    const items = [
      { label: "Pipeline Latency", value: stats.pipeline_latency_s + "s" },
      { label: "Frame Processing", value: stats.frame_processing_s + "s" },
      { label: "Inference", value: stats.inference_latency_s + "s" },
      { label: "Effective FPS", value: stats.effective_fps },
      { label: "Processed Frames", value: `${stats.processed_frames}/${stats.total_frames}` },
      { label: "Frame Skip", value: `every ${stats.frame_skip}${stats.frame_skip > 1 ? " (adaptive)" : ""}` },
      { label: "Resolution", value: stats.resolution },
      { label: "Models", value: stats.models_used },
    ];
    items.forEach(s => {
      const el = document.createElement("div");
      el.className = "stat-item";
      el.innerHTML = `<div class="stat-label">${s.label}</div><div class="stat-value edge-stat-value">${s.value}</div>`;
      grid.appendChild(el);
    });
  }

  // ─── Lightbox ───────────────────────────────────────────

  function openLightbox(b64) { lightboxImg.src = `data:image/jpeg;base64,${b64}`; lightbox.hidden = false; document.body.style.overflow = "hidden"; }
  function closeLightbox() { lightbox.hidden = true; document.body.style.overflow = ""; }
  lightboxClose.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", e => { if (e.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !lightbox.hidden) closeLightbox(); });

  // ─── Sections ───────────────────────────────────────────

  function showSection(name) {
    uploadSection.hidden = name !== "upload";
    progressSection.hidden = name !== "progress";
    resultsSection.hidden = name !== "results";
    errorSection.hidden = name !== "error";
    const hero = $("#hero-section");
    if (hero) hero.hidden = name !== "upload";
  }

  function resetToUpload() {
    clearFile(); queryInput.value = ""; updateBtn();
    showSection("upload"); updateProgress(0, "");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showError(msg) { showSection("error"); $("#error-msg").textContent = msg; }

  newAnalysisBtn.addEventListener("click", resetToUpload);
  errorRetryBtn.addEventListener("click", resetToUpload);
  showSection("upload");
})();
