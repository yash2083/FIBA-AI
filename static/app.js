/**
 * FIBA AI — Frontend Logic
 * =========================
 * Handles: file upload, drag-drop, API calls, SSE progress, result rendering.
 * Pure vanilla JS — no React/Node dependencies.
 */

(function () {
  "use strict";

  // ─── DOM References ──────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dropZone      = $("#drop-zone");
  const videoInput     = $("#video-input");
  const browseLink     = $("#browse-link");
  const uploadIcon     = $("#upload-icon");
  const uploadText     = $("#upload-text");
  const filePreview    = $("#file-preview");
  const fileName       = $("#file-name");
  const fileSize       = $("#file-size");
  const fileClear      = $("#file-clear");
  const queryInput     = $("#query-input");
  const processBtn     = $("#process-btn");
  const exampleChips   = $$(".example-chip");

  const uploadSection  = $("#upload-section");
  const progressSection = $("#progress-section");
  const resultsSection = $("#results-section");
  const errorSection   = $("#error-section");

  const progressBar    = $("#progress-bar");
  const progressPct    = $("#progress-pct");
  const progressMsg    = $("#progress-msg");

  const lightbox       = $("#lightbox");
  const lightboxImg    = $("#lightbox-img");
  const lightboxClose  = $("#lightbox-close");

  const newAnalysisBtn = $("#new-analysis-btn");
  const errorRetryBtn  = $("#error-retry-btn");

  // State
  let selectedFile = null;

  // ─── File Selection ──────────────────────────────────────

  function selectFile(file) {
    if (!file) return;
    // Validate type
    if (!file.type.startsWith("video/")) {
      alert("Please select a video file.");
      return;
    }
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    filePreview.hidden = false;
    uploadIcon.style.display = "none";
    uploadText.style.display = "none";
    $(".upload-hint").style.display = "none";
    dropZone.classList.add("has-file");
    updateProcessBtn();
  }

  function clearFile() {
    selectedFile = null;
    videoInput.value = "";
    filePreview.hidden = true;
    uploadIcon.style.display = "";
    uploadText.style.display = "";
    $(".upload-hint").style.display = "";
    dropZone.classList.remove("has-file");
    updateProcessBtn();
  }

  function updateProcessBtn() {
    processBtn.disabled = !(selectedFile && queryInput.value.trim());
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  }

  // File input
  browseLink.addEventListener("click", (e) => {
    e.preventDefault();
    videoInput.click();
  });

  videoInput.addEventListener("change", () => {
    if (videoInput.files[0]) selectFile(videoInput.files[0]);
  });

  fileClear.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });

  // Drop zone click
  dropZone.addEventListener("click", (e) => {
    if (e.target.closest(".file-preview") || e.target.closest(".file-clear")) return;
    if (!selectedFile) videoInput.click();
  });

  // Drag and drop
  ["dragenter", "dragover"].forEach((type) => {
    dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    dropZone.addEventListener(type, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  // Query input
  queryInput.addEventListener("input", updateProcessBtn);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !processBtn.disabled) startProcessing();
  });

  // Example chips
  exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.query;
      updateProcessBtn();
      queryInput.focus();
    });
  });

  // ─── Processing ──────────────────────────────────────────

  processBtn.addEventListener("click", startProcessing);

  async function startProcessing() {
    if (!selectedFile || !queryInput.value.trim()) return;

    // Show progress
    showSection("progress");

    const formData = new FormData();
    formData.append("video", selectedFile);
    formData.append("query", queryInput.value.trim());

    try {
      const resp = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || "Upload failed");
      }

      const data = await resp.json();
      const jobId = data.job_id;

      // Start SSE for progress
      pollProgress(jobId);
    } catch (err) {
      showError(err.message);
    }
  }

  function pollProgress(jobId) {
    // Try SSE first, fall back to polling
    const evtSource = new EventSource(`/api/stream/${jobId}`);
    let lastPct = 0;

    evtSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          evtSource.close();
          showError(data.error);
          return;
        }

        updateProgress(data.progress, data.message);
        lastPct = data.progress;

        if (data.done) {
          evtSource.close();
          // Fetch final result
          fetchResult(jobId);
        }
      } catch (e) {
        // ignore parse errors
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      // Fall back to polling
      pollFallback(jobId);
    };
  }

  async function pollFallback(jobId) {
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`/api/status/${jobId}`);
        const data = await resp.json();

        updateProgress(data.progress, data.message);

        if (data.done) {
          clearInterval(interval);
          if (data.error) {
            showError(data.error);
          } else if (data.result) {
            renderResults(data.result);
          }
        }
      } catch (e) {
        clearInterval(interval);
        showError("Connection lost");
      }
    }, 800);
  }

  async function fetchResult(jobId) {
    try {
      const resp = await fetch(`/api/status/${jobId}`);
      const data = await resp.json();

      if (data.error) {
        showError(data.error);
      } else if (data.result) {
        renderResults(data.result);
      } else {
        showError("No result received");
      }
    } catch (e) {
      showError("Failed to fetch results");
    }
  }

  // ─── Progress UI ─────────────────────────────────────────

  function updateProgress(pct, msg) {
    progressBar.style.width = pct + "%";
    progressPct.textContent = pct + "%";
    progressMsg.textContent = msg || "";

    // Stage indicators
    const stages = [
      { id: "stage-parse", min: 0, max: 15 },
      { id: "stage-detect", min: 15, max: 45 },
      { id: "stage-track", min: 45, max: 72 },
      { id: "stage-infer", min: 72, max: 85 },
      { id: "stage-render", min: 85, max: 100 },
    ];

    stages.forEach((s) => {
      const el = $(`#${s.id}`);
      if (pct >= s.max) {
        el.className = "stage done";
      } else if (pct >= s.min) {
        el.className = "stage active";
      } else {
        el.className = "stage";
      }
    });
  }

  // ─── Results Rendering ───────────────────────────────────

  function renderResults(result) {
    showSection("results");

    // Banner
    const banner = $("#result-banner");
    const bannerIcon = $("#banner-icon");
    const bannerTitle = $("#banner-title");
    const bannerSubtitle = $("#banner-subtitle");
    const confidenceValue = $("#confidence-value");
    const confidenceCircle = $("#confidence-circle");

    if (result.action_detected) {
      banner.className = "result-banner detected";
      bannerIcon.textContent = "✅";
      bannerTitle.textContent = "Action Detected";
      bannerTitle.style.color = "var(--success)";
    } else {
      banner.className = "result-banner not-detected";
      bannerIcon.textContent = "❌";
      bannerTitle.textContent = "Not Detected";
      bannerTitle.style.color = "var(--error)";
    }

    bannerSubtitle.textContent =
      `"${result.query_info.raw}" → ${result.action_label} (${result.action_category})`;

    // Confidence ring
    const confPct = Math.round(result.confidence * 100);
    confidenceValue.textContent = confPct + "%";
    const circumference = 2 * Math.PI * 35; // r=35
    const offset = circumference - (result.confidence * circumference);
    confidenceCircle.style.strokeDasharray = circumference;

    // Animate
    requestAnimationFrame(() => {
      confidenceCircle.style.transition = "stroke-dashoffset 1s ease";
      confidenceCircle.style.strokeDashoffset = offset;
    });

    if (result.action_detected) {
      confidenceCircle.style.stroke = "var(--success)";
      confidenceValue.style.color = "var(--success)";
    } else {
      confidenceCircle.style.stroke = "var(--error)";
      confidenceValue.style.color = "var(--error)";
    }

    // Evidence
    $("#evidence-text").textContent = result.evidence || "No evidence generated.";

    // Key frames
    const kfGrid = $("#keyframes-grid");
    kfGrid.innerHTML = "";
    if (result.key_frames && result.key_frames.length > 0) {
      result.key_frames.forEach((b64, idx) => {
        const item = document.createElement("div");
        item.className = "keyframe-item";
        item.innerHTML = `
          <img src="data:image/jpeg;base64,${b64}" alt="Key frame ${idx + 1}" loading="lazy" />
          <div class="keyframe-label">Key Frame ${idx + 1}</div>
        `;
        item.addEventListener("click", () => openLightbox(b64));
        kfGrid.appendChild(item);
      });
    }

    // Trajectory
    if (result.trajectory) {
      $("#trajectory-img").src = `data:image/jpeg;base64,${result.trajectory}`;
      $("#trajectory-card").hidden = false;
    } else {
      $("#trajectory-card").hidden = true;
    }

    // Motion stats
    renderMotionStats(result.motion_summary);

    // Query info
    renderQueryInfo(result.query_info);

    // Meta
    const metaParts = [];
    if (result.total_frames) metaParts.push(`${result.total_frames} frames`);
    if (result.fps) metaParts.push(`${result.fps.toFixed(1)} FPS`);
    if (result.processing_time_s) metaParts.push(`processed in ${result.processing_time_s}s`);
    $("#result-meta").textContent = metaParts.join(" · ");

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderMotionStats(summary) {
    const grid = $("#stats-grid");
    grid.innerHTML = "";

    if (!summary) return;

    const stats = [
      { label: "Rotation", value: summary.rotation_deg, unit: "°" },
      { label: "Displacement", value: summary.displacement_px, unit: "px" },
      { label: "Contact Events", value: summary.contact_events, unit: "" },
      { label: "Area Change", value: summary.area_change_ratio, unit: "×" },
      { label: "State Change", value: summary.state_change, unit: "" },
      { label: "Vertical Motion", value: summary.vertical_motion, unit: "" },
      { label: "Motion Speed", value: summary.motion_speed_px_per_frame, unit: "px/f" },
      { label: "Contact Freq", value: summary.contact_frequency, unit: "" },
    ];

    stats.forEach((s) => {
      if (s.value == null) return;
      const item = document.createElement("div");
      item.className = "stat-item";
      item.innerHTML = `
        <div class="stat-label">${s.label}</div>
        <div class="stat-value">${typeof s.value === "number" ? s.value.toFixed(1) : s.value}<span class="stat-unit">${s.unit}</span></div>
      `;
      grid.appendChild(item);
    });
  }

  function renderQueryInfo(info) {
    const grid = $("#query-detail-grid");
    grid.innerHTML = "";

    if (!info) return;

    const fields = [
      { label: "Query", value: info.raw },
      { label: "Verb", value: info.verb },
      { label: "Category", value: info.category },
      { label: "Object", value: info.object },
      { label: "Tool", value: info.tool || "—" },
    ];

    fields.forEach((f) => {
      const item = document.createElement("div");
      item.className = "query-detail-item";
      item.innerHTML = `
        <div class="query-detail-label">${f.label}</div>
        <div class="query-detail-value">${f.value}</div>
      `;
      grid.appendChild(item);
    });
  }

  // ─── Lightbox ────────────────────────────────────────────

  function openLightbox(b64) {
    lightboxImg.src = `data:image/jpeg;base64,${b64}`;
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    lightbox.hidden = true;
    document.body.style.overflow = "";
  }

  lightboxClose.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
  });

  // ─── Section Switching ───────────────────────────────────

  function showSection(name) {
    uploadSection.hidden = name !== "upload";
    progressSection.hidden = name !== "progress";
    resultsSection.hidden = name !== "results";
    errorSection.hidden = name !== "error";

    // Keep hero visible for upload, hidden otherwise
    const hero = $("#hero-section");
    if (hero) hero.hidden = name !== "upload";
  }

  function resetToUpload() {
    clearFile();
    queryInput.value = "";
    updateProcessBtn();
    showSection("upload");
    // Reset progress
    updateProgress(0, "");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showError(msg) {
    showSection("error");
    $("#error-msg").textContent = msg;
  }

  newAnalysisBtn.addEventListener("click", resetToUpload);
  errorRetryBtn.addEventListener("click", resetToUpload);

  // ─── Init ────────────────────────────────────────────────
  showSection("upload");
})();
