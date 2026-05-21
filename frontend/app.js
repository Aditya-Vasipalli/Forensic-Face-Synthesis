/* ─────────────────────────────────────────────
   ForenSynth — app.js
   Stubbed endpoints for sketch→face models.
   POST /api/cyclegan/generate
   POST /api/pix2pix/generate
   POST /api/clarify/enhance
   Each accepts FormData { image } → { output_url }

   Live endpoint for aging model (Flask, port 7001):
   POST http://localhost:7001/api/aging/generate
   FormData { image, source_age, target_age } → { output_url }
───────────────────────────────────────────── */

const AGING_API = "http://localhost:7001/api/aging/generate";

// ── API stubs ──────────────────────────────────────────────────────────────
const API = {
  async generate(model, imageFile) {
    // Endpoint: POST /api/${model}/generate
    // FormData: { image: File }
    // Response: { output_url: string, inference_ms: number }
    // ----- STUB: returns the input image after simulated delay -----
    await delay(1400 + Math.random() * 800);
    return {
      output_url: await fileToDataURL(imageFile),
      inference_ms: Math.round(1200 + Math.random() * 600),
    };
  },

  async enhance(imageFile, scale, denoise, sharpen) {
    // Endpoint: POST /api/clarify/enhance
    // FormData: { image: File, scale: number, denoise: bool, sharpen: bool }
    // Response: { output_url: string }
    // ----- STUB: returns input image after simulated delay -----
    await delay(1800 + Math.random() * 600);
    return {
      output_url: await fileToDataURL(imageFile),
    };
  },
};

// ── Utils ──────────────────────────────────────────────────────────────────
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const fileToDataURL = (file) =>
  new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = (e) => res(e.target.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });

let toastTimer;
function toast(msg, duration = 2800) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), duration);
}

// ── Router ─────────────────────────────────────────────────────────────────
const pages = ["generate", "compare", "clarify", "aging"];

function navigate(page) {
  if (!pages.includes(page)) page = "generate";
  pages.forEach((p) => {
    document.getElementById(`page-${p}`).classList.toggle("active", p === page);
    document
      .querySelector(`[data-page="${p}"]`)
      ?.classList.toggle("active", p === page);
  });
}

document.querySelectorAll(".nav-link").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const pg = a.dataset.page;
    history.pushState({}, "", `#${pg}`);
    navigate(pg);
  });
});

window.addEventListener("popstate", () => {
  navigate(location.hash.replace("#", "") || "generate");
});

navigate(location.hash.replace("#", "") || "generate");

// ── Drop zone helper ───────────────────────────────────────────────────────
function makeDropZone({ zoneEl, inputEl, innerEl, previewEl, onFile }) {
  function load(file) {
    if (!file || !file.type.startsWith("image/")) {
      toast("Only image files accepted.");
      return;
    }
    const url = URL.createObjectURL(file);
    previewEl.src = url;
    previewEl.classList.remove("hidden");
    innerEl.classList.add("hidden");
    zoneEl.classList.add("has-image");
    onFile(file);
  }

  zoneEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    zoneEl.classList.add("drag-over");
  });
  zoneEl.addEventListener("dragleave", () =>
    zoneEl.classList.remove("drag-over"),
  );
  zoneEl.addEventListener("drop", (e) => {
    e.preventDefault();
    zoneEl.classList.remove("drag-over");
    load(e.dataTransfer.files[0]);
  });
  zoneEl.addEventListener("click", () => inputEl.click());
  inputEl.addEventListener("change", () => load(inputEl.files[0]));
}

// ── Compare Slider helper ──────────────────────────────────────────────────
function makeSlider({ wrapEl, afterEl, handleEl }) {
  let dragging = false;

  function setPos(pct) {
    pct = Math.max(2, Math.min(98, pct));
    afterEl.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handleEl.style.left = `${pct}%`;
  }

  setPos(50);

  const getX = (e) => (e.touches ? e.touches[0].clientX : e.clientX);

  wrapEl.addEventListener("mousedown", () => (dragging = true));
  wrapEl.addEventListener("touchstart", () => (dragging = true), {
    passive: true,
  });
  window.addEventListener("mouseup", () => (dragging = false));
  window.addEventListener("touchend", () => (dragging = false));

  function onMove(e) {
    if (!dragging) return;
    const rect = wrapEl.getBoundingClientRect();
    const pct = ((getX(e) - rect.left) / rect.width) * 100;
    setPos(pct);
  }

  wrapEl.addEventListener("mousemove", onMove);
  wrapEl.addEventListener("touchmove", onMove, { passive: true });
}

// ── GENERATE PAGE ──────────────────────────────────────────────────────────
let generateState = {
  model: "cyclegan",
  file: null,
  outputUrl: null,
};

// Tabs
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab")
      .forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    generateState.model = tab.dataset.model;
    document.getElementById("model-badge").textContent =
      generateState.model === "cyclegan" ? "CycleGAN" : "Pix2Pix";
  });
});

// Drop zone
makeDropZone({
  zoneEl: document.getElementById("drop-zone"),
  inputEl: document.getElementById("file-input"),
  innerEl: document.getElementById("drop-inner"),
  previewEl: document.getElementById("preview-img"),
  onFile(file) {
    generateState.file = file;
    document.getElementById("generate-btn").disabled = false;
    // reset output
    document.getElementById("output-placeholder").classList.remove("hidden");
    document.getElementById("output-result").classList.add("hidden");
    generateState.outputUrl = null;
  },
});

// Slider for generate output
makeSlider({
  wrapEl: document.getElementById("compare-wrap"),
  afterEl: document.getElementById("cs-after"),
  handleEl: document.getElementById("cs-handle"),
});

// Generate button
document.getElementById("generate-btn").addEventListener("click", async () => {
  if (!generateState.file) return;
  const btn = document.getElementById("generate-btn");
  const label = btn.querySelector(".btn-label");
  const spinner = btn.querySelector(".btn-spinner");

  btn.disabled = true;
  label.textContent = "Generating…";
  spinner.classList.remove("hidden");

  try {
    const result = await API.generate(generateState.model, generateState.file);
    generateState.outputUrl = result.output_url;

    // Set slider images
    const sketchUrl = URL.createObjectURL(generateState.file);
    document.getElementById("cs-before-img").src = sketchUrl;
    document.getElementById("cs-after-img").src = result.output_url;
    document.getElementById("model-badge").textContent =
      generateState.model === "cyclegan" ? "CycleGAN" : "Pix2Pix";
    document.getElementById("download-btn").href = result.output_url;

    document.getElementById("output-placeholder").classList.add("hidden");
    document.getElementById("output-result").classList.remove("hidden");
    toast(`Done — ${result.inference_ms}ms`);
  } catch (err) {
    toast("Generation failed. Check API connection.");
    console.error(err);
  } finally {
    btn.disabled = false;
    label.textContent = "Generate";
    spinner.classList.add("hidden");
  }
});

// Send to Compare
document.getElementById("send-compare-btn").addEventListener("click", () => {
  if (!generateState.file) return;
  // Pre-load file into compare page drop zone
  _compareLoadFile(generateState.file);
  history.pushState({}, "", "#compare");
  navigate("compare");
  toast("Sketch loaded on Compare page.");
});

// Send to Clarify
document.getElementById("send-clarify-btn").addEventListener("click", () => {
  if (!generateState.outputUrl) return;
  fetch(generateState.outputUrl)
    .then((r) => r.blob())
    .then((blob) => {
      const file = new File([blob], "output.png", { type: "image/png" });
      _clarifyLoadFile(file);
      history.pushState({}, "", "#clarify");
      navigate("clarify");
      toast("Output sent to Clarify.");
    });
});

// ── COMPARE PAGE ───────────────────────────────────────────────────────────
let compareFile = null;

function _compareLoadFile(file) {
  compareFile = file;
  const url = URL.createObjectURL(file);
  const preview = document.getElementById("compare-preview");
  preview.src = url;
  preview.classList.remove("hidden");
  document.getElementById("compare-drop-inner").classList.add("hidden");
  document.getElementById("compare-drop").classList.add("has-image");
  document.getElementById("compare-generate-btn").disabled = false;
}

makeDropZone({
  zoneEl: document.getElementById("compare-drop"),
  inputEl: document.getElementById("compare-file-input"),
  innerEl: document.getElementById("compare-drop-inner"),
  previewEl: document.getElementById("compare-preview"),
  onFile(file) {
    compareFile = file;
    document.getElementById("compare-generate-btn").disabled = false;
    // reset results
    document.getElementById("cmp-cyclegan-img").classList.add("hidden");
    document.getElementById("cmp-pix2pix-img").classList.add("hidden");
    ["cyclegan", "pix2pix"].forEach((m) => {
      document.getElementById(`cmp-${m}-time`).classList.add("hidden");
      document.getElementById(`cmp-${m}-dl`).classList.add("hidden");
    });
    document.getElementById("compare-empty").classList.remove("hidden");
  },
});

document
  .getElementById("compare-generate-btn")
  .addEventListener("click", async () => {
    if (!compareFile) return;
    const btn = document.getElementById("compare-generate-btn");
    btn.disabled = true;
    btn.textContent = "Running…";

    document.getElementById("compare-empty").classList.add("hidden");

    // Show loaders
    document.getElementById("cmp-cyclegan-loader").classList.remove("hidden");
    document.getElementById("cmp-pix2pix-loader").classList.remove("hidden");

    try {
      const [cg, p2p] = await Promise.all([
        API.generate("cyclegan", compareFile),
        API.generate("pix2pix", compareFile),
      ]);

      // CycleGAN
      const cgImg = document.getElementById("cmp-cyclegan-img");
      cgImg.src = cg.output_url;
      cgImg.classList.remove("hidden");
      document.getElementById("cmp-cyclegan-loader").classList.add("hidden");
      document.getElementById("cmp-cyclegan-time").textContent =
        `${cg.inference_ms}ms`;
      document.getElementById("cmp-cyclegan-time").classList.remove("hidden");
      const cgDl = document.getElementById("cmp-cyclegan-dl");
      cgDl.href = cg.output_url;
      cgDl.classList.remove("hidden");

      // Pix2Pix
      const p2pImg = document.getElementById("cmp-pix2pix-img");
      p2pImg.src = p2p.output_url;
      p2pImg.classList.remove("hidden");
      document.getElementById("cmp-pix2pix-loader").classList.add("hidden");
      document.getElementById("cmp-pix2pix-time").textContent =
        `${p2p.inference_ms}ms`;
      document.getElementById("cmp-pix2pix-time").classList.remove("hidden");
      const p2pDl = document.getElementById("cmp-pix2pix-dl");
      p2pDl.href = p2p.output_url;
      p2pDl.classList.remove("hidden");

      toast("Both models complete.");
    } catch (err) {
      toast("One or more models failed.");
      document.getElementById("cmp-cyclegan-loader").classList.add("hidden");
      document.getElementById("cmp-pix2pix-loader").classList.add("hidden");
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Both Models";
    }
  });

// ── CLARIFY PAGE ───────────────────────────────────────────────────────────
let clarifyState = { file: null, scale: 2 };

function _clarifyLoadFile(file) {
  clarifyState.file = file;
  const url = URL.createObjectURL(file);
  const preview = document.getElementById("clarify-preview");
  preview.src = url;
  preview.classList.remove("hidden");
  document.getElementById("clarify-drop-inner").classList.add("hidden");
  document.getElementById("clarify-drop").classList.add("has-image");
  document.getElementById("clarify-btn").disabled = false;
  // reset output
  document.getElementById("clarify-placeholder").classList.remove("hidden");
  document.getElementById("clarify-result").classList.add("hidden");
}

makeDropZone({
  zoneEl: document.getElementById("clarify-drop"),
  inputEl: document.getElementById("clarify-file-input"),
  innerEl: document.getElementById("clarify-drop-inner"),
  previewEl: document.getElementById("clarify-preview"),
  onFile: _clarifyLoadFile,
});

// Scale buttons
document.querySelectorAll(".scale-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".scale-btn")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    clarifyState.scale = parseInt(btn.dataset.scale);
  });
});

// Slider for clarify output
makeSlider({
  wrapEl: document.getElementById("clarify-compare-wrap"),
  afterEl: document.getElementById("clarify-cs-after"),
  handleEl: document.getElementById("clarify-cs-handle"),
});

// Clarify button
document.getElementById("clarify-btn").addEventListener("click", async () => {
  if (!clarifyState.file) return;
  const btn = document.getElementById("clarify-btn");
  const label = btn.querySelector(".btn-label");
  const spinner = btn.querySelector(".btn-spinner");

  btn.disabled = true;
  label.textContent = "Enhancing…";
  spinner.classList.remove("hidden");

  const denoise = document.getElementById("denoise-toggle").checked;
  const sharpen = document.getElementById("sharpen-toggle").checked;

  try {
    const result = await API.enhance(
      clarifyState.file,
      clarifyState.scale,
      denoise,
      sharpen,
    );

    const origUrl = URL.createObjectURL(clarifyState.file);
    document.getElementById("clarify-before-img").src = origUrl;
    document.getElementById("clarify-after-img").src = result.output_url;

    document.getElementById("clarify-badge").textContent =
      `${clarifyState.scale}× Upscale`;
    document.getElementById("clarify-dl-btn").href = result.output_url;

    document.getElementById("clarify-placeholder").classList.add("hidden");
    document.getElementById("clarify-result").classList.remove("hidden");
    toast(`Enhanced at ${clarifyState.scale}×`);
  } catch (err) {
    toast("Enhancement failed. Check API connection.");
    console.error(err);
  } finally {
    btn.disabled = false;
    label.textContent = "Enhance Image";
    spinner.classList.add("hidden");
  }
});

// ── AGING PAGE ─────────────────────────────────────────────────────────────
let agingState = { file: null, outputUrl: null };

// Age sliders — live value display + diff label
function _updateAgeDiff() {
  const src = parseInt(document.getElementById("source-age").value);
  const tgt = parseInt(document.getElementById("target-age").value);
  document.getElementById("source-age-val").textContent = src;
  document.getElementById("target-age-val").textContent = tgt;
  const diff = tgt - src;
  const sign = diff >= 0 ? "+" : "";
  document.getElementById("age-diff-label").textContent =
    diff === 0 ? "No change" : `${sign}${diff} years`;
}

document.getElementById("source-age").addEventListener("input", _updateAgeDiff);
document.getElementById("target-age").addEventListener("input", _updateAgeDiff);
_updateAgeDiff();

// Drop zone
function _agingLoadFile(file) {
  agingState.file = file;
  const url = URL.createObjectURL(file);
  const preview = document.getElementById("aging-preview");
  preview.src = url;
  preview.classList.remove("hidden");
  document.getElementById("aging-drop-inner").classList.add("hidden");
  document.getElementById("aging-drop").classList.add("has-image");
  document.getElementById("aging-btn").disabled = false;
  // reset output
  document.getElementById("aging-placeholder").classList.remove("hidden");
  document.getElementById("aging-result").classList.add("hidden");
  agingState.outputUrl = null;
}

makeDropZone({
  zoneEl: document.getElementById("aging-drop"),
  inputEl: document.getElementById("aging-file-input"),
  innerEl: document.getElementById("aging-drop-inner"),
  previewEl: document.getElementById("aging-preview"),
  onFile: _agingLoadFile,
});

// Slider for aging output
makeSlider({
  wrapEl: document.getElementById("aging-compare-wrap"),
  afterEl: document.getElementById("aging-cs-after"),
  handleEl: document.getElementById("aging-cs-handle"),
});

// Generate button — calls the live Flask API
document.getElementById("aging-btn").addEventListener("click", async () => {
  if (!agingState.file) return;
  const btn = document.getElementById("aging-btn");
  const label = btn.querySelector(".btn-label");
  const spinner = btn.querySelector(".btn-spinner");
  const sourceAge = parseInt(document.getElementById("source-age").value);
  const targetAge = parseInt(document.getElementById("target-age").value);

  btn.disabled = true;
  label.textContent = "Generating…";
  spinner.classList.remove("hidden");

  try {
    const fd = new FormData();
    fd.append("image", agingState.file);
    fd.append("source_age", sourceAge);
    fd.append("target_age", targetAge);

    const res = await fetch(AGING_API, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    const data = await res.json();
    agingState.outputUrl = data.output_url;

    // Populate comparison slider
    const origUrl = URL.createObjectURL(agingState.file);
    document.getElementById("aging-before-img").src = origUrl;
    document.getElementById("aging-after-img").src = data.output_url;
    document.getElementById("aging-label-before").textContent =
      `Age ${sourceAge}`;
    document.getElementById("aging-label-after").textContent =
      `Age ${targetAge}`;
    document.getElementById("aging-badge").textContent =
      `${sourceAge} → ${targetAge}`;
    document.getElementById("aging-dl-btn").href = data.output_url;

    document.getElementById("aging-placeholder").classList.add("hidden");
    document.getElementById("aging-result").classList.remove("hidden");
    toast(`Aged: ${sourceAge} → ${targetAge}`);
  } catch (err) {
    toast(`Aging failed: ${err.message}`);
    console.error(err);
  } finally {
    btn.disabled = false;
    label.textContent = "Generate Aged Face";
    spinner.classList.add("hidden");
  }
});

// Send aged output to Clarify
document
  .getElementById("aging-send-clarify-btn")
  .addEventListener("click", () => {
    if (!agingState.outputUrl) return;
    fetch(agingState.outputUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const file = new File([blob], "aged_face.png", { type: "image/png" });
        _clarifyLoadFile(file);
        history.pushState({}, "", "#clarify");
        navigate("clarify");
        toast("Aged face sent to Clarify.");
      });
  });
