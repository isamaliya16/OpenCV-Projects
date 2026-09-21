(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = Number(document.body.dataset.maxBytes) || 15 * 1024 * 1024;
  const OK_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
  const BUTTON_TEXT = "Start AI Scan";

  /* ------------------------------------------------------------ model status badge */
  const badge = $("status");

  function paintStatus(data) {
    const object = data.object_model.state;
    const face = data.face_model.state;
    const set = (state, text) => {
      badge.dataset.state = state;
      badge.textContent = "\u25CF " + text;
    };
    if (object === "error") return set("error", "Object model unavailable");
    if (object === "loading" || face === "loading") return set("loading", "Loading models\u2026");
    if (face === "error") return set("warn", "Ready \u00B7 face detection unavailable");
    if (object === "idle" || face === "idle") return set("ready", "Ready \u00B7 models load on first scan");
    return set("ready", "AI scanner ready");
  }

  async function pollStatus(attempt = 0) {
    try {
      const response = await fetch(badge.dataset.url, { cache: "no-store" });
      const data = await response.json();
      paintStatus(data);
      const loading = data.object_model.state === "loading" || data.face_model.state === "loading";
      if (loading && attempt < 120) setTimeout(() => pollStatus(attempt + 1), 1500);
    } catch (error) {
      badge.dataset.state = "unknown";
      badge.textContent = "\u25CF Status unavailable";
    }
  }
  if (badge) pollStatus();

  /* ------------------------------------------------------------ confirm dialogs */
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  /* ------------------------------------------------------------ upload form */
  const form = $("scanForm");
  if (!form) return;

  const input = $("image");
  const drop = $("drop");
  const nameLabel = $("fileName");
  const preview = $("preview");
  const message = $("clientMsg");
  const button = $("scan");
  const progress = $("progress");
  let previewUrl = null;

  const showError = (text) => {
    message.textContent = text;
    message.hidden = !text;
  };
  const formatSize = (bytes) =>
    bytes < 1024 * 1024
      ? `${Math.max(1, Math.round(bytes / 1024))} KB`
      : `${+(bytes / 1024 / 1024).toFixed(1)} MB`;
  const extensionOf = (name) => (name.includes(".") ? name.split(".").pop().toLowerCase() : "");

  function problemWith(file) {
    if (!OK_EXTENSIONS.includes(extensionOf(file.name))) {
      return "Unsupported format. Use JPG, JPEG, PNG or WEBP.";
    }
    if (file.size > MAX_BYTES) {
      return `That file is ${formatSize(file.size)}. The limit is ${formatSize(MAX_BYTES)}.`;
    }
    return "";
  }

  function resetPreview() {
    nameLabel.textContent = "No file selected";
    preview.hidden = true;
    preview.removeAttribute("src");
    drop.classList.remove("has-file");
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  }

  function clearInput() {
    input.value = "";
    resetPreview();
  }

  /* Show the chosen file, or explain what is wrong with it. The caller decides
     whether to keep the input's file, so this never touches input.files. */
  function selectFile(file) {
    resetPreview();
    const problem = problemWith(file);
    showError(problem);
    if (problem) return false;
    nameLabel.textContent = `${file.name} (${formatSize(file.size)})`;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    drop.classList.add("has-file");
    return true;
  }

  input.addEventListener("change", () => {
    if (!input.files.length) {
      showError("");
      resetPreview();
      return;
    }
    if (!selectFile(input.files[0])) clearInput();
  });

  /* drag and drop */
  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("dragover");
    })
  );
  drop.addEventListener("dragleave", (event) => {
    if (!drop.contains(event.relatedTarget)) drop.classList.remove("dragover");
  });
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("dragover");
    const files = event.dataTransfer && event.dataTransfer.files;
    if (!files || !files.length) return;
    const file = files[0];
    try {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
    } catch (error) {
      showError("Drag and drop is not supported in this browser. Use Choose an image instead.");
      return;
    }
    if (!selectFile(file)) clearInput();
  });
  // A file dropped outside the drop zone must not make the browser open it.
  ["dragover", "drop"].forEach((type) =>
    window.addEventListener(type, (event) => event.preventDefault())
  );

  /* sliders show their value */
  document.querySelectorAll("input[type=range][data-out]").forEach((range) => {
    const out = $(range.dataset.out);
    const paint = () => { out.textContent = `${range.value}%`; };
    range.addEventListener("input", paint);
    paint();
  });

  /* submit */
  form.addEventListener("submit", (event) => {
    if (!input.files.length) {
      event.preventDefault();
      showError("Choose an image first.");
      return;
    }
    const problem = problemWith(input.files[0]);
    if (problem) {
      event.preventDefault();
      showError(problem);
      return;
    }
    button.disabled = true;
    button.firstElementChild.textContent = "Scanning image\u2026";
    progress.hidden = false;
    form.setAttribute("aria-busy", "true");
  });

  // Coming back with the Back button restores the page from cache with the
  // button still disabled; reset it.
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    button.disabled = false;
    button.firstElementChild.textContent = BUTTON_TEXT;
    progress.hidden = true;
    form.removeAttribute("aria-busy");
  });
})();
