/* FaceKey login: captures camera frames, sends them to the server, and
   renders the server's verdict. The browser never decides "verified" itself. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const stage = $("#stage");
  const video = $("#video");
  const faceBox = $("#faceBox");
  const statusEl = $("#status");
  const detailEl = $("#detail");
  const veilText = $("#veil-text");
  const actionBtn = $("#action");
  const meter = $("#meter");
  const steps = {
    camera: $("#step-camera"),
    face: $("#step-face"),
    verified: $("#step-verified"),
  };
  // (no local firstName — the matched account's name comes back from the server on "verified")

  const FRAME_INTERVAL_MS = 280;   // ~3.5 frames per second
  const MAX_SEND_WIDTH = 640;      // matches the size the model was trained on
  const JPEG_QUALITY = 0.85;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  let stream = null;
  let attemptId = null;
  let running = false;
  let required = Number(meter.dataset.required) || 5;
  let threshold = null;
  let netErrors = 0;
  let lockTimer = null;

  /* ---------------- small UI helpers ---------------- */
  function setState(state) { stage.dataset.state = state; }

  function setStatus(text, tone) {
    statusEl.textContent = text;
    statusEl.classList.toggle("is-bad", tone === "bad");
    statusEl.classList.toggle("is-ok", tone === "ok");
  }

  function setSteps(camera, face, verified) {
    steps.camera.dataset.status = camera;
    steps.face.dataset.status = face;
    steps.verified.dataset.status = verified;
  }

  function buildMeter(n) {
    meter.textContent = "";
    for (let i = 0; i < n; i++) {
      const seg = document.createElement("span");
      seg.className = "seg";
      meter.appendChild(seg);
    }
    meter.setAttribute("aria-valuemax", String(n));
    fillMeter(0);
  }

  function fillMeter(count, ok) {
    [...meter.children].forEach((seg, i) => seg.classList.toggle("on", i < count));
    meter.classList.toggle("ok", !!ok);
    meter.setAttribute("aria-valuenow", String(count));
  }

  function placeBox(face) {
    if (!face) { faceBox.hidden = true; return; }
    // The preview is mirrored, so mirror the box horizontally too.
    const left = 1 - (face.x + face.w);
    faceBox.style.left = `${left * 100}%`;
    faceBox.style.top = `${face.y * 100}%`;
    faceBox.style.width = `${face.w * 100}%`;
    faceBox.style.height = `${face.h * 100}%`;
    faceBox.hidden = false;
  }

  function showDistance(distance, limit) {
    if (distance == null) { detailEl.textContent = ""; return; }
    detailEl.textContent = limit != null
      ? `Match score ${distance.toFixed(1)}. Accepted up to ${limit}.`
      : `Match score ${distance.toFixed(1)}.`;
  }

  function setButton(label, kind) {
    actionBtn.textContent = label;
    actionBtn.dataset.kind = kind;
    actionBtn.classList.toggle("btn-primary", kind !== "stop");
    actionBtn.classList.toggle("btn-quiet", kind === "stop");
    actionBtn.disabled = false;
  }

  /* ---------------- camera ---------------- */
  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return fail("Camera access needs a secure page. Open this site over HTTPS, or at localhost.");
    }
    setSteps("active", "pending", "pending");
    setStatus("Waiting for camera permission...");
    detailEl.textContent = "";
    actionBtn.disabled = true;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
    } catch (err) {
      const messages = {
        NotAllowedError: "Camera permission is blocked. Allow camera access from your browser's address bar, then try again.",
        NotFoundError: "No camera was found on this device.",
        NotReadableError: "Another app is using the camera. Close it and try again.",
      };
      return fail(messages[err.name] || "The camera could not be started.");
    }

    video.srcObject = stream;
    try { await video.play(); } catch (_) { /* autoplay is muted, this should not throw */ }
    if (video.videoWidth) stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    setSteps("done", "active", "pending");
    setState("searching");
    await beginAttempt();
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    placeBox(null);
  }

  /* ---------------- attempt loop ---------------- */
  async function beginAttempt() {
    let res;
    try {
      res = await fetch("/api/login/start", { method: "POST" });
    } catch (_) {
      return fail("Could not reach the server. Check your connection and try again.");
    }
    const data = await res.json().catch(() => ({}));
    if (res.status === 429 && data.state === "locked") return showLocked(data.retry_after);
    if (res.status === 409 && data.state === "no_users") {
      return fail("No accounts exist yet. Create one first.");
    }
    if (!res.ok) return fail("Could not start verification. Try again.");

    attemptId = data.attempt_id;
    required = data.required;
    threshold = data.threshold;
    buildMeter(required);
    netErrors = 0;
    running = true;
    setButton("Stop camera", "stop");
    setStatus("Looking for your face. Face the camera in even light.");
    loop();
  }

  function captureFrame() {
    return new Promise((resolve) => {
      const scale = Math.min(1, MAX_SEND_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);   // unmirrored, like the training data
      canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY);
    });
  }

  async function loop() {
    while (running) {
      const started = performance.now();
      try {
        if (video.videoWidth) {
          const blob = await captureFrame();
          if (!running) break;
          await sendFrame(blob);
        }
        netErrors = 0;
      } catch (_) {
        netErrors += 1;
        if (netErrors >= 4) return fail("Lost connection to the server. Try again.");
      }
      const wait = Math.max(30, FRAME_INTERVAL_MS - (performance.now() - started));
      await new Promise((r) => setTimeout(r, wait));
    }
  }

  async function sendFrame(blob) {
    const res = await fetch("/api/login/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg", "X-Attempt-Id": attemptId },
      body: blob,
    });
    const data = await res.json().catch(() => ({}));

    if (res.status === 429 && data.error === "slow_down") return;
    if (res.status === 429 && data.state === "locked") return showLocked(data.retry_after);
    if (res.status === 410) return fail("This attempt timed out. Try again.");
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    render(data);
  }

  /* ---------------- render the server's verdict ---------------- */
  function render(d) {
    switch (d.state) {
      case "searching":
        setState("searching");
        placeBox(null);
        setSteps("done", "active", "pending");
        fillMeter(0);
        setStatus("Looking for your face. Move closer and face the camera.");
        detailEl.textContent = "";
        break;

      case "checking":
        setState("checking");
        placeBox(d.face);
        setSteps("done", "done", "active");
        fillMeter(d.streak);
        setStatus(`Checking... ${d.streak} of ${d.required}`);
        showDistance(d.distance, threshold);
        break;

      case "mismatch":
        setState("mismatch");
        placeBox(d.face);
        setSteps("done", "done", "active");
        fillMeter(0);
        setStatus("This face doesn't match. Face the camera in even light.", "bad");
        showDistance(d.distance, threshold);
        break;

      case "verified":
        setState("verified");
        placeBox(d.face);
        setSteps("done", "done", "done");
        fillMeter(d.required, true);
        setStatus(`Verified. Welcome back${d.name ? ", " + d.name.split(" ")[0] : ""}.`, "ok");
        detailEl.textContent = "Taking you to your dashboard...";
        stopLoopKeepPreview();
        setTimeout(() => { window.location.href = d.redirect || "/dashboard"; }, 1100);
        break;

      case "failed":
        fail("Face not recognised. Check the lighting, face the camera, and try again.");
        break;

      case "locked":
        showLocked(d.retry_after);
        break;

      case "expired":
        fail("Verification timed out. Try again.");
        break;
    }
  }

  function stopLoopKeepPreview() { running = false; actionBtn.hidden = true; }

  function fail(message) {
    stopCamera();
    setState("error");
    setSteps("pending", "pending", "pending");
    fillMeter(0);
    veilText.textContent = "Camera is off";
    setStatus(message, "bad");
    detailEl.textContent = "";
    actionBtn.hidden = false;
    setButton("Try again", "start");
  }

  function showLocked(seconds) {
    stopCamera();
    setState("error");
    setSteps("pending", "pending", "pending");
    fillMeter(0);
    detailEl.textContent = "Too many unsuccessful attempts.";
    actionBtn.hidden = false;
    actionBtn.disabled = true;
    actionBtn.textContent = "Try again";
    let left = Number(seconds) || 0;
    clearInterval(lockTimer);
    const tick = () => {
      if (left <= 0) {
        clearInterval(lockTimer);
        setStatus("You can try again now.");
        detailEl.textContent = "";
        setButton("Try again", "start");
        return;
      }
      const m = Math.floor(left / 60);
      const s = String(left % 60).padStart(2, "0");
      setStatus(`Locked for security. Try again in ${m}:${s}.`, "bad");
      left -= 1;
    };
    tick();
    lockTimer = setInterval(tick, 1000);
  }

  /* ---------------- controls ---------------- */
  actionBtn.addEventListener("click", () => {
    if (actionBtn.dataset.kind === "stop") {
      stopCamera();
      setState("idle");
      setSteps("pending", "pending", "pending");
      fillMeter(0);
      setStatus("Camera stopped. Press Start camera when you're ready.");
      detailEl.textContent = "";
      setButton("Start camera", "start");
    } else {
      setState("idle");
      startCamera();
    }
  });

  window.addEventListener("pagehide", stopCamera);

  buildMeter(required);
  setButton("Start camera", "start");
})();
