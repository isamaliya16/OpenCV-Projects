/* FaceKey registration: collects account details, then captures a run of
   face samples and sends each to the server, which decides what counts as a
   good sample and when enrollment is complete. The browser never writes to
   the dataset or model itself. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const root = $("#registerRoot");
  const target = Number(root.dataset.target) || 20;

  /* ---------------- step 1: account form ---------------- */
  const formStep = $("#formStep");
  const captureStep = $("#captureStep");
  const uName = $("#fUsername");
  const uDisplay = $("#fName");
  const uEmail = $("#fEmail");
  const formError = $("#formError");
  const formSubmit = $("#formSubmit");
  const stepAccount = $("#step-account");
  const pageTitle = $("#pageTitle");
  const pageLede = $("#pageLede");

  function fieldError(id, message) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = message || "";
    el.hidden = !message;
  }

  function clearFieldErrors() {
    ["fUsernameError", "fNameError", "fEmailError"].forEach((id) => fieldError(id, ""));
    fieldError("formError", "");
  }

  let usernameCheckTimer = null;
  uName.addEventListener("input", () => {
    uName.value = uName.value.toLowerCase().replace(/[^a-z0-9._-]/g, "");
    fieldError("fUsernameError", "");
    clearTimeout(usernameCheckTimer);
    const val = uName.value;
    if (val.length < (Number(root.dataset.unameMin) || 3)) return;
    usernameCheckTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/register/username-available?u=${encodeURIComponent(val)}`);
        const data = await res.json();
        if (data.available === false && uName.value === val) {
          fieldError("fUsernameError", "That username is already taken.");
        }
      } catch (_) { /* offline check is best-effort only; server re-validates on submit */ }
    }, 350);
  });

  let enrollmentId = null;

  formStep.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFieldErrors();
    formSubmit.disabled = true;
    formSubmit.textContent = "Checking...";

    try {
      const res = await fetch("/api/register/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: uName.value.trim(),
          display_name: uDisplay.value.trim(),
          email: uEmail.value.trim(),
        }),
      });
      const data = await res.json().catch(() => ({}));

      if (res.status === 422 && data.errors) {
        if (data.errors.username) fieldError("fUsernameError", data.errors.username);
        if (data.errors.display_name) fieldError("fNameError", data.errors.display_name);
        if (data.errors.email) fieldError("fEmailError", data.errors.email);
        return;
      }
      if (res.status === 429) {
        fieldError("formError", "Too many accounts created from this network recently. Try again later.");
        return;
      }
      if (!res.ok) {
        fieldError("formError", "Something went wrong. Try again.");
        return;
      }

      enrollmentId = data.enrollment_id;
      enterCaptureStep();
    } catch (_) {
      fieldError("formError", "Could not reach the server. Check your connection.");
    } finally {
      formSubmit.disabled = false;
      formSubmit.textContent = "Continue to camera";
    }
  });

  function enterCaptureStep() {
    formStep.hidden = true;
    captureStep.hidden = false;
    stepAccount.dataset.status = "done";
    stepCamera.dataset.status = "active";
    pageTitle.textContent = "Capture your face";
    pageLede.textContent = `We'll take ${target} quick photos as you move your head slightly. Look at the camera in good light.`;
    setStatus("Press Start camera to begin.");
  }

  /* ---------------- step 2: camera capture ---------------- */
  const stage = $("#stage");
  const video = $("#video");
  const faceBox = $("#faceBox");
  const statusEl = $("#status");
  const detailEl = $("#detail");
  const veilText = $("#veil-text");
  const actionBtn = $("#action");
  const backBtn = $("#backBtn");
  const meter = $("#meter");
  const dupCard = $("#dupCard");
  const dupText = $("#dupText");
  const dupCancel = $("#dupCancel");
  const dupContinue = $("#dupContinue");
  const stepCamera = $("#step-camera");
  const stepCapture = $("#step-capture");

  const HINTS = [
    "Look straight at the camera.",
    "Turn your head slightly left.",
    "Turn your head slightly right.",
    "Tilt your chin up a little.",
    "Tilt your chin down a little.",
    "Back to center, relax your face.",
  ];

  const FRAME_INTERVAL_MS = 220;
  const MAX_SEND_WIDTH = 640;
  const JPEG_QUALITY = 0.85;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  let stream = null;
  let running = false;
  let netErrors = 0;
  let hintTimer = null;
  let lastCount = 0;

  function setState(state) { stage.dataset.state = state; }
  function setStatus(text, tone) {
    statusEl.textContent = text;
    statusEl.classList.toggle("is-bad", tone === "bad");
    statusEl.classList.toggle("is-ok", tone === "ok");
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
    const left = 1 - (face.x + face.w);
    faceBox.style.left = `${left * 100}%`;
    faceBox.style.top = `${face.y * 100}%`;
    faceBox.style.width = `${face.w * 100}%`;
    faceBox.style.height = `${face.h * 100}%`;
    faceBox.hidden = false;
  }

  function setButton(label, kind) {
    actionBtn.textContent = label;
    actionBtn.dataset.kind = kind;
    actionBtn.classList.toggle("btn-primary", kind !== "stop");
    actionBtn.classList.toggle("btn-quiet", kind === "stop");
    actionBtn.disabled = false;
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return fail("Camera access needs a secure page. Open this site over HTTPS, or at localhost.");
    }
    stepCamera.dataset.status = "done";
    stepCapture.dataset.status = "active";
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
    try { await video.play(); } catch (_) {}
    if (video.videoWidth) stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    setState("searching");
    running = true;
    netErrors = 0;
    buildMeter(target);
    setButton("Stop camera", "stop");
    setStatus("Looking for your face. Face the camera in even light.");
    hintTimer = setInterval(rotateHint, 2600);
    loop();
  }

  function rotateHint() {
    if (!running) return;
    const i = Math.floor(Math.random() * HINTS.length);
    detailEl.textContent = HINTS[i];
  }

  function stopCamera() {
    running = false;
    clearInterval(hintTimer);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    placeBox(null);
  }

  function captureFrame() {
    return new Promise((resolve) => {
      const scale = Math.min(1, MAX_SEND_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
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
    const res = await fetch("/api/register/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg", "X-Enrollment-Id": enrollmentId },
      body: blob,
    });
    const data = await res.json().catch(() => ({}));

    if (res.status === 429 && data.error === "slow_down") return;
    if (res.status === 410) return fail("This session timed out. Start over.");
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    render(data);
  }

  function render(d) {
    if (typeof d.count === "number" && d.count !== lastCount) {
      lastCount = d.count;
      fillMeter(d.count);
    }

    switch (d.state) {
      case "searching":
        setState("searching");
        placeBox(null);
        setStatus(`Looking for your face. ${lastCount}/${target} captured.`);
        break;

      case "multiple_faces":
        setState("mismatch");
        placeBox(d.face);
        setStatus("Only one person should be in frame.", "bad");
        break;

      case "ok":
        setState("checking");
        placeBox(d.face);
        setStatus(`Capturing... ${lastCount}/${target}`);
        break;

      case "ready":
        finishCapture();
        break;

      case "duplicate_hold":
        finishCapture();
        break;

      case "expired":
        fail("This session took too long. Start over.");
        break;
    }
  }

  async function finishCapture() {
    running = false;
    clearInterval(hintTimer);
    setState("verified");
    fillMeter(target, true);
    setStatus("Got everything needed. Finishing up...");
    detailEl.textContent = "";

    try {
      const res = await fetch("/api/register/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enrollment_id: enrollmentId }),
      });
      const data = await res.json().catch(() => ({}));

      if (res.status === 409 && data.error === "duplicate_unconfirmed") {
        return showDuplicateCheck();
      }
      if (res.status === 409 && data.error === "username_taken") {
        return fail("That username was just taken by someone else. Go back and pick another.");
      }
      if (!res.ok) {
        return fail("Could not finish creating the account. Try again.");
      }
      window.location.href = data.redirect || "/login";
    } catch (_) {
      fail("Could not reach the server to finish setting up your account.");
    }
  }

  async function showDuplicateCheck() {
    // Re-fetch the flagged candidate via one more (throttle-safe) frame poll
    // is unnecessary — the server already told us on the last "ready" frame.
    setState("mismatch");
    dupCard.hidden = false;
    dupText.textContent = "This face looks similar to an account that's already registered. If that's not you, you can continue anyway.";
  }

  dupCancel.addEventListener("click", () => window.location.reload());
  dupContinue.addEventListener("click", async () => {
    dupCard.hidden = true;
    setStatus("Finishing up...");
    try {
      const res = await fetch("/api/register/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enrollment_id: enrollmentId, force: true }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) return fail("Could not finish creating the account. Try again.");
      window.location.href = data.redirect || "/login";
    } catch (_) {
      fail("Could not reach the server.");
    }
  });

  function fail(message) {
    stopCamera();
    setState("error");
    veilText.textContent = "Camera is off";
    setStatus(message, "bad");
    actionBtn.hidden = false;
    setButton("Try again", "start");
  }

  actionBtn.addEventListener("click", () => {
    if (actionBtn.dataset.kind === "stop") {
      stopCamera();
      setState("idle");
      setStatus("Camera stopped. Press Start camera when you're ready.");
      detailEl.textContent = "";
      setButton("Start camera", "start");
    } else {
      setState("idle");
      startCamera();
    }
  });

  backBtn.addEventListener("click", () => {
    stopCamera();
    captureStep.hidden = true;
    formStep.hidden = false;
    stepAccount.dataset.status = "active";
    stepCamera.dataset.status = "pending";
    stepCapture.dataset.status = "pending";
    pageTitle.textContent = "Create your account";
    pageLede.textContent = "Pick a username, then capture a set of face photos. We'll use these to recognise you next time you sign in.";
  });

  window.addEventListener("pagehide", stopCamera);
})();
