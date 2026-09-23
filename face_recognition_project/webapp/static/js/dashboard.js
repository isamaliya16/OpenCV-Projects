/* FaceKey dashboard: purely cosmetic client-side behaviour.
   The session itself is enforced server-side (cookie + expiry check on every request). */
(() => {
  "use strict";

  // Render UTC timestamps from the server in the visitor's own locale/timezone.
  document.querySelectorAll(".js-time").forEach((el) => {
    const d = new Date(el.getAttribute("datetime"));
    if (!isNaN(d)) {
      el.textContent = d.toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
      });
      el.title = d.toString();
    }
  });

  // Countdown is a convenience display only; expiry is checked server-side.
  const el = document.getElementById("countdown");
  if (el) {
    let left = Number(el.dataset.seconds) || 0;
    const tick = () => {
      const m = Math.floor(Math.max(0, left) / 60);
      const s = String(Math.max(0, left) % 60).padStart(2, "0");
      el.textContent = `${m}:${s}`;
      if (left <= 0) {
        clearInterval(timer);
        window.location.reload();  // server will bounce to /login once the session is gone
        return;
      }
      left -= 1;
    };
    tick();
    const timer = setInterval(tick, 1000);
  }

  // Position the gauge marker from data attributes (avoids inline styles / CSP issues).
  const gauge = document.getElementById("gauge");
  if (gauge) {
    const distance = parseFloat(gauge.dataset.distance || "0");
    const limit = parseFloat(gauge.dataset.limit || "1");
    const pct = Math.max(2, Math.min(98, (distance / (limit * 2)) * 100));
    gauge.querySelector(".gauge-marker").style.setProperty("--pos", pct + "%");
  }

  // ---------------- account: update profile ----------------
  const form = document.getElementById("accountForm");
  if (form) {
    const nameField = document.getElementById("aName");
    const emailField = document.getElementById("aEmail");
    const saveBtn = document.getElementById("accountSave");
    const savedNote = document.getElementById("accountSaved");

    const setError = (id, message) => {
      const el = document.getElementById(id);
      el.textContent = message || "";
      el.hidden = !message;
    };

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      setError("aNameError", "");
      setError("aEmailError", "");
      savedNote.hidden = true;
      saveBtn.disabled = true;
      saveBtn.textContent = "Saving...";

      try {
        const res = await fetch("/api/account/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ display_name: nameField.value.trim(), email: emailField.value.trim() }),
        });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
          if (data.errors) {
            if (data.errors.display_name) setError("aNameError", data.errors.display_name);
            if (data.errors.email) setError("aEmailError", data.errors.email);
          } else {
            setError("aNameError", "Could not save changes. Try again.");
          }
          return;
        }

        document.querySelectorAll(".user-name").forEach((el) => { el.textContent = data.display_name; });
        savedNote.hidden = false;
      } catch (_) {
        setError("aNameError", "Could not reach the server.");
      } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = "Save changes";
      }
    });
  }

  // ---------------- account: delete ----------------
  const deleteBtn = document.getElementById("deleteAccount");
  if (deleteBtn) {
    deleteBtn.addEventListener("click", async () => {
      const sure = window.confirm(
        "Delete your account? This removes your profile and enrolled face photos, and can't be undone."
      );
      if (!sure) return;

      deleteBtn.disabled = true;
      deleteBtn.textContent = "Deleting...";
      try {
        const res = await fetch("/api/account/delete", { method: "POST" });
        const data = await res.json().catch(() => ({}));
        window.location.href = (res.ok && data.redirect) ? data.redirect : "/login";
      } catch (_) {
        deleteBtn.disabled = false;
        deleteBtn.textContent = "Delete my account";
        window.alert("Could not reach the server. Try again.");
      }
    });
  }
})();
