# System design — multi-user registration + face-verified login

This document covers the web application in `webapp/`, built on top of the
project's collect → train → recognize pipeline (`01_collect_faces.py`,
`02_train_model.py`, `03_face_recognition.py`). The web app **replaces**
those scripts' single-hardcoded-person model with real registration: anyone
can create an account through the browser, and login identifies *which*
registered account a face belongs to, not just yes/no for one person.

## 1. Goal and scope

Let any number of people register an account by showing their face to a
camera, and sign back in the same way — with the server, not the browser,
deciding both **whether** a face is enrolled and **whose** it is. This is a
real multi-user system built entirely on the project's existing tools
(OpenCV Haar cascade + LBPH), not a mockup: registering, retraining, storing
accounts, and verifying logins all actually happen, and are covered by an
automated test suite that drives real HTTP requests with real face images
(see §11).

**Threat model, stated plainly:** this is meaningfully stronger than the
single-user demo it replaces (a stranger can no longer get in just because
one hardcoded name matched), but LBPH still has no liveness check. A printed
photo or a video of an enrolled user's face passes the same checks a live
face would. §8 and §10 spell out exactly where the line is.

## 2. High-level architecture

```
┌──────────────┐     HTTPS (frames as JPEG POSTs)      ┌─────────────────────────────┐
│   Browser    │ ─────────────────────────────────────▶│   Flask app (webapp/)       │
│ getUserMedia │◀───────────────────────────────────── │                              │
│ canvas.toBlob│   JSON: {state, face box, count/streak}│  routes.py                  │
└──────────────┘                                        │   ├─ registration endpoints │
                                                          │   └─ login endpoints        │
                                                          │  face_engine.py  ────────┐  │
                                                          │  enrollment.py           │  │
                                                          │  security.py             │  │
                                                          │  users_db.py  (accounts) │  │
                                                          │  db.py (audit log)       │  │
                                                          └──────────────────────────┼──┘
                                                                                     ▼
                                            ┌───────────────────────────────────────────────┐
                                            │ haarcascade_frontalface_default.xml            │
                                            │ trainer/face_model.yml  (LBPH, ALL users)       │
                                            │ dataset/<username>/*.jpg (per-user face crops)  │
                                            └───────────────────────────────────────────────┘
```

Two parallel flows share the same face engine and the same trust-boundary
rule (§3): **registration** (`/register`, `/api/register/*`) turns a run of
camera frames into a new account and folds it into the shared model;
**login** (`/login`, `/api/login/*`) turns a run of camera frames into a
verified session for whichever account they match.

## 3. Trust boundary — why the server decides, not the browser

The client is assumed hostile: a browser can be scripted, its JS overridden,
its DOM edited live. So for both registration and login, the server never
accepts a claim from the client about what a camera frame contains — only
raw pixels (a JPEG) — and computes the result itself, every time.

- `POST /api/login/frame` and `POST /api/register/frame` both take a JPEG
  body and an opaque, server-issued id (`X-Attempt-Id` /
  `X-Enrollment-Id`). They return a **description** (state, face box, match
  distance) — never something the client could forge into "verified".
- The session cookie that marks someone as logged in is only ever set inside
  the login route, immediately after the server's own count of consecutive
  matching frames — for one specific account — reaches the required streak.
  There is no client-writable field that flips this.
- The dataset files a registration produces, the account row in
  `users.db`, and the retrained model are all written by the server, from
  images the server itself decoded and cropped — the client sends frames,
  never files or crops.
- The cookie is `HttpOnly` + `SameSite=Lax` and signed (Flask's default
  itsdangerous signing with `SECRET_KEY`), so JS on the page can't read or
  forge it.

## 4. Identity model — from one hardcoded person to a users table

The original scripts baked `PERSON_NAME` / `PERSON_LABEL` into `config.py`.
The web app moves identity into a **SQLite `users` table**
(`webapp/users_db.py`): each row is `(label, username, display_name, email,
created_at, sample_count, source)`, where `label` is the same integer LBPH
uses internally. Registration allocates the next free label; nothing else
about the recognizer changes.

**The pre-existing single-user dataset keeps working.** On first run,
`sync_legacy_user()` checks whether the `users` table is empty and
`dataset/ayush/` (from `config.PERSON_DIR_NAME`) has images; if so, it
imports that person as the first account — same label (`config.PERSON_LABEL`
= 1), same name, marked `source='legacy'`. This runs once; every later start
sees the table is non-empty and does nothing. Nothing about the original
`01_collect_faces.py` / `02_train_model.py` scripts changes or is required
going forward — they're just no longer the *only* way to add someone.

## 5. Registration pipeline

1. **`POST /api/register/start`** — the person submits a username, display
   name, and optional email. The server validates format
   (`enrollment.validate_registration`: lowercase-slug username, 3-32 chars;
   non-empty name; email regex if present) and uniqueness
   (`UsersStore.username_taken` / `email_taken`, case-insensitive), then
   opens an `EnrollmentSession` (`enrollment.py`) — an in-memory record, not
   yet anything on disk.
2. **`POST /api/register/frame`** (repeated) — each camera frame runs
   through `FaceEngine.detect_for_enrollment`, which is **stricter than
   login's detector**: it requires exactly one person in frame. A second
   face that's both reasonably confident (≥50% of the primary detection's
   Haar confidence) and reasonably sized (≥35% of its area) aborts that
   frame with `"multiple_faces"` — a bystander can never get folded into
   someone else's enrollment. A good frame is cropped to `FACE_SIZE` and,
   if enough time has passed since the last accepted sample
   (`ENROLL_MIN_SAMPLE_INTERVAL`, natural pose variety instead of 20 near-
   identical frames), kept in memory. Once `ENROLL_TARGET_SAMPLES` (20) are
   collected, the session is `"ready"`.
3. **Duplicate-account guard** — the instant a session goes `"ready"`, up to
   5 of its captured samples are run through the *current* model's
   `predict()` (§7). If they consistently land close to an existing
   account (stricter threshold than login's — see §7), the session becomes
   `"duplicate_hold"` instead: the person sees who it resembles and can
   either stop or explicitly confirm they're a different person.
4. **`POST /api/register/complete`** — only now does anything reach disk:
   the captured crops are written to `dataset/<username>/`, a row is
   inserted into `users`, and `FaceEngine.retrain()` rebuilds the LBPH model
   from **every** account's saved crops and atomically swaps it in
   (`os.replace`, so concurrent logins never see a half-written model file).
   A `"duplicate_hold"` session requires `force: true` in this call to
   proceed — the guard warns, it doesn't block.

Registration also has its own throttle (`RegistrationThrottle`, IP-based:
`MAX_REGISTRATIONS_PER_IP` per hour) separate from login's lockout — a
different kind of abuse (spamming new accounts) with a different, milder
response (no accounts created, not "locked out").

## 6. Login pipeline

Unchanged in spirit from the single-user version, generalized to identify
*whichever* account matches:

1. `FaceEngine.analyze()` runs the Haar cascade, takes the single
   highest-confidence detection (never falls back to a second, weaker face),
   crops it, and calls `LBPHFaceRecognizer.predict()` — which, trained on
   every account's images, returns the *nearest* label across all of them
   plus a distance.
2. A frame counts as a match only if that distance is within
   `MATCH_THRESHOLD` (70).
3. **The streak is per-identity, not just "matched someone".** `Attempt`
   (`security.py`) tracks `matched_label` alongside the streak count: every
   frame in a row must match the **same** label. If the best-matching
   identity changes mid-streak (e.g. the camera catches a different
   registered person, or a borderline frame flips to a different nearest
   label), that streak is abandoned and a new one starts from that frame —
   a login can never be assembled by borrowing frames across two different
   people. This is covered directly by a unit test
   (`test_streak_does_not_cross_identities`).
4. Once the streak reaches `REQUIRED_STREAK` (5), the server looks up that
   label in `users`, clears and reissues the session (fixation-safe — see
   the single-user doc's §6 reasoning, unchanged), and returns the
   account's display name so the UI can say "Welcome back, \_\_\_" without
   the browser ever having known who it was checking for in advance.

If `users` is empty (fresh install, nobody registered yet), `POST
/api/login/start` returns `409 {"state": "no_users"}` and the login page
shows a "no one is registered yet" state with a link to `/register` instead
of a camera button — there's nothing to verify against.

## 7. Abuse resistance

Mostly unchanged from the single-user version (per-IP lockout after
`MAX_FAILED_ATTEMPTS`, frame-rate throttling, one live attempt per client,
1MB frame cap — see the route code in `routes.py` and `security.py`), with
one addition specific to multi-user:

- **Failed attempts are attributed to the account they most resembled.**
  `Attempt.best_label` tracks the label with the single lowest distance seen
  across the *whole* attempt (even frames that didn't clear the match
  threshold), separately from the streak's `matched_label`. A failed or
  locked-out login is logged against that nearest account — so a person
  can see on their own dashboard "5 failed attempts most closely resembled
  your face" as a real security signal, the way "failed login attempts"
  shows up on real account-security pages. It's informational only: it
  never contributes to that account's streak or grants anything.
- **Duplicate-account guard threshold vs. login threshold.** These are
  deliberately two different numbers doing two different jobs.
  `MATCH_THRESHOLD` (70) answers "is this frame this account, for login
  purposes" and errs toward not locking someone out over one bad-lighting
  frame, since the required streak (§6) is what actually protects login.
  `DUPLICATE_GUARD_THRESHOLD` (45, stricter) answers "should we interrupt
  registration to ask 'is this you already?'" — it only has to be confident
  enough to warn, and the person can always confirm and continue, so it can
  afford to be pickier and still fail safe in both directions (a false
  warning costs one click; a missed one just means two accounts exist for
  the same face, which login handles as described next).
- **What a duplicate-guard override means for login afterward.** If someone
  deliberately continues past the warning, two accounts now share
  essentially the same face. LBPH's `predict()` still only returns one
  nearest label, so that face will reliably log in as *one* of those
  accounts (whichever the model scores marginally closer) rather than
  either nondeterministically or both — but which one is not something the
  person chooses at login time. This is a known, narrow edge case of
  allowing the override at all, not a security hole: it can't be used to
  access an account that wasn't already confirmed to belong to the same
  face.

## 8. What this still doesn't cover

Unchanged from the single-user version's honest limitations:

- **No liveness/anti-spoof detection.** A printed photo or a replayed video
  of an enrolled user's face passes the same checks a live face would, for
  both login and — now also — registration (someone could enroll a photo of
  someone else). Real deployments need passive (texture/frequency analysis)
  or active (prompted blink/head-turn) liveness checks.
- **Lockout and lockout throttling are per-IP, in-process memory.** Fine
  for a single worker/demo; see §9 for the path to shared state.
- **The duplicate-account guard is a warning, not a hard block** — by
  design (§7), but worth restating: it doesn't prevent someone from
  enrolling a face that isn't their own if they're willing to click through.

## 9. Data handling and audit log

Unchanged in principle from the single-user version, extended with identity:

- Camera frames are decoded in memory and discarded — for both login and
  registration — never written to disk. Only the final, confirmed
  registration crops are saved (`dataset/<username>/`), and only after the
  person completes signup.
- The audit log (`webapp/db.py`, SQLite, WAL mode) now records which
  account each event relates to (`label`, nullable — e.g. a lockout that
  never resolved to a clear "closest" account has none), plus a
  `"registered"` event type. Each dashboard only ever queries its own
  account's `label` (`recent_for_label` / `stats_for_label`) — nobody sees
  another account's activity.
- No images, face embeddings, or biometric templates are stored anywhere
  outside the LBPH model file itself and the saved training crops, both of
  which already existed in the original single-user project.

## 10. Deployment notes

Unchanged from the single-user version — dev server only
(`run_web.py`/Flask's built-in server), browsers require `https://` or
`localhost` for camera access, and `AttemptStore` / `LockoutTracker` /
`EnrollmentStore` / `RegistrationThrottle` are all in-process memory (fine
for one worker; move to Redis for multiple). One addition:

- **`FaceEngine.retrain()` is the one place multi-user adds real
  contention risk.** It's called synchronously at the end of
  `/api/register/complete`, holds the engine's lock only for the brief
  moment it swaps in the newly trained recognizer (not for the training
  itself, which runs on a freshly created object first), and writes the
  model file via `os.replace` so concurrent login requests either see the
  old model or the new one, never a partial file. At the dataset sizes this
  project deals with (tens to low hundreds of images per user), a full
  retrain is fast (well under a second in testing); if the user base grew
  into the thousands, this is the first thing to move off the request path
  (e.g. a background job queue) rather than optimize in place.

## 11. Where things live

```
webapp/
  __init__.py      app factory: wires users/engine/enrollment/audit stores,
                     runs the one-time legacy-user import at startup
  settings.py       config (shared detection params from config.py, plus
                     registration/enrollment policy)
  face_engine.py    detect + recognize (Haar + LBPH); retrain(); the
                     stricter single-face check used only by registration
  users_db.py       accounts table: create/lookup, username & email
                     uniqueness, legacy-dataset import
  enrollment.py     registration session state machine + field validation
  security.py       login Attempt state machine (per-identity streak),
                     per-IP lockout, registration rate limiter
  db.py             per-account audit log (SQLite)
  routes.py         /login, /register, /dashboard, /logout,
                     /api/login/*, /api/register/*
  templates/        login.html, register.html, dashboard.html, base.html, error.html
  static/           css/app.css, js/login.js, js/register.js, js/dashboard.js, fonts/, img/
run_web.py          dev-server launcher
tests/              pytest suite — engine, users_db, enrollment, security,
                     and full registration+login HTTP flows with real face images
```

## 12. Test coverage (what's actually been verified, not just written)

63 automated tests (`pytest tests/ -q`), all exercising real face images
(the project's own `dataset/ayush/*.jpg`, plus a public-domain sample photo
as a "different person") through the real Haar+LBPH pipeline — not mocked:

- **Engine** (`test_face_engine.py`): loads the real pretrained model;
  correctly matches/rejects real owner vs. stranger frames; the stricter
  enrollment path correctly accepts a single clear face and rejects a
  genuine two-person frame.
- **Users store** (`test_users_db.py`): label allocation, case-insensitive
  username/email uniqueness, and the legacy-import-runs-exactly-once
  behavior.
- **Security** (`test_security.py`): a streak verifies; a mismatch resets
  it; **a streak cannot cross from one identity to another**; lockout
  triggers and resets correctly; the registration throttle caps at its
  limit.
- **Enrollment** (`test_enrollment.py`): sample acceptance/target logic,
  field validation rules.
- **Full HTTP flows** (`test_routes.py`): register a brand-new account
  end-to-end and immediately log in as them; confirm registering a new user
  doesn't break an existing user's login; **register two different real
  identities and confirm each logs in as themselves, never as each other**;
  username/email conflicts rejected; a two-person frame rejected during
  registration; the duplicate-account guard both fires and can be
  overridden with `force`; a completely empty system correctly serves its
  first registration and that account's subsequent login; cold-start and
  no-attempt-id edge cases.

This was also driven manually through a real browser (Playwright, with the
camera mocked from a static image) end to end: registration's camera
capture UI, the success banner, logging in as the newly created account,
re-confirming the pre-existing legacy account still logs in correctly after
a retrain, and the duplicate-guard warning card with its "continue anyway"
path.
