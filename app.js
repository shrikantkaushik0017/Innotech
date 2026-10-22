// SilentAlert - simple vanilla JavaScript (no build step needed)
import { AudioClassifier, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/+esm";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";
const THRESHOLD = 0.25;   // minimum model score to raise an alert (lower = more sensitive)
const COOLDOWN_MS = 6000; // ignore the same sound again for 6 seconds

// Our sound categories and the YAMNet labels that belong to each one
const SOUNDS = {
  doorbell: { name: "Doorbell", icon: "🔔", labels: ["Doorbell", "Ding-dong"] },
  knocking: { name: "Knocking", icon: "🚪", labels: ["Knock"] },
  barking:  { name: "Dog barking", icon: "🐕", labels: ["Bark", "Dog", "Bow-wow"] },
  siren:    { name: "Siren", icon: "🚨", labels: ["Siren"] },
};
const PRI = { low: "▽ LOW", medium: "◆ MEDIUM", high: "▲ HIGH" };

const $ = (id) => document.getElementById(id);
const store = {
  get(k, fallback) { try { return JSON.parse(localStorage.getItem(k)) ?? fallback; } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ---------- Settings, profiles and history (saved in LocalStorage) ----------
function defaultProfiles() {
  const mk = (on) => Object.fromEntries(Object.keys(SOUNDS).map((k) => [k, { on: on.includes(k), pri: "medium" }]));
  const p = { home: mk(["doorbell", "knocking", "siren"]), focus: mk(["doorbell", "siren"]), custom: mk(Object.keys(SOUNDS)) };
  for (const n of Object.keys(p)) { p[n].barking.pri = "low"; p[n].siren.pri = "high"; }
  return p;
}
let profiles = store.get("sa_profiles", defaultProfiles());
let current = store.get("sa_profile", "home");
let settings = { size: "16", hc: false, rm: false, flash: false, vib: false, speak: false, notify: true, ...store.get("sa_settings", {}) };
let history = store.get("sa_history", []);
let mode = "live";

// ---------- Audio state ----------
let classifier = null, stream = null, ctx = null, analyser = null, proc = null, running = false;
const lastSeen = {};

// ---------- Visualizer bars ----------
const BAR_COUNT = 28;
for (let i = 0; i < BAR_COUNT; i++) $("bars").appendChild(document.createElement("span"));
const bars = [...$("bars").children];

function drawBars() {
  if (!running) { bars.forEach((b) => (b.style.height = "4px")); $("level").style.width = "0"; return; }
  const f = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(f);            // real microphone data
  const step = Math.floor(f.length / BAR_COUNT / 2) || 1;
  let sum = 0;
  bars.forEach((b, i) => { const v = f[i * step]; sum += v; b.style.height = 4 + (v / 255) * 86 + "px"; });
  $("level").style.width = Math.min(100, (sum / BAR_COUNT / 255) * 250) + "%";
  requestAnimationFrame(drawBars);
}

// ---------- Start / stop listening ----------
async function loadModel() {
  if (classifier) return;
  $("aiStatus").textContent = "Loading AI model...";
  const files = await FilesetResolver.forAudioTasks(WASM);
  classifier = await AudioClassifier.createFromOptions(files, {
    baseOptions: { modelAssetPath: MODEL }, maxResults: 10,
  });
  $("aiStatus").textContent = "AI model ready (YAMNet, running on this device).";
  $("pModel").textContent = "ready";
}

async function startListening() {
  prepareAlerts();
  if (!navigator.mediaDevices?.getUserMedia) return fail("This browser cannot use the microphone (or the page is not on HTTPS).");
  $("startBtn").disabled = true;
  $("statusText").textContent = "Starting...";
  try {
    await loadModel();
  } catch (e) {
    $("aiStatus").textContent = "AI model failed to load: " + e.message + ". Check your internet, or use Demo mode.";
    $("startBtn").disabled = false; $("statusText").textContent = "Not listening"; return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true }); // asked only after the click
  } catch (e) {
    const msg = e.name === "NotFoundError" ? "No microphone was found." :
      e.name === "NotAllowedError" ? "Microphone permission was denied. Allow it in the browser address bar." :
      "Could not open the microphone: " + e.message;
    $("startBtn").disabled = false; return fail(msg);
  }
  ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  analyser = ctx.createAnalyser(); analyser.fftSize = 512;
  proc = ctx.createScriptProcessor(16384, 1, 1); // gives us chunks of raw audio for the model
  proc.onaudioprocess = (e) => {
    if (!running) return;
    const result = classifier.classify(e.inputBuffer.getChannelData(0), ctx.sampleRate); // MediaPipe resamples to 16 kHz
    handlePrediction(result[0]?.classifications[0]?.categories || []);
  };
  source.connect(analyser); source.connect(proc); proc.connect(ctx.destination);
  running = true;
  $("statusText").textContent = "● Listening (microphone active)";
  $("stopBtn").disabled = false;
  updatePrivacy(); drawBars();
}

function stopListening() {
  running = false;
  if (proc) { proc.disconnect(); proc.onaudioprocess = null; proc = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; } // releases the mic
  if (ctx) { ctx.close(); ctx = null; }
  $("statusText").textContent = "Not listening";
  $("startBtn").disabled = mode !== "live"; $("stopBtn").disabled = true;
  $("prediction").textContent = "Latest model prediction: none yet.";
  updatePrivacy(); drawBars();
}

function fail(msg) { $("statusText").textContent = "⚠ " + msg; }


// ---------- Sound + desktop pop-up (works when you switch tabs) ----------
let beepCtx = null;
function prepareAlerts() { // call this inside a button click so the browser allows sound and pop-ups
  if (!settings.notify) return;
  try { beepCtx = beepCtx || new AudioContext(); beepCtx.resume(); } catch {}
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
}
function beep(pri) {
  if (!beepCtx) return;
  const count = pri === "high" ? 3 : pri === "medium" ? 2 : 1;
  for (let i = 0; i < count; i++) {
    const o = beepCtx.createOscillator(), g = beepCtx.createGain();
    const t = beepCtx.currentTime + i * 0.3;
    o.frequency.value = pri === "high" ? 988 : 784;
    o.connect(g); g.connect(beepCtx.destination);
    g.gain.setValueAtTime(0.2, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    o.start(t); o.stop(t + 0.22);
  }
}
function desktopPopup(title, body) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const n = new Notification(title, { body, tag: "silentalert" });
  n.onclick = () => { window.focus(); n.close(); };
}

// ---------- Turning predictions into alerts ----------
function handlePrediction(cats) {
  if (!cats.length) return;
  const top = cats[0];
  $("prediction").textContent = `Latest model prediction: ${top.categoryName} (${Math.round(top.score * 100)}%)`;
  for (const [key, s] of Object.entries(SOUNDS)) {
    const hits = cats.filter((c) => s.labels.includes(c.categoryName));
    if (!hits.length) continue;
    const best = Math.max(...hits.map((h) => h.score));
    const now = Date.now();
    if (best < THRESHOLD || !profiles[current][key].on) continue;   // filter really controls alerts
    if (now - (lastSeen[key] || 0) < COOLDOWN_MS) continue;          // avoid duplicate alerts
    lastSeen[key] = now;
    raiseAlert(key, "live", best);
  }
}

function raiseAlert(key, kind, score) {
  const s = SOUNDS[key], pri = profiles[current][key].pri;
  const ev = { id: Date.now(), sound: key, time: new Date().toISOString(), mode: kind, score: score ?? null, pri };
  history.unshift(ev); history = history.slice(0, 100); store.set("sa_history", history);
  const when = new Date(ev.time).toLocaleTimeString();
  $("alertPanel").hidden = false;
  $("alertPanel").className = "alert p-" + pri;
  $("alertPanel").hidden = false;
  $("alertTitle").textContent = `${s.icon} ${s.name}`;
  $("alertPri").textContent = PRI[pri];
  $("alertText").textContent = kind === "live"
    ? `The model heard a sound like "${s.name}". This is a classification, not proof.`
    : `Sample event for demonstration.`;
  $("alertMeta").textContent = `${when} · ${kind === "live" ? "Detected live · model score " + Math.round(score * 100) + "%" : "SIMULATED (demo)"}`;
  if (settings.flash) { document.body.classList.remove("flash"); void document.body.offsetWidth; document.body.classList.add("flash"); }
  if (settings.vib && navigator.vibrate) navigator.vibrate(pri === "high" ? [200, 100, 200] : 200);
  if (settings.speak && "speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(`${s.name}. ${pri} priority.${kind === "demo" ? " Demo." : ""}`));
  if (settings.notify) {
    beep(pri);
    if (document.hidden || !document.hasFocus()) desktopPopup(`${s.icon} ${s.name} (${pri} priority)`, kind === "live" ? "Detected live by SilentAlert" : "Demo: simulated event");
  }
  renderHistory();
}

// ---------- Rendering ----------
function renderSounds() {
  $("soundList").innerHTML = "";
  for (const [key, s] of Object.entries(SOUNDS)) {
    const row = document.createElement("div"); row.className = "sound";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = profiles[current][key].on; cb.id = "on_" + key;
    const lab = document.createElement("label"); lab.htmlFor = cb.id; lab.append(cb, ` ${s.icon} ${s.name}`);
    const sel = document.createElement("select"); sel.setAttribute("aria-label", s.name + " priority");
    for (const p of Object.keys(PRI)) sel.add(new Option(PRI[p], p, false, p === profiles[current][key].pri));
    cb.onchange = () => { profiles[current][key].on = cb.checked; store.set("sa_profiles", profiles); };
    sel.onchange = () => { profiles[current][key].pri = sel.value; store.set("sa_profiles", profiles); };
    row.append(lab, sel); $("soundList").append(row);
  }
}

function renderHistory() {
  const f = $("historyFilter").value;
  const shown = history.filter((e) => f === "all" || e.sound === f);
  $("historyList").innerHTML = "";
  if (!shown.length) $("historyList").innerHTML = "<li class='muted'>No events yet.</li>";
  for (const e of shown) {
    const li = document.createElement("li");
    li.textContent = `${SOUNDS[e.sound].icon} ${SOUNDS[e.sound].name} · ${new Date(e.time).toLocaleString()} · ` +
      `${e.mode === "live" ? "Live, " + Math.round(e.score * 100) + "%" : "Demo (simulated)"} · ${PRI[e.pri]}`;
    $("historyList").append(li);
  }
  const live = history.filter((e) => e.mode === "live"), counts = {};
  live.forEach((e) => (counts[e.sound] = (counts[e.sound] || 0) + 1));
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  $("stats").textContent = `Saved events: ${history.length} (live ${live.length}, demo ${history.length - live.length}). ` +
    (top ? `Most frequent live sound: ${SOUNDS[top[0]].name} (${top[1]}).` : "No live detections yet.");
}

function updatePrivacy() {
  $("pMic").textContent = running ? "ON" : "off";
  $("pLive").textContent = running ? "active" : "inactive";
  $("pModel").textContent = classifier ? "ready" : "not loaded";
  $("privacyChip").textContent = running ? "● Mic on" : "● Mic off";
}

function applySettings() {
  document.documentElement.style.fontSize = settings.size + "px";
  document.body.classList.toggle("hc", settings.hc);
  document.body.classList.toggle("rm", settings.rm);
  $("textSize").value = settings.size; $("hc").checked = settings.hc; $("rm").checked = settings.rm;
  $("flash").checked = settings.flash; $("vib").checked = settings.vib; $("speak").checked = settings.speak; $("notify").checked = settings.notify;
}

function setMode(m) {
  if (running) stopListening();
  mode = m;
  $("modeBadge").textContent = m === "live" ? "LIVE MODE" : "DEMO MODE (simulated)";
  $("modeBadge").className = "badge " + m;
  $("demoBox").hidden = m !== "demo";
  $("startBtn").disabled = m !== "live";
}

// ---------- Wire up buttons ----------
for (const k of Object.keys(SOUNDS)) $("historyFilter").add(new Option(SOUNDS[k].name, k));
$("startBtn").onclick = startListening;
$("stopBtn").onclick = stopListening;
$("privStop").onclick = stopListening;
$("dismissBtn").onclick = () => ($("alertPanel").hidden = true);
$("clearBtn").onclick = () => { history = []; store.set("sa_history", history); renderHistory(); };
$("historyFilter").onchange = renderHistory;
$("profileSelect").onchange = (e) => { current = e.target.value; store.set("sa_profile", current); renderSounds(); };
$("resetBtn").onclick = () => { profiles = defaultProfiles(); store.set("sa_profiles", profiles); renderSounds(); };
document.querySelectorAll("input[name=mode]").forEach((r) => (r.onchange = () => setMode(r.value)));
document.querySelectorAll(".demo").forEach((b) => (b.onclick = () => {
  prepareAlerts();
  if (!profiles[current][b.dataset.sound].on) { $("alertPanel").hidden = true; return alert("That sound is switched off in the current profile."); }
  raiseAlert(b.dataset.sound, "demo", null);
}));
$("textSize").onchange = (e) => save("size", e.target.value);
for (const id of ["hc", "rm", "flash", "vib", "speak", "notify"]) $(id).onchange = (e) => save(id, e.target.checked);
function save(k, v) { settings[k] = v; store.set("sa_settings", settings); applySettings(); }

if (!navigator.vibrate) { $("vib").disabled = true; $("vibNote").textContent = "(not supported on this device)"; }
if (!("speechSynthesis" in window)) { $("speak").disabled = true; $("speakNote").textContent = "(not supported in this browser)"; }
window.addEventListener("pagehide", stopListening);
$("profileSelect").value = current;
applySettings(); renderSounds(); renderHistory(); setMode("live"); updatePrivacy(); drawBars();