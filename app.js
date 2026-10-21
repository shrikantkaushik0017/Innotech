
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const STORAGE_KEY = "silentalert-events-v1";
const PREF_KEY = "silentalert-preferences-v1";

// YAMNet model and MediaPipe WASM assets
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";

const soundDetails = {
  doorbell: { name: "Doorbell", emoji: "🔔" },
  knocking: { name: "Knocking", emoji: "🚪" },
  barking: { name: "Dog barking", emoji: "🐕" },
  siren: { name: "Siren", emoji: "🚨" }
};

// These are candidate YAMNet label matches, not guarantees.
// Doorbell and knocking performance must be tested.
const soundAliases = {
  doorbell: ["doorbell", "ding-dong", "ding dong"],
  knocking: ["knock", "door knock", "tap"],
  barking: ["bark", "barking", "dog"],
  siren: ["siren", "civil defense siren", "alarm"]
};

let stream = null;
let audioContext = null;
let analyser = null;
let source = null;
let processor = null;
let silentGain = null;
let animationId = null;
let classifier = null;
let modelPromise = null;
let isStarting = false;
let isListening = false;
let processingAudio = false;
let audioChunks = [];
let audioSamples = 0;
let lastAlertTimes = {};
let events = readEvents();

function readEvents() {
  try {
    const saved = JSON.parse(
      localStorage.getItem(STORAGE_KEY) || "[]"
    );
    return Array.isArray(saved) ? saved.slice(0, 50) : [];
  } catch {
    return [];
  }
}

function saveEvents() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(events.slice(0, 50))
    );
  } catch {
    showMessage("Browser storage is unavailable.");
  }
}

function showMessage(message) {
  $("#monitorMessage").textContent = message;
}

function setAIStatus(message) {
  $("#aiStatus").textContent = message;
}

function setListening(active) {
  isListening = active;
  $("#startBtn").disabled = active || isStarting;
  $("#stopBtn").disabled = !active;
  $("#statusBadge").classList.toggle("active", active);
  $("#statusText").textContent = active
    ? "Listening"
    : "Not listening";
  $("#soundVisual").classList.toggle("active", active);
}

function updateLevel(level) {
  const value = Math.round(level);
  $("#levelBar").style.width = `${value}%`;
  $("#levelText").textContent = `Input level: ${value}%`;
  $("#levelTrack").setAttribute("aria-valuenow", String(value));
}

function monitorAudio() {
  if (!analyser || !isListening) return;

  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);

  let sum = 0;

  for (const sample of samples) {
    const normalized = (sample - 128) / 128;
    sum += normalized * normalized;
  }

  const rms = Math.sqrt(sum / samples.length);
  updateLevel(Math.min(100, rms * 300));

  animationId = requestAnimationFrame(monitorAudio);
}

async function loadAI() {
  if (classifier) return classifier;
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    setAIStatus("Loading AI model… This may take a moment.");

    if (!window.AudioClassifier || !window.FilesetResolver) {
      throw new Error(
        "MediaPipe library was not loaded. Check your internet connection and the script in index.html."
      );
    }

    const fileset = await window.FilesetResolver.forAudioTasks(
      WASM_URL
    );

    classifier = await window.AudioClassifier.createFromOptions(
      fileset,
      {
        baseOptions: { modelAssetPath: MODEL_URL },
        maxResults: 5,
        scoreThreshold: 0.05
      }
    );

    setAIStatus("AI model loaded. Waiting for microphone audio.");
    return classifier;
  })();

  try {
    return await modelPromise;
  } catch (error) {
    modelPromise = null;
    classifier = null;
    throw error;
  }
}

function getTopCategories(result) {
  const classifications = result?.[0]?.classifications || [];

  return classifications
    .flatMap((group) => group.categories || [])
    .map((category) => ({
      label: category.displayName || category.categoryName || "",
      score: Number(category.score) || 0
    }))
    .sort((a, b) => b.score - a.score);
}

function matchSound(label) {
  const normalized = label.toLowerCase().trim();

  for (const [key, aliases] of Object.entries(soundAliases)) {
    const match = aliases.some((alias) => {
      if (alias === "dog") {
        return normalized === "dog" ||
          normalized.includes("dog bark") ||
          normalized.includes("dog growl");
      }

      return normalized === alias ||
        normalized.includes(alias);
    });

    if (match) return key;
  }

  return null;
}

function showPrediction(categories) {
  const best = categories[0];

  if (!best) {
    $("#predictionText").textContent =
      "No prediction available yet.";
    return;
  }

  $("#predictionText").textContent =
    `AI candidate: ${best.label} (${(best.score * 100).toFixed(1)}%). ` +
    "This is a model score, not a guarantee.";
}

function handlePrediction(categories) {
  showPrediction(categories);

  // Require a reasonably strong score before creating an alert.
  const MIN_CONFIDENCE = 0.35;
  const best = categories[0];

  if (!best || best.score < MIN_CONFIDENCE) return;

  const key = matchSound(best.label);
  if (!key) return;

  const enabled = $$('input[name="sound"]:checked')
    .map((input) => input.value);

  if (!enabled.includes(key)) return;

  // Avoid repeated alerts for the same class.
  const now = Date.now();
  if (now - (lastAlertTimes[key] || 0) < 8000) return;

  lastAlertTimes[key] = now;
  showAlert(key, false, best.label, best.score);
}

function classifyAudio(samples, sampleRate) {
  if (!classifier || processingAudio || !isListening) return;

  processingAudio = true;

  try {
    // The clip classifier handles audio clips and sample-rate
    // preprocessing. We supply the actual AudioContext sample rate.
    const result = classifier.classify(samples, sampleRate);
    const categories = getTopCategories(result);
    handlePrediction(categories);
  } catch (error) {
    console.error("Audio classification failed:", error);
    setAIStatus("Classification error. Check the browser console.");
  } finally {
    processingAudio = false;
  }
}

function handleAudioProcess(event) {
  if (!isListening || !classifier) return;

  const input = event.inputBuffer.getChannelData(0);
  audioChunks.push(new Float32Array(input));
  audioSamples += input.length;

  // Classify approximately one second at a time.
  const targetSamples = audioContext.sampleRate;

  if (audioSamples < targetSamples) return;

  const clip = new Float32Array(audioSamples);
  let offset = 0;

  for (const chunk of audioChunks) {
    clip.set(chunk, offset);
    offset += chunk.length;
  }

  audioChunks = [];
  audioSamples = 0;

  classifyAudio(clip, audioContext.sampleRate);
}

async function startListening() {
  if (isStarting || isListening) return;
  isStarting = true;
  $("#startBtn").disabled = true;

  try {
    showMessage("Loading the AI model…");
    setAIStatus("Preparing model and microphone.");

    // Load the model before requesting microphone audio.
    await loadAI();

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Microphone access is unavailable. Use Chrome on localhost or HTTPS."
      );
    }

    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      },
      video: false
    });

    const AudioContextClass =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error("Web Audio is not supported in this browser.");
    }

    audioContext = new AudioContextClass();
    await audioContext.resume();

    source = audioContext.createMediaStreamSource(stream);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    // ScriptProcessor is deprecated, but useful for this simple
    // beginner prototype. A production version should use AudioWorklet.
    processor = audioContext.createScriptProcessor(
      16384, 1, 1
    );
    processor.onaudioprocess = handleAudioProcess;

    // Keep the processor active without playing microphone audio
    // back through the speakers.
    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    audioChunks = [];
    audioSamples = 0;
    lastAlertTimes = {};

    setListening(true);
    setAIStatus("AI is running. Predictions will appear below.");
    showMessage("Listening for supported environmental sounds.");
    monitorAudio();
  } catch (error) {
    console.error("Could not start SilentAlert:", error);

    stopListening();

    setAIStatus("AI or microphone could not start.");

    showMessage(
      error.name === "NotAllowedError"
        ? "Microphone permission denied. Allow access in Chrome and retry."
        : error.message || "Could not start monitoring."
    );
  } finally {
    isStarting = false;
    $("#startBtn").disabled = isListening;
  }
}

function stopListening() {
  isListening = false;

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
  }
  animationId = null;

  if (processor) {
    processor.onaudioprocess = null;
    processor.disconnect();
  }
  processor = null;

  if (source) source.disconnect();
  source = null;

  if (silentGain) silentGain.disconnect();
  silentGain = null;
  analyser = null;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = null;

  if (audioContext && audioContext.state !== "closed") {
    audioContext.close().catch(() => {});
  }
  audioContext = null;

  audioChunks = [];
  audioSamples = 0;
  processingAudio = false;

  updateLevel(0);
  setListening(false);
  showMessage("Listening stopped.");

  if (classifier) {
    setAIStatus("AI model loaded. Start listening to resume.");
  }
}

function showAlert(soundKey, isDemo = true, label = "", score = 0) {
  const sound = soundDetails[soundKey];
  if (!sound) return;

  const enabled = $$('input[name="sound"]:checked')
    .map((input) => input.value);

  if (!enabled.includes(soundKey)) {
    showMessage(`${sound.name} alerts are disabled.`);
    return;
  }

  const event = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    sound: soundKey,
    time: new Date().toISOString(),
    source: isDemo ? "Demo" : "AI",
    label,
    score
  };

  events.unshift(event);
  events = events.slice(0, 50);
  saveEvents();
  renderHistory();

  $("#alertTitle").textContent = isDemo
    ? `${sound.name} demo alert`
    : `${sound.name} candidate detected`;

  $("#alertDescription").textContent = isDemo
    ? "Simulated event — not detected by the microphone or AI."
    : `Model label: ${label}. Score: ${(score * 100).toFixed(1)}%. Verify the sound yourself.`;

  $("#alertPanel").hidden = false;

  showMessage(
    isDemo
      ? `Demo notification displayed: ${sound.name}.`
      : `AI candidate detected: ${sound.name}.`
  );
}

function dismissAlert() {
  $("#alertPanel").hidden = true;
}

function renderHistory() {
  const list = $("#historyList");
  list.replaceChildren();

  if (events.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent =
      "No events yet. Try a demo alert or start AI monitoring.";
    list.append(empty);
    return;
  }

  for (const event of events) {
    const sound = soundDetails[event.sound];
    if (!sound) continue;

    const item = document.createElement("div");
    item.className = "history-item";

    const emoji = document.createElement("span");
    emoji.className = "history-emoji";
    emoji.textContent = sound.emoji;

    const details = document.createElement("div");
    details.className = "history-details";

    const name = document.createElement("strong");
    name.textContent = sound.name;

    const time = document.createElement("small");
    time.textContent = new Date(event.time).toLocaleString();

    details.append(name, time);

    const tag = document.createElement("span");
    tag.className = "history-tag";
    tag.textContent = event.source === "Demo"
      ? "SIMULATED"
      : "AI CANDIDATE";

    item.append(emoji, details, tag);
    list.append(item);
  }
}

function loadPreferences() {
  try {
    const preferences = JSON.parse(
      localStorage.getItem(PREF_KEY) || "{}"
    );

    $$('input[name="sound"]').forEach((input) => {
      if (typeof preferences[input.value] === "boolean") {
        input.checked = preferences[input.value];
      }
    });
  } catch {
    // Use defaults if saved preferences cannot be read.
  }
}

function savePreferences() {
  const preferences = {};

  $$('input[name="sound"]').forEach((input) => {
    preferences[input.value] = input.checked;
  });

  try {
    localStorage.setItem(PREF_KEY, JSON.stringify(preferences));
  } catch {
    showMessage("Preferences could not be saved.");
  }
}

$("#startBtn").addEventListener("click", startListening);
$("#stopBtn").addEventListener("click", stopListening);
$("#dismissBtn").addEventListener("click", dismissAlert);

$("#clearBtn").addEventListener("click", () => {
  events = [];
  saveEvents();
  renderHistory();
});

$$(".demo-btn").forEach((button) => {
  button.addEventListener("click", () => {
    showAlert(button.dataset.sound, true);
  });
});

$$('input[name="sound"]').forEach((input) => {
  input.addEventListener("change", savePreferences);
});

window.addEventListener("pagehide", stopListening);

loadPreferences();
renderHistory();
setListening(false);
setAIStatus("AI model not loaded yet.");
