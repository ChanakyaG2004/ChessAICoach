"use strict";

const csrf = document.querySelector('meta[name="coach-csrf"]').content;
const el = (id) => document.getElementById(id);
const els = {
  dot: el("connection-dot"), connection: el("connection-text"),
  turn: el("turn-title"), tag: el("turn-tag"), instruction: el("instruction"),
  checkMove: el("check-move-button"), checkStatus: el("move-check-status"),
  messages: el("messages"), thinking: el("thinking-indicator"),
  question: el("question"), send: el("send-button"),
  cameraTab: el("camera-tab"), expectedTab: el("expected-tab"),
  cameraBoard: el("camera-board"), expectedBoard: el("expected-board"),
  boardImage: el("board-image"), boardPlaceholder: el("board-placeholder"),
  digitalBoard: el("digital-board"), boardPill: el("board-live-pill"),
  boardExplainer: el("board-explainer"),
  human: el("human-side"), engine: el("engine-status"),
  detection: el("detection-status"), feedback: el("feedback"),
  moves: el("moves"), fen: el("fen"),
  recovery: el("recovery-panel"), recoveryPrompt: el("recovery-prompt"),
  confirm: el("confirm-button"), actionStatus: el("action-status"),
  manual: el("manual-move"), special: el("special-controls"),
  promotion: el("promotion-controls"), rook: el("rook-confirm"),
  mic: el("mic-button"), voiceStatus: el("voice-status"),
  voiceReplies: el("voice-replies"), stopSpeaking: el("stop-speaking")
};

let current = null;
let actionSending = false;
let messageSignature = "";
let imageRevision = -1;
let lastSpokenId = 0;
let voicePlayer = null;
let voiceObjectUrl = null;
let speakGeneration = 0;
let recorder = null;
let recordingStarting = false;
let transcribing = false;
let boardTab = "camera";
const extraSpeech = { input: null, output: null };

// A future local speech adapter can register these without changing chat.
window.CoachSpeech = {
  registerInputProvider(provider) { extraSpeech.input = provider; updateVoiceControls(); },
  registerOutputProvider(provider) { extraSpeech.output = provider; updateVoiceControls(); }
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    credentials: "same-origin",
    ...options,
    headers: { ...(options.headers || {}), "X-Coach-CSRF": csrf }
  });
  if (!response.ok) {
    let error = `Request failed (${response.status})`;
    try { error = (await response.json()).error || error; } catch (_) { /* keep status */ }
    throw new Error(error);
  }
  return response;
}

async function postJson(path, value) {
  return (await api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value)
  })).json();
}

function updateConnection(connected) {
  els.dot.classList.toggle("connected", connected);
  els.connection.textContent = connected ? "Connected to local session" : "Waiting for local session";
  if (!connected) {
    els.checkMove.disabled = true;
    els.confirm.disabled = true;
  }
}

function showActionStatus(text) {
  els.actionStatus.textContent = text;
}

function setBoardTab(name) {
  boardTab = name;
  const camera = name === "camera";
  els.cameraBoard.hidden = !camera;
  els.expectedBoard.hidden = camera;
  els.cameraTab.setAttribute("aria-selected", String(camera));
  els.expectedTab.setAttribute("aria-selected", String(!camera));
  els.boardPill.textContent = camera ? "CAMERA" : "EXPECTED";
}

const glyphs = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟"
};

function renderDigitalBoard(pieces) {
  const cells = [];
  for (let rank = 8; rank >= 1; rank--) {
    for (let file = 0; file < 8; file++) {
      const square = String.fromCharCode(97 + file) + rank;
      const piece = pieces[square] || "";
      const cell = document.createElement("div");
      cell.className = `square ${((rank + file) % 2) === 0 ? "light" : "dark"}`;
      if (piece) cell.classList.add(piece === piece.toUpperCase() ? "white-piece" : "black-piece");
      cell.setAttribute("aria-label", `${square}: ${piece || "empty"}`);
      cell.textContent = glyphs[piece] || "";
      if (file === 0) {
        const coord = document.createElement("span");
        coord.className = "coord";
        coord.textContent = String(rank);
        cell.append(coord);
      }
      if (rank === 1) {
        const coord = document.createElement("span");
        coord.className = "coord file-coord";
        coord.textContent = String.fromCharCode(97 + file);
        cell.append(coord);
      }
      cells.push(cell);
    }
  }
  els.digitalBoard.replaceChildren(...cells);
}

function renderMessages(messages) {
  const signature = messages.map((m) => `${m.id}:${m.status}:${m.text}`).join("\u0001");
  if (signature === messageSignature) return;
  const nearBottom = els.messages.scrollHeight - els.messages.scrollTop - els.messages.clientHeight < 100;
  messageSignature = signature;
  if (!messages.length) return;
  const nodes = messages.map((message) => {
    const card = document.createElement("article");
    card.className = `message ${message.role} ${message.status}`;
    const heading = document.createElement("div");
    heading.className = "message-label";
    const label = document.createElement("span");
    label.textContent = message.role === "user" ? "You" : "Coach";
    heading.append(label);
    if (message.role === "assistant" && message.status === "done") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "speak-button";
      button.textContent = "Listen";
      button.setAttribute("aria-label", "Read this reply aloud");
      button.addEventListener("click", () => speakText(message.text));
      heading.append(button);
    }
    const body = document.createElement("p");
    body.className = "message-body";
    body.textContent = message.text;
    card.append(heading, body);
    return card;
  });
  els.messages.replaceChildren(...nodes);
  if (nearBottom) els.messages.scrollTop = els.messages.scrollHeight;
}

function maybeAutoSpeak(messages) {
  for (const message of messages) {
    if (message.role !== "assistant" || message.status !== "done") continue;
    if (message.id <= lastSpokenId) continue;
    lastSpokenId = message.id;
    if (els.voiceReplies.checked) speakText(message.text);
  }
}

function renderState(state) {
  current = state;
  updateConnection(true);
  const turn = state.turn === "white" ? "White" : "Black";
  els.turn.textContent = state.game_over ? `Game over · ${state.result}` : `${turn} to move${state.in_check ? " · check" : ""}`;
  els.tag.textContent = state.pending_action ? "SETUP" : state.game_over ? "FINAL" : "LIVE";
  els.instruction.textContent = state.pending_action || state.instruction || "Wait for the board to settle.";
  els.human.textContent = `${state.human_color} at the top of the camera board`;
  els.engine.textContent = state.engine_status || "Unavailable";
  els.detection.textContent = state.detection_status || "Waiting";
  els.feedback.textContent = state.feedback || "Waiting for the next move";
  els.moves.textContent = state.moves.length ? state.moves.join(" · ") : "No moves yet";
  els.fen.textContent = state.fen;
  els.thinking.hidden = !state.busy;
  els.recovery.hidden = !state.pending_action;
  els.recoveryPrompt.textContent = state.pending_action || "";
  els.confirm.hidden = !state.pending_action;
  els.confirm.disabled = !state.recovery_ready || state.action_queued || actionSending;
  els.checkMove.hidden = Boolean(state.pending_action);
  els.checkMove.disabled = !state.move_check_ready || state.action_queued || actionSending;
  els.checkMove.textContent = state.move_check_pending ? "Checking…" : "Check my move";
  els.checkStatus.textContent = state.pending_action ? "Confirm the physical position first." :
    state.game_over ? "Start a new game to play again." :
    state.move_check_result || "After each move, clear your hand and press Check my move.";

  if (state.image_url) {
    if (state.image_revision !== imageRevision) {
      imageRevision = state.image_revision;
      els.boardImage.src = state.image_url;
    }
    els.boardImage.hidden = false;
    els.boardPlaceholder.hidden = true;
  } else {
    imageRevision = state.image_revision;
    els.boardImage.hidden = true;
    els.boardImage.removeAttribute("src");
    els.boardPlaceholder.hidden = false;
  }
  renderDigitalBoard(state.expected_pieces || state.pieces || {});
  els.boardExplainer.textContent = state.expected_fen && state.expected_fen !== state.fen ?
    "Expected pieces show the setup to arrange before confirming. The recorded position changes after you confirm." :
    "Make one complete move, clear your hand, then press Check my move. The camera checks the changed squares against the recorded position.";
  renderMessages(state.messages || []);
  maybeAutoSpeak(state.messages || []);

  const diagnostic = (state.detection_status || "").toLowerCase();
  const promotion = diagnostic.includes("promotion:");
  const rook = diagnostic.includes("unfinished castle") || diagnostic.includes("rook move");
  els.special.hidden = !promotion && !rook;
  els.promotion.hidden = !promotion;
  els.rook.hidden = !rook;
  updateVoiceControls();
}

async function refresh() {
  try {
    const state = await (await api("/api/state")).json();
    renderState(state);
  } catch (error) {
    updateConnection(false);
    els.voiceStatus.textContent = `Local session unavailable: ${error.message}`;
  }
}

async function sendQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed) return;
  els.send.disabled = true;
  try {
    await postJson("/api/chat", { question: trimmed });
    els.question.value = "";
    await refresh();
  } catch (error) {
    showActionStatus(error.message);
  } finally {
    els.send.disabled = false;
  }
}

async function requestAction(action, extra = {}) {
  if (!current || actionSending) return;
  actionSending = true;
  els.checkMove.disabled = true;
  els.confirm.disabled = true;
  try {
    await postJson("/api/action", { action, expected_fen: current.fen, ...extra });
    showActionStatus(action === "check" ? "Move check requested." : "Request sent. Follow the physical board instructions.");
  } catch (error) {
    showActionStatus(error.message);
  } finally {
    actionSending = false;
    await refresh();
  }
}

function speechInputAvailable() {
  const local = current?.speech?.available && navigator.mediaDevices?.getUserMedia &&
                (window.AudioContext || window.webkitAudioContext);
  return Boolean(local || extraSpeech.input);
}

function speechOutputAvailable() {
  return Boolean(current?.speech?.tts_available || extraSpeech.output || localBrowserVoice());
}

function localBrowserVoice() {
  return window.speechSynthesis?.getVoices().find((voice) => voice.localService && voice.lang.startsWith("en"));
}

function updateVoiceControls() {
  els.mic.disabled = recordingStarting || transcribing || (!speechInputAvailable() && !recorder);
  els.mic.title = recorder ? "Stop recording" : speechInputAvailable() ?
    "Record locally; review transcript before sending" : "Local voice input unavailable";
  els.voiceReplies.disabled = !speechOutputAvailable();
  if (!recorder && current && !speechInputAvailable()) {
    els.voiceStatus.textContent = current.speech?.reason || "Voice input unavailable. Text chat is ready.";
  }
}

function encodeWav(samples, rate) {
  const targetRate = 16000;
  const length = Math.min(30 * targetRate, Math.floor(samples.length * targetRate / rate));
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
  write(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); write(36, "data"); view.setUint32(40, length * 2, true);
  for (let i = 0; i < length; i++) {
    const from = Math.floor(i * rate / targetRate);
    const to = Math.max(from + 1, Math.floor((i + 1) * rate / targetRate));
    let sum = 0;
    for (let j = from; j < Math.min(to, samples.length); j++) sum += samples[j];
    const sample = Math.max(-1, Math.min(1, sum / Math.max(1, to - from)));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function startRecording() {
  if (recordingStarting || recorder || transcribing) return;
  if (extraSpeech.input && !current?.speech?.available) {
    try {
      const transcript = await extraSpeech.input.start();
      if (transcript) els.question.value = `${els.question.value} ${transcript}`.trim();
      els.question.focus();
    } catch (error) { els.voiceStatus.textContent = error.message; }
    return;
  }
  if (!speechInputAvailable()) return;
  recordingStarting = true;
  updateVoiceControls();
  stopSpeaking();
  let stream = null;
  let context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: false
    });
    const Context = window.AudioContext || window.webkitAudioContext;
    context = new Context();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silence = context.createGain();
    silence.gain.value = 0;
    const chunks = [];
    processor.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    source.connect(processor); processor.connect(silence); silence.connect(context.destination);
    const timer = setTimeout(stopRecording, 29000);
    recorder = { stream, context, source, processor, silence, chunks, timer };
    els.mic.classList.add("recording");
    els.mic.textContent = "■";
    els.voiceStatus.textContent = "Recording locally. Click the microphone again to stop; your transcript stays editable.";
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (context && context.state !== "closed") await context.close();
    els.voiceStatus.textContent = `Microphone unavailable: ${error.message}`;
  } finally {
    recordingStarting = false;
    updateVoiceControls();
  }
}

async function stopRecording() {
  const active = recorder;
  if (!active) return;
  recorder = null;
  transcribing = true;
  updateVoiceControls();
  clearTimeout(active.timer);
  active.stream.getTracks().forEach((track) => track.stop());
  active.source.disconnect(); active.processor.disconnect(); active.silence.disconnect();
  await active.context.close();
  els.mic.classList.remove("recording");
  els.mic.textContent = "🎙";
  const total = active.chunks.reduce((count, chunk) => count + chunk.length, 0);
  if (!total) {
    els.voiceStatus.textContent = "No audio was captured. Try again or type your question.";
    transcribing = false;
    updateVoiceControls();
    return;
  }
  const samples = new Float32Array(total);
  let cursor = 0;
  for (const chunk of active.chunks) { samples.set(chunk, cursor); cursor += chunk.length; }
  const wav = encodeWav(samples, active.context.sampleRate);
  els.voiceStatus.textContent = "Transcribing on this computer… first use may take a little longer.";
  try {
    const response = await api("/api/transcribe", {
      method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav
    });
    const { transcript } = await response.json();
    els.question.value = `${els.question.value} ${transcript || ""}`.trim();
    els.question.focus();
    els.voiceStatus.textContent = "Transcript ready. Edit it, then press Send question.";
  } catch (error) {
    els.voiceStatus.textContent = `Could not transcribe: ${error.message}`;
  } finally {
    transcribing = false;
  }
  updateVoiceControls();
}

function stopSpeaking() {
  speakGeneration += 1;
  if (voicePlayer) {
    voicePlayer.pause();
    voicePlayer.currentTime = 0;
    voicePlayer = null;
  }
  if (voiceObjectUrl) {
    URL.revokeObjectURL(voiceObjectUrl);
    voiceObjectUrl = null;
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  if (extraSpeech.output?.stop) extraSpeech.output.stop();
  els.stopSpeaking.hidden = true;
  els.voiceStatus.textContent = "Speech stopped. Text chat remains available.";
}

async function speakText(text) {
  stopSpeaking();
  const generation = speakGeneration;
  els.stopSpeaking.hidden = false;
  els.voiceStatus.textContent = "Preparing this spoken reply locally…";
  if (extraSpeech.output && !current?.speech?.tts_available) {
    try { await extraSpeech.output.speak(text); } catch (error) { els.voiceStatus.textContent = error.message; }
    return;
  }
  if (current?.speech?.tts_available) {
    try {
      const response = await api("/api/speak", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.slice(0, 3000) })
      });
      const blob = await response.blob();
      if (generation !== speakGeneration) return;
      voiceObjectUrl = URL.createObjectURL(blob);
      const audio = new Audio(voiceObjectUrl);
      voicePlayer = audio;
      audio.addEventListener("ended", () => { if (generation === speakGeneration) stopSpeaking(); }, { once: true });
      audio.addEventListener("error", () => { if (generation === speakGeneration) stopSpeaking(); }, { once: true });
      await audio.play();
      els.voiceStatus.textContent = "Reading this reply aloud. Use Stop speaking to stop.";
      return;
    } catch (error) { els.voiceStatus.textContent = `Voice reply unavailable: ${error.message}`; }
  }
  const browserVoice = localBrowserVoice();
  if (browserVoice && generation === speakGeneration) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 3000));
    utterance.voice = browserVoice;
    utterance.onend = () => { if (generation === speakGeneration) stopSpeaking(); };
    window.speechSynthesis.speak(utterance);
    els.voiceStatus.textContent = "Reading this reply aloud. Use Stop speaking to stop.";
  } else {
    els.stopSpeaking.hidden = true;
  }
}

el("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(els.question.value);
});
els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(els.question.value); }
});
document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
  els.question.value = chip.dataset.question;
  els.question.focus();
}));
els.cameraTab.addEventListener("click", () => setBoardTab("camera"));
els.expectedTab.addEventListener("click", () => setBoardTab("expected"));
els.confirm.addEventListener("click", () => requestAction("confirm"));
els.checkMove.addEventListener("click", () => requestAction("check"));
el("cancel-button").addEventListener("click", () => requestAction("cancel"));
document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => requestAction(button.dataset.action)));
el("manual-form").addEventListener("submit", (event) => {
  event.preventDefault();
  requestAction("manual", { move: els.manual.value.trim().toLowerCase() });
});
document.querySelectorAll("[data-promotion]").forEach((button) => button.addEventListener("click", () =>
  requestAction("promotion", { choice: Number(button.dataset.promotion) })));
els.rook.addEventListener("click", () => requestAction("rook"));
els.mic.addEventListener("click", () => recorder ? stopRecording() : startRecording());
els.stopSpeaking.addEventListener("click", stopSpeaking);
els.voiceReplies.addEventListener("change", () => {
  if (!els.voiceReplies.checked) stopSpeaking();
  els.voiceStatus.textContent = els.voiceReplies.checked ?
    "Voice replies enabled. You can switch them off any time." : "Voice replies off. Text chat is always available.";
});

setBoardTab("camera");
window.speechSynthesis?.addEventListener("voiceschanged", updateVoiceControls);
refresh();
setInterval(refresh, 1000);
