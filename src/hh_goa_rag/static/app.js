const TARGET_SAMPLE_RATE = 16000;
const MAX_RECORDING_SECONDS = 30;
const PIPELINE_STAGES = [
  "Transcribing",
  "Checking query",
  "Retrieving evidence",
  "Generating answer",
  "Validating grounding",
  "Complete",
];

const elements = {
  healthChip: document.querySelector("#health-chip"),
  healthText: document.querySelector("#health-text"),
  recordButton: document.querySelector("#record-button"),
  recordLabel: document.querySelector("#record-label"),
  stopButton: document.querySelector("#stop-button"),
  recordVisual: document.querySelector("#record-visual"),
  recordHint: document.querySelector("#record-hint"),
  upload: document.querySelector("#audio-upload"),
  alert: document.querySelector("#alert"),
  progressPanel: document.querySelector("#progress-panel"),
  activeStage: document.querySelector("#active-stage"),
  stageList: document.querySelector("#stage-list"),
  resultPanel: document.querySelector("#result-panel"),
  routeBadge: document.querySelector("#route-badge"),
  latency: document.querySelector("#latency"),
  transcript: document.querySelector("#transcript"),
  answerLabel: document.querySelector("#answer-label"),
  answer: document.querySelector("#answer"),
  reasonCode: document.querySelector("#reason-code"),
  evidenceBlock: document.querySelector("#evidence-block"),
  sourceCount: document.querySelector("#source-count"),
  evidenceList: document.querySelector("#evidence-list"),
  metricGrid: document.querySelector("#metric-grid"),
  retrievedIds: document.querySelector("#retrieved-ids"),
  newQueryButton: document.querySelector("#new-query-button"),
};

const recording = {
  active: false,
  chunks: [],
  context: null,
  stream: null,
  source: null,
  worklet: null,
  silence: null,
  timer: null,
  startedAt: 0,
};

let progressRequestId = null;

elements.recordButton.addEventListener("click", startRecording);
elements.stopButton.addEventListener("click", () => stopRecording(true));
elements.upload.addEventListener("change", handleUpload);
elements.newQueryButton.addEventListener("click", resetForNewQuery);
window.addEventListener("pagehide", releaseMicrophone);

checkHealth();

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error("unavailable");
    elements.healthChip.classList.add("ready");
    elements.healthText.textContent = "System ready";
  } catch (_error) {
    elements.healthChip.classList.add("error");
    elements.healthText.textContent = "System unavailable";
  }
}

async function startRecording() {
  clearAlert();
  elements.resultPanel.hidden = true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showAlert("Microphone capture is not supported in this browser. Try uploading audio instead.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const context = new AudioContext();
    await context.audioWorklet.addModule("/static/audio-worklet.js");
    const source = context.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(context, "pcm-capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const silence = context.createGain();
    silence.gain.value = 0;
    recording.chunks = [];
    worklet.port.onmessage = (event) => recording.chunks.push(new Float32Array(event.data));
    source.connect(worklet);
    worklet.connect(silence);
    silence.connect(context.destination);

    Object.assign(recording, {
      active: true,
      context,
      stream,
      source,
      worklet,
      silence,
      startedAt: Date.now(),
    });
    recording.timer = window.setTimeout(() => stopRecording(true), MAX_RECORDING_SECONDS * 1000);
    setRecordingUi(true);
    updateRecordingClock();
  } catch (error) {
    releaseMicrophone();
    if (error && (error.name === "NotAllowedError" || error.name === "SecurityError")) {
      showAlert(
        "Microphone permission was denied. Allow microphone access for this site, or upload an audio file.",
      );
    } else if (error && error.name === "NotFoundError") {
      showAlert("No microphone was found. Connect one or upload an audio file.");
    } else {
      showAlert("Recording could not start. Check the microphone and try again.");
    }
  }
}

function updateRecordingClock() {
  if (!recording.active) return;
  const seconds = Math.min((Date.now() - recording.startedAt) / 1000, MAX_RECORDING_SECONDS);
  elements.recordHint.textContent = `Recording · ${seconds.toFixed(1)}s / ${MAX_RECORDING_SECONDS}s`;
  window.setTimeout(updateRecordingClock, 100);
}

async function stopRecording(processAudio) {
  if (!recording.active) return;
  recording.active = false;
  setRecordingUi(false);
  window.clearTimeout(recording.timer);
  const inputRate = recording.context.sampleRate;
  const chunks = recording.chunks.slice();
  releaseMicrophone();
  if (!processAudio) return;
  try {
    const input = mergeChunks(chunks);
    if (input.length < inputRate * 0.15 || signalRms(input) < 0.0003) {
      throw new UserFacingError("The recording appears empty. Speak closer to the microphone and try again.");
    }
    if (input.length / inputRate > MAX_RECORDING_SECONDS + 0.25) {
      throw new UserFacingError("The recording exceeds the 30-second limit.");
    }
    const pcm = resampleMono(input, inputRate, TARGET_SAMPLE_RATE);
    await submitAudio(encodePcm16Wav(pcm, TARGET_SAMPLE_RATE));
  } catch (error) {
    handleUserError(error, "The recording could not be prepared. Please try again.");
  }
}

function releaseMicrophone() {
  window.clearTimeout(recording.timer);
  if (recording.worklet) recording.worklet.disconnect();
  if (recording.source) recording.source.disconnect();
  if (recording.silence) recording.silence.disconnect();
  if (recording.stream) recording.stream.getTracks().forEach((track) => track.stop());
  if (recording.context && recording.context.state !== "closed") recording.context.close();
  recording.context = null;
  recording.stream = null;
  recording.source = null;
  recording.worklet = null;
  recording.silence = null;
  recording.timer = null;
}

async function handleUpload(event) {
  clearAlert();
  const file = event.target.files && event.target.files[0];
  event.target.value = "";
  if (!file) return;
  if (file.size === 0) {
    showAlert("The selected audio file is empty.");
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    showAlert("The selected file is too large. Upload at most 30 seconds of audio.");
    return;
  }
  setInputDisabled(true);
  elements.progressPanel.hidden = false;
  elements.activeStage.textContent = "Preparing audio";
  try {
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(await file.arrayBuffer());
    } finally {
      await context.close();
    }
    if (!decoded.length || decoded.duration < 0.15 || signalRms(decoded.getChannelData(0)) < 0.0003) {
      throw new UserFacingError("The selected audio appears empty or silent.");
    }
    if (decoded.duration > MAX_RECORDING_SECONDS + 0.01) {
      throw new UserFacingError("The selected audio exceeds the 30-second limit.");
    }
    const mono = mixToMono(decoded);
    const pcm = resampleMono(mono, decoded.sampleRate, TARGET_SAMPLE_RATE);
    await submitAudio(encodePcm16Wav(pcm, TARGET_SAMPLE_RATE));
  } catch (error) {
    setInputDisabled(false);
    elements.progressPanel.hidden = true;
    handleUserError(error, "The audio file could not be decoded. Try WAV, MP3, M4A, or WebM.");
  }
}

async function submitAudio(wavBlob) {
  clearAlert();
  setInputDisabled(true);
  elements.resultPanel.hidden = true;
  resetStages();
  elements.progressPanel.hidden = false;
  elements.activeStage.textContent = "Preparing audio";
  const requestId = makeRequestId();
  progressRequestId = requestId;
  pollProgress(requestId);
  const form = new FormData();
  form.append("audio", wavBlob, "voice-query.wav");
  try {
    const response = await fetch("/api/query/audio", {
      method: "POST",
      headers: { "X-Request-ID": requestId },
      body: form,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new UserFacingError(payload.detail || "The backend could not process this audio.");
    }
    progressRequestId = null;
    markProgress({ stage: "Complete", history: ["Complete"], complete: true });
    renderResult(payload);
  } catch (error) {
    progressRequestId = null;
    elements.progressPanel.hidden = true;
    handleUserError(error, "The backend could not process the request. Please try again.");
  } finally {
    setInputDisabled(false);
  }
}

async function pollProgress(requestId) {
  while (progressRequestId === requestId) {
    try {
      const response = await fetch(`/api/query/status/${encodeURIComponent(requestId)}`, {
        cache: "no-store",
      });
      if (response.ok) markProgress(await response.json());
    } catch (_error) {
      // The query response remains authoritative; brief polling failures need no UI error.
    }
    await delay(220);
  }
}

function markProgress(progress) {
  const seen = new Set(progress.history || []);
  elements.activeStage.textContent = progress.stage || "Processing";
  elements.stageList.querySelectorAll("li").forEach((item) => {
    const stage = item.dataset.stage;
    item.classList.toggle("seen", seen.has(stage));
    item.classList.toggle("active", stage === progress.stage);
  });
}

function renderResult(data) {
  window.setTimeout(() => {
    elements.progressPanel.hidden = true;
    elements.resultPanel.hidden = false;
    elements.routeBadge.textContent = data.route || "SYSTEM_ERROR";
    elements.routeBadge.className = "route-badge";
    if (data.route !== "ANSWER") elements.routeBadge.classList.add("refusal");
    if (data.route === "SYSTEM_ERROR" || data.route === "STT_FAILURE") {
      elements.routeBadge.classList.add("error");
    }
    elements.latency.textContent = `${formatMs(data.total_latency_ms)} measured request latency`;
    elements.transcript.textContent = data.transcript || "No transcript was produced.";
    elements.answerLabel.textContent = data.route === "ANSWER" ? "Answer" : "Response";
    elements.answer.textContent = responseText(data);
    elements.reasonCode.textContent = `Reason code · ${data.reason_code || "NONE"}`;
    renderEvidence(data);
    renderTechnicalDetails(data);
    elements.resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 120);
}

function responseText(data) {
  if (data.route === "ANSWER") return data.answer || "A grounded answer was not returned.";
  const friendly = {
    INSUFFICIENT_CONTEXT:
      "I could not find enough evidence in the provided knowledge base to answer reliably.",
    OFF_TOPIC: "I answer questions that can be supported by the provided knowledge base.",
    UNSAFE: "I can’t help with that request.",
    STT_FAILURE: "I could not transcribe that audio reliably. Please try recording again.",
    SYSTEM_ERROR: "The system could not complete this request. Please try again shortly.",
  };
  return friendly[data.route] || "The request could not be completed.";
}

function renderEvidence(data) {
  elements.evidenceList.replaceChildren();
  const evidence = data.metadata && Array.isArray(data.metadata.retrieved)
    ? data.metadata.retrieved
    : [];
  const citations = new Set(data.citations || []);
  elements.evidenceBlock.hidden = data.route !== "ANSWER" || evidence.length === 0;
  elements.sourceCount.textContent = `${evidence.length} passages · ${citations.size} cited`;
  evidence.forEach((item) => {
    const details = document.createElement("details");
    details.className = "evidence-item";
    const cited = citations.has(item.parent_id);
    if (cited) {
      details.classList.add("cited");
      details.open = true;
    }
    const summary = document.createElement("summary");
    const id = document.createElement("span");
    id.className = "evidence-id";
    id.textContent = `#${item.rank} · ${item.parent_id}`;
    const meta = document.createElement("span");
    meta.className = "evidence-meta";
    if (cited) meta.append(makeChip("Cited", "cited-chip"));
    meta.append(makeChip(`Score ${Number(item.score || 0).toFixed(4)}`, "score"));
    summary.append(id, meta);
    const passage = document.createElement("p");
    passage.className = "evidence-text";
    passage.textContent = item.text || "Passage text unavailable.";
    const chunk = document.createElement("span");
    chunk.className = "chunk-id";
    chunk.textContent = `Chunk ${item.chunk_id || "unknown"}`;
    passage.append(chunk);
    details.append(summary, passage);
    elements.evidenceList.append(details);
  });
}

function makeChip(text, className) {
  const chip = document.createElement("span");
  chip.className = className;
  chip.textContent = text;
  return chip;
}

function renderTechnicalDetails(data) {
  elements.metricGrid.replaceChildren();
  const timings = data.stage_latencies_ms || {};
  const metrics = [
    ["STT", timings.stt],
    ["Embedding", timings.query_embedding ?? timings.embedding],
    ["Search", timings.vector_search ?? timings.retrieval],
    ["Guardrails", timings.guardrails],
    ["Generation", timings.generation],
    ["Total", data.total_latency_ms],
  ];
  metrics.forEach(([label, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const name = document.createElement("span");
    name.textContent = label;
    const amount = document.createElement("strong");
    amount.textContent = formatMs(value);
    metric.append(name, amount);
    elements.metricGrid.append(metric);
  });

  elements.retrievedIds.replaceChildren();
  const citations = new Set(data.citations || []);
  (data.retrieved_ids || []).forEach((id) => {
    const code = document.createElement("code");
    code.textContent = id;
    if (citations.has(id)) code.className = "cited";
    elements.retrievedIds.append(code);
  });
  if (!(data.retrieved_ids || []).length) {
    const none = document.createElement("span");
    none.textContent = "No retrieval was run for this route.";
    none.className = "summary-note";
    elements.retrievedIds.append(none);
  }
}

function setRecordingUi(active) {
  elements.recordButton.disabled = active;
  elements.stopButton.disabled = !active;
  elements.upload.disabled = active;
  elements.recordVisual.classList.toggle("recording", active);
  elements.recordLabel.textContent = active ? "Recording" : "Record";
  if (!active) elements.recordHint.textContent = "Record up to 30 seconds";
}

function setInputDisabled(disabled) {
  elements.recordButton.disabled = disabled;
  elements.stopButton.disabled = true;
  elements.upload.disabled = disabled;
}

function resetForNewQuery() {
  elements.resultPanel.hidden = true;
  elements.progressPanel.hidden = true;
  clearAlert();
  resetStages();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function resetStages() {
  elements.stageList.querySelectorAll("li").forEach((item) => item.classList.remove("seen", "active"));
}

function showAlert(message) {
  elements.alert.textContent = message;
  elements.alert.hidden = false;
}

function clearAlert() {
  elements.alert.textContent = "";
  elements.alert.hidden = true;
}

function handleUserError(error, fallback) {
  showAlert(error instanceof UserFacingError ? error.message : fallback);
}

class UserFacingError extends Error {}

function mergeChunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function mixToMono(audioBuffer) {
  const mono = new Float32Array(audioBuffer.length);
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    const input = audioBuffer.getChannelData(channel);
    for (let index = 0; index < input.length; index += 1) mono[index] += input[index];
  }
  const scale = 1 / audioBuffer.numberOfChannels;
  for (let index = 0; index < mono.length; index += 1) mono[index] *= scale;
  return mono;
}

function resampleMono(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return input.slice();
  const ratio = sourceRate / targetRate;
  const output = new Float32Array(Math.floor(input.length / ratio));
  for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
    const start = Math.floor(outputIndex * ratio);
    const end = Math.min(Math.floor((outputIndex + 1) * ratio), input.length);
    let total = 0;
    for (let inputIndex = start; inputIndex < Math.max(start + 1, end); inputIndex += 1) {
      total += input[inputIndex];
    }
    output[outputIndex] = total / Math.max(1, end - start);
  }
  return output;
}

function signalRms(input) {
  let sum = 0;
  for (let index = 0; index < input.length; index += 1) sum += input[index] * input[index];
  return Math.sqrt(sum / Math.max(1, input.length));
}

function encodePcm16Wav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function formatMs(value) {
  const number = Number(value || 0);
  if (number >= 1000) return `${(number / 1000).toFixed(2)} s`;
  return `${number.toFixed(number >= 10 ? 1 : 2)} ms`;
}

function makeRequestId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const values = crypto.getRandomValues(new Uint32Array(4));
  return Array.from(values, (value) => value.toString(16).padStart(8, "0")).join("");
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
