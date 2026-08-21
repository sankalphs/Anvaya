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
  voiceModeButton: document.querySelector("#voice-mode-button"),
  textModeButton: document.querySelector("#text-mode-button"),
  voiceInputPanel: document.querySelector("#voice-input-panel"),
  textInputPanel: document.querySelector("#text-input-panel"),
  textQuery: document.querySelector("#text-query"),
  promptChips: document.querySelectorAll(".prompt-chip"),
  textCount: document.querySelector("#text-count"),
  textSubmitButton: document.querySelector("#text-submit-button"),
  languageSelect: document.querySelector("#language-select"),
  languageDropdownButton: null,
  languageDropdownMenu: null,
  formatChip: document.querySelector(".voice-only-format"),
  alert: document.querySelector("#alert"),
  queryPanel: document.querySelector(".query-panel"),
  progressPanel: document.querySelector("#progress-panel"),
  progressDetail: document.querySelector("#progress-detail"),
  activeStage: document.querySelector("#active-stage"),
  traceLabel: document.querySelector("#trace-label"),
  stageList: document.querySelector("#stage-list"),
  resultPanel: document.querySelector("#result-panel"),
  routeBadge: document.querySelector("#route-badge"),
  latency: document.querySelector("#latency"),
  requestId: document.querySelector("#request-id"),
  transcript: document.querySelector("#transcript"),
  queryKind: document.querySelector("#query-kind"),
  answerLabel: document.querySelector("#answer-label"),
  answerVisibilityLabel: document.querySelector("#answer-visibility-label"),
  answer: document.querySelector("#answer"),
  reasonCode: document.querySelector("#reason-code"),
  answerExplainer: document.querySelector("#answer-explainer"),
  groundingBadge: document.querySelector("#grounding-badge"),
  groundingSummary: document.querySelector("#grounding-summary"),
  answerExplanation: document.querySelector("#answer-explanation"),
  evidenceBlock: document.querySelector("#evidence-block"),
  evidenceTitle: document.querySelector("#evidence-title"),
  evidenceHelper: document.querySelector("#evidence-helper"),
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
let lastQueryKind = "voice";

function pinAmbientDecorationsToViewport() {
  const viewportRoot = document.documentElement;
  document.querySelectorAll(".ambient").forEach((ambient) => {
    if (ambient.parentElement !== viewportRoot) viewportRoot.appendChild(ambient);
  });
}

pinAmbientDecorationsToViewport();

function initializeLanguageDropdown() {
  const select = elements.languageSelect;
  if (!select || select.dataset.customized === "true") return;

  const wrapper = document.createElement("div");
  wrapper.className = "language-select-wrap";
  select.parentNode.insertBefore(wrapper, select);
  wrapper.appendChild(select);
  select.classList.add("native-language-select");
  select.dataset.customized = "true";

  const button = document.createElement("button");
  button.className = "language-dropdown-button";
  button.type = "button";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");
  button.innerHTML = '<span class="language-dropdown-value"></span><span class="language-dropdown-arrow" aria-hidden="true">⌄</span>';

  const menu = document.createElement("div");
  menu.className = "language-dropdown-menu";
  menu.setAttribute("role", "listbox");
  menu.hidden = true;

  [...select.options].forEach((option) => {
    const item = document.createElement("button");
    item.className = "language-dropdown-option";
    item.type = "button";
    item.dataset.value = option.value;
    item.setAttribute("role", "option");
    item.textContent = option.textContent;
    item.addEventListener("click", () => {
      select.value = option.value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      closeMenu();
    });
    menu.appendChild(item);
  });

  wrapper.append(button, menu);
  elements.languageDropdownButton = button;
  elements.languageDropdownMenu = menu;

  function syncLanguageDropdown() {
    const selected = select.options[select.selectedIndex];
    button.querySelector(".language-dropdown-value").textContent = selected?.textContent || "Select language";
    menu.querySelectorAll(".language-dropdown-option").forEach((item) => {
      const active = item.dataset.value === select.value;
      item.classList.toggle("selected", active);
      item.setAttribute("aria-selected", String(active));
    });
  }

  function closeMenu() {
    menu.hidden = true;
    button.setAttribute("aria-expanded", "false");
  }

  button.addEventListener("click", () => {
    if (button.disabled) return;
    const opening = menu.hidden;
    menu.hidden = !opening;
    button.setAttribute("aria-expanded", String(opening));
  });
  select.addEventListener("change", syncLanguageDropdown);
  document.addEventListener("click", (event) => {
    if (!wrapper.contains(event.target)) closeMenu();
  });
  syncLanguageDropdown();
}

initializeLanguageDropdown();

elements.recordButton.addEventListener("click", startRecording);
elements.stopButton.addEventListener("click", () => stopRecording(true));
elements.upload.addEventListener("change", handleUpload);
elements.voiceModeButton.addEventListener("click", () => setInputMode("voice"));
elements.textModeButton.addEventListener("click", () => setInputMode("text"));
elements.textQuery.addEventListener("input", updateTextCount);
elements.textQuery.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") submitText();
});
elements.textSubmitButton.addEventListener("click", submitText);
elements.promptChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    elements.textQuery.value = chip.dataset.prompt || "";
    updateTextCount();
    setInputMode("text");
    elements.textQuery.focus({ preventScroll: true });
  });
});
elements.newQueryButton.addEventListener("click", resetForNewQuery);
window.addEventListener("pagehide", releaseMicrophone);

updateTextCount();
checkHealth();

function setInputMode(mode) {
  const textMode = mode === "text";
  elements.voiceModeButton.classList.toggle("active", !textMode);
  elements.textModeButton.classList.toggle("active", textMode);
  elements.voiceModeButton.setAttribute("aria-selected", String(!textMode));
  elements.textModeButton.setAttribute("aria-selected", String(textMode));
  elements.voiceInputPanel.hidden = textMode;
  elements.textInputPanel.hidden = !textMode;
  elements.formatChip.hidden = textMode;
  clearAlert();
}

function updateTextCount() {
  const count = elements.textQuery.value.length;
  elements.textCount.textContent = `${count.toLocaleString()} / 2,000 characters`;
}

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
  preserveViewport(() => { elements.resultPanel.hidden = true; });
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showAlert("Microphone capture is not supported in this browser. Try uploading audio instead.");
    return;
  }
  let pendingStream = null;
  let pendingContext = null;
  try {
    pendingStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    pendingContext = new AudioContext();
    await pendingContext.audioWorklet.addModule("/static/audio-worklet.js");
    const source = pendingContext.createMediaStreamSource(pendingStream);
    const worklet = new AudioWorkletNode(pendingContext, "pcm-capture-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const silence = pendingContext.createGain();
    silence.gain.value = 0;
    recording.chunks = [];
    worklet.port.onmessage = (event) => recording.chunks.push(new Float32Array(event.data));
    source.connect(worklet);
    worklet.connect(silence);
    silence.connect(pendingContext.destination);

    Object.assign(recording, {
      active: true,
      context: pendingContext,
      stream: pendingStream,
      source,
      worklet,
      silence,
      startedAt: Date.now(),
    });
    recording.timer = window.setTimeout(() => stopRecording(true), MAX_RECORDING_SECONDS * 1000);
    setRecordingUi(true);
    updateRecordingClock();
  } catch (error) {
    if (pendingStream) pendingStream.getTracks().forEach((track) => track.stop());
    if (pendingContext && pendingContext.state !== "closed") await pendingContext.close();
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
    preserveViewport(() => {
      elements.progressPanel.hidden = true;
      elements.progressPanel.setAttribute("aria-busy", "false");
      elements.queryPanel.setAttribute("aria-busy", "false");
    });
    handleUserError(error, "The audio file could not be decoded. Try WAV, MP3, M4A, or WebM.");
  }
}

async function submitAudio(wavBlob) {
  clearAlert();
  setInputDisabled(true);
  preserveViewport(() => {
    elements.resultPanel.hidden = true;
    resetStages();
    elements.progressPanel.hidden = false;
    elements.progressPanel.setAttribute("aria-busy", "true");
    elements.queryPanel.setAttribute("aria-busy", "true");
    elements.activeStage.textContent = "Preparing audio";
  });
  lastQueryKind = "voice";
  elements.traceLabel.textContent = "Voice path · observed stages only";
  const requestId = makeRequestId();
  progressRequestId = requestId;
  pollProgress(requestId);
  const form = new FormData();
  form.append("audio", wavBlob, "voice-query.wav");
  form.append("language_code", elements.languageSelect.value);
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
    payload.request_id = response.headers.get("X-Request-ID") || requestId;
    progressRequestId = null;
    await showFinalProgress(requestId);
    renderResult(payload);
  } catch (error) {
    progressRequestId = null;
    preserveViewport(() => {
      elements.progressPanel.hidden = true;
      elements.progressPanel.setAttribute("aria-busy", "false");
      elements.queryPanel.setAttribute("aria-busy", "false");
    });
    handleUserError(error, "The backend could not process the request. Please try again.");
  } finally {
    setInputDisabled(false);
    elements.progressPanel.setAttribute("aria-busy", "false");
    elements.queryPanel.setAttribute("aria-busy", "false");
  }
}

async function submitText() {
  clearAlert();
  const text = elements.textQuery.value.trim();
  if (!text) {
    showAlert("Type a question before retrieving an answer.");
    elements.textQuery.focus();
    return;
  }
  setInputDisabled(true);
  preserveViewport(() => {
    elements.resultPanel.hidden = true;
    resetStages();
    elements.progressPanel.hidden = false;
    elements.progressPanel.setAttribute("aria-busy", "true");
    elements.queryPanel.setAttribute("aria-busy", "true");
    elements.activeStage.textContent = "Preparing text";
  });
  lastQueryKind = "text";
  elements.traceLabel.textContent = "Text path · STT skipped · observed stages only";
  const requestId = makeRequestId();
  progressRequestId = requestId;
  pollProgress(requestId);
  try {
    const response = await fetch("/api/query/text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": requestId,
      },
      body: JSON.stringify({ text, language_code: elements.languageSelect.value }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new UserFacingError(payload.detail || "The backend could not process this question.");
    }
    payload.request_id = response.headers.get("X-Request-ID") || requestId;
    progressRequestId = null;
    await showFinalProgress(requestId);
    renderResult(payload);
  } catch (error) {
    progressRequestId = null;
    preserveViewport(() => {
      elements.progressPanel.hidden = true;
      elements.progressPanel.setAttribute("aria-busy", "false");
      elements.queryPanel.setAttribute("aria-busy", "false");
    });
    handleUserError(error, "The backend could not process the question. Please try again.");
  } finally {
    setInputDisabled(false);
    elements.progressPanel.setAttribute("aria-busy", "false");
    elements.queryPanel.setAttribute("aria-busy", "false");
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

async function showFinalProgress(requestId) {
  try {
    const response = await fetch(`/api/query/status/${encodeURIComponent(requestId)}`, {
      cache: "no-store",
    });
    if (response.ok) {
      markProgress(await response.json());
      return;
    }
  } catch (_error) {
    // The response remains authoritative if the final observation is unavailable.
  }
  markProgress({ stage: "Complete", history: ["Complete"], complete: true });
}

function markProgress(progress) {
  const seen = new Set(progress.history || []);
  elements.activeStage.textContent = progress.stage || "Processing";
  elements.progressDetail.textContent = progress.stage
    ? `${progress.stage} · the request remains on this page while the next checkpoint is observed.`
    : "Your question is moving through the visible checkpoints below.";
  elements.stageList.querySelectorAll("li").forEach((item) => {
    const stage = item.dataset.stage;
    item.classList.toggle("seen", seen.has(stage));
    item.classList.toggle("active", stage === progress.stage);
    const skipped = lastQueryKind === "text" && stage === "Transcribing";
    item.classList.toggle("skipped", skipped);
    item.setAttribute("aria-label", skipped ? `${stage}, skipped for text input` : stage);
  });
}

function renderResult(data) {
  preserveViewport(() => {
    elements.progressPanel.hidden = true;
    elements.progressPanel.setAttribute("aria-busy", "false");
    elements.queryPanel.setAttribute("aria-busy", "false");
    elements.resultPanel.hidden = false;
    elements.routeBadge.textContent = data.route || "SYSTEM_ERROR";
    elements.routeBadge.className = "route-badge";
    if (data.route !== "ANSWER") elements.routeBadge.classList.add("refusal");
    if (data.route === "SYSTEM_ERROR" || data.route === "STT_FAILURE") {
      elements.routeBadge.classList.add("error");
    }
    elements.latency.textContent = `${formatMs(data.total_latency_ms)} measured request latency`;
    elements.requestId.textContent = data.request_id ? `Trace ${data.request_id}` : "";
    elements.queryKind.textContent = `${lastQueryKind === "text" ? "Text" : "Voice"} input · final route`;
    elements.transcript.textContent = data.transcript || "No transcript was produced.";
    elements.answerLabel.textContent = data.route === "ANSWER" ? "Answer" : "Response";
    const answerMode = data.metadata?.generation?.answer_mode;
    elements.answerVisibilityLabel.textContent = data.route === "ANSWER"
      ? answerMode === "extractive_fallback"
        ? "Quoted from retrieved evidence · model fallback"
        : "Generated from cited evidence"
      : "No answer generated · retrieval evidence shown";
    elements.answer.textContent = responseText(data);
    elements.reasonCode.textContent = `Reason code · ${data.reason_code || "NONE"}`;
    renderAnswerTransparency(data);
    renderEvidence(data);
    renderTechnicalDetails(data);
    elements.resultPanel.setAttribute("aria-busy", "false");
    elements.resultPanel.focus({ preventScroll: true });
  });
}

function renderAnswerTransparency(data) {
  const evidence = data.metadata && Array.isArray(data.metadata.retrieved)
    ? data.metadata.retrieved
    : [];
  const citations = new Set(data.citations || []);
  const decision = data.metadata?.evidence_decision || {};
  const generation = data.metadata?.generation || {};
  const grounding = data.metadata?.grounding || {};
  const isAnswer = data.route === "ANSWER";
  const isGrounded = isAnswer && grounding.valid !== false && citations.size > 0;
  elements.answerExplainer.hidden = false;
  elements.groundingBadge.className = `signal-badge ${isGrounded ? "grounded" : "limited"}`;
  elements.groundingBadge.textContent = isGrounded ? "Grounded" : "Guardrail decision";
  if (isAnswer) {
    const sourceWord = citations.size === 1 ? "source" : "sources";
    const generationContextIds = data.metadata?.generation_context_ids || [];
    elements.groundingSummary.textContent = `${citations.size} cited ${sourceWord} · ${evidence.length} retrieved · ${generationContextIds.length || Math.min(evidence.length, 3)} sent to model`;
    const model = generation.model || "the configured local model";
    const runtime = generation.runtime ? ` on ${generation.runtime}` : "";
    if (generation.answer_mode === "extractive_fallback") {
      elements.answerExplanation.textContent = `The model answer did not pass validation, so this is a direct passage from the retrieved knowledge base, not a fresh model summary. The failed check was ${generation.fallback_reason || "not disclosed"}. Open the evidence rows to inspect it.`;
    } else {
      elements.answerExplanation.textContent = `This answer passed the evidence and grounding checks. It was produced by ${model}${runtime}. Similarity scores and citations are signals, not calibrated truth or confidence scores. Open the evidence rows to inspect the exact passages.`;
    }
  } else {
    const rule = decision.decision_rule ? ` Rule: ${decision.decision_rule}.` : "";
    elements.groundingSummary.textContent = "No supported answer was generated";
    elements.answerExplanation.textContent = `The system stopped before presenting an unsupported answer.${rule} Retrieved passages, when available, are shown below so you can see what was considered.`;
  }
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
  elements.evidenceBlock.hidden = evidence.length === 0;
  if (data.route === "ANSWER") {
    elements.evidenceTitle.textContent = "What supports this answer";
    elements.evidenceHelper.textContent =
      "Green-marked passages were cited by the answer. Expand any row to inspect the exact retrieved text.";
  } else {
      elements.evidenceTitle.textContent = "What the retriever found";
      elements.evidenceHelper.textContent =
      "No answer was generated from these passages. Scores are vector similarity, not answer confidence; the KB guardrail must also find query-term evidence.";
  }
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
  const checkpoints = data.latency_checkpoints || data.metadata?.latency_checkpoints || {};
  const metrics = [
    ["STT", lastQueryKind === "text" ? "Skipped" : timings.stt],
    ["Input check", timings.input_validation],
    ["Route check", timings.route_check],
    ["Embedding", timings.query_embedding ?? timings.embedding],
    ["Search", timings.vector_search ?? timings.retrieval],
    ["Evidence gate", timings.evidence_guardrail],
    [data.route === "ANSWER" ? "Generation" : "Generation decision", timings.generation],
    ...(checkpoints.qwen_model_load_ms != null
      ? [["Qwen load", checkpoints.qwen_model_load_ms]]
      : []),
    ...(checkpoints.qwen_generation_ms != null
      ? [["Qwen decode", checkpoints.qwen_generation_ms]]
      : []),
    ...(checkpoints.qwen_time_to_first_token_ms != null
      ? [["Qwen TTFT", checkpoints.qwen_time_to_first_token_ms]]
      : []),
    ["Grounding", timings.grounding_validation],
    ["Total", data.total_latency_ms],
  ];
  metrics.forEach(([label, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const name = document.createElement("span");
    name.textContent = label;
    const amount = document.createElement("strong");
    amount.textContent = typeof value === "string" ? value : formatStageMs(value, label);
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
  setModeDisabled(active);
  elements.recordVisual.classList.toggle("recording", active);
  elements.recordLabel.textContent = active ? "Recording" : "Record a question";
  if (!active) elements.recordHint.textContent = "Record up to 30 seconds";
}

function setInputDisabled(disabled) {
  elements.recordButton.disabled = disabled;
  elements.stopButton.disabled = true;
  elements.upload.disabled = disabled;
  elements.textQuery.disabled = disabled;
  elements.textSubmitButton.disabled = disabled;
  elements.languageSelect.disabled = disabled;
  if (elements.languageDropdownButton) elements.languageDropdownButton.disabled = disabled;
  setModeDisabled(disabled);
  elements.textSubmitButton.innerHTML = disabled
    ? "Generating…"
    : 'Generate answer <span aria-hidden="true">↗</span>';
}

function setModeDisabled(disabled) {
  elements.voiceModeButton.disabled = disabled;
  elements.textModeButton.disabled = disabled;
}

function resetForNewQuery() {
  preserveViewport(() => {
    elements.resultPanel.hidden = true;
    elements.progressPanel.hidden = true;
    elements.resultPanel.setAttribute("aria-busy", "false");
    elements.queryPanel.setAttribute("aria-busy", "false");
    clearAlert();
    resetStages();
    elements.textQuery.value = "";
    updateTextCount();
  });
  const focusTarget = lastQueryKind === "text" ? elements.textQuery : elements.recordButton;
  focusTarget.focus({ preventScroll: true });
}

function resetStages() {
  elements.stageList.querySelectorAll("li").forEach((item) => item.classList.remove("seen", "active", "skipped"));
}

function showAlert(message) {
  elements.alert.textContent = message;
  elements.alert.hidden = false;
  elements.alert.focus({ preventScroll: true });
}

function preserveViewport(update) {
  const left = window.scrollX;
  const top = window.scrollY;
  update();
  const restore = () => window.scrollTo({ left, top, behavior: "auto" });
  requestAnimationFrame(() => {
    restore();
    requestAnimationFrame(restore);
  });
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

function formatStageMs(value, _label) {
  const number = Number(value || 0);
  return _label === "Total" || number > 0 ? formatMs(number) : "Not reached";
}

function makeRequestId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const values = crypto.getRandomValues(new Uint32Array(4));
  return Array.from(values, (value) => value.toString(16).padStart(8, "0")).join("");
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
