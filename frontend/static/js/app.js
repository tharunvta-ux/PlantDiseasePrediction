(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const chooseBtn = document.getElementById("chooseBtn");
  const previewArea = document.getElementById("previewArea");
  const previewImage = document.getElementById("previewImage");
  const fileNameEl = document.getElementById("fileName");
  const removeImageBtn = document.getElementById("removeImageBtn");
  const removeBtn = document.getElementById("removeBtn");
  const predictBtn = document.getElementById("predictBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  const statusBadge = document.getElementById("statusBadge");
  const resultSkeleton = document.getElementById("resultSkeleton");
  const skeletonMessage = document.getElementById("skeletonMessage");
  const resultContent = document.getElementById("resultContent");
  const resultClass = document.getElementById("resultClass");
  const resultConfidence = document.getElementById("resultConfidence");
  const top3List = document.getElementById("top3List");
  const resultError = document.getElementById("resultError");
  const errorMessage = document.getElementById("errorMessage");
  const resultEyebrow = document.getElementById("resultEyebrow");
  const levelBanner = document.getElementById("levelBanner");
  const cropChip = document.getElementById("cropChip");
  const healthChip = document.getElementById("healthChip");
  const resultMessage = document.getElementById("resultMessage");
  const qualityWarnings = document.getElementById("qualityWarnings");
  const top3Eyebrow = document.getElementById("top3Eyebrow");
  const resultChips = document.getElementById("resultChips");
  const plantChip = document.getElementById("plantChip");
  const analysisNotes = document.getElementById("analysisNotes");
  const top3Section = document.getElementById("top3Section");
  const observationPanel = document.getElementById("observationPanel");

  const treatmentPanel = document.getElementById("treatmentPanel");
  const treatmentLoading = document.getElementById("treatmentLoading");
  const treatmentUnavailable = document.getElementById("treatmentUnavailable");
  const treatmentUnavailableText = document.getElementById("treatmentUnavailableText");
  const treatmentError = document.getElementById("treatmentError");
  const treatmentErrorText = document.getElementById("treatmentErrorText");
  const treatmentRetryBtn = document.getElementById("treatmentRetryBtn");
  const treatmentContent = document.getElementById("treatmentContent");

  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  const languageSelect = document.getElementById("languageSelect");
  const explainSection = document.getElementById("explainSection");
  const explainBtn = document.getElementById("explainBtn");
  const explainLoading = document.getElementById("explainLoading");
  const explainFigure = document.getElementById("explainFigure");
  const explainImage = document.getElementById("explainImage");
  const explainCaption = document.getElementById("explainCaption");
  const explainError = document.getElementById("explainError");
  const reportBtn = document.getElementById("reportBtn");
  const printReport = document.getElementById("printReport");
  const speakBtn = document.getElementById("speakBtn");
  const speakNote = document.getElementById("speakNote");

  let selectedFile = null;
  let objectUrl = null;
  let lastPrediction = null;
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  let recommendationController = null;

  // Latest results, kept for the PDF report / read-aloud.
  let lastAnalysis = null;
  let lastGuidance = null;
  let lastExplanation = null;
  let languages = [{ code: "en", name: "English", native_name: "English", speech_lang: "en-IN" }];

  const LANGUAGE_KEY = "verdant.language";

  function storedLanguage() {
    try {
      return localStorage.getItem(LANGUAGE_KEY) || "en";
    } catch (_) {
      return "en";
    }
  }

  function currentLanguage() {
    return languageSelect.value || "en";
  }

  async function loadLanguages() {
    try {
      const response = await fetch("/languages");
      if (response.ok) languages = await response.json();
    } catch (_) {
      /* keep English only */
    }

    languageSelect.innerHTML = "";
    languages.forEach((lang) => {
      const option = document.createElement("option");
      option.value = lang.code;
      option.textContent = lang.code === "en" ? lang.name : `${lang.native_name} (${lang.name})`;
      languageSelect.appendChild(option);
    });

    const saved = storedLanguage();
    languageSelect.value = languages.some((l) => l.code === saved) ? saved : "en";
  }

  const LEVEL_TEXT = {
    high: "High confidence",
    uncertain: "Uncertain",
    low: "Low confidence",
  };

  const LEVEL_BANNER = {
    high: "High-confidence prediction.",
    uncertain:
      "Uncertain prediction — the model is not sure. Treat this as a possible diagnosis and compare the alternatives below.",
    low:
      "Result uncertain — no diagnosis shown. Please upload a clearer, well-lit photo of a single leaf.",
  };

  function formatClassName(rawName) {
    return rawName
      .split("___")
      .map((part) =>
        part
          .replace(/_/g, " ")
          .trim()
          .replace(/\b\w/g, (c) => c.toUpperCase())
      )
      .join(" — ");
  }

  function setStatus(phase) {
    if (phase === "idle") {
      statusBadge.textContent = "Awaiting image";
      statusBadge.classList.remove("error");
    } else if (phase === "loading") {
      statusBadge.textContent = "Running";
      statusBadge.classList.remove("error");
    } else if (phase === "result") {
      statusBadge.textContent = "Complete";
      statusBadge.classList.remove("error");
    } else if (phase === "error") {
      statusBadge.textContent = "Error";
      statusBadge.classList.add("error");
    }
  }

  function resetResultPanel(message) {
    setStatus("idle");
    resetTreatmentPanel();
    observationPanel.classList.add("hidden");
    resultSkeleton.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
    skeletonMessage.textContent =
      message || "Upload a photo and press Predict to see the results here.";
  }

  function showLoading() {
    setStatus("loading");
    lastPrediction = null;
    resetTreatmentPanel();
    resultSkeleton.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
    skeletonMessage.textContent = "Identifying the plant and inspecting the leaf…";
    observationPanel.classList.add("hidden");
    resetExplanation();
    lastAnalysis = null;
  }

  /* ---------- "Where did the model look?" ---------- */

  function resetExplanation() {
    lastExplanation = null;
    explainSection.classList.add("hidden");
    explainLoading.classList.add("hidden");
    explainFigure.classList.add("hidden");
    explainError.classList.add("hidden");
    explainBtn.disabled = false;
    explainBtn.classList.remove("hidden");
  }

  async function requestExplanation() {
    if (!selectedFile || !lastPrediction) return;

    explainBtn.disabled = true;
    explainError.classList.add("hidden");
    explainLoading.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("target_class", lastPrediction.predicted_class);

    try {
      const response = await fetch("/explain", { method: "POST", body: formData });
      let data = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (!response.ok || !data || !data.overlay_png) {
        throw new Error((data && data.error) || "Could not create the explanation.");
      }

      lastExplanation = data;
      explainImage.src = data.overlay_png;
      explainCaption.textContent =
        `${data.explanation} (Explaining: ${formatClassName(data.target_class)}; ` +
        `${data.grid[0]}×${data.grid[1]} occlusion test.)`;
      explainFigure.classList.remove("hidden");
      explainBtn.classList.add("hidden");
    } catch (err) {
      explainError.textContent = err.message || "Could not create the explanation.";
      explainError.classList.remove("hidden");
      explainBtn.disabled = false;
    } finally {
      explainLoading.classList.add("hidden");
    }
  }

  function showResult(data) {
    setStatus("result");
    resultSkeleton.classList.add("hidden");
    resultError.classList.add("hidden");
    resultContent.classList.remove("hidden");
    top3Section.classList.remove("hidden");
    resultConfidence.classList.remove("hidden");

    const level = data.confidence_level || null;
    const shownConfidence =
      typeof data.calibrated_confidence === "number"
        ? data.calibrated_confidence
        : data.confidence;

    levelBanner.className = "level-banner hidden";
    resultConfidence.className = "confidence-badge";

    if (level) {
      levelBanner.textContent = LEVEL_BANNER[level] || "";
      levelBanner.classList.add(`level-${level}`);
      levelBanner.classList.toggle("hidden", level === "high");
      resultConfidence.classList.add(`level-${level}`);
    }

    if (level === "low") {
      resultEyebrow.textContent = "Result";
      resultClass.textContent = "Uncertain result";
      resultConfidence.textContent = LEVEL_TEXT.low;
      top3Eyebrow.textContent = "Possible matches (not a diagnosis)";
    } else {
      resultEyebrow.textContent =
        level === "uncertain" ? "Possible diagnosis" : "Predicted disease";
      resultClass.textContent = formatClassName(data.predicted_class);
      resultConfidence.textContent =
        `${shownConfidence.toFixed(2)}%` + (level ? ` · ${LEVEL_TEXT[level]}` : "");
      top3Eyebrow.textContent = "Top predictions (model scores)";
    }

    renderChips(data, level);

    resultMessage.textContent = data.message || "";
    resultMessage.classList.toggle("hidden", !data.message);

    qualityWarnings.innerHTML = "";
    (data.quality_warnings || []).forEach((warning) => {
      const li = document.createElement("li");
      li.textContent = warning.message;
      qualityWarnings.appendChild(li);
    });
    qualityWarnings.classList.toggle(
      "hidden",
      !(data.quality_warnings && data.quality_warnings.length)
    );

    top3List.innerHTML = "";
    data.top3_predictions.forEach((item, index) => {
      const highlight = index === 0 && level !== "low" ? " highlight" : "";

      const row = document.createElement("div");
      row.className = "bar-row";

      const top = document.createElement("div");
      top.className = "bar-row-top";

      const label = document.createElement("span");
      label.className = "bar-label" + highlight;
      label.textContent = formatClassName(item.class);

      const value = document.createElement("span");
      value.className = "bar-value" + highlight;
      value.textContent = `${item.confidence.toFixed(2)}%`;

      top.appendChild(label);
      top.appendChild(value);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill" + highlight;
      fill.style.width = `${Math.min(item.confidence, 100)}%`;
      track.appendChild(fill);

      row.appendChild(top);
      row.appendChild(track);
      top3List.appendChild(row);
    });
  }

  function renderChips(data, level) {
    const hasDetails = Boolean(data.crop && data.health_status);
    resultChips.classList.toggle("hidden", !hasDetails);
    if (!hasDetails) return;

    cropChip.classList.remove("hidden");
    cropChip.textContent =
      level === "low"
        ? `Crop: uncertain`
        : typeof data.crop_confidence === "number"
          ? `Crop: ${data.crop} (${data.crop_confidence.toFixed(1)}%)`
          : `Crop: ${data.crop}`;

    const status = data.health_status;
    healthChip.className = `chip chip-${status}`;
    healthChip.textContent =
      status === "healthy" ? "Appears healthy"
        : status === "diseased" ? "Signs of disease"
          : "Health status uncertain";
  }

  /* ---------- Hybrid analysis (CNN + AI plant identification) ---------- */

  function fillNotes(notes) {
    analysisNotes.innerHTML = "";
    (notes || []).forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      analysisNotes.appendChild(li);
    });
    analysisNotes.classList.toggle("hidden", !(notes && notes.length));
  }

  function renderPlantChip(analysis) {
    const id = analysis.plant_identification;
    plantChip.classList.remove("hidden");

    if (id && id.is_plant) {
      const sci = id.plant_scientific_name ? ` (${id.plant_scientific_name})` : "";
      plantChip.textContent =
        `🌿 ${id.plant_common_name || "Unknown plant"}${sci} · AI-identified, ${id.identification_confidence} confidence`;
    } else if (id && !id.is_plant) {
      plantChip.textContent = "No plant detected · AI";
    } else {
      plantChip.textContent = "Plant type not verified (AI identification unavailable)";
    }
  }

  function showNoDiagnosis(analysis) {
    const id = analysis.plant_identification || {};
    setStatus("result");
    resultSkeleton.classList.add("hidden");
    resultError.classList.add("hidden");
    resultContent.classList.remove("hidden");

    levelBanner.className = "level-banner level-uncertain";
    resultChips.classList.remove("hidden");
    top3Section.classList.add("hidden");
    cropChip.classList.add("hidden");

    if (analysis.route === "not_plant") {
      levelBanner.className = "level-banner level-low";
      levelBanner.textContent = "No plant detected — nothing to diagnose.";
      resultEyebrow.textContent = "Result";
      resultClass.textContent = "No plant detected";
      resultConfidence.classList.add("hidden");
      healthChip.className = "chip hidden";
    } else {
      levelBanner.textContent =
        "This plant is outside the disease model's 14 crops — showing an unverified AI observation instead.";
      resultEyebrow.textContent = "Identified plant";
      resultClass.textContent = id.plant_common_name || "Unknown plant";
      resultConfidence.classList.remove("hidden");
      resultConfidence.className = "confidence-badge level-uncertain";
      resultConfidence.textContent = "Not covered by model";

      const healthText = { yes: "AI: looks healthy", no: "AI: possible problem", unclear: "AI: health unclear" };
      const healthClass = { yes: "chip-healthy", no: "chip-diseased", unclear: "chip-uncertain" };
      healthChip.className = `chip ${healthClass[id.appears_healthy] || "chip-uncertain"}`;
      healthChip.textContent = healthText[id.appears_healthy] || "AI: health unclear";
    }

    resultMessage.textContent = analysis.message || "";
    resultMessage.classList.toggle("hidden", !analysis.message);

    const warnings = (analysis.prediction && analysis.prediction.quality_warnings) || [];
    qualityWarnings.innerHTML = "";
    warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w.message;
      qualityWarnings.appendChild(li);
    });
    qualityWarnings.classList.toggle("hidden", warnings.length === 0);
  }

  function renderObservation(analysis) {
    const id = analysis.plant_identification;
    if (!analysis.show_llm_observation || !id) {
      observationPanel.classList.add("hidden");
      return;
    }

    observationPanel.classList.remove("hidden");
    const name = id.plant_common_name || "this plant";
    const healthy = { yes: "looks healthy", no: "shows possible problems", unclear: "has an unclear health status" };
    document.getElementById("obsIntro").textContent =
      `The AI thinks this is ${name} and that it ${healthy[id.appears_healthy] || "has an unclear health status"}. ` +
      "This is an AI observation of the photo, not a model diagnosis — confirm with a local expert." +
      (id.image_notes ? ` Photo note: ${id.image_notes}` : "");

    fillList(document.getElementById("obsSymptoms"), id.visible_symptoms);
    fillList(document.getElementById("obsIssues"), id.possible_issues);
    fillList(document.getElementById("obsAdvice"), id.general_advice);
    document.getElementById("obsMeta").textContent =
      id.generated_by ? `Generated by ${id.generated_by.model}` : "";
  }

  function showAnalysis(analysis) {
    const p = analysis.prediction;
    const d = analysis.diagnosis;
    const id = analysis.plant_identification;

    if (d) {
      showResult({
        predicted_class: d.predicted_class,
        confidence: d.confidence,
        calibrated_confidence: d.confidence,
        confidence_level: d.confidence_level,
        top3_predictions: d.top3_predictions,
        // Crop as recognised by the trained model; the AI's view is in the plant chip.
        crop: p.crop,
        crop_confidence: p.crop_confidence,
        health_status: d.confidence_level === "low" ? "uncertain" : d.health_status,
        message: analysis.message,
        quality_warnings: p.quality_warnings,
      });

      if (d.source === "cnn_within_identified_crop") {
        top3Eyebrow.textContent = `Model's top matches within ${id.supported_crop}`;
      }
    } else {
      showNoDiagnosis(analysis);
    }

    renderPlantChip(analysis);
    fillNotes(analysis.notes);
    renderObservation(analysis);

    lastAnalysis = analysis;
    explainSection.classList.toggle("hidden", !(d && d.confidence_level !== "low"));
  }

  function showError(message) {
    setStatus("error");
    resultSkeleton.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.remove("hidden");
    errorMessage.textContent = message || "Something went wrong. Please try again.";
  }

  /* ---------- Treatment guidance ---------- */

  function hideTreatmentStates() {
    treatmentLoading.classList.add("hidden");
    treatmentUnavailable.classList.add("hidden");
    treatmentError.classList.add("hidden");
    treatmentContent.classList.add("hidden");
    lastGuidance = null;
    discardSpeechAudio();
    speakBtn.classList.add("hidden");
    speakNote.classList.add("hidden");
  }

  /* ---------- Read aloud ----------
   * English: the browser's own voice (every device has one).
   * Other languages: audio generated on the server (Gemini TTS), because
   * most devices have no Tamil / Telugu / ... voice installed; a device
   * voice is only used if the server audio fails.
   */

  const speech = "speechSynthesis" in window ? window.speechSynthesis : null;
  let speechAudio = null; // HTMLAudioElement for server audio
  let speechAudioUrl = null; // object URL, reused while the guidance is unchanged
  let speechController = null;

  function setSpeakState(state) {
    const labels = { idle: "🔊 Read aloud", loading: "⏳ Preparing audio…", playing: "⏹ Stop reading" };
    speakBtn.textContent = labels[state];
    speakBtn.setAttribute("aria-pressed", state === "playing" ? "true" : "false");
    speakBtn.disabled = state === "loading";
  }

  function stopSpeaking() {
    if (speech) speech.cancel();
    if (speechAudio) speechAudio.pause();
    if (speechController) speechController.abort();
    speechController = null;
    setSpeakState("idle");
  }

  function discardSpeechAudio() {
    stopSpeaking();
    if (speechAudioUrl) URL.revokeObjectURL(speechAudioUrl);
    speechAudioUrl = null;
    speechAudio = null;
  }

  function guidanceSentences(result) {
    const g = result.guidance;
    const parts = [g.disease, g.what_it_means];
    if (result.confidence_level === "uncertain") parts.push(g.uncertainty_note);
    [g.common_symptoms, g.immediate_steps, g.recommended_treatment, g.preventive_measures, g.when_to_seek_expert_help]
      .forEach((list) => (list || []).forEach((item) => parts.push(item)));
    return parts.filter(Boolean);
  }

  function deviceVoice(langTag) {
    if (!speech) return null;
    const prefix = langTag.split("-")[0].toLowerCase();
    return speech.getVoices().find((v) => v.lang && v.lang.toLowerCase().startsWith(prefix)) || null;
  }

  function speakWithBrowser(langTag, voice) {
    speech.cancel();
    // One utterance per sentence: Chrome cuts long utterances off after ~15 s.
    const sentences = guidanceSentences(lastGuidance);
    sentences.forEach((text, index) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = langTag;
      if (voice) utterance.voice = voice;
      utterance.rate = 0.95;
      utterance.onerror = stopSpeaking;
      if (index === sentences.length - 1) utterance.onend = () => setSpeakState("idle");
      speech.speak(utterance);
    });
    setSpeakState("playing");
  }

  function playAudio() {
    speechAudio.currentTime = 0;
    speechAudio.play().then(() => setSpeakState("playing")).catch(() => {
      setSpeakState("idle");
      speakNote.textContent = "The browser blocked audio playback. Press Read aloud again.";
      speakNote.classList.remove("hidden");
    });
  }

  async function speakWithServer(langTag) {
    if (speechAudio) {
      playAudio();
      return;
    }

    setSpeakState("loading");
    speakNote.textContent = "Generating audio — this can take 10–20 seconds the first time.";
    speakNote.classList.remove("hidden");

    const controller = new AbortController();
    speechController = controller;

    try {
      const response = await fetch("/speech", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastPrediction, language: lastGuidance.language || currentLanguage() }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let message = "Could not create the audio.";
        try {
          message = (await response.json()).error || message;
        } catch (_) {
          /* keep default */
        }
        throw new Error(message);
      }

      const blob = await response.blob();
      if (controller !== speechController) return;

      speechAudioUrl = URL.createObjectURL(blob);
      speechAudio = new Audio(speechAudioUrl);
      speechAudio.onended = () => setSpeakState("idle");
      speakNote.classList.add("hidden");
      speechController = null;
      playAudio();
    } catch (err) {
      if (err.name === "AbortError") return;
      speechController = null;

      const voice = deviceVoice(langTag);
      if (voice) {
        speakNote.textContent = "Server audio unavailable — using this device's voice instead.";
        speakWithBrowser(langTag, voice);
      } else {
        setSpeakState("idle");
        speakNote.textContent = `${err.message} Please try again in a minute.`;
      }
    }
  }

  function speakGuidance() {
    if (!lastGuidance || !lastPrediction) return;

    if (speakBtn.getAttribute("aria-pressed") === "true") {
      stopSpeaking();
      return;
    }

    speakNote.classList.add("hidden");
    const langTag = lastGuidance.speech_lang || "en-IN";
    const isEnglish = langTag.toLowerCase().startsWith("en");

    if (isEnglish && speech) {
      speakWithBrowser(langTag, deviceVoice(langTag));
    } else {
      speakWithServer(langTag);
    }
  }

  function resetTreatmentPanel() {
    if (recommendationController) recommendationController.abort();
    recommendationController = null;
    hideTreatmentStates();
    treatmentPanel.classList.add("hidden");
  }

  function fillList(element, items) {
    element.innerHTML = "";
    const list = items && items.length ? items : null;
    if (!list) {
      const li = document.createElement("li");
      li.className = "empty-item";
      li.textContent = "No specific items.";
      element.appendChild(li);
      return;
    }
    list.forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      element.appendChild(li);
    });
  }

  function showTreatment(result) {
    const g = result.guidance;
    hideTreatmentStates();
    treatmentContent.classList.remove("hidden");

    document.getElementById("guidanceDisease").textContent = g.disease;
    document.getElementById("guidanceMeaning").textContent = g.what_it_means;
    document.getElementById("guidanceUncertainty").textContent =
      result.confidence_level === "uncertain" ? g.uncertainty_note : "";

    fillList(document.getElementById("guidanceSymptoms"), g.common_symptoms);
    fillList(document.getElementById("guidanceImmediate"), g.immediate_steps);
    fillList(document.getElementById("guidanceTreatment"), g.recommended_treatment);
    fillList(document.getElementById("guidancePrevention"), g.preventive_measures);
    fillList(document.getElementById("guidanceExpert"), g.when_to_seek_expert_help);

    const grounding = result.grounding || {};
    const meta = [];
    if (grounding.pathogen_type && grounding.pathogen_type !== "none") {
      meta.push(`Pathogen type: ${grounding.pathogen_type}`);
    }
    if (grounding.causal_agent) meta.push(`Causal agent: ${grounding.causal_agent}`);
    if (grounding.knowledge_expert_reviewed === false) {
      meta.push("Reference facts not yet reviewed by an agronomist");
    }
    if (result.generated_by) meta.push(`Generated by ${result.generated_by.model}`);
    document.getElementById("guidanceMeta").textContent = meta.join(" · ");
    document.getElementById("guidanceDisclaimer").textContent = result.disclaimer || "";

    lastGuidance = result;
    speakBtn.classList.remove("hidden"); // server audio works even without device voices
  }

  /* ---------- PDF report (browser "Save as PDF") ---------- */

  const esc = (value) =>
    String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const listHtml = (items) =>
    items && items.length ? `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : "<p>—</p>";

  function buildReport() {
    const a = lastAnalysis;
    if (!a) return "";

    const d = a.diagnosis;
    const id = a.plant_identification;
    const now = new Date();
    const parts = [];

    parts.push(`<h1>Plant health report</h1>`);
    parts.push(
      `<p class="meta">Generated ${esc(now.toLocaleString())} · File: ${esc(selectedFile ? selectedFile.name : "")} · ` +
      `Model: ${esc(a.prediction.model_version)}</p>`
    );

    const images = [`<figure><img src="${esc(previewImage.src)}" alt="Uploaded photo"><figcaption>Uploaded photo</figcaption></figure>`];
    if (lastExplanation) {
      images.push(
        `<figure><img src="${esc(lastExplanation.overlay_png)}" alt="Model attention heat map">` +
        `<figcaption>Where the model looked (red = most important)</figcaption></figure>`
      );
    }
    parts.push(`<div class="images">${images.join("")}</div>`);

    const plantLine = id
      ? id.is_plant
        ? `${esc(id.plant_common_name || "Unknown plant")}${id.plant_scientific_name ? ` (<i>${esc(id.plant_scientific_name)}</i>)` : ""} — AI identification, ${esc(id.identification_confidence)} confidence`
        : "No plant detected"
      : "Not verified (AI identification unavailable)";

    const rows = [["Plant", plantLine]];
    if (d) {
      const shown = d.confidence_level === "low" ? "Uncertain — no diagnosis" : formatClassName(d.predicted_class);
      rows.push(["Diagnosis (trained model)", esc(shown)]);
      rows.push(["Confidence", `${esc(d.confidence.toFixed(1))}% · ${esc(LEVEL_TEXT[d.confidence_level] || d.confidence_level)}`]);
      const health = { healthy: "Appears healthy", diseased: "Signs of disease", uncertain: "Uncertain" };
      rows.push(["Health", esc(health[d.confidence_level === "low" ? "uncertain" : d.health_status])]);
    } else {
      rows.push(["Diagnosis (trained model)", "Not available for this photo"]);
    }
    parts.push(`<section><h2>Result</h2><table>${rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`).join("")}</table>`);
    parts.push(`<p>${esc(a.message)}</p>`);
    const notes = [...(a.notes || []), ...((a.prediction.quality_warnings || []).map((w) => w.message))];
    if (notes.length) parts.push(listHtml(notes));
    parts.push(`</section>`);

    if (d && d.top3_predictions) {
      parts.push(
        `<section><h2>Top model predictions</h2><table>` +
        d.top3_predictions.map((p) => `<tr><td>${esc(formatClassName(p.class))}</td><td>${esc(p.confidence.toFixed(1))}%</td></tr>`).join("") +
        `</table></section>`
      );
    }

    if (a.show_llm_observation && id) {
      parts.push(
        `<section><h2>AI observation (unverified)</h2>` +
        `<p><b>Visible symptoms</b></p>${listHtml(id.visible_symptoms)}` +
        `<p><b>Possible issues</b></p>${listHtml(id.possible_issues)}` +
        `<p><b>General care steps</b></p>${listHtml(id.general_advice)}</section>`
      );
    }

    if (lastGuidance) {
      const g = lastGuidance.guidance;
      parts.push(
        `<section><h2>Treatment guidance (AI-generated)</h2>` +
        `<p><b>${esc(g.disease)}</b></p><p>${esc(g.what_it_means)}</p>` +
        (lastGuidance.confidence_level === "uncertain" ? `<p>${esc(g.uncertainty_note)}</p>` : "") +
        `<p><b>Common symptoms</b></p>${listHtml(g.common_symptoms)}` +
        `<p><b>Immediate steps</b></p>${listHtml(g.immediate_steps)}` +
        `<p><b>Recommended treatment</b></p>${listHtml(g.recommended_treatment)}` +
        `<p><b>Preventive measures</b></p>${listHtml(g.preventive_measures)}` +
        `<p><b>When to seek expert help</b></p>${listHtml(g.when_to_seek_expert_help)}</section>`
      );
    }

    parts.push(
      `<p class="disclaimer">This report is produced automatically by an AI system and is not a confirmed diagnosis. ` +
      `Treatment depends on the crop, severity and local regulations — always follow product labels and consult a local agricultural expert.</p>`
    );

    return parts.join("");
  }

  function downloadReport() {
    if (!lastAnalysis) return;

    printReport.innerHTML = buildReport();

    const originalTitle = document.title;
    const stamp = new Date().toISOString().slice(0, 10);
    document.title = `plant-report-${stamp}`; // default PDF file name
    window.addEventListener("afterprint", () => { document.title = originalTitle; }, { once: true });

    // Wait for the report images to load before opening the print dialog.
    const pending = [...printReport.querySelectorAll("img")]
      .filter((img) => !img.complete)
      .map((img) => new Promise((resolve) => { img.onload = img.onerror = resolve; }));

    Promise.all(pending).then(() => window.print());
  }

  function showTreatmentError(message) {
    hideTreatmentStates();
    treatmentError.classList.remove("hidden");
    treatmentErrorText.textContent =
      message || "Could not load treatment guidance. Please try again.";
  }

  async function requestRecommendation(prediction) {
    if (recommendationController) recommendationController.abort();
    const controller = new AbortController();
    recommendationController = controller;

    treatmentPanel.classList.remove("hidden");
    hideTreatmentStates();

    if (!prediction.confidence_level) {
      treatmentUnavailable.classList.remove("hidden");
      treatmentUnavailableText.textContent =
        "Treatment guidance needs a confidence assessment, which this server did not return.";
      return;
    }

    if (prediction.confidence_level === "low") {
      treatmentUnavailable.classList.remove("hidden");
      treatmentUnavailableText.textContent =
        "Treatment guidance is only generated when the prediction is reliable enough. " +
        "Upload a clearer photo of a single leaf to get guidance.";
      return;
    }

    treatmentLoading.classList.remove("hidden");

    try {
      const response = await fetch("/recommendation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          predicted_class: prediction.predicted_class,
          confidence_level: prediction.confidence_level,
          top3_predictions: prediction.top3_predictions,
          language: currentLanguage(),
        }),
        signal: controller.signal,
      });

      let data = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (controller !== recommendationController) return;

      if (!response.ok || !data || !data.guidance) {
        showTreatmentError(data && data.error);
        return;
      }

      showTreatment(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      showTreatmentError("Could not reach the server for treatment guidance. Please try again.");
    }
  }

  function handleFiles(files) {
    if (!files || !files[0]) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      showError("File is too large. Maximum size is 10 MB.");
      return;
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);

    selectedFile = file;
    previewImage.src = objectUrl;
    fileNameEl.textContent = file.name;

    dropzone.classList.add("hidden");
    previewArea.classList.remove("hidden");
    loadingOverlay.classList.add("hidden");
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict disease";

    resetResultPanel();
  }

  function removeImage() {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    selectedFile = null;
    fileInput.value = "";

    dropzone.classList.remove("hidden");
    previewArea.classList.add("hidden");
    loadingOverlay.classList.add("hidden");

    resetResultPanel();
  }

  async function predict() {
    if (!selectedFile) return;

    loadingOverlay.classList.remove("hidden");
    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing…";
    showLoading();

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("language", currentLanguage());

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch (_) {
        data = null;
      }

      if (!response.ok || !data) {
        showError(data && data.error ? data.error : `Server error (${response.status}). Please try again.`);
        return;
      }

      showAnalysis(data);

      if (data.diagnosis) {
        lastPrediction = {
          predicted_class: data.diagnosis.predicted_class,
          confidence_level: data.diagnosis.confidence_level,
          top3_predictions: data.diagnosis.top3_predictions,
        };
        requestRecommendation(lastPrediction);
      } else if (data.route === "unsupported_plant") {
        lastPrediction = null;
        treatmentPanel.classList.remove("hidden");
        hideTreatmentStates();
        treatmentUnavailable.classList.remove("hidden");
        treatmentUnavailableText.textContent =
          "Model-based treatment guidance is only available for the 14 supported crops. " +
          "See the AI observation above and consult a local agricultural expert.";
      }
    } catch (err) {
      showError("Could not reach the prediction server. Please try again.");
    } finally {
      loadingOverlay.classList.add("hidden");
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict disease";
    }
  }

  chooseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => handleFiles(e.target.files));

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragging");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
    handleFiles(e.dataTransfer.files);
  });

  removeImageBtn.addEventListener("click", removeImage);
  removeBtn.addEventListener("click", removeImage);
  predictBtn.addEventListener("click", predict);
  treatmentRetryBtn.addEventListener("click", () => {
    if (lastPrediction) requestRecommendation(lastPrediction);
  });

  explainBtn.addEventListener("click", requestExplanation);
  reportBtn.addEventListener("click", downloadReport);
  speakBtn.addEventListener("click", speakGuidance);

  languageSelect.addEventListener("change", () => {
    try {
      localStorage.setItem(LANGUAGE_KEY, currentLanguage());
    } catch (_) {
      /* preference just isn't remembered */
    }
    // Re-generate guidance for the current result in the new language.
    if (lastPrediction && !treatmentPanel.classList.contains("hidden")) {
      requestRecommendation(lastPrediction);
    }
  });

  // Chrome loads the voice list asynchronously; asking early fills it.
  if (speech) speech.getVoices();

  loadLanguages();
})();
