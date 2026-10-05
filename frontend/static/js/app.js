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

  let selectedFile = null;
  let objectUrl = null;
  let lastPrediction = null;
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  let recommendationController = null;

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
        crop: id && id.supported_crop !== "none" ? id.supported_crop : p.crop,
        crop_confidence: id && id.supported_crop !== "none" ? null : p.crop_confidence,
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
})();
