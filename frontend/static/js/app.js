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

  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  let selectedFile = null;
  let objectUrl = null;

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
    resultSkeleton.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
    skeletonMessage.textContent =
      message || "Upload a photo and press Predict to see the results here.";
  }

  function showLoading() {
    setStatus("loading");
    resultSkeleton.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
    skeletonMessage.textContent = "Model is inspecting leaf texture, edges and lesions…";
  }

  function showResult(data) {
    setStatus("result");
    resultSkeleton.classList.add("hidden");
    resultError.classList.add("hidden");
    resultContent.classList.remove("hidden");

    resultClass.textContent = formatClassName(data.predicted_class);
    resultConfidence.textContent = `${data.confidence.toFixed(2)}%`;

    top3List.innerHTML = "";
    data.top3_predictions.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "bar-row";

      const top = document.createElement("div");
      top.className = "bar-row-top";

      const label = document.createElement("span");
      label.className = "bar-label" + (index === 0 ? " highlight" : "");
      label.textContent = formatClassName(item.class);

      const value = document.createElement("span");
      value.className = "bar-value" + (index === 0 ? " highlight" : "");
      value.textContent = `${item.confidence.toFixed(2)}%`;

      top.appendChild(label);
      top.appendChild(value);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill" + (index === 0 ? " highlight" : "");
      fill.style.width = `${Math.min(item.confidence, 100)}%`;
      track.appendChild(fill);

      row.appendChild(top);
      row.appendChild(track);
      top3List.appendChild(row);
    });
  }

  function showError(message) {
    setStatus("error");
    resultSkeleton.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.remove("hidden");
    errorMessage.textContent = message || "Something went wrong. Please try again.";
  }

  function handleFiles(files) {
    if (!files || !files[0]) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) return;

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
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error);
        return;
      }

      showResult(data);
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
})();
