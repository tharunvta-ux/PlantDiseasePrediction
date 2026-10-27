/*
 * Treatment plan, accounts and treatment history.
 *
 * app.js announces each analysis ("verdant:analysis") and resets
 * ("verdant:reset"); this file shows the first step of the plan, saves it
 * as a tracked case, and lets the grower report "worked / did not work"
 * at each check.
 *
 * Language: plan and case texts arrive translated from the server
 * (?language=…), interface labels via /ui-text, and 🔊 reads a plan or a
 * case aloud in the chosen language. Product names and doses are never
 * translated.
 */
(() => {
  const $ = (id) => document.getElementById(id);

  const planPanel = $("planPanel");
  const planStep = $("planStep");
  const planActions = $("planActions");
  const planSpeakBtn = $("planSpeakBtn");
  const trackBtn = $("trackBtn");
  const trackNote = $("trackNote");

  const signInBtn = $("signInBtn");
  const signOutBtn = $("signOutBtn");
  const accountEmail = $("accountEmail");
  const dueCounts = [$("dueCount"), $("dueCountMobile")];

  const historySignedOut = $("historySignedOut");
  const historySignedIn = $("historySignedIn");
  const historySignInBtn = $("historySignInBtn");
  const dueBanner = $("dueBanner");
  const historyEmpty = $("historyEmpty");
  const historyError = $("historyError");
  const historyTranslating = $("historyTranslating");
  const caseList = $("caseList");

  const authDialog = $("authDialog");
  const authForm = $("authForm");
  const authTitle = $("authTitle");
  const authEmail = $("authEmail");
  const authPassword = $("authPassword");
  const authError = $("authError");
  const authSubmit = $("authSubmit");
  const authSwitchText = $("authSwitchText");
  const authSwitchBtn = $("authSwitchBtn");
  const authCloseBtn = $("authCloseBtn");

  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  const KISAN_CALL_CENTRE = "1800-180-1551";
  const mapsSearch = (query) =>
    `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;

  let user = null;
  let cases = [];
  let authMode = "login";
  let pendingTrack = false;

  // Latest analysis from app.js and what the plan panel shows for it.
  let lastAnalysis = null;
  let lastFile = null;
  let lastPlan = null;
  let planState = null; // { mode: "confirm" | "checks" | "plan", fields, answer, extent }

  /* ---------- helpers ---------- */

  // Builds DOM nodes; all text goes through textContent.
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([key, value]) => {
      if (value === null || value === undefined || value === false) return;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    });
    [].concat(children).forEach((child) => {
      if (child === null || child === undefined || child === false) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  async function api(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }
    if (!response.ok) {
      const error = new Error((data && data.error) || `Server error (${response.status}). Please try again.`);
      error.status = response.status;
      error.code = data && data.code;
      throw error;
    }
    return data;
  }

  const postJson = (url, body) =>
    api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  function language() {
    const select = $("languageSelect");
    return (select && select.value) || "en";
  }

  /* ---------- interface labels ---------- */

  let uiStrings = {};
  let uiLanguage = "en";

  // Translated label; {name} placeholders are filled from `vars`.
  function T(template, vars = {}) {
    const text = (uiLanguage === language() && uiStrings[template]) || template;
    return text.replace(/\{([a-z_]+)\}/g, (match, key) => (key in vars ? String(vars[key]) : match));
  }

  const countText = (n, word) => T(n === 1 ? `{n} ${word}` : `{n} ${word}s`, { n });

  async function loadUiText() {
    const lang = language();
    if (lang === "en") {
      uiStrings = {};
      uiLanguage = "en";
      return;
    }
    try {
      const data = await api(`/ui-text?language=${encodeURIComponent(lang)}`);
      if (lang !== language()) return;
      uiStrings = data.strings || {};
      uiLanguage = lang;
    } catch (_) {
      /* labels stay in English */
    }
  }

  // Static page text marked data-i18n="…English…".
  function applyStaticText() {
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = T(node.dataset.i18n, { n: node.dataset.i18nN || "" });
    });
  }

  function locale() {
    const lang = language();
    return lang === "en" ? undefined : `${lang}-IN`;
  }

  function formatDate(iso, withYear = true) {
    if (!iso) return "";
    const options = withYear ? { day: "numeric", month: "short", year: "numeric" } : { day: "numeric", month: "short" };
    try {
      return new Date(iso).toLocaleDateString(locale(), options);
    } catch (_) {
      return new Date(iso).toLocaleDateString(undefined, options);
    }
  }

  function startOfDay(date) {
    return new Date(date.getFullYear(), date.getMonth(), date.getDate());
  }

  function dueInfo(step) {
    if (!step || !step.check_due_at) return null;
    const due = new Date(step.check_due_at);
    const days = Math.round((startOfDay(due) - startOfDay(new Date())) / 86400000);
    const date = formatDate(step.check_due_at, false);

    if (days > 1) return { due: false, text: T("Check again in {days} ({date})", { days: countText(days, "day"), date }) };
    if (days === 1) return { due: false, text: T("Check again tomorrow ({date})", { date }) };
    if (days === 0) return { due: true, text: T("Check due today") };
    return { due: true, text: T("Check overdue by {days}", { days: countText(-days, "day") }) };
  }

  const isDue = (c) => c.awaiting_check_in && Boolean((dueInfo(c.current_step) || {}).due);

  /* ---------- read aloud ----------
   * The server builds the spoken text from its own plan / case data in the
   * chosen language. English uses the browser's voice; other languages use
   * server audio (most devices have no Tamil, Telugu, … voice), one short
   * part at a time while the next part is prepared. */

  const speech = "speechSynthesis" in window ? window.speechSynthesis : null;
  const player = new Audio();
  let speaking = null; // { button, session, finish }
  let speakSession = 0;

  function speakButton(onclick) {
    const button = el("button", { type: "button", class: "btn-outline btn-pill speak-btn", "aria-pressed": "false", onclick });
    setSpeakState(button, "idle");
    return button;
  }

  function setSpeakState(button, state) {
    const labels = { idle: "🔊 Read aloud", loading: "⏳ Preparing audio…", playing: "⏹ Stop reading" };
    button.textContent = T(labels[state]);
    button.setAttribute("aria-pressed", state === "playing" ? "true" : "false");
    button.disabled = state === "loading";
  }

  function stopSpeaking() {
    speakSession += 1;
    if (speech) speech.cancel();
    player.pause();
    if (speaking) {
      setSpeakState(speaking.button, "idle");
      if (speaking.finish) speaking.finish();
    }
    speaking = null;
  }

  async function readAloud(button, url, body) {
    if (speaking && speaking.button === button) {
      stopSpeaking();
      return;
    }
    stopSpeaking();

    const session = speakSession;
    const lang = language();
    speaking = { button, session };
    setSpeakState(button, "loading");

    try {
      const info = await postJson(url, { ...body, language: lang });
      if (session !== speakSession) return;

      if (info.speech_lang.toLowerCase().startsWith("en") && speech) {
        speech.cancel();
        info.parts.forEach((text, index) => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = info.speech_lang;
          if (index === info.parts.length - 1) utterance.onend = () => { if (session === speakSession) stopSpeaking(); };
          speech.speak(utterance);
        });
        setSpeakState(button, "playing");
        return;
      }

      const fetchPart = async (part) => {
        const response = await fetch(url, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, language: lang, part }),
        });
        if (!response.ok) throw new Error("audio");
        return URL.createObjectURL(await response.blob());
      };

      let next = fetchPart(0);
      for (let index = 0; index < info.parts.length; index += 1) {
        const src = await next;
        if (session !== speakSession) return URL.revokeObjectURL(src);
        if (index + 1 < info.parts.length) next = fetchPart(index + 1);

        player.src = src;
        await player.play();
        setSpeakState(button, "playing");
        await new Promise((resolve) => {
          player.onended = resolve;
          speaking.finish = resolve;
        });
        URL.revokeObjectURL(src);
        if (session !== speakSession) return;
      }
      stopSpeaking();
    } catch (_) {
      if (session !== speakSession) return;
      stopSpeaking();
      button.title = T("Could not play the audio. Please try again in a minute.");
      window.alert(button.title);
    }
  }

  /* ---------- step rendering (plan panel + history) ---------- */

  function expertLinks() {
    return el("div", { class: "expert-links" }, [
      el("a", { class: "btn-primary btn-pill", href: `tel:${KISAN_CALL_CENTRE.replace(/-/g, "")}`, text: T("📞 Kisan Call Centre {number}", { number: KISAN_CALL_CENTRE }) }),
      el("a", { class: "btn-outline btn-pill", href: mapsSearch("Krishi Vigyan Kendra near me"), target: "_blank", rel: "noopener", text: T("📍 Krishi Vigyan Kendra nearby") }),
      el("a", { class: "btn-outline btn-pill", href: mapsSearch("agriculture department office near me"), target: "_blank", rel: "noopener", text: T("📍 Agriculture office nearby") }),
    ]);
  }

  function productCard(step) {
    if (!step.products || !step.products.length) return null;

    return el("div", { class: "product-card" }, [
      el("p", {
        class: "product-title",
        text: T(step.products.length > 1 ? "Use ONE of these TNAU-recommended options:" : "TNAU-recommended option:"),
      }),
      el("table", { class: "product-table" }, [
        el("tbody", {}, step.products.map((p) =>
          el("tr", {}, [
            // Name and dose exactly as TNAU lists them, in every language.
            el("td", {}, [el("strong", { text: p.name }), el("span", { class: "product-kind", text: `${p.kind} · ${p.group}` })]),
            el("td", { class: "product-dose", text: p.dose }),
          ])
        )),
      ]),
      step.source_url
        ? el("p", { class: "treatment-meta" }, [
            `${T("Source:")} `,
            el("a", { href: step.source_url, target: "_blank", rel: "noopener", text: "TNAU Agritech Portal" }),
          ])
        : null,
    ]);
  }

  // inHistory: the check-in form below already asks the check question.
  function stepBody(step, inHistory = false) {
    const days = step.wait_days ? countText(step.wait_days, "day") : null;
    const waitText = days
      ? inHistory
        ? T("Then wait {days} and check again.", { days })
        : T("Then wait {days} and check: {question}", { days, question: step.check_question })
      : null;

    return el("div", { class: `step-body step-${step.kind}` }, [
      el("h4", { class: "step-title", text: step.title }),
      step.reason ? el("p", { class: "step-reason", text: step.reason }) : null,
      step.actions && step.actions.length
        ? el("ul", { class: "step-actions" }, step.actions.map((a) => el("li", { text: a })))
        : null,
      productCard(step),
      step.kind === "expert" ? expertLinks() : null,
      waitText ? el("p", { class: "step-wait", text: waitText }) : null,
    ]);
  }

  /* ---------- plan panel ---------- */

  function diagnosisFields(analysis) {
    const d = analysis.diagnosis;
    const id = analysis.plant_identification || {};

    if (d) {
      if (d.confidence_level === "low" || d.is_healthy) return null;
      return {
        predicted_class: d.predicted_class,
        confidence_level: d.confidence_level,
        // The AI's plant name only when it agrees with the disease model.
        plant_name: analysis.route === "supported_crop" ? id.plant_common_name || "" : "",
      };
    }

    if (analysis.route === "unsupported_plant" && id.is_plant && id.appears_healthy !== "yes") {
      return {
        predicted_class: "",
        confidence_level: "",
        plant_name: id.plant_common_name || "Plant",
        general_advice: id.general_advice || [],
      };
    }

    return null;
  }

  function hidePlan() {
    lastPlan = null;
    planState = null;
    planPanel.classList.add("hidden");
    planSpeakBtn.classList.add("hidden");
  }

  function resetTrackButton() {
    trackBtn.disabled = false;
    trackBtn.textContent = T(user ? "Track this treatment" : "Sign in to track this treatment");
  }

  async function showPlan(analysis, file) {
    lastAnalysis = analysis;
    lastFile = file;
    hidePlan();

    if (analysis.ai_condition && analysis.route === "unsupported_plant") {
      planState = { mode: "confirm" };
      renderPlanState();
      return;
    }

    const fields = diagnosisFields(analysis);
    if (fields) {
      planState = { mode: "plan", fields, answer: null, extent: null };
      await loadPlan();
    }
  }

  // Re-draw the plan panel (after an answer, or in a new language).
  function renderPlanState() {
    if (!planState || !lastAnalysis) return;
    if (planState.mode === "confirm") askCrop();
    else if (planState.mode === "checks") symptomChecks();
    else loadPlan();
  }

  /* An AI-matched disease of a crop the disease model does not cover.
   * TNAU products are only shown after the grower confirms it:
   * 1. "Is this a <crop> plant?" (growers know what they planted),
   * 2. simple yes/no checks they can do by looking at / touching a leaf. */

  const ANSWERS = {
    yes: "Your answers match this disease.",
    unsure: "Your answers only partly match, so safe care steps come first.",
    no: "Your answers do not match this disease, so general care steps are shown.",
    wrong_crop: "You said this is not the plant the AI named, so general care steps are shown.",
  };

  function answerMatch(choice) {
    const condition = lastAnalysis.ai_condition;
    const id = lastAnalysis.plant_identification || {};
    const plantName = id.plant_common_name || condition.crop;
    const general = choice === "no" || choice === "wrong_crop";

    planState = {
      mode: "plan",
      answer: choice,
      extent: null,
      fields: general
        ? { predicted_class: "", confidence_level: "", plant_name: choice === "wrong_crop" ? "Plant" : plantName, general_advice: id.general_advice || [] }
        : { predicted_class: condition.id, confidence_level: choice === "yes" ? "high" : "uncertain", plant_name: plantName },
    };
    loadPlan();
  }

  function restartMatch() {
    planState = { mode: "confirm" };
    renderPlanState();
  }

  function askExpert() {
    return el("details", { class: "ask-expert" }, [
      el("summary", { text: T("Not sure? Ask an expert first") }),
      el("p", { class: "treatment-meta", text: T("An expert can confirm the disease from the plant or a photo before you spend money on any treatment.") }),
      expertLinks(),
    ]);
  }

  function askCrop() {
    const condition = lastAnalysis.ai_condition;
    const crop = condition.crop.toLowerCase();

    lastPlan = null;
    planSpeakBtn.classList.add("hidden");
    planStep.replaceChildren(
      el("div", { class: "match-card" }, [
        el("p", { class: "result-eyebrow", text: T("Possible match found by AI") }),
        el("h4", { class: "step-title", text: condition.display_name }),
        el("p", { class: "checkin-question", text: T("First: is this a {crop} plant?", { crop }) }),
        el("div", { class: "match-buttons" }, [
          el("button", {
            type: "button",
            class: "btn-primary btn-pill",
            text: T("Yes, it is {crop}", { crop }),
            onclick: () => { planState = { mode: "checks" }; renderPlanState(); },
          }),
          el("button", { type: "button", class: "btn-outline btn-pill", text: T("No"), onclick: () => answerMatch("wrong_crop") }),
        ]),
        condition.has_products
          ? el("p", { class: "treatment-meta", text: T("TNAU lists a treatment for this; it is shown once your answers match.") })
          : null,
      ])
    );
    planActions.classList.add("hidden");
    planPanel.classList.remove("hidden");
  }

  function symptomChecks() {
    const condition = lastAnalysis.ai_condition;
    const answers = condition.checks.map(() => null);
    const showButton = el("button", { type: "button", class: "btn-primary btn-pill", text: T("Show what to do") });

    const rows = condition.checks.map((question, index) =>
      el("div", { class: "check-row" }, [
        el("p", { class: "check-question", text: question }),
        el("div", { class: "check-buttons", role: "radiogroup" }, ["yes", "no"].map((value) =>
          el("label", { class: `check-option check-${value}` }, [
            el("input", { type: "radio", name: `check-${index}`, value, onchange: () => { answers[index] = value; } }),
            el("span", { text: T(value === "yes" ? "Yes" : "No") }),
          ])
        )),
      ])
    );

    showButton.addEventListener("click", () => {
      const yes = answers.filter((a) => a === "yes").length;
      const no = answers.filter((a) => a === "no").length;
      // All yes confirms; any yes (or unanswered) is "not sure"; all no rules it out.
      answerMatch(yes === answers.length ? "yes" : no === answers.length ? "no" : "unsure");
    });

    planStep.replaceChildren(
      el("div", { class: "match-card" }, [
        el("p", { class: "result-eyebrow", text: T("Possible match found by AI") }),
        el("h4", { class: "step-title", text: condition.display_name }),
        el("p", { class: "checkin-question", text: T("Look closely at the plant and answer:") }),
        ...rows,
        el("p", { class: "treatment-meta", text: T("Skip a question if you cannot tell.") }),
        el("div", { class: "match-buttons" }, [showButton]),
        el("p", { class: "treatment-meta" }, [
          `${T("Compare with")} `,
          el("a", { href: condition.source_url, target: "_blank", rel: "noopener", class: "inline-link", text: T("TNAU's page with photos") }),
          ".",
        ]),
        askExpert(),
      ])
    );
  }

  const EXTENTS = [
    ["few_leaves", "A few leaves"],
    ["about_half", "About half the plant"],
    ["most", "Most of the plant"],
  ];
  const OUTLOOK_ICON = { good: "🟢", fair: "🟡", poor: "🔴" };
  const TREND_TEXT = { improving: "↓ Improving", same: "→ About the same", worsening: "↑ Getting worse" };
  const GRADE_TEXT = { mild: "Mild", moderate: "Moderate", severe: "Severe" };

  function outlookBox(outlook) {
    return outlook
      ? el("p", { class: `outlook outlook-${outlook.level}`, text: `${OUTLOOK_ICON[outlook.level] || ""} ${outlook.text}` })
      : null;
  }

  function severityText(severity) {
    if (!severity || !severity.grade) return "";
    const grade = T(GRADE_TEXT[severity.grade] || severity.grade);
    return severity.affected_percent != null
      ? T("{grade} (about {percent}% of the leaf)", { grade, percent: Math.round(severity.affected_percent) })
      : grade;
  }

  // "How much of the plant has this?" + photo measurement + outlook.
  function severityCard(severity, photo, selected) {
    return el("div", { class: "severity-card" }, [
      el("p", { class: "checkin-question", text: T("How much of the plant has this?") }),
      el("div", { class: "extent-options" }, EXTENTS.map(([value, label]) =>
        el("button", {
          type: "button",
          class: `extent-option${selected === value ? " selected" : ""}`,
          "aria-pressed": selected === value ? "true" : "false",
          text: T(label),
          onclick: () => { planState.extent = value; loadPlan(); },
        })
      )),
      photo && photo.reliable
        ? el("p", { class: "treatment-meta", text: T("Photo check: about {percent}% of the visible leaf area looks affected (approximate).", { percent: Math.round(photo.affected_percent) }) })
        : null,
      severity && severity.grade ? el("p", { class: "severity-grade", text: T("Severity: {text}", { text: severityText(severity) }) }) : null,
      outlookBox(severity && severity.outlook),
    ]);
  }

  function planRequest() {
    return {
      ...planState.fields,
      photo_severity: lastAnalysis.photo_severity || null,
      extent: planState.extent,
    };
  }

  let planToken = 0;

  // English first (fast), then the chosen language when it differs.
  async function loadPlan() {
    const analysis = lastAnalysis;
    const state = planState;
    const token = ++planToken;
    const lang = language();

    for (const pass of lang === "en" ? ["en"] : ["en", lang]) {
      let data;
      try {
        data = await postJson("/treatment-plan", { ...planRequest(), language: pass });
      } catch (_) {
        return; // the rest of the result is still useful without a plan
      }
      if (token !== planToken || analysis !== lastAnalysis || state !== planState) return;
      if (!data.trackable || !data.step) return;
      renderPlan(data, pass !== lang);
    }
  }

  function renderPlan(data, translating) {
    const intro = planState.answer
      ? el("p", { class: "match-answer" }, [
          `${lastAnalysis.ai_condition.display_name}: ${T(ANSWERS[planState.answer])} `,
          el("button", { type: "button", class: "link-button", text: T("Change answers"), onclick: restartMatch }),
        ])
      : null;

    lastPlan = { fields: planState.fields, step: data.step, extent: planState.extent };
    planStep.replaceChildren(...[
      intro,
      translating ? el("p", { class: "treatment-meta translating", text: T("Translating…") }) : null,
      stepBody(data.step),
      severityCard(data.severity, lastAnalysis.photo_severity, planState.extent),
    ].filter(Boolean));
    planActions.classList.remove("hidden");
    planSpeakBtn.classList.remove("hidden");
    if (!speaking || speaking.button !== planSpeakBtn) setSpeakState(planSpeakBtn, "idle");

    const meta = data.meta || {};
    trackNote.textContent =
      T("Tracking saves this plan to your history and shows when to check whether it worked.") +
      (data.step.products.length && meta.checked_on
        ? ` ${T("Product options copied from TNAU (checked {date}); not yet reviewed by an agronomist.", { date: meta.checked_on })}`
        : "");

    resetTrackButton();
    planPanel.classList.remove("hidden");
  }

  async function track() {
    if (!lastPlan || !lastFile) return;

    if (!user) {
      pendingTrack = true;
      openAuth("login");
      return;
    }

    trackBtn.disabled = true;
    trackBtn.textContent = T("Saving…");

    const form = new FormData();
    form.append("file", lastFile);
    Object.entries(lastPlan.fields).forEach(([key, value]) =>
      form.append(key, Array.isArray(value) ? JSON.stringify(value) : value)
    );
    if (lastPlan.extent) form.append("extent", lastPlan.extent);
    form.append("language", language());

    try {
      await api("/cases", { method: "POST", body: form });
      trackBtn.textContent = T("✓ Tracking started");
      trackNote.replaceChildren(`${T("Saved.")} `, el("a", { href: "#history", class: "inline-link", text: T("Open your treatment history") }));
      await loadCases();
    } catch (err) {
      if (err.status === 401) {
        user = null;
        renderAccount();
      }
      resetTrackButton();
      trackNote.textContent = err.message;
    }
  }

  /* ---------- history ---------- */

  const STATUS = {
    active: ["In treatment", "chip-uncertain"],
    recovered: ["Recovered", "chip-healthy"],
    expert: ["See an expert", "chip-diseased"],
  };

  const LEVEL = { high: "diagnosis high confidence", uncertain: "diagnosis uncertain" };
  const AI_MATCH_LEVEL = { high: "AI match, confirmed by you", uncertain: "AI match, not confirmed" };

  function diagnosisLabel(c) {
    if (!c.confidence_level) return "";
    const labels = c.diagnosis_source === "ai_match" ? AI_MATCH_LEVEL : LEVEL;
    return ` · ${T(labels[c.confidence_level] || c.confidence_level)}`;
  }

  function latestFeedback(c) {
    const steps = c.steps;
    if (c.status === "recovered") return { text: steps[steps.length - 1].feedback, positive: true };
    const previous = steps.length > 1 ? steps[steps.length - 2] : null;
    if (!previous || !previous.feedback) return null;

    // The feedback ends with the new step's reason, which the step shows itself.
    const reason = c.current_step.reason || "";
    let text = previous.feedback;
    if (reason && text.endsWith(reason)) text = text.slice(0, -reason.length).trim();
    return text ? { text, positive: false } : null;
  }

  function checkInForm(c, step) {
    const name = `outcome-${c.id}`;
    const error = el("p", { class: "explain-error hidden" });
    const submit = el("button", { type: "submit", class: "btn-primary btn-pill", text: T("Save check") });
    const notes = el("textarea", { name: "notes", rows: "2", maxlength: "1000", placeholder: T("e.g. new spots on the top leaves, or the spots are drying up") });
    const photo = el("input", { type: "file", name: "file", accept: "image/jpeg,image/png", capture: "environment" });
    const info = dueInfo(step);
    const extentName = `extent-${c.id}`;

    const form = el("form", { class: "checkin-form", novalidate: true }, [
      el("p", { class: "checkin-question", text: step.check_question || T("Did it work?") }),
      info && !info.due
        ? el("p", { class: "treatment-meta", text: T("You can check early if the result is already clear.") })
        : null,
      el("div", { class: "outcome-options", role: "radiogroup" }, [
        el("label", { class: "outcome-option outcome-worked" }, [
          el("input", { type: "radio", name, value: "worked" }),
          el("span", { text: T("It worked") }),
        ]),
        el("label", { class: "outcome-option outcome-not-worked" }, [
          el("input", { type: "radio", name, value: "not_worked" }),
          el("span", { text: T("It did not work") }),
        ]),
      ]),
      el("label", { class: "field" }, [
        el("span", { text: T("What do you see? (optional: the app replies to your note)") }),
        notes,
      ]),
      el("div", { class: "field" }, [
        el("span", { text: T("How much of the plant has it now? (optional)") }),
        el("div", { class: "extent-options", role: "radiogroup" }, EXTENTS.map(([value, label]) =>
          el("label", { class: "extent-option" }, [el("input", { type: "radio", name: extentName, value }), el("span", { text: T(label) })])
        )),
      ]),
      el("label", { class: "field" }, [
        el("span", { text: T("New photo (recommended: the app measures whether the disease is shrinking and re-checks the diagnosis)") }),
        photo,
        el("span", { class: "photo-tip", text: T("Tip: photograph the same plant from about the same distance as last time.") }),
      ]),
      error,
      submit,
    ]);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      error.classList.add("hidden");

      const choice = form.querySelector(`input[name="${name}"]:checked`);
      if (!choice) {
        error.textContent = T("Choose whether it worked.");
        error.classList.remove("hidden");
        return;
      }

      const file = photo.files[0];
      if (file && file.size > MAX_UPLOAD_BYTES) {
        error.textContent = T("The photo is too large. Maximum size is 10 MB.");
        error.classList.remove("hidden");
        return;
      }

      const body = new FormData();
      body.append("outcome", choice.value);
      body.append("notes", notes.value);
      body.append("language", language());
      const extent = form.querySelector(`input[name="${extentName}"]:checked`);
      if (extent) body.append("extent", extent.value);
      if (file) body.append("file", file);

      submit.disabled = true;
      submit.textContent = T(file ? "Re-checking the new photo…" : "Saving…");

      try {
        const data = await api(`/cases/${c.id}/checkin`, { method: "POST", body });
        cases = cases.map((item) => (item.id === c.id ? data.case : item));
        renderHistory();
        const card = $(`case-${c.id}`);
        if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (err) {
        error.textContent = err.message;
        error.classList.remove("hidden");
        submit.disabled = false;
        submit.textContent = T("Save check");
      }
    });

    return form;
  }

  function timeline(c) {
    const done = c.steps.filter((s) => s.outcome !== "pending");
    if (!done.length) return null;

    return el("details", { class: "case-timeline" }, [
      el("summary", { text: T("Steps so far ({n})", { n: done.length }) }),
      el("ol", {}, done.map((s) => {
        const f = s.followup;
        return el("li", { class: "timeline-item" }, [
          el("div", { class: "timeline-head" }, [
            el("strong", { text: T("Step {n}: {title}", { n: s.number, title: s.title }) }),
            el("span", {
              class: `level-pill ${s.outcome === "worked" ? "level-high" : "level-low"}`,
              text: T(s.outcome === "worked" ? "Worked" : "Did not work"),
            }),
          ]),
          el("p", { class: "treatment-meta", text: T("Started {date} · checked {checked}", { date: formatDate(s.created_at), checked: formatDate(s.outcome_at) }) }),
          s.products.length
            ? el("p", { class: "timeline-text", text: T("Products suggested: {products}", { products: s.products.map((p) => p.name).join(" / ") }) })
            : null,
          s.notes ? el("p", { class: "timeline-text", text: T("Your notes: {notes}", { notes: s.notes }) }) : null,
          s.severity && s.severity.grade
            ? el("p", { class: "timeline-text" }, [
                T("Severity: {text}", { text: severityText(s.severity) }),
                s.severity.trend ? ` — ${T(TREND_TEXT[s.severity.trend.direction])} (${s.severity.trend.text})` : "",
              ])
            : null,
          f && f.display_name
            ? el("p", { class: "timeline-text", text: T("New photo: {result}", { result: f.display_name }) })
            : null,
          s.followup_photo_url
            ? el("img", { class: "timeline-photo", src: s.followup_photo_url, alt: `Check-in photo, step ${s.number}`, loading: "lazy" })
            : null,
          s.feedback ? el("p", { class: "timeline-feedback", text: s.feedback }) : null,
        ]);
      })),
    ]);
  }

  async function deleteCase(c) {
    if (!window.confirm(T("Delete the treatment record for {name}? This cannot be undone.", { name: c.display_name }))) return;
    try {
      await api(`/cases/${c.id}`, { method: "DELETE" });
      cases = cases.filter((item) => item.id !== c.id);
      renderHistory();
    } catch (err) {
      window.alert(err.message);
    }
  }

  const caseSpeakButtons = new Map(); // case id -> button (kept across re-renders while playing)

  function caseSpeakButton(c) {
    let button = caseSpeakButtons.get(c.id);
    if (!button || !(speaking && speaking.button === button)) {
      button = speakButton(() => readAloud(button, `/cases/${c.id}/speech`, {}));
      caseSpeakButtons.set(c.id, button);
    }
    return button;
  }

  function caseCard(c) {
    const [statusText, statusClass] = STATUS[c.status] || STATUS.active;
    const step = c.current_step;
    const info = c.awaiting_check_in ? dueInfo(step) : null;
    const feedback = latestFeedback(c);
    const severity = c.latest_severity;
    const outlook = c.status === "recovered" ? null : (severity || c.severity || {}).outlook;
    const trend = severity && severity.trend;

    return el("article", { class: `case-card glass${isDue(c) ? " case-due" : ""}`, id: `case-${c.id}` }, [
      isDue(c) ? el("p", { class: "due-ribbon", text: T("⏰ {due} - answer below", { due: info.text }) }) : null,
      el("div", { class: "case-head" }, [
        el("img", { class: "case-photo", src: c.photo_url, alt: c.plant_name, loading: "lazy" }),
        el("div", { class: "case-info" }, [
          el("div", { class: "case-title-row" }, [
            el("h3", { text: c.display_name }),
            caseSpeakButton(c),
          ]),
          el("p", {
            class: "treatment-meta",
            text: T("{plant} · started {date} · step {n}", { plant: c.plant_name, date: formatDate(c.created_at), n: step.number }) + diagnosisLabel(c),
          }),
          el("div", { class: "result-chips" }, [
            el("span", { class: `chip ${statusClass}`, text: T(statusText) }),
            info ? el("span", { class: `chip ${info.due ? "chip-diseased" : ""}`, text: info.text }) : null,
            severity && severity.grade
              ? el("span", { class: `chip severity-${severity.grade}`, text: T("Severity: {text}", { text: severityText(severity) }) })
              : null,
            trend ? el("span", { class: `chip trend-${trend.direction}`, text: T(TREND_TEXT[trend.direction]) }) : null,
          ]),
        ]),
      ]),
      outlookBox(outlook),
      feedback && feedback.text
        ? el("p", { class: `case-feedback${feedback.positive ? " positive" : ""}`, text: feedback.text })
        : null,
      c.awaiting_check_in
        ? el("div", { class: "current-step" }, [
            el("p", { class: "result-eyebrow", text: T("Current step {n}", { n: step.number }) }),
            stepBody(step, true),
            checkInForm(c, step),
          ])
        : null,
      timeline(c),
      el("div", { class: "case-footer" }, [
        el("button", { type: "button", class: "link-button danger", text: T("Delete record"), onclick: () => deleteCase(c) }),
      ]),
    ]);
  }

  function renderHistory() {
    historySignedOut.classList.toggle("hidden", Boolean(user));
    historySignedIn.classList.toggle("hidden", !user);

    const dueCases = user ? cases.filter(isDue) : [];
    const due = dueCases.length;
    const checks = countText(due, "check");
    const dueTitle = due ? `${T("{checks} due - tell us whether the treatment worked:", { checks })} ${dueCases.map((c) => c.display_name).join(", ")}` : "";

    dueCounts.forEach((badge) => {
      badge.textContent = String(due);
      badge.title = dueTitle;
      badge.classList.toggle("hidden", due === 0);
    });

    // The badge's link goes straight to the (first) case that needs a check.
    document.querySelectorAll('a[href^="#history"], a[href^="#case-"]').forEach((link) => {
      if (link.closest(".nav-links, .nav-actions")) {
        link.setAttribute("href", due ? `#case-${dueCases[0].id}` : "#history");
        link.title = dueTitle;
      }
    });

    dueBanner.replaceChildren(
      `${T("{checks} due - tell us whether the treatment worked:", { checks })} `,
      ...dueCases.flatMap((c, index) => [
        index ? ", " : "",
        el("a", { href: `#case-${c.id}`, class: "inline-link", text: c.display_name, onclick: (event) => focusCase(event, c.id) }),
      ]),
    );
    dueBanner.classList.toggle("hidden", due === 0);

    if (!user) {
      caseList.replaceChildren();
      return;
    }

    historyEmpty.classList.toggle("hidden", cases.length > 0);
    // Cases waiting for a check first, then the rest (newest first).
    const ordered = [...dueCases, ...cases.filter((c) => !isDue(c))];
    caseList.replaceChildren(...ordered.map(caseCard));
  }

  function focusCase(event, id) {
    const card = $(`case-${id}`);
    if (!card) return;
    if (event) event.preventDefault();
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    card.classList.remove("case-flash");
    void card.offsetWidth; // restart the animation
    card.classList.add("case-flash");
  }

  // Following the nav badge link (#case-<id>) also highlights the card.
  window.addEventListener("hashchange", () => {
    const match = location.hash.match(/^#case-(\d+)$/);
    if (match) focusCase(null, match[1]);
  });

  let casesToken = 0;

  // English first (fast), then the chosen language when it differs.
  async function loadCases() {
    const token = ++casesToken;

    if (!user) {
      cases = [];
      renderHistory();
      return;
    }

    const lang = language();

    for (const pass of lang === "en" ? ["en"] : ["en", lang]) {
      historyTranslating.classList.toggle("hidden", pass === lang);
      try {
        const data = await api(`/cases?language=${encodeURIComponent(pass)}`);
        if (token !== casesToken) return;
        cases = data.cases;
        historyError.classList.add("hidden");
      } catch (err) {
        if (token !== casesToken) return;
        if (err.status === 401) {
          user = null;
          renderAccount();
        }
        historyError.textContent = err.message;
        historyError.classList.remove("hidden");
        renderHistory();
        break;
      }
      renderHistory();
    }

    if (token === casesToken) historyTranslating.classList.add("hidden");
  }

  /* ---------- account ---------- */

  function renderAccount() {
    signInBtn.classList.toggle("hidden", Boolean(user));
    signOutBtn.classList.toggle("hidden", !user);
    accountEmail.classList.toggle("hidden", !user);
    accountEmail.textContent = user ? user.email : "";
    if (lastPlan && !trackBtn.disabled) resetTrackButton();
  }

  function setAuthMode(mode) {
    authMode = mode;
    const login = mode === "login";
    authTitle.textContent = login ? "Sign in" : "Create an account";
    authSubmit.textContent = login ? "Sign in" : "Create account";
    authSwitchText.textContent = login ? "No account yet?" : "Already have an account?";
    authSwitchBtn.textContent = login ? "Create one" : "Sign in";
    authPassword.setAttribute("autocomplete", login ? "current-password" : "new-password");
    authError.classList.add("hidden");
  }

  function openAuth(mode) {
    setAuthMode(mode);
    authDialog.showModal();
    authEmail.focus();
  }

  async function submitAuth(event) {
    event.preventDefault();
    authError.classList.add("hidden");
    authSubmit.disabled = true;

    try {
      const data = await postJson(authMode === "login" ? "/auth/login" : "/auth/register", {
        email: authEmail.value,
        password: authPassword.value,
      });
      user = data.user;
      authPassword.value = "";
      authDialog.close();
      renderAccount();
      await loadCases();
      if (pendingTrack) {
        pendingTrack = false;
        track();
      }
    } catch (err) {
      authError.textContent = err.message;
      authError.classList.remove("hidden");
    } finally {
      authSubmit.disabled = false;
    }
  }

  async function signOut() {
    try {
      await api("/auth/logout", { method: "POST" });
    } catch (_) {
      /* the session is cleared server-side on next sign-in anyway */
    }
    user = null;
    cases = [];
    renderAccount();
    renderHistory();
  }

  /* ---------- language ---------- */

  async function onLanguageChange() {
    stopSpeaking();
    // The first switch to a language can take a few seconds (later ones are cached).
    historyTranslating.classList.toggle("hidden", language() === "en");
    await loadUiText();
    applyStaticText();
    renderHistory(); // labels right away, then translated case texts
    loadCases();
    renderPlanState();
  }

  /* ---------- wiring ---------- */

  document.addEventListener("verdant:analysis", (event) => showPlan(event.detail.analysis, event.detail.file));
  document.addEventListener("verdant:reset", () => {
    stopSpeaking();
    lastAnalysis = null;
    lastFile = null;
    hidePlan();
  });
  // app.js fills the language list after its first request.
  document.addEventListener("verdant:languages", onLanguageChange);

  const languageSelect = $("languageSelect");
  if (languageSelect) languageSelect.addEventListener("change", onLanguageChange);

  planSpeakBtn.addEventListener("click", () => {
    if (lastPlan) readAloud(planSpeakBtn, "/treatment-plan/speech", planRequest());
  });
  setSpeakState(planSpeakBtn, "idle");

  trackBtn.addEventListener("click", track);
  signInBtn.addEventListener("click", () => openAuth("login"));
  historySignInBtn.addEventListener("click", () => openAuth("login"));
  signOutBtn.addEventListener("click", signOut);
  authForm.addEventListener("submit", submitAuth);
  authSwitchBtn.addEventListener("click", () => setAuthMode(authMode === "login" ? "register" : "login"));
  authCloseBtn.addEventListener("click", () => authDialog.close());
  authDialog.addEventListener("close", () => {
    if (!user) pendingTrack = false;
  });

  (async () => {
    try {
      user = (await api("/auth/me")).user;
    } catch (_) {
      user = null;
    }
    renderAccount();
    loadCases();
  })();
})();
