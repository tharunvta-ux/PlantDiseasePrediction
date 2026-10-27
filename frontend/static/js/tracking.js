/*
 * Treatment plan, accounts and treatment history.
 *
 * app.js announces each analysis ("verdant:analysis") and resets
 * ("verdant:reset"); this file shows the first step of the plan, saves it
 * as a tracked case, and lets the grower report "worked / did not work"
 * at each check.
 */
(() => {
  const $ = (id) => document.getElementById(id);

  const planPanel = $("planPanel");
  const planStep = $("planStep");
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

  // Latest analysis from app.js and its plan preview.
  let lastAnalysis = null;
  let lastFile = null;
  let lastPlan = null;

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

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  function formatDate(iso) {
    return iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";
  }

  function startOfDay(date) {
    return new Date(date.getFullYear(), date.getMonth(), date.getDate());
  }

  function dueInfo(step) {
    if (!step || !step.check_due_at) return null;
    const due = new Date(step.check_due_at);
    const days = Math.round((startOfDay(due) - startOfDay(new Date())) / 86400000);
    const date = due.toLocaleDateString(undefined, { day: "numeric", month: "short" });

    if (days > 1) return { due: false, text: `Check again in ${days} days (${date})` };
    if (days === 1) return { due: false, text: `Check again tomorrow (${date})` };
    if (days === 0) return { due: true, text: "Check due today" };
    return { due: true, text: `Check overdue by ${plural(-days, "day")}` };
  }

  const isDue = (c) => c.awaiting_check_in && Boolean((dueInfo(c.current_step) || {}).due);

  /* ---------- step rendering (plan panel + history) ---------- */

  function expertLinks() {
    return el("div", { class: "expert-links" }, [
      el("a", { class: "btn-primary btn-pill", href: `tel:${KISAN_CALL_CENTRE.replace(/-/g, "")}`, text: `📞 Kisan Call Centre ${KISAN_CALL_CENTRE}` }),
      el("a", { class: "btn-outline btn-pill", href: mapsSearch("Krishi Vigyan Kendra near me"), target: "_blank", rel: "noopener", text: "📍 Krishi Vigyan Kendra nearby" }),
      el("a", { class: "btn-outline btn-pill", href: mapsSearch("agriculture department office near me"), target: "_blank", rel: "noopener", text: "📍 Agriculture office nearby" }),
    ]);
  }

  function productCard(step) {
    if (!step.products || !step.products.length) return null;

    return el("div", { class: "product-card" }, [
      el("p", {
        class: "product-title",
        text: step.products.length > 1 ? "Use ONE of these TNAU-recommended options:" : "TNAU-recommended option:",
      }),
      el("table", { class: "product-table" }, [
        el("tbody", {}, step.products.map((p) =>
          el("tr", {}, [
            el("td", {}, [el("strong", { text: p.name }), el("span", { class: "product-kind", text: `${p.kind} · ${p.group}` })]),
            el("td", { class: "product-dose", text: p.dose }),
          ])
        )),
      ]),
      step.source_url
        ? el("p", { class: "treatment-meta" }, [
            "Source: ",
            el("a", { href: step.source_url, target: "_blank", rel: "noopener", text: "TNAU Agritech Portal" }),
          ])
        : null,
    ]);
  }

  // inHistory: the check-in form below already asks the check question.
  function stepBody(step, inHistory = false) {
    const waitText = step.wait_days
      ? `Then wait ${plural(step.wait_days, "day")} and check` +
        (inHistory ? " again." : `: ${step.check_question}`)
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
    planPanel.classList.add("hidden");
  }

  function resetTrackButton() {
    trackBtn.disabled = false;
    trackBtn.textContent = user ? "Track this treatment" : "Sign in to track this treatment";
  }

  async function showPlan(analysis, file) {
    lastAnalysis = analysis;
    lastFile = file;
    hidePlan();

    const fields = diagnosisFields(analysis);
    if (!fields) return;

    let data;
    try {
      data = await postJson("/treatment-plan", fields);
    } catch (_) {
      return; // the rest of the result is still useful without a plan
    }

    if (analysis !== lastAnalysis || !data.trackable || !data.step) return;

    lastPlan = { fields, step: data.step };
    planStep.replaceChildren(stepBody(data.step));

    const meta = data.meta || {};
    trackNote.textContent =
      "Tracking saves this plan to your history and shows when to check whether it worked." +
      (data.step.products.length && meta.checked_on
        ? ` Product options copied from TNAU (checked ${meta.checked_on}); not yet reviewed by an agronomist.`
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
    trackBtn.textContent = "Saving…";

    const form = new FormData();
    form.append("file", lastFile);
    Object.entries(lastPlan.fields).forEach(([key, value]) =>
      form.append(key, Array.isArray(value) ? JSON.stringify(value) : value)
    );

    try {
      await api("/cases", { method: "POST", body: form });
      trackBtn.textContent = "✓ Tracking started";
      trackNote.replaceChildren("Saved. ", el("a", { href: "#history", class: "inline-link", text: "Open your treatment history" }), ".");
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

  const LEVEL = { high: "high confidence", uncertain: "uncertain" };

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
    const submit = el("button", { type: "submit", class: "btn-primary btn-pill", text: "Save check" });
    const notes = el("textarea", { name: "notes", rows: "2", maxlength: "1000", placeholder: "e.g. new spots on the top leaves, or the spots are drying up" });
    const photo = el("input", { type: "file", name: "file", accept: "image/jpeg,image/png", capture: "environment" });
    const info = dueInfo(step);

    const form = el("form", { class: "checkin-form", novalidate: true }, [
      el("p", { class: "checkin-question", text: step.check_question || "Did it work?" }),
      info && !info.due
        ? el("p", { class: "treatment-meta", text: "You can check early if the result is already clear." })
        : null,
      el("div", { class: "outcome-options", role: "radiogroup" }, [
        el("label", { class: "outcome-option outcome-worked" }, [
          el("input", { type: "radio", name, value: "worked" }),
          el("span", { text: "It worked" }),
        ]),
        el("label", { class: "outcome-option outcome-not-worked" }, [
          el("input", { type: "radio", name, value: "not_worked" }),
          el("span", { text: "It did not work" }),
        ]),
      ]),
      el("label", { class: "field" }, [
        el("span", { text: "What do you see? (optional: the app replies to your note)" }),
        notes,
      ]),
      el("label", { class: "field" }, [
        el("span", { text: "New photo (optional, recommended if it did not work: the app re-checks the diagnosis)" }),
        photo,
      ]),
      error,
      submit,
    ]);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      error.classList.add("hidden");

      const choice = form.querySelector(`input[name="${name}"]:checked`);
      if (!choice) {
        error.textContent = "Choose whether it worked.";
        error.classList.remove("hidden");
        return;
      }

      const file = photo.files[0];
      if (file && file.size > MAX_UPLOAD_BYTES) {
        error.textContent = "The photo is too large. Maximum size is 10 MB.";
        error.classList.remove("hidden");
        return;
      }

      const body = new FormData();
      body.append("outcome", choice.value);
      body.append("notes", notes.value);
      body.append("language", language());
      if (file) body.append("file", file);

      submit.disabled = true;
      submit.textContent = file ? "Re-checking the new photo…" : "Saving…";

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
        submit.textContent = "Save check";
      }
    });

    return form;
  }

  function timeline(c) {
    const done = c.steps.filter((s) => s.outcome !== "pending");
    if (!done.length) return null;

    return el("details", { class: "case-timeline" }, [
      el("summary", { text: `Steps so far (${done.length})` }),
      el("ol", {}, done.map((s) => {
        const f = s.followup;
        return el("li", { class: "timeline-item" }, [
          el("div", { class: "timeline-head" }, [
            el("strong", { text: `Step ${s.number}: ${s.title}` }),
            el("span", {
              class: `level-pill ${s.outcome === "worked" ? "level-high" : "level-low"}`,
              text: s.outcome === "worked" ? "Worked" : "Did not work",
            }),
          ]),
          el("p", { class: "treatment-meta", text: `Started ${formatDate(s.created_at)} · checked ${formatDate(s.outcome_at)}` }),
          s.products.length
            ? el("p", { class: "timeline-text", text: `Products suggested: ${s.products.map((p) => p.name).join(" or ")}` })
            : null,
          s.notes ? el("p", { class: "timeline-text", text: `Your notes: ${s.notes}` }) : null,
          f && f.display_name
            ? el("p", { class: "timeline-text", text: `New photo: ${f.display_name} (${LEVEL[f.confidence_level] || f.confidence_level || "unclear"})` })
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
    if (!window.confirm(`Delete the treatment record for ${c.display_name}? This cannot be undone.`)) return;
    try {
      await api(`/cases/${c.id}`, { method: "DELETE" });
      cases = cases.filter((item) => item.id !== c.id);
      renderHistory();
    } catch (err) {
      window.alert(err.message);
    }
  }

  function caseCard(c) {
    const [statusText, statusClass] = STATUS[c.status] || STATUS.active;
    const step = c.current_step;
    const info = c.awaiting_check_in ? dueInfo(step) : null;
    const feedback = latestFeedback(c);

    return el("article", { class: `case-card glass${isDue(c) ? " case-due" : ""}`, id: `case-${c.id}` }, [
      el("div", { class: "case-head" }, [
        el("img", { class: "case-photo", src: c.photo_url, alt: `Photo of ${c.plant_name}`, loading: "lazy" }),
        el("div", { class: "case-info" }, [
          el("h3", { text: c.display_name }),
          el("p", {
            class: "treatment-meta",
            text: `${c.plant_name} · started ${formatDate(c.created_at)} · step ${step.number}` +
              (c.confidence_level ? ` · diagnosis ${LEVEL[c.confidence_level] || c.confidence_level}` : ""),
          }),
          el("div", { class: "result-chips" }, [
            el("span", { class: `chip ${statusClass}`, text: statusText }),
            info ? el("span", { class: `chip ${info.due ? "chip-diseased" : ""}`, text: info.text }) : null,
          ]),
        ]),
      ]),
      feedback && feedback.text
        ? el("p", { class: `case-feedback${feedback.positive ? " positive" : ""}`, text: feedback.text })
        : null,
      c.awaiting_check_in
        ? el("div", { class: "current-step" }, [
            el("p", { class: "result-eyebrow", text: `Current step ${step.number}` }),
            stepBody(step, true),
            checkInForm(c, step),
          ])
        : null,
      timeline(c),
      el("div", { class: "case-footer" }, [
        el("button", { type: "button", class: "link-button danger", text: "Delete record", onclick: () => deleteCase(c) }),
      ]),
    ]);
  }

  function renderHistory() {
    historySignedOut.classList.toggle("hidden", Boolean(user));
    historySignedIn.classList.toggle("hidden", !user);

    const due = user ? cases.filter(isDue).length : 0;
    dueCounts.forEach((badge) => {
      badge.textContent = String(due);
      badge.classList.toggle("hidden", due === 0);
    });
    dueBanner.textContent = `${plural(due, "check")} due. Tell us whether the treatment worked so the next step can be suggested.`;
    dueBanner.classList.toggle("hidden", due === 0);

    if (!user) {
      caseList.replaceChildren();
      return;
    }

    historyEmpty.classList.toggle("hidden", cases.length > 0);
    caseList.replaceChildren(...cases.map(caseCard));
  }

  async function loadCases() {
    if (!user) {
      cases = [];
      renderHistory();
      return;
    }

    try {
      cases = (await api("/cases")).cases;
      historyError.classList.add("hidden");
    } catch (err) {
      if (err.status === 401) {
        user = null;
        renderAccount();
      }
      historyError.textContent = err.message;
      historyError.classList.remove("hidden");
    }

    renderHistory();
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

  /* ---------- wiring ---------- */

  document.addEventListener("verdant:analysis", (event) => showPlan(event.detail.analysis, event.detail.file));
  document.addEventListener("verdant:reset", () => {
    lastAnalysis = null;
    lastFile = null;
    hidePlan();
  });

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
