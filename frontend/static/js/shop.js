/*
 * Treatment-kit shop: buy the product a treatment step recommends, sized
 * for the grower's field, and pay with PayPal (sandbox).
 *
 * tracking.js calls VerdantShop.kitPanel(case, step, helpers) for every
 * open step that lists products, and VerdantShop.ordersBlock(case,
 * helpers) for each case. The server computes every quantity and price;
 * this file only shows them and runs the PayPal buttons.
 *
 *   1. field size: typed in the form, told to the AI assistant in any
 *      language, or remembered from the grower's last order
 *   2. POST /shop/cases/<id>/kit          -> kit (packs, total)
 *   3. PayPal button -> POST /shop/cases/<id>/orders -> PayPal order id
 *   4. grower approves in the PayPal window
 *   5. POST /shop/orders/<paypal id>/capture -> paid
 */
(() => {
  const SPRAY_CHOICES = [1, 2, 3, 4];
  const AREA_UNITS = [
    ["acre", "acres"],
    ["cent", "cents"],
    ["hectare", "hectares"],
    ["sqm", "square metres"],
  ];

  let configPromise = null;
  let sdkPromise = null;
  let ordersPromise = null;
  let orders = [];

  // case id -> { step, field, product, quote, reply, error, loading, remembered, message }
  const states = new Map();

  function config(h) {
    if (!configPromise) {
      configPromise = h.api("/shop/config").catch(() => ({ enabled: false }));
    }
    return configPromise;
  }

  function loadSdk(cfg) {
    if (window.paypal) return Promise.resolve(window.paypal);
    if (!sdkPromise) {
      sdkPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        const params = new URLSearchParams({
          "client-id": cfg.client_id,
          currency: cfg.currency,
          intent: "capture",
          components: "buttons",
        });
        script.src = `https://www.paypal.com/sdk/js?${params}`;
        script.onload = () => resolve(window.paypal);
        script.onerror = () => {
          sdkPromise = null;
          reject(new Error("PayPal could not be loaded."));
        };
        document.head.appendChild(script);
      });
    }
    return sdkPromise;
  }

  function loadOrders(h, force = false) {
    if (!ordersPromise || force) {
      ordersPromise = h
        .api("/shop/orders")
        .then((data) => {
          orders = data.orders || [];
          return orders;
        })
        .catch(() => orders);
    }
    return ordersPromise;
  }

  function money(value, currency) {
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency }).format(Number(value));
    } catch (_) {
      return `${value} ${currency}`;
    }
  }

  function stateFor(c, step) {
    let state = states.get(c.id);
    if (!state || state.step !== step.number) {
      // A new step (e.g. after "did not work") keeps the field size.
      state = {
        step: step.number,
        field: state ? state.field : null,
        product: null,
        quote: null,
        reply: null,
        error: null,
        loading: false,
        remembered: false,
        message: "",
        started: false,
      };
      states.set(c.id, state);
    }
    return state;
  }

  /* ---------- kit requests ---------- */

  async function fetchKit(c, state, h, rerender, body) {
    state.loading = true;
    state.error = null;
    rerender();
    try {
      const data = await h.postJson(`/shop/cases/${c.id}/kit`, body);
      state.quote = data.quote;
      state.remembered = data.remembered;
      if (data.field) state.field = data.field;
      if (data.quote) state.product = data.quote.product;
    } catch (err) {
      state.error = err.message;
    } finally {
      state.loading = false;
      rerender();
    }
  }

  async function askAssistant(c, state, h, rerender) {
    if (!state.message.trim()) {
      state.error = h.T("Tell the assistant about your field first.");
      rerender();
      return;
    }
    state.loading = true;
    state.error = null;
    rerender();
    try {
      const data = await h.postJson(`/shop/cases/${c.id}/assistant`, {
        message: state.message,
        language: h.language(),
        ...(state.field || {}),
      });
      state.reply = data.reply;
      state.remembered = false;
      if (data.field) state.field = data.field;
      if (data.quote) {
        state.quote = data.quote;
        state.product = data.quote.product;
      } else if (!data.field) {
        state.quote = null;
      }
    } catch (err) {
      state.error = err.message;
    } finally {
      state.loading = false;
      rerender();
    }
  }

  /* ---------- rendering ---------- */

  function kitForm(c, state, h, rerender) {
    const field = state.field || { area: "", area_unit: "acre", sprays: 2 };
    const area = h.el("input", { type: "number", min: "0", step: "any", inputmode: "decimal", "aria-label": h.T("Area to treat") });
    area.value = field.area;
    const unit = h.el("select", { "aria-label": h.T("Unit") }, AREA_UNITS.map(([value, label]) =>
      h.el("option", { value, text: h.T(label), selected: value === field.area_unit })
    ));
    const sprays = h.el("select", { "aria-label": h.T("Number of sprays") }, SPRAY_CHOICES.map((n) =>
      h.el("option", { value: String(n), text: h.T(n === 1 ? "{n} spray" : "{n} sprays", { n }), selected: n === field.sprays })
    ));
    const options = (c.current_step.products || []).map((p) => p.name);
    const product = options.length > 1
      ? h.el("select", { "aria-label": h.T("Product") }, [
          h.el("option", { value: "", text: h.T("Cheapest listed option") }),
          ...options.map((name) => h.el("option", { value: name, text: name, selected: name === state.product })),
        ])
      : null;

    const submit = (event) => {
      event.preventDefault();
      state.field = { area: area.value, area_unit: unit.value, sprays: Number(sprays.value) };
      state.product = product ? product.value || null : null;
      state.reply = null;
      fetchKit(c, state, h, rerender, { ...state.field, product: state.product });
    };

    return h.el("form", { class: "kit-form", novalidate: true, onsubmit: submit }, [
      h.el("label", { class: "field kit-area" }, [h.el("span", { text: h.T("Area to treat") }), h.el("div", { class: "kit-area-row" }, [area, unit])]),
      h.el("label", { class: "field" }, [h.el("span", { text: h.T("Sprays") }), sprays]),
      product ? h.el("label", { class: "field kit-product" }, [h.el("span", { text: h.T("Product") }), product]) : null,
      h.el("button", { type: "submit", class: "btn-outline btn-pill kit-update", disabled: state.loading, text: h.T(state.quote ? "Update kit" : "Calculate kit") }),
    ]);
  }

  function assistantBox(c, state, h, rerender, cfg) {
    if (!cfg.assistant) return null;
    const input = h.el("textarea", {
      rows: "2",
      maxlength: "500",
      placeholder: h.T("e.g. I have half an acre of tomatoes and will spray twice. I already have mancozeb."),
      oninput: (event) => { state.message = event.target.value; },
    });
    input.value = state.message;
    const ask = (event) => {
      event.preventDefault();
      state.message = input.value;
      askAssistant(c, state, h, rerender);
    };
    return h.el("form", { class: "kit-assistant", onsubmit: ask }, [
      h.el("label", { class: "field" }, [h.el("span", { text: h.T("✨ Tell the AI assistant about your field, in your own words and language") }), input]),
      h.el("button", { type: "submit", class: "btn-primary btn-pill", disabled: state.loading, text: h.T(state.loading ? "Working…" : "Ask the assistant") }),
    ]);
  }

  function quoteBlock(state, h) {
    const q = state.quote;
    if (!q) return null;
    const cur = q.currency;
    const sprays = q.field.sprays;
    const need = q.spray_litres
      ? h.T("You need about {needed} for {sprays} ({litres} litres of spray each time).", {
          needed: q.needed_text,
          sprays: h.T(sprays === 1 ? "{n} spray" : "{n} sprays", { n: sprays }),
          litres: Math.max(1, Math.round(q.spray_litres)),
        })
      : h.T("You need about {needed} for {sprays}.", {
          needed: q.needed_text,
          sprays: h.T(sprays === 1 ? "{n} spray" : "{n} sprays", { n: sprays }),
        });

    return h.el("div", { class: "kit-quote" }, [
      h.el("p", { class: "kit-quote-title" }, [h.el("strong", { text: q.product }), h.el("span", { class: "product-kind", text: `${q.kind} · ${q.group} · ${q.dose}` })]),
      h.el("p", { class: "kit-need", text: `${need} ${h.T("The kit holds {packed}.", { packed: q.packed_text })}` }),
      h.el("table", { class: "product-table kit-table" }, [
        h.el("tbody", {}, [
          ...q.items.map((item) => h.el("tr", {}, [
            h.el("td", { text: `${item.quantity} × ${item.name}` }),
            h.el("td", { class: "product-dose", text: money(item.line_total, cur) }),
          ])),
          h.el("tr", { class: "kit-total" }, [
            h.el("td", { text: h.T("Total") }),
            h.el("td", { class: "product-dose", text: money(q.total, cur) }),
          ]),
        ]),
      ]),
      q.note ? h.el("p", { class: "treatment-meta", text: q.note }) : null,
      q.alternatives && q.alternatives.length
        ? h.el("p", { class: "treatment-meta", text: `${h.T("Other listed options for this field:")} ${q.alternatives.map((a) => `${a.product} ${money(a.total, cur)}`).join(" · ")}` })
        : null,
    ]);
  }

  async function renderButtons(container, c, state, h, rerender, cfg) {
    let paypal;
    try {
      paypal = await loadSdk(cfg);
    } catch (err) {
      container.replaceChildren(h.el("p", { class: "explain-error", text: h.T("PayPal could not be loaded. Check your connection and try again.") }));
      return;
    }
    if (!container.isConnected) return;

    const field = { ...state.field };
    const product = state.product;
    const reply = state.reply;

    paypal
      .Buttons({
        style: { layout: "vertical", shape: "pill", label: "pay", height: 44 },
        createOrder: async () => {
          state.error = null;
          const data = await h.postJson(`/shop/cases/${c.id}/orders`, { ...field, product, assistant: reply });
          return data.paypal_order_id;
        },
        onApprove: async (data, actions) => {
          try {
            const result = await h.postJson(`/shop/orders/${encodeURIComponent(data.orderID)}/capture`, {});
            orders = [result.order, ...orders.filter((o) => o.id !== result.order.id)];
            state.reorder = false;
            state.reply = null;
            h.refresh(); // redraws the case card: receipt + order list
          } catch (err) {
            if (err.code === "INSTRUMENT_DECLINED") return actions.restart();
            state.error = err.message;
            rerender();
          }
          return undefined;
        },
        onCancel: () => {
          state.error = h.T("Payment cancelled. Nothing was charged.");
          rerender();
        },
        onError: (err) => {
          state.error = (err && err.message && !/^Expected an order id/.test(err.message))
            ? err.message
            : h.T("The payment could not be started. Please try again.");
          rerender();
        },
      })
      .render(container)
      .catch(() => { /* the panel was re-rendered meanwhile */ });
  }

  function statusText(order, h) {
    if (order.status === "paid") return h.T("✓ Paid");
    if (order.status === "refunded") return h.T("Refunded");
    return h.T("Payment failed");
  }

  function orderRow(order, h, rerender) {
    const refund = async () => {
      if (!window.confirm(h.T("Cancel this order and refund {amount} to your PayPal account?", { amount: money(order.total, order.currency) }))) return;
      try {
        const data = await h.postJson(`/shop/orders/${order.id}/refund`, {});
        orders = orders.map((o) => (o.id === order.id ? data.order : o));
        h.refresh(); // the step's kit can be ordered again
        rerender();
      } catch (err) {
        window.alert(err.message);
      }
    };

    return h.el("li", { class: `kit-order kit-order-${order.status}` }, [
      h.el("div", { class: "kit-order-head" }, [
        h.el("strong", { text: `${statusText(order, h)} · ${money(order.total, order.currency)}` }),
        h.el("span", { class: "treatment-meta", text: h.T("Step {n} · {date}", { n: order.step_number, date: h.formatDate(order.paid_at || order.created_at) }) }),
      ]),
      h.el("p", { class: "timeline-text", text: `${order.product} — ${order.field_text || ""}` }),
      order.shipping && order.shipping.address
        ? h.el("p", { class: "treatment-meta", text: `${h.T("Ships to:")} ${[order.shipping.name, order.shipping.address.address_line_1, order.shipping.address.admin_area_2, order.shipping.address.postal_code].filter(Boolean).join(", ")}` })
        : null,
      h.el("p", { class: "treatment-meta", text: `${h.T("PayPal order")} ${order.paypal_order_id} · ${h.T("invoice")} ${order.invoice_id}` }),
      order.status === "paid"
        ? h.el("button", { type: "button", class: "link-button danger", text: h.T("Cancel order and refund"), onclick: refund })
        : null,
    ]);
  }

  /* ---------- public ---------- */

  function kitPanel(c, step, h) {
    const panel = h.el("section", { class: "kit-panel", id: `kit-${c.id}` });
    const state = stateFor(c, step);

    const rerender = () => {
      if (!panel.isConnected && panel.dataset.ready) return;
      config(h).then((cfg) => fill(cfg));
    };

    function fill(cfg) {
      panel.dataset.ready = "1";
      if (!cfg.enabled) {
        panel.replaceChildren(
          h.el("h4", { class: "kit-title", text: h.T("🛒 Get this treatment delivered") }),
          h.el("p", { class: "treatment-meta", text: h.T("Checkout is not configured on this server.") }),
        );
        return;
      }

      const head = h.el("div", { class: "kit-head" }, [
        h.el("h4", { class: "kit-title", text: h.T("🛒 Get this treatment delivered") }),
        cfg.sandbox ? h.el("span", { class: "chip", text: h.T("PayPal sandbox") }) : null,
      ]);
      const disclaimer = h.el("p", { class: "treatment-meta kit-disclaimer", text: `${h.T("Demo store: sandbox payments only, no real money is taken and nothing is shipped.")} ${cfg.prices_note || ""}` });

      // This step is already paid: show the receipt, not a second checkout.
      const paid = orders.find((o) => o.case_id === c.id && o.step_number === step.number && o.status === "paid");
      if (paid && !state.reorder) {
        panel.replaceChildren(...[
          head,
          h.el("p", { class: "case-feedback positive", text: h.T("✓ Paid {amount} for {product} ({field}). Your kit for step {n} is on its way. Spray as the step says, then check back.", {
            amount: money(paid.total, paid.currency),
            product: paid.product,
            field: paid.field_text || "",
            n: step.number,
          }) }),
          h.el("ul", { class: "kit-paid-items" }, (paid.items || []).map((item) => h.el("li", { text: `${item.quantity} × ${item.name}` }))),
          h.el("button", {
            type: "button",
            class: "btn-outline btn-pill kit-update",
            text: h.T("Order more for this step"),
            onclick: () => { state.reorder = true; rerender(); },
          }),
          disclaimer,
        ]);
        return;
      }

      // Field size carried over from an order for an earlier step.
      const earlier = orders.find((o) => o.case_id === c.id && o.status === "paid" && o.step_number < step.number);
      const rememberedText = !state.remembered || !state.quote || paid
        ? null
        : earlier
          ? h.T("Sized for your field from your step {n} order ({field}). That step did not work, so this kit uses a product from a different chemical group.", { n: earlier.step_number, field: state.quote.field_text })
          : h.T("Sized for your field from your last order ({field}).", { field: state.quote.field_text });

      const buttons = h.el("div", { class: "paypal-buttons" });

      panel.replaceChildren(...[
        head,
        h.el("p", { class: "treatment-meta", text: h.T("The app works out how much you need from the plan's dose and your field size, then picks the cheapest packs.") }),
        paid
          ? h.el("p", { class: "treatment-meta" }, [
              h.T("You already paid for this step's kit."),
              " ",
              h.el("button", { type: "button", class: "link-button", text: h.T("Back to the receipt"), onclick: () => { state.reorder = false; rerender(); } }),
            ])
          : null,
        assistantBox(c, state, h, rerender, cfg),
        state.reply ? h.el("p", { class: "assistant-reply", text: state.reply }) : null,
        rememberedText ? h.el("p", { class: "assistant-reply", text: rememberedText }) : null,
        kitForm(c, state, h, rerender),
        state.loading ? h.el("p", { class: "treatment-meta translating", text: h.T("Working…") }) : null,
        state.error ? h.el("p", { class: "explain-error", text: state.error }) : null,
        quoteBlock(state, h),
        state.quote && !state.loading ? buttons : null,
        disclaimer,
      ].filter(Boolean));

      if (state.quote && !state.loading) renderButtons(buttons, c, state, h, rerender, cfg);
    }

    Promise.all([config(h), loadOrders(h)]).then(([cfg]) => {
      fill(cfg);
      // First view of this step: price it for the remembered field, if any.
      if (cfg.enabled && !state.started) {
        state.started = true;
        fetchKit(c, state, h, rerender, state.field ? { ...state.field } : {});
      }
    });

    return panel;
  }

  function ordersBlock(c, h) {
    const block = h.el("div", { class: "kit-orders" });

    const fill = () => {
      const mine = orders.filter((o) => o.case_id === c.id);
      block.replaceChildren(...(mine.length
        ? [
            h.el("p", { class: "result-eyebrow", text: h.T("Treatment kit orders ({n})", { n: mine.length }) }),
            h.el("ul", { class: "kit-order-list" }, mine.map((o) => orderRow(o, h, fill))),
          ]
        : []));
    };

    loadOrders(h).then(fill);
    return block;
  }

  window.VerdantShop = {
    kitPanel,
    ordersBlock,
    refresh: (h) => loadOrders(h, true),
    reset: () => {
      states.clear();
      orders = [];
      ordersPromise = null;
    },
  };
})();
