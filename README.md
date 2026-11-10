# Verdant: AI Crop Doctor + PayPal Treatment Kits

**Snap a leaf. Verdant's AI diagnoses the crop disease, gives a safe
step-by-step treatment plan in the farmer's own language, works out the
exact spray kit for their field, and lets them buy it with PayPal.**

| | |
|---|---|
| 🌐 **Live demo** | [plant-disease-detection-p7f3.onrender.com/app](https://plant-disease-detection-p7f3.onrender.com/app/) (the free server sleeps when idle; the first load can take about a minute) |
| 🎬 **Video (under 3 min)** | _link added on submission_ |
| 🧪 **PayPal sandbox buyer** | email `sb-cziqj53836512@personal.example.com` · password `a__kS0hT` (fake sandbox account, no real money) |
| 📄 **License** | [MIT](LICENSE) |

Built for the **PayPal AI Hackathon** (Devpost, Oct 1 – Nov 12, 2026).

---

## The problem

A smallholder farmer sees spots on a leaf. Three questions decide
whether the crop is saved:

1. **What is it?** Most farmers have no plant doctor nearby.
2. **What should I spray, and is it safe?** Shop advice is often wrong,
   and banned or fake chemicals are a real problem.
3. **How much do I buy?** Doses are written as "0.25%" or "200 ml per
   acre". Turning that into packs for *my* half acre and two sprays is
   hard, so farmers buy too little (the treatment fails) or too much
   (money wasted, extra chemicals in the field).

Verdant answers all three, and then lets the farmer buy the right kit.

## What it does

1. **Photo → diagnosis.** Take a photo of a leaf (the phone camera opens
   directly). Gemini identifies the plant; a CNN trained on 38 classes
   (14 crops) diagnoses the disease, with calibrated confidence and a
   heat map showing where it looked.
2. **A safe, step-by-step plan.** Products and doses come **only** from
   curated official sources (TNAU Agritech Portal; Government of India
   CIB&RC registered uses), never from the AI. An uncertain diagnosis
   gets low-risk care steps only; banned chemicals and antibiotics are
   never suggested.
3. **In the farmer's language.** Every screen, plan and history entry is
   available in 12 languages (English, Hindi, Tamil, Telugu, Kannada,
   Malayalam, Marathi, Bengali, Gujarati, Punjabi, Spanish, French), with
   🔊 read-aloud for farmers who prefer listening.
4. **🛒 The treatment kit + PayPal.** The farmer tells the **AI assistant**
   about their field in their own words, for example *"எனக்கு அரை ஏக்கர்
   தக்காளி, ரெண்டு தடவை தெளிக்கணும்"* (Tamil for "half an acre of
   tomato, spraying twice"). The assistant works out 0.5 acre and 2
   sprays and chooses among the listed products. Verdant computes the
   exact quantity and the cheapest combination of packs, and the farmer
   pays with **PayPal**.
5. **Tracking that closes the loop.** Each step has a "check again on"
   date. If the farmer answers *"It did not work"*, the plan moves to a
   product from a **different chemical group** (resistance management),
   and the kit for the next step is **priced automatically for the same
   field**: one tap to re-order. Paid kits can be cancelled and refunded.

## How PayPal is used

All server-side, with the [PayPal REST APIs](https://developer.paypal.com/api/rest/) in the sandbox
(`backend/services/paypal_service.py`, `backend/routes/shop_routes.py`):

| PayPal feature | Use in Verdant |
|---|---|
| **OAuth 2.0 client credentials** | Server access token, cached and renewed automatically |
| **Orders v2: create** (`intent: CAPTURE`) | Itemised order: one line per pack (SKU, quantity, unit price), `item_total` breakdown, `reference_id` = case + treatment step, unique `invoice_id`, `brand_name`, shipping from the buyer's PayPal account |
| **PayPal JS SDK buttons** | PayPal and card buttons; the buyer approves in the PayPal window |
| **Orders v2: capture** | After approval. The server checks the captured **amount and currency match** its own quote before marking the order paid, and stores the payer and shipping address |
| **Payments v2: refund** | "Cancel order and refund" for a paid kit |
| **`PayPal-Request-Id`** | Idempotent create, capture and refund, so a retried request is never applied twice |
| **Error handling** | `INSTRUMENT_DECLINED` restarts the PayPal flow so the buyer can choose another funding source; `ORDER_ALREADY_CAPTURED` is treated as success |

**The browser never sends prices.** When the order is created, the
server recomputes the kit from the case's current treatment step and the
catalog, so a manipulated request cannot change what is charged or add
products the plan does not list.

## How AI is used

| AI tool | What it does |
|---|---|
| **Custom CNN** (TensorFlow/Keras, served as an 8-bit TFLite model) | Disease diagnosis for 38 PlantVillage classes. 93% accuracy on held-out PlantVillage validation images (lab photos; real-field accuracy is lower and not yet measured). Temperature-calibrated confidence levels; occlusion heat maps |
| **Google Gemini** (`gemini-3.8-flash`, with a fallback chain of free models) | Plant identification from the photo (vision); plain-language explanation of the diagnosis grounded in the plan; replies to check-in notes; translation of plans and screens into 12 languages; text-to-speech for Indian languages |
| **Gemini as a shopping assistant** (`backend/services/shop_assistant_service.py`) | Reads the farmer's field description in any of the 12 languages and returns structured JSON (area, unit, sprays, chosen option, a product the farmer already has). It may only choose between the step's **listed** products, by number; it never writes doses or prices (a reply that contains one is replaced) |
| **OpenCV** | Severity: share of the leaf that is spotted or discoloured, combined with the farmer's estimate |

**Agentic commerce with guardrails.** The AI *proposes* (field size,
which listed product); deterministic code *decides* (quantity from the
official dose, cheapest packs, price); the farmer *approves* every
payment in PayPal. The agent cannot invent a product, change a dose or
pay on its own.

## Try it (judges)

**Hosted demo:** open https://plant-disease-detection-p7f3.onrender.com/app/

1. **Sign in → Create one** (any email and an 8+ character password).
2. Upload a clear leaf photo. Good test images: any photo from the
   public [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color),
   for example from the `Corn_(maize)___Common_rust_` folder. Then click
   **Predict disease**.
3. Under **What to do now**, click **Track this treatment**.
4. Go to **History**. In **🛒 Get this treatment delivered**, type
   *"half an acre, 2 sprays"* (in any language) and click **Ask the
   assistant**, or enter the area and click **Calculate kit**.
5. Click **PayPal** and log in with the sandbox buyer above. You'll see
   **✓ Paid** and the order with its shipping address.
6. Optional: in the check-in form choose **It did not work → Save check**.
   Step 2's kit (a different chemical group) is priced for the same
   field automatically. Try **🌐** to switch language, and
   **Cancel order and refund**.

> If the diagnosis is *uncertain*, Verdant deliberately offers only
> low-risk care steps and nothing to buy. Use a clearer photo.

## Run it locally

Requirements: Python 3.12, a free [Gemini API key](https://aistudio.google.com/apikey),
and a free [PayPal sandbox app](https://developer.paypal.com/dashboard/applications/sandbox).

```bash
git clone https://github.com/tharunvta-ux/PlantDiseasePrediction.git
cd PlantDiseasePrediction
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements-render.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Fill in `.env`:

```
LLM_API_KEY=<your Gemini key>
PAYPAL_CLIENT_ID=<sandbox app client id>
PAYPAL_CLIENT_SECRET=<sandbox app secret>
MODEL_PATH=saved_models/plant_disease_cnn_int8.tflite
```

```bash
flask --app backend.app run --port 5000
```

Open **http://127.0.0.1:5000/app/**.

**PayPal sandbox tip:** create the sandbox **business** (merchant) and
**personal** (buyer) accounts with country **United States**, and link
your sandbox app to that business account. Sandbox accounts created in
India are blocked by PayPal's India regulations (see *Limitations*).

**Tests:** `pip install pytest` then `pytest -m "not model and not live"`
(350+ tests; PayPal and Gemini are mocked).

### Deploy on Render (free)

`render.yaml` is a Render Blueprint: **New → Blueprint**, pick this repo,
then enter `LLM_API_KEY`, `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` and a
free Postgres `DATABASE_URL` (e.g. [Neon](https://neon.tech); Render's own
disk is wiped on every restart). The model runs as TFLite in about 150 MB of RAM.

## Built during the hackathon (since Oct 1, 2026)

The project started in July 2026 as a CNN disease classifier with a
simple web page. During the submission period we added:

- **PayPal treatment-kit checkout** (Orders v2 create and capture,
  refunds, idempotency, server-side pricing) and the **AI shopping
  assistant**
- Dose parser and kit sizing for every product in the plans; pack
  optimiser; automatic re-pricing for the next treatment step
- Hybrid analysis: Gemini plant identification + CNN diagnosis
- Treatment plans from TNAU / CIB&RC, treatment tracking with check-ins,
  AI replies to check-in notes
- Severity and recovery outlook; heat-map explanations; model
  performance page; PDF report
- 12 languages for every screen, with read-aloud
- Gemini fallback chain for the free tier; Render deployment with a
  TFLite model

Full history: `git log --since=2026-10-01`.

## Data sources and safety

- **Products and doses:** [TNAU Agritech Portal: Crop Protection](https://agritech.tnau.ac.in/crop_protection/)
  (checked 2026-10-27), and where TNAU has no page, the exact crop + disease
  entry in the Government of India
  [CIB&RC "Major uses of pesticides"](https://ppqs.gov.in/sites/default/files/fungicides_31.03.2024.pdf)
  list. Each plan links to its source.
- **Never suggested:** chemicals banned in India (captafol, tridemorph),
  and antibiotic sprays (streptocycline etc.), because of antimicrobial
  resistance. See `_meta` in `backend/knowledge/treatment_plans.json`.
- Diseases with no official product for that crop get care steps and
  then an expert (Kisan Call Centre, Krishi Vigyan Kendra), never an
  invented product.
- The plan files have **not yet been reviewed by an agronomist**.
- **Regional knowledge packs:** plans (`treatment_plans.json`,
  `ai_condition_plans.json`) and the shop catalog (`shop_catalog.json`)
  are data files. India is the first pack; another country is added
  with its own vetted pack, without code changes.

## Limitations (honest)

- **PayPal and India:** PayPal does not currently process domestic
  payments within India. This demo runs entirely in the PayPal sandbox
  with US test accounts. In production, the kit would be paid by family
  abroad sending support home, or by farmers in countries where PayPal
  handles local payments.
- **Demo store:** prices are approximate demo prices in USD; nothing is
  shipped.
- **Model:** trained on PlantVillage lab photos; field photos with busy
  backgrounds are harder, which is why the app shows confidence levels
  and refuses to suggest chemicals when unsure.

## Project structure

```
backend/
  app.py                      Flask app
  routes/                     analysis, tracking, shop (PayPal), auth, model report
  services/
    kit_service.py            dose parsing, kit sizing, pack optimiser
    paypal_service.py         PayPal REST client (Orders v2, refunds)
    shop_assistant_service.py Gemini shopping assistant
    treatment_plan_service.py curated plan ladder
    llm/                      Gemini provider with fallback chain
  knowledge/                  curated plans, disease info, shop catalog
  prediction/                 CNN inference, calibration, severity, heat maps
frontend/                     HTML, CSS, JS (app.js, tracking.js, shop.js)
tests/                        pytest suite
```

### Regenerate the TFLite model and its calibration

```
python -m backend.training.export_tflite --variant int8
set MODEL_PATH=saved_models/plant_disease_cnn_int8.tflite
python -m backend.evaluation.collect_predictions
python -m backend.evaluation.calibrate
```

## License

[MIT](LICENSE)
