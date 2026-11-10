# Plant Disease Detection and Treatment Recommendation System

## Project Overview

This project is a Deep Learning based web application that detects plant diseases from leaf images.

## Technology Stack

- Python
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy
- SQLite
- HTML
- CSS
- JavaScript

## Project Structure

backend/
frontend/
dataset/
saved_models/
uploads/

## Current Status

✅ Module 1 Completed

## How an image is analysed (`POST /analyze`)

1. **Plant identification** (Gemini vision): is it a plant, which plant,
   is it one of the 14 crops the disease model supports?
2. **Disease diagnosis** (trained CNN): only for supported crops, with
   calibrated confidence levels. If the AI identifies a different crop
   than the CNN, the CNN's best match *within* that crop is shown as
   uncertain.
3. **Other plants**: an AI observation of visible symptoms, clearly
   marked as unverified.
4. **Treatment guidance** (`POST /recommendation`): Gemini explains the
   CNN's diagnosis using curated disease facts.

If Gemini is unavailable, `/analyze` falls back to the CNN result alone.
`POST /predict` returns the CNN result only (unchanged API).

When the AI and the CNN name different crops, the CNN's result is kept
but marked uncertain: on PlantVillage leaves Gemini named the correct
crop for 15 of 20 test images (e.g. tomato leaves called potato).

## Other features

- **Where did the model look?** (`POST /explain`): occlusion-sensitivity
  heat map. On sample images it shows the model partly relies on the
  background and leaf edges, a known weakness of lab-photo training.
- **Model performance page** (`/app/model`, data from `GET /model/report`):
  held-out accuracy, confidence levels, per-class accuracy, confusion
  matrix, calibration and training curves.
- **Guidance in 12 languages** (`GET /languages`, `language` field on
  `/analyze` and `/recommendation`) with a read-aloud button: English uses
  the browser's voice; other languages use server-generated audio
  (`POST /speech`, Gemini TTS) because most devices have no Tamil, Telugu,
  etc. voice installed.
- **PDF report**: the browser print dialog's "Save as PDF" produces a
  one-page report with the photo, heat map, result and guidance.

## Treatment plans and tracking

- **Camera capture**: "Take photo" opens the back camera on phones.
- **Step-by-step plan** (`POST /treatment-plan`): after a diagnosis the
  app suggests a first step. Plans come from
  `backend/knowledge/treatment_plans.json`, not from the LLM:
  - uncertain diagnosis → low-risk care steps only;
  - confident diagnosis → care steps plus a **TNAU-listed** fungicide /
    insecticide with TNAU's dose (TNAU Agritech Portal, checked
    2026-10-27, source link per disease);
  - didn't work → a product from a *different* chemical group, then
    **consult an expert nearby** (Kisan Call Centre 1800-180-1551,
    Krishi Vigyan Kendra map search);
  - viral diseases → remove infected plants and control the insect
    vector, then expert; citrus greening → expert straight away.
  Chemicals TNAU lists but that are banned or doubtful in India
  (captafol, tridemorph, antibiotic sprays such as streptocycline) are
  left out; see `_meta.excluded_from_tnau`. The plan file has **not been
  reviewed by an agronomist** yet.
- **Other crops** (`backend/knowledge/ai_condition_plans.json`): TNAU
  plans for 20 diseases of papaya, banana, mango, brinjal, chilli, rice,
  coconut and groundnut. Gemini may match a photo to one of them
  (`ai_condition` in the `/analyze` response). The grower is shown TNAU's
  symptom description and asked "Does your plant look like this?":
  *Yes* gives the TNAU treatment, *Not sure* gives care steps first,
  *No* gives general care.
- **Accounts + history** (`/auth/*`, `/cases*`): sign in with email and
  password and track a plan. Each step has a "check again on" date.
  At the check the grower ticks *It worked* (case closed with a positive
  message) or *It did not work*, optionally with a new photo. The photo
  is re-analysed: a confident *different* disease starts that disease's
  plan, and a confirmed uncertain diagnosis unlocks the TNAU treatment.
  Otherwise the next step is suggested.

- **Confirming an AI match:** the grower first confirms the crop ("Is
  this a papaya plant?"), then answers 2–3 yes/no symptom checks
  (`checks` in `ai_condition_plans.json`). All yes → TNAU treatment;
  partly → care steps first; all no → general care. "Not sure? Ask an
  expert first" offers the Kisan Call Centre and KVK links.
- **Severity and outlook** (`backend/prediction/severity.py`,
  `backend/services/severity_service.py`): share of leaf area that is
  spotted / yellow / brown / powdery (OpenCV colour segmentation,
  approximate), combined with the grower's "How much of the plant has
  this?" (the worse one counts) → mild / moderate / severe. The outlook
  is good / fair / poor with its reason, deliberately not a percentage.
  At check-ins the new photo is compared with the last one: a shrinking
  disease keeps the current step going once instead of switching
  chemicals; a fast-spreading severe one goes straight to an expert.
  The photo measure underestimates pale powdery mildew and virus
  mottling, which is why the grower's estimate is included.

- **Every screen in the farmer's language** (🌐 picker in the top bar):
  plans, history cases, check-in forms and AI-match questions are
  translated server-side (`translation_service.py`, `localize_service.py`,
  labels from the allowlist in `ui_text.py`). Product names and doses are
  never translated, and a translation that changes any number or
  placeholder is rejected. Translations are cached in the `translations`
  table, so each text costs one Gemini call per language.
- **🔊 Read aloud** on "What to do now" and on every history case
  (`POST /treatment-plan/speech`, `POST /cases/<id>/speech`): the server
  builds the spoken text from its own data, so the key cannot be used
  as a free TTS service.

## Treatment kit checkout (PayPal sandbox + AI assistant)

Knowing *what* to spray is only half the problem: the grower also has to
work out *how much* to buy for their field and get it. Every treatment
step that lists a TNAU product now has a **🛒 Get this treatment
delivered** panel in the treatment history:

1. **Size the kit.** Type the field size (acres, cents, hectares or square
   metres) and the number of sprays, or tell the **AI assistant** in your
   own words and language ("arai acre thakkali, rendu thadava", "I have 20
   cents of chilli and already have mancozeb"). Gemini reads the area,
   unit and sprays, picks one of the step's listed options (skipping one
   the grower already has) and explains its choice in their language.
2. **Kit maths** (`backend/services/kit_service.py`, deterministic): the
   TNAU dose text is parsed ("0.25% (2.5 g per litre)", "200 ml per
   acre", "50 g per 100 litres") and multiplied by the spray volume
   (500 litres per hectare) or the area, times the sprays. The cheapest
   combination of catalog packs that covers it is chosen
   (`backend/knowledge/shop_catalog.json`, demo prices).
3. **Pay with PayPal** (`backend/services/paypal_service.py`, Orders v2):
   the server creates the order with itemised packs, reference
   `case-<id>-step-<n>` and a unique invoice id; the grower approves it
   in the PayPal window; the server captures it and checks the captured
   amount matches before marking the order paid. The shipping address
   comes from the buyer's PayPal account.
4. **Next step, new kit.** If the step did not work, the plan moves to a
   product from a different chemical group and the panel re-prices a kit
   for the same field automatically (remembered from the last order).
5. **Cancel and refund** a paid kit (Payments v2 full refund).

Guardrails: the browser never sends prices (the server recomputes the
kit from the curated plan when the order is created); the assistant can
only choose between the step's listed products by number and may not
write doses or product names; uncertain diagnoses get care steps only,
so nothing can be bought until a diagnosis is confident.

Routes: `GET /shop/config`, `POST /shop/cases/<id>/kit`,
`POST /shop/cases/<id>/assistant`, `POST /shop/cases/<id>/orders`,
`POST /shop/orders/<paypal_order_id>/capture`, `GET /shop/orders`,
`POST /shop/orders/<id>/refund`.

Setup: create a sandbox app at
[developer.paypal.com](https://developer.paypal.com/dashboard/applications/sandbox)
and set `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET` (see `.env.example`).
Pay with a sandbox **personal** account from the same dashboard. This is
a demo store: no real money is taken and nothing is shipped.

Data is stored with SQLAlchemy: SQLite at `instance/verdant.db` locally,
Postgres in production via `DATABASE_URL`.

## Deploy on Render (free plan)

The web service runs the model as a compact TFLite file
(`saved_models/plant_disease_cnn_int8.tflite`) with the LiteRT runtime,
so it uses ~150 MB RAM instead of ~1.2 GB with full TensorFlow.

1. Push this repository to GitHub.
2. In Render: **New → Blueprint**, pick the repository. `render.yaml`
   configures the build, start command and environment.
3. When asked, enter your Gemini key for `LLM_API_KEY` (never commit it).
4. Create a free Postgres database at [Neon](https://neon.tech) or
   [Supabase](https://supabase.com) and enter its connection string for
   `DATABASE_URL`. Render's own disk is wiped on every deploy and restart,
   so without this, accounts and history are lost.
   (`SECRET_KEY` is generated by Render automatically.)
5. Open `https://<your-service>.onrender.com/app/`.

Free instances sleep after 15 minutes without traffic; the first visit
afterwards takes about a minute.

To regenerate the TFLite model and its calibration:

```
python -m backend.training.export_tflite --variant int8
set MODEL_PATH=saved_models/plant_disease_cnn_int8.tflite
python -m backend.evaluation.collect_predictions
python -m backend.evaluation.calibrate
```

## Future Modules

- Dataset Preparation
- Image Preprocessing
- CNN Model
- Training
- Prediction
- Treatment Recommendation
