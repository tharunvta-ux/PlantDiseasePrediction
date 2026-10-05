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

## Deploy on Render (free plan)

The web service runs the model as a compact TFLite file
(`saved_models/plant_disease_cnn_int8.tflite`) with the LiteRT runtime,
so it uses ~150 MB RAM instead of ~1.2 GB with full TensorFlow.

1. Push this repository to GitHub.
2. In Render: **New → Blueprint**, pick the repository. `render.yaml`
   configures the build, start command and environment.
3. When asked, enter your Gemini key for `LLM_API_KEY` (never commit it).
4. Open `https://<your-service>.onrender.com/app/`.

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
- History Database
