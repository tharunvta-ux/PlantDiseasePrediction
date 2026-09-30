# Test image fixtures

Real-world images are **not** included. The related tests are skipped
until images are added here.

```
tests/fixtures/
    real_world/            # phone photos taken in the field / garden
        Tomato___Early_blight/
            img001.jpg
        Tomato___healthy/
        _unsupported/      # plants/objects outside the 38 model classes
    web/                   # images downloaded from the internet
        <exact class folder name>/
```

- Folder names must match the model classes exactly
  (see `backend/model_artifacts/plant_disease_cnn/class_names.json`).
- Only add images you have the right to use.
- Test: `pytest tests/test_real_model.py -k confidently_wrong`
  fails if any image is predicted wrongly at the "high" confidence level.
