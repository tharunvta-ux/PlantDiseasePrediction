"""
Deployment checks for the low-memory TFLite configuration (Render).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT, first_val_image

TFLITE_MODEL = PROJECT_ROOT / "saved_models" / "plant_disease_cnn_int8.tflite"

SCRIPT = r"""
import io, sys, json
from pathlib import Path
from backend.app import create_app
client = create_app().test_client()
img = Path(sys.argv[1])
r = client.post("/predict", data={"file": (io.BytesIO(img.read_bytes()), img.name)},
                content_type="multipart/form-data")
print(json.dumps({"status": r.status_code, "body": r.get_json(),
                  "tensorflow_loaded": "tensorflow" in sys.modules}))
"""


@pytest.mark.skipif(not TFLITE_MODEL.exists(), reason="TFLite model not exported.")
def test_tflite_app_runs_without_tensorflow():
    """Fresh process: serving with the TFLite model must not import TF."""

    import json

    image = first_val_image("Tomato___healthy")
    env = {
        **os.environ,
        "MODEL_PATH": str(TFLITE_MODEL),
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }

    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(image)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2000:]

    output = json.loads(result.stdout.strip().splitlines()[-1])

    assert output["status"] == 200
    assert output["body"]["model_version"] == "plant_disease_cnn_int8"
    assert output["tensorflow_loaded"] is False


def test_render_requirements_are_utf8_and_pinned():
    lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements-render.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert all("==" in line for line in lines)
    assert not any(line.lower().startswith("tensorflow") for line in lines)
    assert any(line.startswith("gunicorn==") for line in lines)


def test_render_model_is_committable():
    """GitHub rejects files > 100 MB."""

    if not TFLITE_MODEL.exists():
        pytest.skip("TFLite model not exported.")

    assert TFLITE_MODEL.stat().st_size < 100 * 1024 * 1024
