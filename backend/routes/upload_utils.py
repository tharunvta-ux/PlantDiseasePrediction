"""
Shared upload handling for image endpoints.
"""

import os
import uuid
from contextlib import contextmanager
from typing import Iterator

from flask import current_app
from flask import jsonify
from flask import request
from PIL import Image, UnidentifiedImageError

from backend.config import ALLOWED_EXTENSIONS


class UploadError(Exception):
    """Invalid upload; `response` is the Flask (body, status) to return."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.response = (jsonify({"error": message}), status)


def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has supported extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@contextmanager
def saved_image_upload(field: str = "file") -> Iterator[str]:
    """
    Validate the uploaded image, save it, yield its path, always delete it.

    Raises:
        UploadError: missing file, unsupported type or not a valid image.
    """

    if field not in request.files:
        raise UploadError("No file uploaded")

    file = request.files[field]

    if file.filename == "":
        raise UploadError("No selected file")

    if not allowed_file(file.filename):
        raise UploadError("Unsupported file type")

    extension = file.filename.rsplit(".", 1)[1].lower()

    filepath = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        f"{uuid.uuid4()}.{extension}",
    )

    try:
        file.save(filepath)

        try:
            with Image.open(filepath) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            raise UploadError("The uploaded file is not a valid image")

        yield filepath

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
