"""
Main Flask application.

Creates and configures the Flask app.
"""

import logging
import os

from flask import Flask

from backend.routes.frontend_routes import frontend_bp
from backend.routes.prediction_routes import prediction_bp


def create_app() -> Flask:
    """
    Create and configure Flask application.

    Returns:
        Flask: Configured Flask app.
    """

    app = Flask(__name__)

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = upload_folder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app.register_blueprint(prediction_bp)
    app.register_blueprint(frontend_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)