"""
Main Flask application.

Creates and configures the Flask app.
"""

import logging
import os

from flask import Flask
from flask import jsonify

from backend.config import MAX_UPLOAD_BYTES
from backend.routes.analysis_routes import analysis_bp
from backend.routes.frontend_routes import frontend_bp
from backend.routes.model_routes import model_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.recommendation_routes import recommendation_bp


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
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app.register_blueprint(prediction_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(frontend_bp)

    @app.errorhandler(413)
    def file_too_large(_error):
        """Return JSON instead of Flask's HTML page for oversized uploads."""

        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)

        return jsonify(
            {"error": f"File is too large. Maximum size is {limit_mb} MB."}
        ), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)