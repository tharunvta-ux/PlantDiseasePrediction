"""
Main Flask application.

Creates and configures the Flask app.
"""

import logging
import os
import secrets
from datetime import timedelta
from urllib.parse import urlparse

from flask import Flask
from flask import jsonify
from flask import request

from backend.config import MAX_UPLOAD_BYTES, get_settings
from backend.database.db import init_db
from backend.routes.analysis_routes import analysis_bp
from backend.routes.auth_routes import auth_bp
from backend.routes.frontend_routes import frontend_bp
from backend.routes.model_routes import model_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.recommendation_routes import recommendation_bp
from backend.routes.tracking_routes import tracking_bp

logger = logging.getLogger(__name__)

# State-changing requests to these paths must come from this site.
SAME_ORIGIN_PREFIXES = ("/auth/", "/cases")


def create_app() -> Flask:
    """
    Create and configure Flask application.

    Returns:
        Flask: Configured Flask app.
    """

    app = Flask(__name__)
    settings = get_settings()

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if settings.secret_key:
        app.config["SECRET_KEY"] = settings.secret_key
    else:
        logger.warning("SECRET_KEY is not set: users are signed out whenever the server restarts.")
        app.config["SECRET_KEY"] = secrets.token_hex(32)

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.session_cookie_secure,
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    )

    init_db(app, settings.database_url)

    @app.before_request
    def same_origin_only():
        """Reject cross-site form posts that would ride on the session cookie."""

        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None

        if not request.path.startswith(SAME_ORIGIN_PREFIXES):
            return None

        origin = request.headers.get("Origin")

        if origin and urlparse(origin).netloc != request.host:
            return jsonify({"error": "Cross-site request blocked.", "code": "BAD_ORIGIN"}), 403

        return None

    app.register_blueprint(prediction_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(tracking_bp)
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
