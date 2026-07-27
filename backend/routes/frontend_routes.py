"""
Frontend UI routes.

Serves the static HTML/CSS/JS interface from the top-level
frontend/ directory. Does not touch the prediction API.
"""

from pathlib import Path

from flask import Blueprint
from flask import render_template

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = PROJECT_ROOT / "frontend"

frontend_bp = Blueprint(
    "frontend",
    __name__,
    url_prefix="/app",
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
    static_url_path="/static",
)


@frontend_bp.route("/", methods=["GET"])
def index():
    """
    Render the plant disease detection UI.
    """

    return render_template("index.html")
