"""
Account routes (email + password, session cookie).

POST /auth/register   {"email", "password"}
POST /auth/login      {"email", "password"}
POST /auth/logout
GET  /auth/me         -> {"user": {...} | null}
"""

from __future__ import annotations

import functools
import re

from flask import Blueprint, g, jsonify, request, session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from backend.database.db import db_session
from backend.database.models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
MAX_PASSWORD = 128

# Compared against when the email is unknown, so a login takes the same
# time whether or not the account exists.
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def _error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


def current_user() -> User | None:
    """Signed-in user for this request, or None."""

    if "current_user" not in g:
        user_id = session.get("user_id")
        g.current_user = db_session().get(User, user_id) if user_id else None

    return g.current_user


def login_required(view):
    """Return 401 JSON unless signed in."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            return _error("Please sign in first.", "NOT_SIGNED_IN", 401)
        return view(*args, **kwargs)

    return wrapper


def _credentials():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return None, None

    email = payload.get("email")
    password = payload.get("password")

    if not isinstance(email, str) or not isinstance(password, str):
        return None, None

    return email.strip().lower(), password


def _sign_in(user: User) -> None:
    session.clear()
    session["user_id"] = user.id
    session.permanent = True


@auth_bp.route("/register", methods=["POST"])
def register():
    email, password = _credentials()

    if not email or len(email) > 254 or not EMAIL_PATTERN.match(email):
        return _error("Please enter a valid email address.", "INVALID_EMAIL", 400)

    if password is None or not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        return _error(
            f"Password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters long.",
            "INVALID_PASSWORD",
            400,
        )

    db = db_session()
    user = User(email=email, password_hash=generate_password_hash(password))
    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return _error("An account with this email already exists.", "EMAIL_TAKEN", 409)

    _sign_in(user)

    return jsonify({"user": user.to_dict()}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    email, password = _credentials()

    if not email or password is None:
        return _error("Email and password are required.", "INVALID_CREDENTIALS", 400)

    user = db_session().scalar(select(User).where(User.email == email))

    if not check_password_hash(user.password_hash if user else _DUMMY_HASH, password) or not user:
        return _error("Wrong email or password.", "INVALID_CREDENTIALS", 401)

    _sign_in(user)

    return jsonify({"user": user.to_dict()}), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()

    return jsonify({"user": None}), 200


@auth_bp.route("/me", methods=["GET"])
def me():
    user = current_user()

    return jsonify({"user": user.to_dict() if user else None}), 200
