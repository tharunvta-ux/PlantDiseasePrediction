"""
backend/database/models.py

Tables for user accounts and treatment tracking.

    users            one row per account
    treatment_cases  one diagnosed plant problem being followed up
    treatment_steps  the steps suggested for a case, with the grower's
                     "worked / did not work" answer for each

Timestamps are stored as naive UTC (SQLite has no time zones).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

CASE_ACTIVE = "active"
CASE_RECOVERED = "recovered"
CASE_EXPERT = "expert"

OUTCOME_PENDING = "pending"
OUTCOME_WORKED = "worked"
OUTCOME_NOT_WORKED = "not_worked"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(value: datetime | None) -> str | None:
    """Naive UTC datetime -> ISO string with Z."""

    return value.isoformat(timespec="seconds") + "Z" if value else None


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    cases: Mapped[List["TreatmentCase"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "email": self.email}


class TreatmentCase(Base):
    __tablename__ = "treatment_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    status: Mapped[str] = mapped_column(String(20), default=CASE_ACTIVE)
    plant_name: Mapped[str] = mapped_column(String(120))
    # None for plants outside the disease model (AI observation only).
    predicted_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    display_name: Mapped[str] = mapped_column(String(200))
    confidence_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    photo: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    # Severity / outlook when tracking started (severity_service.assess).
    severity: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    user: Mapped[User] = relationship(back_populates="cases")
    steps: Mapped[List["TreatmentStep"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="TreatmentStep.number",
    )

    @property
    def current_step(self) -> "TreatmentStep | None":
        return self.steps[-1] if self.steps else None


class TreatmentStep(Base):
    __tablename__ = "treatment_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        ForeignKey("treatment_cases.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Diagnosis this step was planned for (changes if a re-check finds
    # a different disease).
    predicted_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    kind: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(200))
    actions: Mapped[list] = mapped_column(JSON, default=list)
    products: Mapped[list] = mapped_column(JSON, default=list)
    wait_days: Mapped[int] = mapped_column(Integer, default=0)
    check_question: Mapped[str] = mapped_column(String(300), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    source_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    check_due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    outcome: Mapped[str] = mapped_column(String(20), default=OUTCOME_PENDING)
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Message shown after the check-in, and what a new photo showed.
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    followup: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    followup_photo: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    # Severity, trend and outlook measured at this step's check-in.
    severity: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    case: Mapped[TreatmentCase] = relationship(back_populates="steps")


ORDER_CREATED = "created"  # PayPal order created, waiting for the buyer
ORDER_PAID = "paid"
ORDER_REFUNDED = "refunded"
ORDER_FAILED = "failed"


class ShopOrder(Base):
    """A treatment kit paid with PayPal (sandbox), linked to a case step."""

    __tablename__ = "shop_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # Kept when the case is deleted (it is still a payment record).
    case_id: Mapped[int | None] = mapped_column(
        ForeignKey("treatment_cases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    step_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    paypal_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    invoice_id: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(20), default=ORDER_CREATED)
    currency: Mapped[str] = mapped_column(String(3))
    total: Mapped[str] = mapped_column(String(20))
    product: Mapped[str] = mapped_column(String(200))
    display_name: Mapped[str] = mapped_column(String(200))
    items: Mapped[list] = mapped_column(JSON, default=list)
    # Area, unit and sprays the kit was sized for (reused for the next step).
    field: Mapped[dict] = mapped_column(JSON, default=dict)
    quote: Mapped[dict] = mapped_column(JSON, default=dict)
    assistant: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    capture_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payer: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    shipping: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    refund_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    refunded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "step_number": self.step_number,
            "paypal_order_id": self.paypal_order_id,
            "invoice_id": self.invoice_id,
            "status": self.status,
            "currency": self.currency,
            "total": self.total,
            "product": self.product,
            "display_name": self.display_name,
            "items": self.items or [],
            "field": self.field or {},
            "field_text": (self.quote or {}).get("field_text"),
            "needed_text": (self.quote or {}).get("needed_text"),
            "payer": self.payer,
            "shipping": self.shipping,
            "created_at": iso(self.created_at),
            "paid_at": iso(self.paid_at),
            "refunded_at": iso(self.refunded_at),
        }


class Translation(Base):
    """Cached machine translation of one app-generated text."""

    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("language", "source_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    language: Mapped[str] = mapped_column(String(10))
    source_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_text: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
