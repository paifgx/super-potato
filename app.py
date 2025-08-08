from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

# Configure minimal structured logging for useful diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("california_housing_api")

# --- Artifact loading (performed once at startup) ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "housing_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order.json"

DEFAULT_FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "RoomsPerPerson",
    "BedrmRoomRatio",
    "LogPopulation",
]


def _load_artifacts() -> tuple[Any, Any, list[str]]:
    """Load model, scaler, and feature order from the local model directory.

    Why: Ensures API inputs are transformed identically to training, avoiding
    silent feature order drift.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        message = (
            "Model/scaler missing. Train first in the dev container by running "
            "the notebook to export 'model/housing_model.pkl' and 'model/scaler.pkl'."
        )
        logger.error("artifact_missing | path_model=%s | path_scaler=%s", MODEL_PATH, SCALER_PATH)
        raise RuntimeError(message)

    model_obj = joblib.load(MODEL_PATH)
    scaler_obj = joblib.load(SCALER_PATH)

    if FEATURE_ORDER_PATH.exists():
        with FEATURE_ORDER_PATH.open("r", encoding="utf-8") as f:
            feature_order_loaded = json.load(f)
        if not isinstance(feature_order_loaded, list) or len(feature_order_loaded) != 9:
            logger.warning("feature_order_invalid | path=%s | using_default=True", FEATURE_ORDER_PATH)
            feature_order_loaded = DEFAULT_FEATURE_ORDER
    else:
        logger.info("feature_order_missing | using_default=True")
        feature_order_loaded = DEFAULT_FEATURE_ORDER

    return model_obj, scaler_obj, feature_order_loaded


model, scaler, feature_order = _load_artifacts()


# --- FastAPI setup ---
app = FastAPI(
    title="California Housing Prediction API",
    version="0.1.0",
    description=(
        "Predicts California Median House Value in USD. Model outputs in 100k "
        "USD are converted to USD."
    ),
)


# --- Schemas ---
class PredictRequest(BaseModel):
    """Payload containing the exact 9 features expected by the model."""

    med_inc: float = Field(..., description="Median income (in $10k units)")
    house_age: float
    ave_rooms: float
    ave_bedrms: float
    population: float
    ave_occup: float
    rooms_per_person: float
    bedrm_room_ratio: float
    log_population: float


class PredictResponse(BaseModel):
    """Model prediction in USD."""

    predicted_price: float
    currency: Literal["USD"] = "USD"


class PredictAutoRequest(BaseModel):
    """User-friendly payload; server derives missing engineered features.

    Why: Keeps client contracts simple while ensuring the model receives the
    exact engineered features in the correct order.
    """

    med_inc: float = Field(..., description="Median income (in $10k units)")
    house_age: float
    rooms: float
    bedrms: float
    persons: float
    population: Optional[float] = None
    location: Optional[str] = Field(
        None, description="Optional location identifier for demo lookup stub"
    )

    @model_validator(mode="after")
    def validate_non_negative(self) -> "PredictAutoRequest":
        if self.rooms is not None and self.rooms <= 0:
            raise ValueError("rooms must be > 0")
        if self.bedrms is not None and self.bedrms < 0:
            raise ValueError("bedrms must be >= 0")
        if self.persons is not None and self.persons <= 0:
            raise ValueError("persons must be > 0")
        return self


# --- Helpers ---
def compose_feature_row_from_api(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert API field names to training feature names and order columns.

    Why: The scaler and model expect columns in the exact training order.
    """
    mapping = {
        "med_inc": "MedInc",
        "house_age": "HouseAge",
        "ave_rooms": "AveRooms",
        "ave_bedrms": "AveBedrms",
        "population": "Population",
        "ave_occup": "AveOccup",
        "rooms_per_person": "RoomsPerPerson",
        "bedrm_room_ratio": "BedrmRoomRatio",
        "log_population": "LogPopulation",
    }
    row = {mapping[key]: payload[key] for key in mapping}
    df = pd.DataFrame([row])
    df = df[feature_order]
    return df


def predict_usd(df_features: pd.DataFrame) -> float:
    """Scale features, run model, and convert 100k USD output to USD."""
    X_scaled = scaler.transform(df_features)
    y_pred_100k = float(model.predict(X_scaled)[0])
    price_usd = round(y_pred_100k * 100000.0, 2)
    return price_usd


def lookup_by_location(location: Optional[str]) -> Dict[str, Optional[float]]:
    """Demo stub: return population and ave_occup by location.

    Why: Illustrates how a real census/DB lookup could fill missing fields.
    """
    if not location:
        return {"population": None, "ave_occup": None}

    demo: Dict[str, Dict[str, float]] = {
        "94103": {"population": 70000.0, "ave_occup": 2.1},
        "90001": {"population": 96000.0, "ave_occup": 3.5},
    }
    return demo.get(location, {"population": None, "ave_occup": None})


# --- Endpoints ---
@app.get("/")
def health() -> Dict[str, Any]:
    """Return service health and model feature order for quick inspection."""
    return {"status": "ok", "model_features": feature_order}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict median house value in USD from the 9 engineered features."""
    try:
        df = compose_feature_row_from_api(request.model_dump())
        price_usd = predict_usd(df)
        logger.info(
            "predict | endpoint=/predict | price_usd=%.2f",
            price_usd,
        )
        return PredictResponse(predicted_price=price_usd)
    except ValidationError as exc:
        logger.exception("validation_error | endpoint=/predict")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction_failed | endpoint=/predict")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict/auto", response_model=PredictResponse)
def predict_auto(request: PredictAutoRequest) -> PredictResponse:
    """Derive engineered features from user inputs and predict price in USD."""
    # 1) Lookup optional fields by location (demo)
    lookup = lookup_by_location(request.location)

    # 2) Resolve population
    population: Optional[float] = (
        request.population if request.population is not None else lookup.get("population")
    )
    if population is None:
        raise HTTPException(
            status_code=422,
            detail="population is required (provide directly or via location)",
        )

    # 3) Resolve ave_occup
    ave_occup: float = (
        float(lookup.get("ave_occup")) if lookup.get("ave_occup") is not None else float(request.persons)
    )

    # 4) Derived features (validate zero division and domain)
    if request.rooms <= 0 or request.persons <= 0:
        raise HTTPException(status_code=422, detail="rooms and persons must be > 0")

    rooms_per_person = float(request.rooms / request.persons)
    bedrm_room_ratio = float(request.bedrms / request.rooms)
    log_population = float(np.log1p(population))

    # 5) Build full payload in API field names
    payload_full: Dict[str, Any] = {
        "med_inc": float(request.med_inc),
        "house_age": float(request.house_age),
        "ave_rooms": float(request.rooms),
        "ave_bedrms": float(request.bedrms),
        "population": float(population),
        "ave_occup": float(ave_occup),
        "rooms_per_person": rooms_per_person,
        "bedrm_room_ratio": bedrm_room_ratio,
        "log_population": log_population,
    }

    try:
        df = compose_feature_row_from_api(payload_full)
        price_usd = predict_usd(df)
        logger.info(
            "predict | endpoint=/predict/auto | price_usd=%.2f",
            price_usd,
        )
        return PredictResponse(predicted_price=price_usd)
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction_failed | endpoint=/predict/auto")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


# Optional: enable `python app.py` local run inside container
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
