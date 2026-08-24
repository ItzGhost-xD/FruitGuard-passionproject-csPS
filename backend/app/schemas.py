from __future__ import annotations

from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    width: int
    height: int
    blur_var: float
    brightness: float
    contrast: float
    mean_saturation: float
    warnings: list[str] = Field(default_factory=list)


class PredictionItem(BaseModel):
    id: str
    fruit: str
    label: str
    status: str
    confidence: float
    summary: str
    look_for: list[str]


class PredictResponse(BaseModel):
    model_available: bool
    model_name: str | None = None
    disclaimer: str
    quality: QualityReport
    top_k: list[PredictionItem] = Field(default_factory=list)
    uncertain: bool = True
    advice: str
