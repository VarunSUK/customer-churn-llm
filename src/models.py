"""
Pydantic models for LLM responses and API data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class CustomerData(BaseModel):
    """Customer data model for prediction requests."""
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None


class FeatureImpact(BaseModel):
    """Feature impact model for SHAP explanations."""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="SHAP impact value")


class ChurnScoreResponse(BaseModel):
    """Response model for churn scoring."""
    churn_probability: float = Field(..., description="Churn probability (0-1)")
    top_reasons: List[FeatureImpact] = Field(..., description="Top contributing features")


class ActionPlan(BaseModel):
    """LLM response model for action planning."""
    action_type: Literal["discount", "concierge_call", "feature_unlock", "education", "no_action"] = Field(
        ..., description="Recommended action type"
    )
    rationale: str = Field(..., description="Clear explanation of why this action was chosen")
    expected_uplift: float = Field(..., ge=0.0, le=1.0, description="Expected reduction in churn probability")
    message_outline: List[str] = Field(..., min_length=2, max_length=6, description="Communication points")


class RecommendRequest(BaseModel):
    """Request model for action recommendation."""
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Customer churn probability")
    top_reasons: List[FeatureImpact] = Field(..., description="Top contributing features")
    customer_segment: Optional[str] = Field(None, description="Customer segment (optional)")
    uplift_by_action: Optional[Dict[str, float]] = Field(None, description="Historical uplift data by action")
    rag_summary: Optional[str] = Field(None, description="Recent issues summary from RAG")


class RecommendResponse(BaseModel):
    """Response model for action recommendation."""
    action_plan: ActionPlan = Field(..., description="Recommended action plan")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    alternative_actions: Optional[List[ActionPlan]] = Field(None, description="Alternative action options")


class MessageDraftRequest(BaseModel):
    """Request model for message drafting."""
    customer_name: str = Field(..., description="Customer name")
    action_type: Literal["discount", "concierge_call", "feature_unlock", "education", "no_action"] = Field(
        ..., description="Action type to draft message for"
    )
    message_outline: List[str] = Field(..., description="Key points to include in message")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Customer churn probability")
    top_reasons: List[FeatureImpact] = Field(..., description="Top contributing features")


class MessageDraftResponse(BaseModel):
    """Response model for message drafting."""
    message: str = Field(..., description="Drafted message (80-120 words)")
    word_count: int = Field(..., description="Actual word count")
    tone: str = Field(..., description="Message tone (empathetic, professional, etc.)")
    call_to_action: str = Field(..., description="Primary call to action")


class UpliftData(BaseModel):
    """Model for uplift modeling data."""
    action_type: str = Field(..., description="Action type")
    uplift_value: float = Field(..., ge=0.0, le=1.0, description="Expected uplift value")
    confidence_interval: Optional[tuple] = Field(None, description="Confidence interval (lower, upper)")
    sample_size: Optional[int] = Field(None, description="Sample size for uplift calculation")


class CustomerProfile(BaseModel):
    """Extended customer profile for comprehensive analysis."""
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Churn probability")
    top_reasons: List[FeatureImpact] = Field(..., description="Top contributing features")
    customer_segment: Optional[str] = Field(None, description="Customer segment")
    tenure_months: Optional[int] = Field(None, description="Customer tenure in months")
    monthly_charges: Optional[float] = Field(None, description="Monthly charges")
    total_charges: Optional[float] = Field(None, description="Total charges")
    contract_type: Optional[str] = Field(None, description="Contract type")
    last_interaction: Optional[str] = Field(None, description="Last customer interaction date")
    rag_summary: Optional[str] = Field(None, description="Recent issues summary")
