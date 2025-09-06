"""
FastAPI application for churn prediction and explanation.
Provides /score endpoint that returns churn probability and top SHAP reasons.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from predict import ChurnPredictor
from shap_explain import ChurnExplainer
from models import (
    CustomerData, FeatureImpact, ChurnScoreResponse, 
    RecommendRequest, RecommendResponse, MessageDraftRequest, MessageDraftResponse
)
from llm_client import llm_client


# Pydantic models are now imported from src/models.py


# Initialize FastAPI app
app = FastAPI(
    title="Churn Copilot API",
    description="API for churn prediction and explanation",
    version="1.0.0"
)

# Global variables for model components
predictor = None
explainer = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model components on startup."""
    global predictor, explainer
    
    model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    
    try:
        predictor = ChurnPredictor(model_path)
        explainer = ChurnExplainer(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py --data data/telco_churn.csv")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Copilot API",
        "version": "1.0.0",
        "endpoints": {
            "/score": "POST - Get churn probability and explanations",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "healthy", "model_loaded": True}


@app.post("/score", response_model=ChurnScoreResponse)
async def score_customer(customer_data: CustomerData):
    """
    Score a customer for churn risk and provide explanations.
    
    Args:
        customer_data: Customer features
        
    Returns:
        Churn probability and top contributing features
    """
    if predictor is None or explainer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert Pydantic model to dict, excluding None values
        customer_dict = customer_data.dict(exclude_none=True)
        
        if not customer_dict:
            raise HTTPException(
                status_code=400, 
                detail="No customer data provided"
            )
        
        # Get churn probability
        churn_probability = predictor.predict_proba(customer_dict)
        
        # Get SHAP explanations
        explanations = explainer.explain_prediction(customer_dict, top_k=5)
        
        # Convert to response format
        top_reasons = [
            FeatureImpact(feature=exp["feature"], impact=exp["impact"])
            for exp in explanations
        ]
        
        return ChurnScoreResponse(
            churn_probability=churn_probability,
            top_reasons=top_reasons
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature names
        feature_names = predictor.feature_names
        
        return {
            "model_path": predictor.model_path,
            "num_features": len(feature_names) if feature_names else 0,
            "feature_names": feature_names[:10] if feature_names else [],  # Show first 10
            "model_type": "LightGBM Pipeline"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_action(request: RecommendRequest):
    """
    Recommend a retention action using LLM analysis.
    
    Args:
        request: Recommendation request with customer risk profile
        
    Returns:
        Recommended action plan with confidence score
    """
    if not llm_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Please configure OPENAI_API_KEY."
        )
    
    try:
        # Convert FeatureImpact objects to dicts for LLM
        top_reasons = [{"feature": r.feature, "impact": r.impact} for r in request.top_reasons]
        
        # Generate action plan using LLM
        action_plan = llm_client.plan_action(
            churn_probability=request.churn_probability,
            top_reasons=top_reasons,
            customer_segment=request.customer_segment,
            uplift_by_action=request.uplift_by_action,
            rag_summary=request.rag_summary
        )
        
        # Calculate confidence based on expected uplift and risk level
        confidence = min(0.9, action_plan.expected_uplift + (request.churn_probability * 0.3))
        
        return RecommendResponse(
            action_plan=action_plan,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendation: {str(e)}"
        )


@app.post("/draft_message", response_model=MessageDraftResponse)
async def draft_message(request: MessageDraftRequest):
    """
    Draft a personalized customer message using LLM.
    
    Args:
        request: Message drafting request with customer info and action details
        
    Returns:
        Drafted message with metadata
    """
    if not llm_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Please configure OPENAI_API_KEY."
        )
    
    try:
        # Convert FeatureImpact objects to dicts for LLM
        top_reasons = [{"feature": r.feature, "impact": r.impact} for r in request.top_reasons]
        
        # Generate message using LLM
        message_response = llm_client.draft_message(
            customer_name=request.customer_name,
            action_type=request.action_type,
            message_outline=request.message_outline,
            churn_probability=request.churn_probability,
            top_reasons=top_reasons
        )
        
        return message_response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error drafting message: {str(e)}"
        )


@app.get("/llm_status")
async def llm_status():
    """Check LLM service availability."""
    return {
        "available": llm_client.is_available(),
        "model": llm_client.model if llm_client.is_available() else None,
        "provider": llm_client.provider if llm_client.is_available() else None,
        "api_key_configured": llm_client.api_key is not None
    }


def main():
    """Run the FastAPI application."""
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
