"""
Unit tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from models import (
    CustomerData, FeatureImpact, ChurnScoreResponse, ActionPlan,
    RecommendRequest, RecommendResponse, MessageDraftRequest, MessageDraftResponse
)


class TestCustomerData:
    """Test CustomerData model."""
    
    def test_valid_customer_data(self):
        """Test valid customer data."""
        data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "Contract": "Month-to-month"
        }
        
        customer = CustomerData(**data)
        
        assert customer.gender == "Male"
        assert customer.SeniorCitizen == 0
        assert customer.tenure == 12
        assert customer.MonthlyCharges == 50.0
        assert customer.Contract == "Month-to-month"
    
    def test_customer_data_optional_fields(self):
        """Test customer data with optional fields."""
        data = {
            "tenure": 12
        }
        
        customer = CustomerData(**data)
        
        assert customer.tenure == 12
        assert customer.gender is None
        assert customer.MonthlyCharges is None


class TestFeatureImpact:
    """Test FeatureImpact model."""
    
    def test_valid_feature_impact(self):
        """Test valid feature impact."""
        impact = FeatureImpact(feature="Contract_Month-to-month", impact=0.5)
        
        assert impact.feature == "Contract_Month-to-month"
        assert impact.impact == 0.5
    
    def test_negative_impact(self):
        """Test negative impact value."""
        impact = FeatureImpact(feature="tenure", impact=-0.3)
        
        assert impact.impact == -0.3


class TestChurnScoreResponse:
    """Test ChurnScoreResponse model."""
    
    def test_valid_churn_score_response(self):
        """Test valid churn score response."""
        response = ChurnScoreResponse(
            churn_probability=0.75,
            top_reasons=[
                FeatureImpact(feature="Contract_Month-to-month", impact=0.5),
                FeatureImpact(feature="tenure", impact=-0.3)
            ]
        )
        
        assert response.churn_probability == 0.75
        assert len(response.top_reasons) == 2
        assert response.top_reasons[0].feature == "Contract_Month-to-month"


class TestActionPlan:
    """Test ActionPlan model."""
    
    def test_valid_action_plan(self):
        """Test valid action plan."""
        plan = ActionPlan(
            action_type="concierge_call",
            rationale="High-risk customer needs personal attention",
            expected_uplift=0.15,
            message_outline=["Acknowledge concerns", "Offer solutions", "Schedule call"]
        )
        
        assert plan.action_type == "concierge_call"
        assert plan.expected_uplift == 0.15
        assert len(plan.message_outline) == 3
    
    def test_invalid_action_type(self):
        """Test invalid action type."""
        with pytest.raises(ValidationError):
            ActionPlan(
                action_type="invalid_action",
                rationale="Test",
                expected_uplift=0.1,
                message_outline=["Test"]
            )
    
    def test_invalid_expected_uplift(self):
        """Test invalid expected uplift values."""
        # Test negative value
        with pytest.raises(ValidationError):
            ActionPlan(
                action_type="discount",
                rationale="Test",
                expected_uplift=-0.1,
                message_outline=["Test"]
            )
        
        # Test value > 1
        with pytest.raises(ValidationError):
            ActionPlan(
                action_type="discount",
                rationale="Test",
                expected_uplift=1.5,
                message_outline=["Test"]
            )
    
    def test_message_outline_length_validation(self):
        """Test message outline length validation."""
        # Test too few items
        with pytest.raises(ValidationError):
            ActionPlan(
                action_type="discount",
                rationale="Test",
                expected_uplift=0.1,
                message_outline=["Only one item"]
            )
        
        # Test too many items
        with pytest.raises(ValidationError):
            ActionPlan(
                action_type="discount",
                rationale="Test",
                expected_uplift=0.1,
                message_outline=["1", "2", "3", "4", "5", "6", "7"]
            )


class TestRecommendRequest:
    """Test RecommendRequest model."""
    
    def test_valid_recommend_request(self):
        """Test valid recommend request."""
        request = RecommendRequest(
            churn_probability=0.75,
            top_reasons=[
                FeatureImpact(feature="Contract_Month-to-month", impact=0.5)
            ],
            customer_segment="high_value",
            uplift_by_action={"discount": 0.15, "concierge_call": 0.25},
            rag_summary="Customer has technical issues"
        )
        
        assert request.churn_probability == 0.75
        assert request.customer_segment == "high_value"
        assert "discount" in request.uplift_by_action
        assert request.rag_summary == "Customer has technical issues"
    
    def test_recommend_request_optional_fields(self):
        """Test recommend request with optional fields."""
        request = RecommendRequest(
            churn_probability=0.5,
            top_reasons=[FeatureImpact(feature="tenure", impact=0.3)]
        )
        
        assert request.customer_segment is None
        assert request.uplift_by_action is None
        assert request.rag_summary is None


class TestMessageDraftRequest:
    """Test MessageDraftRequest model."""
    
    def test_valid_message_draft_request(self):
        """Test valid message draft request."""
        request = MessageDraftRequest(
            customer_name="John Doe",
            action_type="concierge_call",
            message_outline=["Acknowledge concerns", "Offer solutions"],
            churn_probability=0.75,
            top_reasons=[FeatureImpact(feature="Contract_Month-to-month", impact=0.5)]
        )
        
        assert request.customer_name == "John Doe"
        assert request.action_type == "concierge_call"
        assert len(request.message_outline) == 2
        assert request.churn_probability == 0.75


class TestMessageDraftResponse:
    """Test MessageDraftResponse model."""
    
    def test_valid_message_draft_response(self):
        """Test valid message draft response."""
        response = MessageDraftResponse(
            message="Hello John, we understand your concerns...",
            word_count=15,
            tone="empathetic",
            call_to_action="contact us"
        )
        
        assert response.word_count == 15
        assert response.tone == "empathetic"
        assert response.call_to_action == "contact us"


if __name__ == "__main__":
    pytest.main([__file__])
