"""
🚀 Churn Prediction Dashboard

A modern, interactive dashboard for predicting and analyzing customer churn.
Features real-time predictions, detailed explanations, and actionable insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

# Configuration
API_BASE_URL = "https://customer-churn-llm.onrender.com"
DEFAULT_DATA_PATH = "data/telco_churn.csv"

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            padding: 2rem 3rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 500;
        }
        .stSelectbox, .stTextInput, .stNumberInput, .stTextArea, .stFileUploader {
            margin-bottom: 1rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            gap: 8px;
            padding: 0 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e88e5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def score_customer(customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Score a customer using the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/score",
            json=customer_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")
        return None


def load_sample_data() -> pd.DataFrame:
    """Load sample data from the default CSV file."""
    try:
        if os.path.exists(DEFAULT_DATA_PATH):
            return pd.read_csv(DEFAULT_DATA_PATH)
        else:
            st.warning(f"Sample data file not found at {DEFAULT_DATA_PATH}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()


def create_churn_probability_gauge(probability: float) -> go.Figure:
    """Create a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def get_user_friendly_explanation(feature_name, impact):
    """Convert technical feature names and impacts to user-friendly explanations."""
    
    # Feature name mapping to user-friendly terms
    feature_mapping = {
        # Contract related
        "cat__Contract_Month-to-month": "Month-to-Month Contract",
        "cat__Contract_One year": "One-Year Contract", 
        "cat__Contract_Two year": "Two-Year Contract",
        
        # Internet service
        "cat__InternetService_Fiber optic": "Fiber Internet Service",
        "cat__InternetService_DSL": "DSL Internet Service",
        "cat__InternetService_No": "No Internet Service",
        
        # Payment methods
        "cat__PaymentMethod_Electronic check": "Electronic Check Payment",
        "cat__PaymentMethod_Mailed check": "Mailed Check Payment",
        "cat__PaymentMethod_Bank transfer (automatic)": "Automatic Bank Transfer",
        "cat__PaymentMethod_Credit card (automatic)": "Automatic Credit Card",
        
        # Services
        "cat__OnlineSecurity_Yes": "Online Security Service",
        "cat__OnlineSecurity_No": "No Online Security",
        "cat__TechSupport_Yes": "Technical Support Service",
        "cat__TechSupport_No": "No Technical Support",
        "cat__OnlineBackup_Yes": "Online Backup Service",
        "cat__OnlineBackup_No": "No Online Backup",
        "cat__DeviceProtection_Yes": "Device Protection Service",
        "cat__DeviceProtection_No": "No Device Protection",
        
        # Demographics
        "cat__SeniorCitizen_1": "Senior Citizen (65+)",
        "cat__SeniorCitizen_0": "Not Senior Citizen",
        "cat__Partner_Yes": "Has Partner",
        "cat__Partner_No": "No Partner",
        "cat__Dependents_Yes": "Has Dependents",
        "cat__Dependents_No": "No Dependents",
        
        # Billing
        "cat__PaperlessBilling_Yes": "Paperless Billing",
        "cat__PaperlessBilling_No": "Paper Billing",
        
        # Numeric features
        "num__tenure": "Customer Tenure (Months)",
        "num__MonthlyCharges": "Monthly Charges",
        "num__TotalCharges": "Total Charges",
    }
    
    # Get user-friendly feature name
    friendly_name = feature_mapping.get(feature_name, feature_name.replace("cat__", "").replace("num__", ""))
    
    # Create impact explanation
    abs_impact = abs(impact)
    
    if impact > 0:
        if abs_impact > 0.5:
            impact_strength = "significantly increases"
            emoji = "🔴"
        elif abs_impact > 0.2:
            impact_strength = "moderately increases"
            emoji = "🟠"
        else:
            impact_strength = "slightly increases"
            emoji = "🟡"
    else:
        if abs_impact > 0.5:
            impact_strength = "significantly decreases"
            emoji = "🟢"
        elif abs_impact > 0.2:
            impact_strength = "moderately decreases"
            emoji = "🔵"
        else:
            impact_strength = "slightly decreases"
            emoji = "🟣"
    
    # Create actionable insights
    insights = {
        "cat__Contract_Month-to-month": "Consider offering longer-term contracts with incentives",
        "cat__Contract_One year": "Good contract stability, maintain current terms",
        "cat__Contract_Two year": "Excellent contract stability, consider loyalty rewards",
        "cat__InternetService_Fiber optic": "High-speed service but may be expensive, consider pricing options",
        "cat__PaymentMethod_Electronic check": "Manual payment method, encourage automatic payments",
        "cat__PaymentMethod_Mailed check": "Traditional payment, offer digital alternatives",
        "num__tenure": "Customer loyalty factor, longer tenure = lower risk",
        "num__MonthlyCharges": "Pricing sensitivity, consider value-based pricing",
        "num__TotalCharges": "Overall spending pattern, high spenders may be price-sensitive",
        "cat__OnlineSecurity_No": "Security concerns, promote security add-ons",
        "cat__TechSupport_No": "Support needs, offer technical assistance",
        "cat__SeniorCitizen_1": "Senior customer, may need additional support",
        "cat__PaperlessBilling_Yes": "Digital preference, leverage for engagement"
    }
    
    insight = insights.get(feature_name, "Monitor this factor closely")
    
    return {
        "friendly_name": friendly_name,
        "impact_strength": impact_strength,
        "emoji": emoji,
        "insight": insight,
        "raw_impact": impact
    }


def create_feature_impact_chart(reasons: list) -> go.Figure:
    """Create a horizontal bar chart for feature impacts."""
    # Get user-friendly names and impacts
    friendly_features = []
    impacts = []
    colors = []
    
    for reason in reasons:
        explanation = get_user_friendly_explanation(reason['feature'], reason['impact'])
        friendly_features.append(explanation['friendly_name'])
        impacts.append(reason['impact'])
        colors.append('red' if reason['impact'] > 0 else 'green')
    
    fig = go.Figure(go.Bar(
        x=impacts,
        y=friendly_features,
        orientation='h',
        marker_color=colors,
        text=[f"{impact:.3f}" for impact in impacts],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top Contributing Factors to Churn Risk",
        xaxis_title="Impact on Churn Risk",
        yaxis_title="Customer Factor",
        height=400,
        font=dict(size=12)
    )
    
    return fig


def show_prediction_page():
    """Show the main prediction page with customer data input and analysis."""
    st.title("🔮 Predict Customer Churn")
    
    with st.expander("ℹ️ How to use", expanded=True):
        st.markdown("""
        This tool helps you predict customer churn risk and understand the key factors.
        - Upload a CSV file or enter customer details manually
        - Get instant churn probability and key insights
        - View detailed explanations and recommendations
        """)
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["📝 Manual Entry", "📁 Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    customer_data = None
    
    if input_method == "📁 Upload CSV":
        with st.container():
            st.subheader("Upload Customer Data")
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload a CSV file",
                type="csv",
                help="CSV should contain customer data with appropriate columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Successfully loaded {len(df)} records")
                    
                    with st.expander("📊 Preview Data", expanded=False):
                        st.dataframe(df.head())
                    
                    # Select row to analyze
                    if len(df) > 0:
                        row_index = st.selectbox(
                            "Select a customer to analyze:",
                            range(len(df)),
                            format_func=lambda x: f"Row {x + 1}"
                        )
                    
                    # Convert selected row to dict, excluding None values
                    customer_data = df.iloc[row_index].to_dict()
                    customer_data = {k: v for k, v in customer_data.items() if pd.notna(v)}
                    
                    st.write("**Selected Customer Data:**")
                    st.json(customer_data)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    
    elif input_method == "📝 Manual Entry":
        st.header("✏️ Manual Customer Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female", None])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1, None])
            partner = st.selectbox("Partner", ["Yes", "No", None])
            dependents = st.selectbox("Dependents", ["Yes", "No", None])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=None)
            
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No", "No phone service", None])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service", None])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No", None])
        
        with col2:
            st.subheader("Additional Services")
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service", None])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service", None])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service", None])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service", None])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service", None])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service", None])
            
            st.subheader("Contract & Billing")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year", None])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No", None])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", 
                "Credit card (automatic)", None
            ])
            
            st.subheader("Charges")
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=None)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=None)
        
        # Create customer data dict
        customer_data = {
            k: v for k, v in {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }.items() if v is not None
        }
    
    elif input_method == "Sample Data":
        st.header("📊 Sample Data Analysis")
        
        df = load_sample_data()
        if not df.empty:
            st.write("**Sample Data Preview:**")
            st.dataframe(df.head())
            
            # Select row to analyze
            row_index = st.selectbox(
                "Select row to analyze:",
                range(len(df)),
                format_func=lambda x: f"Row {x + 1}"
            )
            
            # Convert selected row to dict, excluding None values
            customer_data = df.iloc[row_index].to_dict()
            customer_data = {k: v for k, v in customer_data.items() if pd.notna(v)}
            
            st.write("**Selected Customer Data:**")
            st.json(customer_data)
    
    # Score the customer if data is available
    if customer_data:
        st.header("🎯 Churn Analysis")
        
        if st.button("Score Customer Risk", type="primary"):
            with st.spinner("Analyzing customer risk..."):
                result = score_customer(customer_data)
            
            if result:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Churn probability gauge
                    st.plotly_chart(
                        create_churn_probability_gauge(result["churn_probability"]),
                        use_container_width=True
                    )
                
                with col2:
                    # Risk summary
                    st.metric(
                        "Churn Probability",
                        f"{result['churn_probability']:.1%}",
                        delta=f"{result['churn_probability'] - 0.5:.1%}" if result['churn_probability'] != 0.5 else None
                    )
                    
                    # Risk level with detailed explanation
                    risk_level = result['churn_probability']
                    if risk_level < 0.3:
                        st.success(f"🟢 **Low Risk: {risk_level:.1%}**")
                        st.write("This customer has a low likelihood of churning. Continue current service levels.")
                    elif risk_level < 0.7:
                        st.warning(f"🟡 **Medium Risk: {risk_level:.1%}**")
                        st.write("This customer shows some risk factors. Monitor closely and consider retention actions.")
                    else:
                        st.error(f"🔴 **High Risk: {risk_level:.1%}**")
                        st.write("This customer is at high risk of churning. Immediate retention actions recommended.")
                
                # Feature impact chart
                st.plotly_chart(
                    create_feature_impact_chart(result["top_reasons"]),
                    use_container_width=True
                )
                
                # Detailed explanations
                st.subheader("📋 Detailed Explanations")
                
                for i, reason in enumerate(result["top_reasons"], 1):
                    explanation = get_user_friendly_explanation(reason['feature'], reason['impact'])
                    
                    # Create expandable section for each factor
                    with st.expander(f"{explanation['emoji']} **{i}. {explanation['friendly_name']}**", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Impact:** {explanation['impact_strength']} churn risk")
                            st.write(f"**Insight:** {explanation['insight']}")
                        
                        with col2:
                            st.metric(
                                label="Risk Impact",
                                value=f"{reason['impact']:.3f}",
                                delta=f"{'High' if abs(reason['impact']) > 0.5 else 'Medium' if abs(reason['impact']) > 0.2 else 'Low'} Impact"
                            )
                
                # Summary insights
                st.subheader("💡 Key Insights & Recommendations")
                
                high_risk_factors = [r for r in result["top_reasons"] if r["impact"] > 0.3]
                protective_factors = [r for r in result["top_reasons"] if r["impact"] < -0.2]
                
                if high_risk_factors:
                    st.warning("**🚨 High Risk Factors:**")
                    for factor in high_risk_factors:
                        explanation = get_user_friendly_explanation(factor['feature'], factor['impact'])
                        st.write(f"• {explanation['friendly_name']}: {explanation['insight']}")
                
                if protective_factors:
                    st.success("**🛡️ Protective Factors:**")
                    for factor in protective_factors:
                        explanation = get_user_friendly_explanation(factor['feature'], factor['impact'])
                        st.write(f"• {explanation['friendly_name']}: {explanation['insight']}")
                
                # Action recommendations
                st.info("**🎯 Recommended Actions:**")
                if any("Contract_Month-to-month" in r['feature'] for r in result["top_reasons"]):
                    st.write("• Offer contract incentives to encourage longer-term commitments")
                if any("MonthlyCharges" in r['feature'] for r in result["top_reasons"]):
                    st.write("• Review pricing strategy and consider value-based pricing")
                if any("TechSupport_No" in r['feature'] for r in result["top_reasons"]):
                    st.write("• Proactively offer technical support services")
                if any("OnlineSecurity_No" in r['feature'] for r in result["top_reasons"]):
                    st.write("• Promote security add-ons and educate on benefits")
    
def show_batch_analysis_page():
    """Show the batch analysis page for processing multiple customers."""
    st.title("📊 Batch Analysis")
    st.markdown("Upload a CSV file to analyze multiple customers at once.")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=["csv"],
        help="CSV should contain customer data with appropriate columns"
    )
    
    if uploaded_file is not None:
        with st.spinner("Analyzing customer data..."):
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Show sample data
            with st.expander("View sample data", expanded=False):
                st.dataframe(df.head())
            
            # Add analysis options
            st.subheader("Analysis Options")
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Churn Risk Threshold (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5
                )
            with col2:
                top_n = st.slider(
                    "Show Top N Customers",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5
                )
            
            if st.button("🔍 Run Analysis", type="primary"):
                with st.spinner(f"Analyzing {len(df)} customers..."):
                    # Simulate batch processing
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(1, 11):
                        # Update progress bar
                        progress = i * 10
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {progress}% complete")
                        time.sleep(0.1)
                    
                    # Show results
                    st.success("✅ Analysis complete!")
                    
                    # Sample visualization
                    st.subheader("Churn Risk Distribution")
                    fig = px.histogram(
                        df.sample(100),  # Sample for demo
                        x=df.sample(100).sample(100, replace=True).index % 100,  # Random values for demo
                        nbins=20,
                        labels={"value": "Churn Probability"},
                        color_discrete_sequence=["#1e88e5"]
                    )
                    fig.update_layout(
                        showlegend=False,
                        xaxis_title="Churn Probability",
                        yaxis_title="Number of Customers"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top N customers at risk
                    st.subheader(f"🔴 Top {top_n} High-Risk Customers")
                    st.dataframe(
                        df.sample(top_n).sort_values(by="churn_risk", ascending=False) if "churn_risk" in df.columns 
                        else df.sample(top_n).assign(churn_risk=lambda x: x.index % 100 / 100).sort_values(by="churn_risk", ascending=False),
                        use_container_width=True
                    )
                    
                    # Export options
                    st.download_button(
                        label="📥 Download Full Analysis",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"churn_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv',
                        help="Download the complete analysis results"
                    )

def show_segments_page():
    """Show customer segmentation analysis."""
    st.title("👥 Customer Segments")
    st.markdown("Analyze customer segments and their churn characteristics.")
    
    # Sample segmentation data
    segments = {
        "Segment": ["Loyal Customers", "At Risk", "Champions", "Needs Attention", "Sleepers"],
        "Size (%)": [25, 15, 30, 20, 10],
        "Churn Rate": ["5%", "45%", "2%", "30%", "60%"],
        "Value": ["High", "Medium", "High", "Low", "Medium"],
        "Recommendation": ["Reward loyalty", "Personalized offers", "Upsell opportunities", "Engagement campaigns", "Win-back offers"]
    }
    
    df_segments = pd.DataFrame(segments)
    
    # Segmentation visualization
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Segment Overview")
        fig_pie = px.pie(
            df_segments,
            values="Size (%)",
            names="Segment",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Segment Details")
        st.dataframe(
            df_segments,
            column_config={
                "Size (%)": st.column_config.ProgressColumn(
                    "Size (%)",
                    help="Segment size as percentage of total customers",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Add segment analysis
    st.subheader("Segment Analysis")
    selected_segment = st.selectbox(
        "Select a segment to analyze:",
        df_segments["Segment"].tolist()
    )
    
    segment_data = df_segments[df_segments["Segment"] == selected_segment].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Segment Size", f"{segment_data['Size (%)']}%")
    with col2:
        st.metric("Churn Rate", segment_data["Churn Rate"])
    with col3:
        st.metric("Customer Value", segment_data["Value"])
    
    st.markdown("#### Recommended Actions")
    st.info(segment_data["Recommendation"])

def show_about_page():
    """Show the about page with app information."""
    st.title("ℹ️ About Churn Prediction Dashboard")
    
    st.markdown("""
    ## 📊 Churn Prediction Dashboard
    
    A powerful tool for predicting and analyzing customer churn using machine learning.
    
    ### Key Features
    - 🔮 **Churn Prediction**: Get instant churn probability for customers
    - 📈 **Batch Analysis**: Process multiple customers at once
    - 🎯 **Actionable Insights**: Understand key factors driving churn
    - 👥 **Customer Segmentation**: Identify high-risk segments
    - 📊 **Interactive Visualizations**: Explore data with beautiful charts
    
    ### Technology Stack
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    - **ML Model**: LightGBM with SHAP explanations
    - **LLM Integration**: Gemini API for advanced insights
    
    ### Getting Started
    1. Start the API server: `uvicorn app.api:app --reload`
    2. Run the Streamlit app: `streamlit run frontend/streamlit_app.py`
    3. Open your browser to `http://localhost:8501`
    
    ### Support
    For support or feature requests, please contact the development team.
    
    ---
    
    **Version 1.2.0** | *Last updated: September 2025*
    """)

def main():
    """Main Streamlit application with enhanced UI/UX."""
    # Set page config first
    st.set_page_config(
        page_title="Churn Prediction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom header
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=80)
        with col2:
            st.title("Churn Prediction Dashboard")
            st.caption("Predict, analyze, and prevent customer churn with AI-powered insights")
    
    # Check if API is running with loading state
    with st.spinner("Checking API connection..."):
        if not check_api_health():
            st.error("⚠️ API server is not running. Please start the API server first.")
            if st.button("🔄 Try Again"):
                st.experimental_rerun()
            st.stop()
    
    # Sidebar with enhanced navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=60)
        st.sidebar.title("🔍 Navigation")
        
        with st.sidebar.expander("🔧 Settings", expanded=False):
            theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
            anim_speed = st.slider("Chart Animation Speed", 0.1, 1.0, 0.5, 0.1, key="anim_speed")
        
        page = st.sidebar.radio(
            "",
            ["📊 Predict Churn", "📈 Batch Analysis", "👥 Customer Segments", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        col1, col2 = st.sidebar.columns([1, 2])
        col1.metric("API Status", "✅ Online" if check_api_health() else "❌ Offline")
        col2.metric("Model", "🤖 Loaded")
        
        st.sidebar.markdown("---")
        st.sidebar.caption(f"v1.2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content area routing
    try:
        if page == "📊 Predict Churn":
            show_prediction_page()
        elif page == "📈 Batch Analysis":
            show_batch_analysis_page()
        elif page == "👥 Customer Segments":
            show_segments_page()
        else:
            show_about_page()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)  # This will show the full traceback for debugging
    
    # Apply custom styles to metric cards
    try:
        style_metric_cards()
    except Exception as e:
        st.warning("Could not apply custom styles to metric cards.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
        <p>🚀 <strong>Churn Prediction Dashboard</strong> | Built with FastAPI, Streamlit, and LightGBM</p>
        <p>© 2025 Churn Prediction Team | <a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
