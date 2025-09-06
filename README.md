# 🔍 Churn Prediction & Prevention Dashboard

A comprehensive machine learning system that predicts customer churn, explains the key factors, and suggests retention strategies using AI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/)
[![API Status](https://img.shields.io/badge/API-Online-brightgreen)](https://your-render-app.onrender.com/health)

## 📋 Project Overview

The Churn Prediction & Prevention Dashboard is an end-to-end machine learning solution designed to help businesses reduce customer churn through data-driven insights. By analyzing customer behavior patterns and service usage, the system identifies at-risk customers and provides actionable recommendations to improve retention. The intuitive Streamlit interface allows non-technical users to explore predictions, understand key drivers of churn through SHAP explanations, and generate personalized retention strategies powered by AI. Built with scalability in mind, the application features a FastAPI backend for model serving and can be easily deployed to the cloud for continuous monitoring and analysis.

## 🎯 Key Features

- **Predictive Analytics**: Machine learning model to predict churn probability
- **Explainable AI**: SHAP-based explanations for model predictions
- **Interactive Dashboard**: User-friendly interface for data exploration
- **Actionable Insights**: AI-generated retention strategies
- **Batch Processing**: Analyze multiple customers at once
- **Customer Segmentation**: Identify high-risk customer segments

## 🛠️ Tech Stack

### Backend
- **Python**: 3.10+
- **ML Framework**: LightGBM, scikit-learn, SHAP
- **API**: FastAPI, Uvicorn
- **Data Processing**: pandas, numpy, joblib

### Frontend
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **UI Components**: Streamlit-Extras

### Deployment
- **Frontend Hosting**: Streamlit Cloud
- **Backend Hosting**: Render
- **Environment Management**: python-dotenv

## 📁 Project Structure

```
churn-analysis-llm/
├── data/                            # Dataset and sample data
│   └── telco_churn.csv              # Sample churn dataset
├── src/                             # Core ML pipeline
│   ├── preprocess.py                # Data preprocessing
│   ├── train.py                     # Model training
│   ├── predict.py                   # Prediction logic
│   └── shap_explain.py              # Model explainability
├── app/                             # API layer
│   └── api.py                       # FastAPI endpoints
├── frontend/                        # Web interface
│   └── streamlit_app.py             # Interactive dashboard
├── prompts/                         # LLM prompt templates
│   ├── action_planner.md            # Retention strategies
│   └── message_generator.md         # Customer communication
├── tests/                           # Test suite
├── .github/                         # CI/CD workflows
│   └── workflows/
│       └── tests.yml
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
```

## 🚀 Live Demo

Explore the live demo: [Churn Prediction Dashboard](https://your-streamlit-app-url.streamlit.app/)

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- Git
- [Poetry](https://python-poetry.org/) (recommended) or pip

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn-analysis-llm.git
cd churn-analysis-llm
```

### 2. Install Dependencies

Using Poetry (recommended):
```bash
poetry install
```

Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py --data data/telco_churn.csv
```

This will:
- Clean and preprocess the data
- Train a LightGBM classifier
- Print performance metrics (ROC-AUC, PR-AUC, Brier Score)
- Save the model to `artifacts/model.pkl`

### 3. Start the API Server

```bash
uvicorn app.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 4. Launch the Demo UI (Optional)

```bash
streamlit run frontend/streamlit_app.py
```

## 🌐 Deployment

### Option 1: One-Click Deployment (Recommended)

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

1. Click the button above to deploy to Streamlit Cloud
2. Connect your GitHub repository
3. Set environment variables in the Streamlit Cloud settings
4. Deploy!

### Option 2: Manual Deployment

#### Backend (Render)

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env.example`

#### Frontend (Streamlit Cloud)

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app" and select your repository
3. Set the path to `frontend/streamlit_app.py`
4. Add environment variables:
   - `API_BASE_URL`: Your Render backend URL
   - Any other required variables

## 📊 API Usage

### Score Endpoint

**POST** `/score`

Request body (any subset of Telco features):
```json
{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "tenure": 5,
  "MonthlyCharges": 95.7,
  "TotalCharges": 475.2
}
```

Response:
```json
{
  "churn_probability": 0.37,
  "top_reasons": [
    {"feature": "Contract_Month-to-month", "impact": 0.19},
    {"feature": "MonthlyCharges", "impact": 0.11},
    {"feature": "TechSupport_No", "impact": 0.07}
  ]
}
```

### Example cURL

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "tenure": 5,
        "MonthlyCharges": 95.7,
        "TotalCharges": 475.2
      }'
```

## 🎨 Demo UI Features

The Streamlit app provides:

- **Upload CSV**: Analyze multiple customers from a file
- **Manual Entry**: Input customer data through forms
- **Sample Data**: Use the built-in Telco dataset
- **Risk Visualization**: Gauge charts and feature impact plots
- **Real-time Scoring**: Instant churn probability and explanations

## 🤖 LLM Integration (Future)

The project includes prompt templates for:

### Action Planner (`prompts/action_planner.md`)
Returns structured JSON with:
- `action_type`: discount, concierge_call, feature_unlock, education, no_action
- `rationale`: Decision explanation
- `expected_uplift`: Expected churn reduction (0-1)
- `message_outline`: Communication points

### Message Generator (`prompts/message_generator.md`)
Creates personalized 80-120 word retention messages based on:
- Customer name and risk profile
- Recommended action type
- Key issues and concerns

## 📈 Model Performance

The LightGBM model typically achieves:
- **ROC-AUC**: ~0.85-0.90
- **PR-AUC**: ~0.60-0.70
- **Brier Score**: ~0.15-0.20

## 🔧 Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Model settings
MODEL_PATH=artifacts/model.pkl
DATA_PATH=data/telco_churn.csv

# API settings
API_HOST=127.0.0.1
API_PORT=8000

# LLM settings (choose one)
GEMINI_API_KEY=your_gemini_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here
# OPENAI_API_KEY=your_openai_key_here
```

### LLM Provider Setup

**Gemini (Default):**
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable: `export GEMINI_API_KEY='your-key-here'`
3. Or create a `.env` file with: `GEMINI_API_KEY=your-key-here`

**DeepSeek (Alternative):**
1. Get API key from [DeepSeek](https://platform.deepseek.com/)
2. Set environment variable: `export DEEPSEEK_API_KEY='your-key-here'`

**OpenAI (Alternative):**
1. Get API key from [OpenAI](https://platform.openai.com/)
2. Set environment variable: `export OPENAI_API_KEY='your-key-here'`

## 🧪 Testing

Run individual components:

```bash
# Test preprocessing
python src/preprocess.py

# Test prediction
python src/predict.py

# Test SHAP explanations
python src/shap_explain.py
```

## 🚧 Next Steps

### Immediate Tasks
1. **Add /recommend endpoint** - LLM-powered action planning
2. **Message drafting endpoint** - Generate retention communications
3. **Unit tests** - Comprehensive test coverage
4. **Uplift modeling** - Estimate action effectiveness

### Advanced Features
1. **RAG integration** - Customer history and context
2. **A/B testing** - Measure retention action effectiveness
3. **Real-time monitoring** - Model performance tracking
4. **Multi-model ensemble** - Improve prediction accuracy

## 📝 Data Schema

The Telco dataset includes:

**Categorical Features:**
- gender, Partner, Dependents, PhoneService, MultipleLines
- InternetService, OnlineSecurity, OnlineBackup, DeviceProtection
- TechSupport, StreamingTV, StreamingMovies, Contract
- PaperlessBilling, PaymentMethod

**Numeric Features:**
- SeniorCitizen, tenure, MonthlyCharges, TotalCharges

**Target:**
- Churn (Yes/No → 1/0)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Telco Customer Churn dataset
- LightGBM and SHAP libraries
- FastAPI and Streamlit communities

## 📊 Project Impact

This project demonstrates:
- End-to-end ML pipeline development
- Model explainability with SHAP
- Production API development with FastAPI
- Interactive dashboard with Streamlit
- Cloud deployment best practices

## 👨‍💻 Author

[Your Name] - [Your Email]

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/yourusername)

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ using Python, FastAPI, and Streamlit
</div>
