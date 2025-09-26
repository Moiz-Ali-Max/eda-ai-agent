# 🚀 Professional EDA AI Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-brightgreen.svg)](https://plotly.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini-2.0--flash-orange.svg)](https://ai.google.dev/)
[![Chainlit](https://img.shields.io/badge/Chainlit-UI-purple.svg)](https://chainlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An AI-powered Exploratory Data Analysis platform that combines advanced data visualization with intelligent insights using Google's Gemini AI.

## 🎯 Overview

This professional-grade EDA tool transforms raw CSV data into comprehensive analytical reports with interactive visualizations and AI-generated insights. Built for data scientists, analysts, and business professionals who need quick, professional data analysis.

## ✨ Key Features

### 📊 **Advanced Analytics**
- **Interactive Correlation Analysis** - Heatmaps with statistical significance
- **Distribution Analysis** - Multi-variate statistical profiling with outlier detection
- **Categorical Data Profiling** - Frequency analysis and pattern detection
- **Relationship Discovery** - Scatter plots with trend lines and correlation finding
- **Professional Outlier Detection** - Box plots with statistical insights

### 🤖 **AI-Powered Insights**
- **Executive Summary Generation** - Business-ready overview reports
- **Context-Aware Analysis** - Specific insights for each visualization type
- **Professional Recommendations** - Actionable business intelligence
- **Comprehensive Final Reports** - Complete analysis with next steps

### 🎨 **Professional User Experience**
- **5-Phase Analysis Workflow** - Clear progress tracking
- **Interactive Visualizations** - Plotly-powered charts and graphs
- **Real-time Processing** - Async operations with status updates
- **Professional Styling** - Corporate-grade appearance
- **Multi-encoding Support** - Handles various CSV file formats

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.8+ | Core application logic |
| **AI Engine** | Google Gemini 2.0 Flash | Natural language insights |
| **Visualizations** | Plotly + Matplotlib + Seaborn | Interactive and static charts |
| **Web Framework** | Chainlit | Real-time streaming UI |
| **Data Processing** | Pandas + NumPy + SciPy | Statistical analysis |
| **ML Components** | Scikit-learn | Advanced analytics |
| **Async Processing** | AsyncIO | Concurrent operations |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (optional but recommended for AI insights)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/eda-ai-agent.git
cd eda-ai-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (Optional for AI features)
```bash
# Windows
set GEMINI_API_KEY=your_gemini_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key_here
```

4. **Run the application**
```bash
chainlit run app.py
```

5. **Access the web interface**
Open your browser and go to `http://localhost:8000`

## 📖 Usage

### Basic Workflow

1. **Upload CSV File** - Drop your data file into the interface
2. **Automatic Processing** - Watch as the system analyzes your data in 5 phases:
   - Phase 1: Data validation and encoding detection
   - Phase 2: Comprehensive data overview generation
   - Phase 3: AI-powered executive summary (if Gemini API is available)
   - Phase 4: Interactive visualizations creation
   - Phase 5: Final comprehensive report generation

3. **Explore Results** - Interactive charts, AI insights, and professional reports

### Sample Analysis Output

The system generates:
- 📋 **Data Quality Assessment** - Missing values, duplicates, memory usage
- 🔥 **Interactive Correlation Matrix** - Relationship patterns
- 📊 **Distribution Analysis** - Statistical profiling with outliers
- 🏷️ **Categorical Analysis** - Frequency patterns and insights
- 🎯 **Outlier Detection** - Statistical anomaly identification
- 💫 **Relationship Analysis** - Trend lines and correlations
- 📋 **Executive Summary** - Business-ready insights
- 🚀 **Recommendations** - Actionable next steps

## 🏗️ Project Structure

```
eda-ai-agent/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .chainlit/            # Chainlit configuration
└── .files/               # Temporary file storage (auto-created)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI insights | Optional* |

*Without the API key, the tool still provides comprehensive data analysis and visualizations, but without AI-generated insights.

### Getting Gemini API Key

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create a new API key
3. Set it as an environment variable

## 📊 Supported Data Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx) - via CSV conversion
- **Multiple encodings** - UTF-8, Latin-1, CP1252, ISO-8859-1
- **Various separators** - Comma, semicolon, tab-delimited

## 🎨 Visualization Types

### Interactive Charts (Plotly)
- **Correlation Heatmaps** - With hover information and color coding
- **Distribution Histograms** - Multi-panel analysis with statistical markers
- **Box Plots** - Outlier detection with quartile information
- **Scatter Plots** - With trend lines and correlation coefficients
- **Bar Charts** - Categorical analysis with percentages

### Professional Styling
- Corporate color schemes
- Responsive design
- Mobile-friendly interface
- High-DPI support for presentations

## 🔍 Technical Features

### Data Processing Pipeline
```python
# Async processing flow
CSV Upload → Encoding Detection → Data Validation → 
Statistical Analysis → Visualization Generation → 
AI Insight Generation → Report Compilation
```

### Advanced Analytics
- **Correlation Analysis** - Pearson correlation with significance testing
- **Distribution Testing** - Normality tests and skewness analysis
- **Outlier Detection** - IQR and Z-score methods
- **Missing Value Analysis** - Pattern detection and impact assessment
- **Data Quality Scoring** - Completeness and consistency metrics

### AI Integration
- **Context-Aware Prompts** - Different prompts for different analysis types
- **Error Handling** - Graceful fallbacks when AI is unavailable
- **Rate Limiting** - Responsible API usage
- **Response Validation** - Ensuring quality insights

## 🚦 Error Handling

The application includes comprehensive error handling for:
- ❌ **File Format Issues** - Unsupported formats or corrupted files
- ❌ **Encoding Problems** - Automatic encoding detection and fallback
- ❌ **API Limitations** - Graceful degradation when AI is unavailable
- ❌ **Memory Constraints** - Large file handling optimization
- ❌ **Network Issues** - Timeout and retry mechanisms

## 🔒 Security & Privacy

- **Local Processing** - Data never leaves your environment except for AI API calls
- **Temporary Storage** - Files automatically cleaned after processing
- **API Key Security** - Environment variable usage
- **No Data Persistence** - Analysis results not stored permanently

## 📈 Performance Optimization

- **Async Processing** - Non-blocking operations for better UX
- **Memory Management** - Efficient handling of large datasets
- **Caching** - Smart caching for repeated operations
- **Progressive Loading** - Incremental result delivery
- **Resource Cleanup** - Automatic temporary file management

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/eda-ai-agent.git
cd eda-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run in development mode
chainlit run app.py --watch
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** - For providing powerful natural language processing
- **Plotly Team** - For excellent interactive visualization library
- **Chainlit** - For the amazing real-time UI framework
- **Pandas Community** - For robust data manipulation tools
- **Open Source Community** - For all the amazing libraries that make this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Moiz-Ali-Max/eda-ai-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Moiz-Ali-Max/eda-ai-agent/discussions)
- **Email**: moizaliafzaal@gmail.com

## 🎯 Roadmap

### Version 2.0 (Planned)
- [ ] **Multi-format Support** - JSON, Parquet, SQL databases
- [ ] **Advanced ML Models** - Clustering and classification insights
- [ ] **Custom Report Templates** - Branded report generation
- [ ] **Data Pipeline Integration** - API endpoints for automation
- [ ] **Team Collaboration** - Shared analysis and comments

### Future Enhancements
- [ ] **Time Series Analysis** - Specialized temporal data insights
- [ ] **Geospatial Analysis** - Map visualizations and spatial statistics
- [ ] **Real-time Data Sources** - Live data stream analysis
- [ ] **Mobile App** - Native mobile experience
- [ ] **Enterprise Features** - SSO, audit logs, advanced security

---

## 💡 About the Architecture

This project represents the intersection of AI and data visualization, solving the challenge of making complex statistical insights accessible through natural language generation. The architecture emphasizes:

- **Scalability** - Async processing for handling large datasets
- **Modularity** - Clean separation of concerns
- **Extensibility** - Easy to add new visualization types and AI models
- **User Experience** - Professional interface with clear progress indicators
- **Performance** - Optimized for speed and memory efficiency

---

**Built with ❤️ for the data science community**

*Star this repository if you find it useful!* ⭐
