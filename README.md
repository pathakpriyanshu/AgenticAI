# 🚀 AgenticAI - Intelligent Financial Assistant

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/AI-Agentic-red.svg" alt="Agentic AI">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</div>

<div align="center">
  <h3>🤖 The Future of Financial Intelligence is Here!</h3>
  <p><em>Powered by cutting-edge Agentic AI technology that thinks, plans, and acts autonomously</em></p>
</div>

---

## 🌟 Why this Repo?

This is a **revolutionary financial intelligence system** that harnesses the power of **Agentic AI** - artificial intelligence that can make decisions, plan multi-step workflows, and take autonomous actions to achieve complex goals. Unlike traditional AI that simply responds to prompts, AgenticAI **thinks ahead, reasons through problems, and executes sophisticated financial analysis workflows** with minimal human intervention.

### 🎯 The Power of Agentic AI

**Agentic AI** represents the next frontier in artificial intelligence - systems that exhibit:
- **🧠 Autonomous Decision-Making**: Makes intelligent choices without constant human oversight
- **🎯 Goal-Oriented Planning**: Breaks down complex tasks into manageable steps
- **🔄 Adaptive Execution**: Learns and adapts strategies based on results
- **🛠️ Tool Integration**: Seamlessly coordinates multiple tools and data sources
- **💭 Multi-Step Reasoning**: Thinks through problems like a human financial analyst

---

## 🚀 Features & Capabilities

### 🔀 **Intelligent Query Routing**
Our sophisticated routing system automatically directs your questions to the most appropriate tool:

| Query Type | Example | Route To |
|------------|---------|----------|
| 📄 **PDF Knowledge** | "What does the annual report say about Q3 earnings?" | Vector Store |
| 🌐 **Real-time Data** | "What's Tesla's current stock price?" | Web Search |
| 📊 **Financial Analysis** | "Calculate Apple's Sharpe ratio for 2024" | Financial Metrics |

### 🏗️ **Three Powerful AI Agents**

#### 1. 📚 **Vector Store Agent** - Your Document Intelligence
- **Purpose**: Extracts insights directly from your PDF documents
- **Capabilities**:
  - Processes annual reports, financial statements, research papers
  - Semantic search through document content
  - Contextual understanding of financial documents
  - Instant answers with source citations

#### 2. 🌐 **Web Search Agent** - Real-time Market Intelligence  
- **Purpose**: Fetches live market data and current information
- **Powered by**: Serper API for Google Search
- **Capabilities**:
  - Real-time stock prices and market data
  - Latest financial news and market trends
  - Company updates and earnings announcements
  - Economic indicators and market sentiment

#### 3. 📈 **Financial Metrics Agent** - Advanced Analytics Engine
- **Purpose**: Performs sophisticated financial calculations and analysis
- **Powered by**: AlphaVantage API
- **Capabilities**:
  - **Sharpe Ratio**: Risk-adjusted return analysis
  - **Batting Average**: Investment success rate measurement
  - **Fundamental Analysis**: P/E ratios, ROE, debt-to-equity


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/pathakpriyanshu/AgenticAI.git
cd AgenticAI

# Install dependencies
pip install -r requirements.txt

# Set up your API keys (create a .env file)
SERPER_API_KEY=your_serper_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

# Run the application
python main.py
```

### 🔑 API Keys Required

1. **Serper API** (Google Search)
   - Visit: [serper.dev](https://serper.dev)
   - Get your free API key
   - 2,500 free searches per month

2. **AlphaVantage API** (Financial Data)
   - Visit: [alphavantage.co](https://www.alphavantage.co)
   - Get your free API key
   - 25 requests per day (free tier)

---

## 🚀 Usage Examples

### Basic Query Examples
```python
# Ask about PDF content
"What were the key findings in the Q3 earnings report?"

# Get real-time market data
"What's the current price of NVIDIA stock?"

# Calculate financial metrics
"Calculate the Sharpe ratio for Apple stock over the past year"

# Complex analysis
"Compare the batting average of Tesla vs Ford stock performance"


```
## 📁 Repository Structure

```
AgenticAI/
├── 📄 main.py                 # Main application entry point
├── 🔧 preprocessing.py        # Data preprocessing utilities  
├── 🔍 query.txt              # Sample queries for testing
├── 📚 readme.md              # This file
├── 📋 requirements.txt       # Python dependencies
├── 🔧 utility.py             # Helper functions
├── 🛠️ app.py                 # Web application interface
├── 📊 check_agent_log.json   # Agent performance logs
├── 🔄 chainlit.md            # Chainlit integration docs
├── 📁 __pycache__/           # Python cache files
├── 🔗 .chainlit/             # Chainlit configuration
├── 📄 .vscode/               # VS Code settings
├── 🌐 chromaDB/              # Vector database storage
├── 📊 PDFs/                  # PDF document storage
├── 🤖 Workflows/             # AI agent workflows
└── 🚫 .gitignore             # Git ignore file
```

---

## 🎨 Why Choose our Repo?

### 🌟 **Next-Generation AI Technology**
- **Autonomous Operation**: Works independently without constant prompting
- **Multi-Modal Intelligence**: Combines text, financial data, and real-time information
- **Contextual Understanding**: Remembers conversation history and context
- **Adaptive Learning**: Improves performance over time

### 💼 **Professional-Grade Financial Analysis**
- **Comprehensive Metrics**: 50+ financial ratios and indicators
- **Real-time Data**: Always up-to-date market information
- **Document Intelligence**: Extract insights from complex financial documents
- **Risk Assessment**: Advanced risk management and analysis tools

### 🚀 **Easy Integration**
- **Simple Setup**: Get started in minutes
- **Flexible Architecture**: Easy to extend and customize
- **API Ready**: Can be integrated into existing systems
- **Scalable**: Handles everything from individual queries to batch processing

---

## 🛡️ Security & Privacy

- 🔐 **API Keys**: Stored securely in environment variables
- 🗄️ **Local Storage**: Your documents stay on your system
- 🛡️ **Secure Connections**: All API calls use HTTPS encryption


## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

---


### 🔮 **Future Vision**
- [ ] Addition of Markowitz Optimization for investment Analysis.

---



<div align="center">
  <p>If you find this repo useful, please give it a ⭐ on GitHub!</p>
   alt="GitHub stars">
</div>

---
