# Voyadore 🌏 - Your AI Travel Companion

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Overview

Voyadore is an AI-powered travel planning assistant that helps you plan your perfect trip. From finding flights to discovering local attractions, Voyadore makes travel planning effortless and enjoyable.

## ✨ Features

- 🤖 **AI Travel Assistant**: Get personalized travel recommendations using Google's Gemini AI
- 📄 **PDF Document Analysis**: Upload and analyze travel documents for quick summaries
- 🎥 **Travel Content**: Find curated YouTube videos and websites about your destination
- 🗺️ **Interactive Maps**: Explore nearby attractions, restaurants, and points of interest
- 🌤️ **Weather Updates**: Get real-time weather information for your destination
- 🔍 **Smart Search**: Find flights, hotels, and travel resources using multiple APIs

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, Google Gemini AI
- **APIs**: 
  - Weather API
  - SerpAPI
  - Tavily Search
  - Overpass API
- **Mapping**: Folium, Geopy
- **Other Tools**: PyPDF2, dotenv

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voyadore.git
cd voyadore
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```plaintext
GeminiAPI=your_gemini_api_key
WeatherAPI=your_weather_api_key
SerpAPI=your_serp_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🚀 Usage

1. Run the Streamlit app:
```bash
streamlit run Voyadore.py
```

2. Enter your travel details in the main form:
   - Source and destination
   - Travel dates
   - Personal preferences
   - Upload travel documents (optional)

3. Explore the features:
   - Chat with AI for travel advice
   - Search for travel content
   - Explore interactive maps
   - Check weather information


---
Made with 🤍 using Streamlit and Gemini AI
