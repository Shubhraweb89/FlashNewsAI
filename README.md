# 📰 FlashNews_AI: News Summarizer App

**FlashNews_AI** is a lightweight and interactive news summarizer web app built using **Streamlit**. It helps users stay updated with trending headlines, search for specific topics, and generate concise summaries for both live news and custom text.

---

## 🔧 Technologies Used

- [Newspaper3k](https://newspaper.readthedocs.io/en/latest/) – For article content extraction and summary generation  
- Google News RSS API – For fetching real-time news articles  
- Streamlit – For building the interactive web app  
- Python – Core programming language

---

## ✨ Features

- 📢 **Trending News**: Get a list of top trending news articles  
- 🔍 **Search News**: Search news on any topic/keyword  
- 🌐 **Favorite Topics**: Filter news by your preferred categories  
- 📄 **Summary Generator**: Automatically summarize full-length articles using NLP  
- ✍️ **Paste & Summarize**: Paste any news text or article manually to get a summary  
- 🔢 **Quantity Control**: Customize how many news articles you want to display  

---

## 🚀 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/Shubhraweb89/FlashNewsAI.git
cd FlashNewsAI

### 2. Create a Virtual Environment
python -m venv venv  

### 3. Activate the Virtual Environment
venv\Scripts\activate  #On Windows:
source venv/bin/activate  #On macOS/Linux:

### 4. Install Required Dependencies
pip install -r requirements.txt

### 5. Run the Streamlit Web Application
streamlit run App.py


