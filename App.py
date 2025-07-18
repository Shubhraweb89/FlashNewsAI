import streamlit as st
import requests
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from newspaper import Article, Config
import io
import nltk
import ssl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Configure SSL and user agent
ssl._create_default_https_context = ssl._create_unverified_context
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Configure newspaper3k
config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10

# Download NLTK data
nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title='InNews🇮🇳: AI News Portal',
    page_icon='./Meta/newspaper.ico',
    layout='centered'
)

# ==================== Summarizer Functions ====================
@st.cache_resource
def load_summarizer_model():
    model_path = "E:/News_AI/InNews/bart_summarizer_with_rl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_summarizer_model()

# Feedback configuration
FEEDBACK_DIR = Path("feedback_data")
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.csv"

def generate_summary(article_text):
    inputs = tokenizer(article_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def log_feedback(article, summary, feedback):
    try:
        new_entry = pd.DataFrame({
            "article": [article],
            "summary": [summary],
            "feedback": [feedback],
            "timestamp": [pd.Timestamp.now()]
        })
        
        if FEEDBACK_LOG.exists():
            new_entry.to_csv(FEEDBACK_LOG, mode='a', header=False, index=False)
        else:
            new_entry.to_csv(FEEDBACK_LOG, mode='w', header=True, index=False)
        
        st.success("Feedback saved successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to save feedback: {str(e)}")
        return False

# ==================== News Portal Functions ====================
def fetch_news_search_topic(topic):
    try:
        site = f'https://news.google.com/rss/search?q={topic}'
        req = Request(site, headers={'User-Agent': USER_AGENT})
        op = urlopen(req)
        rd = op.read()
        op.close()
        sp_page = BeautifulSoup(rd, 'xml')
        return sp_page.find_all('item')
    except Exception as e:
        st.error(f"Error fetching news for topic {topic}: {str(e)}")
        return []

def fetch_top_news():
    try:
        site = 'https://news.google.com/news/rss'
        req = Request(site, headers={'User-Agent': USER_AGENT})
        op = urlopen(req)
        rd = op.read()
        op.close()
        sp_page = BeautifulSoup(rd, 'xml')
        return sp_page.find_all('item')
    except Exception as e:
        st.error(f"Error fetching top news: {str(e)}")
        return []

def fetch_category_news(topic):
    try:
        site = f'https://news.google.com/news/rss/headlines/section/topic/{topic}'
        req = Request(site, headers={'User-Agent': USER_AGENT})
        op = urlopen(req)
        rd = op.read()
        op.close()
        sp_page = BeautifulSoup(rd, 'xml')
        return sp_page.find_all('item')
    except Exception as e:
        st.error(f"Error fetching {topic} news: {str(e)}")
        return []

def fetch_news_poster(poster_link):
    try:
        if not poster_link:
            raise ValueError("No image URL provided")
            
        if not poster_link.startswith(('http://', 'https://')):
            poster_link = f'https://{poster_link}'
            
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(poster_link, headers=headers, timeout=10)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            st.image(image, use_container_width=True)
            return
            
    except Exception as e:
        st.warning(f"Couldn't fetch image: {str(e)}")
    
    try:
        fallback_img = Image.open('./Meta/no_image.jpg')
        st.image(fallback_img, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't load fallback image: {str(e)}")

def get_article_content(url):
    """Extract article content using multiple methods"""
    # Method 1: newspaper3k
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        if len(article.text) > 300:
            return article.text
    except:
        pass
    
    # Method 2: Direct HTML request
    try:
        req = Request(url, headers={'User-Agent': USER_AGENT})
        html = urlopen(req).read().decode('utf-8')
        soup_obj = BeautifulSoup(html, 'html.parser')
        paragraphs = soup_obj.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        if len(text) > 300:
            return text
    except:
        pass
    
    return None

def display_news(list_of_news, news_quantity):
    if not list_of_news:
        st.warning("No news articles found")
        return
        
    for idx, news in enumerate(list_of_news[:news_quantity], 1):
        st.markdown(f"### ({idx}) {news.title.text}")
        article_url = news.link.text
        
        # Display image
        try:
            article = Article(article_url, config=config)
            article.download()
            article.parse()
            if hasattr(article, 'top_image') and article.top_image:
                fetch_news_poster(article.top_image)
            else:
                st.warning("No image available")
        except:
            pass
        
        # Generate summaries
        article_content = get_article_content(article_url)
        
        if article_content:
            # AI Summary
            with st.spinner("Generating AI summary..."):
                try:
                    ai_summary = generate_summary(article_content)
                    st.session_state.article = article_content
                    st.session_state.summary = ai_summary
                    
                    with st.expander("🤖 AI Summary (Advanced)"):
                        st.markdown(f'<div style="text-align: justify;">{ai_summary}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Feedback for AI summary
                        feedback = st.radio("Was this summary helpful?",
                                          ("👍 Like", "👎 Dislike"),
                                          key=f"ai_feedback_{idx}")
                        if st.button("Submit Feedback", key=f"ai_feedback_btn_{idx}"):
                            feedback_value = 1 if feedback == "👍 Like" else 0
                            log_feedback(article_content[:500], ai_summary, feedback_value)
                            
                except Exception as e:
                    st.warning(f"AI summary failed: {str(e)}")
            
            # Basic Summary
            try:
                article.nlp()
                if article.summary and len(article.summary) > 50:
                    with st.expander("📰 Basic Summary"):
                        st.markdown(f'<div style="text-align: justify;">{article.summary}</div>',
                                   unsafe_allow_html=True)
            except:
                pass
        else:
            st.warning("Couldn't extract article content")
        
        st.markdown(f"[Read full article at {news.source.text}...]({article_url})")
        st.success(f"Published: {news.pubDate.text}")
        st.markdown("---")

# ==================== Main App ====================
def main():
    # App title and logo
    
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            logo = Image.open('./Meta/newspaper.png')
            st.image(logo, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load logo image: {str(e)}")
    
    # Tab navigation
    tab1, tab2 = st.tabs(["📝 AI Summarizer", "🗞️ News Portal"])
    
    with tab1:
        st.header("🧠 Personalized AI News Summarizer")
        st.markdown("Paste any news article below to get an AI-generated summary")
        
        article_input = st.text_area("✍️ Paste your news article here:", height=250, key="summarizer_input")
        
        if st.button("🔍 Generate Summary", key="summarize_btn"):
            if not article_input.strip():
                st.warning("Please enter a valid article.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary_text = generate_summary(article_input)
                        st.session_state.article = article_input
                        st.session_state.summary = summary_text
                        st.subheader("📝 AI Summary:")
                        st.markdown(f'<div style="text-align: justify;">{summary_text}</div>', 
                                   unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Summary generation failed: {str(e)}")
        
        # Feedback section
        if 'summary' in st.session_state:
            st.markdown("---")
            feedback = st.radio("Was this summary helpful?", 
                               ("👍 Like", "👎 Dislike"),
                               key="manual_feedback")
            if st.button("📩 Submit Feedback", key="submit_feedback"):
                if feedback == "👍 Like":
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("We're sorry to hear that. Please let us know if you have any suggestions for improvement.")

                feedback_value = 1 if feedback == "👍 Like" else 0
                if log_feedback(st.session_state.article[:500], 
                              st.session_state.summary, 
                              feedback_value):
                    del st.session_state.article
                    del st.session_state.summary
                    st.rerun()
    
    with tab2:
        st.header("🌐 News Portal")
        st.markdown("Browse the latest news from various categories")
        
        category = st.selectbox(
            'Select News Category',
            ['--Select--', 'Trending🔥 News', 'Favourite💙 Topics', 'Search🔍 Topic'],
            key="category_select"
        )
        
        if category == '--Select--':
            st.warning('Please select a news category')
        else:
            no_of_news = st.slider('Number of News:', min_value=1, max_value=25, value=5, key="news_slider")
            
            if category == 'Trending🔥 News':
                st.subheader("✅ Trending News")
                news_list = fetch_top_news()
                display_news(news_list, no_of_news)
            elif category == 'Favourite💙 Topics':
                st.subheader("Choose Topic")
                topics = ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 
                         'ENTERTAINMENT', 'SPORTS', 'SCIENCE', 'HEALTH']
                chosen_topic = st.selectbox("Select Topic", topics, key="topic_select")
                
                if st.button("Get News", key="get_news_btn"):
                    st.subheader(f"✅ {chosen_topic} News")
                    news_list = fetch_category_news(chosen_topic)
                    display_news(news_list, no_of_news)
            elif category == 'Search🔍 Topic':
                user_topic = st.text_input("Enter topic to search", key="search_input")
                
                if st.button("Search News", key="search_btn") and user_topic:
                    st.subheader(f"✅ News about {user_topic.capitalize()}")
                    news_list = fetch_news_search_topic(user_topic.replace(' ', ''))
                    display_news(news_list, no_of_news)
                elif st.button("Search News", key="search_btn"):
                    st.warning("Please enter a search term")

if __name__ == "__main__":
    main()