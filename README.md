# Naver Movie Reviews Sentiment Analysis

This project focuses on scraping movie reviews from Naver, processing them, and performing sentiment analysis using machine learning models. It includes data preprocessing, TF-IDF vectorization, Latent Dirichlet Allocation (LDA) for topic modeling, and logistic regression for sentiment classification.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Steps](#steps)
   - [Scraping Reviews](#scraping-reviews)
   - [Data Cleaning](#data-cleaning)
   - [TF-IDF Vectorization](#tf-idf-vectorization)
   - [Topic Modeling with LDA](#topic-modeling-with-lda)
   - [Sentiment Analysis](#sentiment-analysis)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Key Findings](#key-findings)

---

## Prerequisites

1. Python 3.x
2. Install the following libraries:
   ```bash
   pip install pandas requests beautifulsoup4 scikit-learn konlpy tqdm

---

## Steps
### Scraping Reviews
1. Create a list of URLs for scraping reviews from Naver:
 ```python
import requests
from bs4 import BeautifulSoup

url_list = []
for page in range(1, 634):
    review_url = f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=181114&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=sympathyScore&page={page}'
    url_list.append(review_url)
```
---

2. Scrape review data:
```python
comments, stars, sympathies = [], [], []
for page in range(1, 635):
    review_url = f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=181114&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=sympathyScore&page={page}'
    res = requests.get(review_url)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'lxml')
        star = soup.select('div.score_result > ul > li > div.star_score > em')
        tds = soup.select('div.score_result > ul > li > div.score_reple > p > span')
        spts = soup.select('div.score_result > ul > li > div.btn_area > a._sympathyButton > strong')
        for st in star:
            stars.append(int(st.text))
        for cmt in tds:
            if cmt.text not in ['관람객', '스포일러가 포함된 감상평입니다. 감상평 보기']:
                comments.append(cmt.text.strip())
        for sympathy in spts:
            sympathies.append(int(sympathy.text))

import pandas as pd
df = pd.DataFrame({"Review": comments, "Rank": stars, "Sympathy": sympathies})
df.to_csv('날씨의아이.csv', index=False)
```
---

### Data Cleaning
1. Remove duplicates and handle missing data:
```python
cm1 = pd.read_csv('날씨의아이.csv')
cm1.dropna(inplace=True)
cm1.drop_duplicates(inplace=True)
```
2. Extract Korean and English words from the reviews:
```python
import re

def extract_word(text):
    hangul = re.compile('[^가-힣|a-zA-Z]')
    return hangul.sub(' ', text)

cm1['Review'] = cm1['Review'].apply(lambda x: extract_word(x))
```
3. Remove stopwords:
```python
with open('stopwords.txt', 'r') as f:
    stopwords = f.read().split(',')

cm1['Review'] = cm1['Review'].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))
```
---

### TF-IDF Vectorization
1. Transform reviews using TF-IDF:
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = CountVectorizer()
```

---

### Topic Modeling with LDA
1. Apply LDA to extract topics
```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=2)  # 2 topics for sentiment
lda.fit(tf_idf_vect)

# Display top words per topic
index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx+1}:", [(index_to_word[i], topic[i].round(3)) for i in topic.argsort()[:-11:-1]])
```

### Sentiment Analysis
1. Label data as positive (1) or negative (0) based on ranks:
```python
cm1['P/N'] = cm1['Rank'].apply(lambda x: 1 if x > 8 else 0)
```
2. Train a logistic regression model:
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(tf_idf_vect, cm1['P/N'])
```
3. Evaluate the model:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pred = lr.predict(tf_idf_vect)
print('Accuracy:', accuracy_score(cm1['P/N'], pred))
print('Precision:', precision_score(cm1['P/N'], pred))
print('Recall:', recall_score(cm1['P/N'], pred))
print('F1 Score:', f1_score(cm1['P/N'], pred))
```
---

### Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1 Score

---

### Key Findings
1. The most frequent words in reviews provide insights into common themes.
2. Logistic regression achieved high accuracy in classifying sentiments.
3. Topic modeling revealed distinct patterns between positive and negative reviews.

---

### Notes
stopwords.txt should contain stopwords separated by commas.
Ensure proper installation of required libraries before running the code.
