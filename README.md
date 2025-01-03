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
