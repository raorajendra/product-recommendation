from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import re
import string
from nltk.stem import WordNetLemmatizer 
import re, nltk, spacy, string
import texthero as hero
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer


df_products=pd.read_csv('sample30.csv')
loaded_model_classifier = pickle.load(open('finalized_model.sav', 'rb'))
df_products_cleaned=df_products.dropna(subset=['user_sentiment'],axis=0)
df_products_cleaned=df_products_cleaned.dropna(subset=['reviews_username'],axis=0)
df_products_cleaned=df_products_cleaned.drop_duplicates(subset=['reviews_text'])
df_products_cleaned_filtered=df_products_cleaned[['id','name','reviews_rating', 'reviews_text', 'user_sentiment', 'reviews_username']]




## text hero is library capable of doing cleaning of text 

from texthero import preprocessing
custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_diacritics,preprocessing.remove_stopwords
                   
                  ]
df_products_cleaned_filtered['reviews_text_clean'] = hero.clean(df_products_cleaned_filtered['reviews_text'], custom_pipeline)
df_products_cleaned_filtered['reviews_text_clean'] = [review.replace('{','') for review in df_products_cleaned_filtered['reviews_text_clean']]
df_products_cleaned_filtered['reviews_text_clean'] = [review.replace('}','') for review in df_products_cleaned_filtered['reviews_text_clean']]
df_products_cleaned_filtered['reviews_text_clean'] = [review.replace('(','') for review in df_products_cleaned_filtered['reviews_text_clean']]
df_products_cleaned_filtered['reviews_text_clean'] = [review.replace(')','') for review in df_products_cleaned_filtered['reviews_text_clean']]
nlp=en_core_web_sm.load()


def lemmatize(sent):
    lemmatize_text=''
    sent=nlp(sent)
    for token in sent:
        lemmatize_text+=token.lemma_+' '        
    return lemmatize_text
    
    
df_products_cleaned_filtered['reviews_text_lemmatize']=df_products_cleaned_filtered['reviews_text_clean'].apply(lambda x:lemmatize(x))



from textblob import TextBlob

def pos_tag(text):
    try:
        return TextBlob(text).tags
    except:
        return None

def get_adjectives(text):
    blob = TextBlob(text)
    #print('blob',blob)
    return ' '.join([ word for (word,tag) in blob.tags if tag == "NN" or tag == "JJ" 
                     or tag == "JJR"  or tag == "JJS" or tag == "RB" or tag == "RBR" or tag == "RBS" ] )

df_products_cleaned_filtered["reviews_text_lemmatize_POS_removed"] =  df_products_cleaned_filtered.apply(lambda x: get_adjectives(x['reviews_text_lemmatize']), axis=1)


vectorizer=TfidfVectorizer()
vectorized_df=vectorizer.fit_transform(df_products_cleaned_filtered['reviews_text_lemmatize_POS_removed'])
predcited_reviews_sentiments =loaded_model_classifier.predict(vectorized_df)
df_products_cleaned_filtered['sentiment_predicted']=predcited_reviews_sentiments

# load the model from disk
loaded_recommendation_df = pickle.load(open('user_final_prediction_df', 'rb'))



def get_top_5_recommondation(user_input):
    #if user_input not in loaded_recommendation_df:
    #   return 'user '+user_input+ ' not available' 
    print(user_input)     
    df_products_cleaned_filtered.head(5)
    try:
        d=loaded_recommendation_df.loc[user_input].sort_values(ascending=False)[0:20]
    except:
           return 'user '+user_input+ ' not available'
    df = pd.DataFrame(columns=['product','score'])
    for prod in d.index:
       temp_df=df_products_cleaned_filtered[df_products_cleaned_filtered['name']==prod]
       df = df.append({'product':prod,'score':temp_df['sentiment_predicted'].sum()}, ignore_index=True)
       df_sorted=df.sort_values(by=['score'],ascending=False)[0:5]
    return  df_sorted
    
    
    

