import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("train.txt",sep=';',header= None,names=["text","emotions"])
print (df.head())
print(df.isnull().sum())
print("duplicates = ",df.duplicated().sum())

#convert emotions--
unique_emotions = df["emotions"].unique()
emotions_number = {}
i = 0
for emo in unique_emotions:
    emotions_number[emo] = i 
    i += 1
df["emotions"] = df["emotions"].map(emotions_number)
print(df.head())
df["text"] = df["text"].apply(lambda x:x.lower()) 

#remove punchuation--
import string
def remove_punc(txt):
    return txt.translate(str.maketrans('','',string.punctuation))
df['text']=df['text'].apply(remove_punc)

#remove numbers--
def remove_number(txt):
    new=""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new
df["text"]=df['text'].apply(remove_number)

#remove url--
def remove_emojis(txt):
    new=""
    for i in txt:
        if not i.isascii():
            new = new + i
    return new
df["text"]=df['text'].apply(remove_emojis)
