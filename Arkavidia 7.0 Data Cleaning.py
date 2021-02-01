import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
factory = StemmerFactory()

##############################################################################
                        #### 1. Preprocessing #####
##############################################################################

# Read Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
stopwords = pd.read_fwf('stopword.txt')
stopword = list(stopwords['stopword'])
stopword.append("nya")
df = train.append(test).reset_index(drop=True)

# Show wordcloud (General)
fungsi = WordCloud(background_color='white', width = 3000, height = 2000,stopwords=stopwords['stopword'])
wordcloud = fungsi.generate(" ".join(df.review_text))
plt.title('Wordcloud of all strings')
plt.imshow(wordcloud)

# Show wordcloud (positive only)
positif = train[train.category == 1].review_text
wordcloud = fungsi.generate(" ".join(positif))
plt.title('Wordcloud of strings which positive category')
plt.imshow(wordcloud)

# Show wordcloud (negative only)
negatif = train[train.category == 0].review_text
wordcloud = fungsi.generate(" ".join(negatif))
plt.title('Wordcloud of strings which negative category')
plt.imshow(wordcloud)

# Remove irrelevant words
def replace_substring(old_words,new_word,string) :
    """
    Fungsi untuk mengganti substring didalam suatu string dengan
    syarat string tersebut sudah di ubah dalam format list terlebih
    dahulu.
    old_words = harus dalam format list
    new_word = single word only
    string = list of substring
    Returns
    -------
    string
    """
    for kata in old_words :
        string = [word.replace(kata,new_word) if word == kata else word for word in string]
    return string

tidaks = ["gak","gk","ga","tdk","ngk","nggak", "engga","tidaj","gaak"
          "ngga","ngak","g","ng","tak","ndk","ndak","enggak"
          "blm","belum","tdak","tida", "gada","gag","gda","tidk"]
baguss = ['oke','ok','bgs','okey','oklah','bagusa','mantap',
          'okee']
kurangs = ['kurng','krg','dikit']
bangets = ['banget','bngt','bgt','terlalu','sekali','sgt','bangeet','sangat'] 
lumayans = ["lmyn","cukup","ckp","mayan"]
lambats = ["lemot","lelet","lamban","lmbn","lambn",'lmban','ngadat']
iyas = ["ya","y","iyya","yaa"]
ajas = ["doank","doang", "ajaa"]
dengans = ["dgn","dg"]
kamars = ['kamr','kmr','kmar','romm','rom']
bauks = ['bauk','bauu'] 
bobroks = ['ancur','hancur', 'ancuran', 'ancurn']
sulits = ['ribet','rumit','susah','ruwet']
jijiks = ["amit","jijik","ilfeel"]
berisiks = ["bising","brisik","berisiik",'berderinyit']
recoms = ['remomended','recomended','rekomendet']
cumas = ['cm','cmn','cuma','cuman']

strings = []
for i in range(0, len(df)) :
    string = re.sub('[^a-zA-Z]',' ',df['review_text'][i])
    string = string.lower()
    string = string.replace("x", "nya")
    string = string.split()
    # Replace substring
    string = replace_substring(cumas,"cuma", string)
    string = replace_substring(tidaks, "tidak", string)
    string = replace_substring(baguss, "bagus", string)
    string = replace_substring(kurangs, "kurang", string)
    string = replace_substring(bangets, "sangat", string)
    string = replace_substring(lumayans, "lumayan", string)
    string = replace_substring(iyas, "iya", string)
    string = replace_substring(ajas, "saja", string)
    string = replace_substring(lambats, "lambat", string)
    string = replace_substring(dengans, "dengan", string)
    string = replace_substring(kamars, "kamar", string)
    string = replace_substring(bauks, "bau", string)
    string = replace_substring(bobroks, "bobrok", string)
    string = replace_substring(sulits, "sulit", string)
    string = replace_substring(jijiks, "jijik", string)
    string = replace_substring(berisiks, "berisik", string)
    string = replace_substring(recoms, "recommended", string)
    string = replace_substring(["ad"], "ada", string)
    string = replace_substring(["nyesel"], "menyesal", string)
    string = replace_substring(["enk"], "enak", string)
    string = replace_substring(["cek","chek"], "check", string)
    string = replace_substring(["sereem","seraam","serem"], "seram", string)
    string = replace_substring(["jutek","judes"], "tidak ramah", string)
    string = replace_substring(['udh','sdh','udah','sdah'], "sudah", string)
    string = replace_substring(["kumuh","dekil","ktr","jorok"], "kotor", string)
    string = replace_substring(["krn","karna"], "karena", string)
    string = replace_substring(['pening','mumet'], "pusing", string)
    string = replace_substring(['mampet'], "mampat", string)
    string = replace_substring(['eror'], "error", string)
    string = replace_substring(['skali'], "sekali", string)
    string = replace_substring(['dn','ama', 'n'], "dan", string)
    string = replace_substring(['tp','tpi','tetapi'], "tapi", string)
    string = replace_substring(['gatel'], "gatal", string)
    string = replace_substring(['ilang'], "hilang", string)
    string = replace_substring(['boking'], "booking", string)
    string = replace_substring(['kran'], "keran", string)
    string = replace_substring(['tdr','tdur'], "tidur", string)
    string = replace_substring(['bs','bsa'], "bisa", string)
    # Remove substring yg tidak ada huruf vokal
    for word in string :
        if ((bool(re.findall('a|i|u|e|o', word)) == False) & (len(word) >= 1) & (word != "tv")) :
            string.remove(word)
    string = [word for word in string if not word in stopword]
    string = ' '.join(string)
    strings.append(string)