import pandas as pd
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

g_var = None

def work(data):
    global g_var
    g_var = data

class cpt():
    def __init__(self):
        #self.df = df
        #self.data = pd.DataFrame({0:df['content']})
        self.stop = stopwords.words('english')


    #def work (self,data):
    #    self.g_var = data

    def get_wordnet_pos(self, pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def clean_mlm_init(self,i):
        text = g_var[i]
        # split into sentences:
        #text = text.lower()
        text_sentence = sent_tokenize(text)
        token_list = []
        table = str.maketrans('', '', string.punctuation)
        # stop_words = stopwords.words('english')

        for sentence in text_sentence:
            sentence = sentence.lower()
            tokens = word_tokenize(sentence)
            tokens = [w.translate(table) for w in tokens]
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [w for w in tokens if not w in self.stop]
            tokens = [w for w in tokens if len(w) > 0]
            pos_tags = pos_tag(tokens)
            tokens = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1])) for t in pos_tags]
            tokens = [t for t in tokens if len(t) > 1]

            token_list.append(tokens)

        return token_list


    def clean_mlm_parallel_init(self,i):
        return (self.clean_mlm_init(i))
