from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

class cp():
    def __init__(self):
        self.stop = stopwords.words('english')

    def get_wordnet_pos(self,pos_tag):
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


    def text_clean(self,text):
        # lower text
        text = text.lower()
        # tokenize text
        text = " ".join([word for word in text.rsplit()])
        # remove punctuation and multiple spaces
        text = "".join([word.strip(string.punctuation) or " " for word in text]).split()
        # remove weird quotation marks
        text = [word.replace('”', "").replace('“', "").replace('’', '') for word in text]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)

        return (text)

    def clean_parallel (self,article):
        return self.text_clean(article)

    def word_clean(self,word):
        final = []
        word = word.strip(string.punctuation).replace('”'," ").replace('“'," ").replace('’',' ').replace('"',' ').replace("'",' ').replace(':',' ').replace(';',' ')
        word = word.split()
        for split in word:
            if not any(c.isdigit() for c in split) and split not in self.stop:
                if len(split)>0:
                    final.append(split)
            else:
                pass
        return final

    def clean_mlm(self,text):

        # split into sentences:
        text = text.lower()
        text_sentence = sent_tokenize(text)
        token_list = []
        table = str.maketrans('', '', string.punctuation)
        #stop_words = stopwords.words('english')

        for sentence in text_sentence:
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

    def clean_mlm_parallel(self,text):
        return (self.clean_mlm(text))


    def clean_mlm_init(self, i):

        text = X[i]
        # split into sentences:
        text = text.lower()
        text_sentence = sent_tokenize(text)
        token_list = []
        table = str.maketrans('', '', string.punctuation)
        #stop_words = stopwords.words('english')

        for sentence in text_sentence:
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

    def clean_mlm_parallel_init(self,text):
        return (self.clean_mlm(text))