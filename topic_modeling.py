import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from pprint import pprint


TEST_PROCESS = 'paper_text_processed'
PAPER_TEXT = 'paper_text'
TOPICS_NUM = 4


class LdaModeling:
    def __init__(self, path: str, num_articles: int):
        self._lda_model = None
        self.__papers = None
        self._dict_of_topics = {}
        self._topics_list = []
        self.__loading_and_cleaning_data(path, num_articles)
        self.__remove_punctuation_and_convert_to_lowercase()
        self.__prepare_data_to_lda()
        self.__top_topics()

    def __loading_and_cleaning_data(self, path: str, num_of_articles: int):
        print("__loading_and_cleaning_data")
        self.__papers = pd.read_csv(path)
        self.__papers = self.__papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(num_of_articles)

    def __remove_punctuation_and_convert_to_lowercase(self):
        print("__remove_punctuation_and_convert_to_lowercase")
        self.__papers[TEST_PROCESS] = self.__papers[PAPER_TEXT].map(lambda x: re.sub('[,\.!?]', '', x))
        self.__papers[TEST_PROCESS] = self.__papers[TEST_PROCESS].map(lambda x: x.lower())

    @staticmethod
    def __sent_to_words(sentences):
        print("__sent_to_words")
        for sentence in sentences:
            # deacc=True removes punctuations
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    @staticmethod
    def __stopwords_string():
        print("__stopwords_string")
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        return stop_words

    def __remove_stopwords(self, texts):
        print("__remove_stopwords")
        stop_words = self.__stopwords_string()
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]

    def __prepare_data_to_lda(self):
        print("__prepare_data_to_lda")
        data = self.__papers.paper_text_processed.values.tolist()
        data_words = list(self.__sent_to_words(data))
        data_words = self.__remove_stopwords(data_words)
        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        texts = data_words
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        self.__lda_model_training(corpus, id2word)

    def __lda_model_training(self, corpus, id2word):
        print("__lda_model_training")
        # Build LDA model
        self._lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=TOPICS_NUM)
        self.__topics_to_dict()

    def __topics_to_dict(self):
        print("__topics_to_dict")
        for index in range(TOPICS_NUM):
            for topic in self._lda_model.show_topic(index):
                if topic[0] not in self._dict_of_topics:
                    self._dict_of_topics[topic[0]] = topic[1]
                else:
                    self._dict_of_topics[topic[0]] += topic[1]

    def __top_topics(self):
        print("__top_topics")
        self._topics_list = sorted(self._dict_of_topics, key=self._dict_of_topics.get, reverse=True)[:TOPICS_NUM]

    # def print_topics(self):
    #     pprint(self._dict_of_topics)
    #     print(len(self._dict_of_topics.keys()))
    #     print(LdaModeling.top_topics(self._dict_of_topics, TOPICS_NUM))

    @property
    def topics_list(self):
        return self._topics_list


def main():
    path = './NIPS_Papers/papers.csv'
    num_of_articles = 100
    lda_temp = LdaModeling(path, num_of_articles)
    print(lda_temp.topics_list)
    # papers = LdaModeling.loading_and_cleaning_data(path, num_of_articles)
    # LdaModeling.remove_punctuation_and_convert_to_lowercase(papers)
    # exploratory_analysis(papers)
    # LdaModeling.prepare_data_to_lda(papers)


if __name__ == "__main__":
    main()
