import json

import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from pprint import pprint

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns

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
        if path.endswith('json'):
            loaded_json = json.load(open(path))
            self.__papers = pd.DataFrame(loaded_json['articles'])
        if path.endswith('csv'):
            self.__papers = pd.read_csv(path)
        # self.papers_json = self.__papers
        # self.__papers = self.__papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(num_of_articles)

    def __remove_punctuation_and_convert_to_lowercase(self):
        print("__remove_punctuation_and_convert_to_lowercase")
        # self.__papers[TEST_PROCESS] = self.__papers[PAPER_TEXT].map(lambda x: re.sub('[,\.!?]', '', x))
        self.processed_papers = self.__papers['abstract'].map(lambda x: re.sub('[,\.!?]', '', x))
        # self.__papers['processed_papers'] = self.__papers['abstract'].map(lambda x: re.sub('[,\.!?()]', '', x))
        self.processed_papers = self.processed_papers.map(lambda x: x.lower())
        # self.__papers['processed_papers'] = self.__papers['processed_papers'].map(lambda x: x.lower())
        # self.papers_json = self.__papers

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
        # data = self.__papers.paper_text_processed.values.tolist()
        data = self.processed_papers.tolist()
        # data = self.__papers['processed_papers'].tolist()
        # self.__papers['as_list'] = self.__papers['processed_papers'].map(lambda x: x.lower())
        # self.__papers['as_list'] = self.__papers['processed_papers']
        data_words = list(self.__sent_to_words(data))
        # data_words = list(self.__sent_to_words(self.__papers['as_list']))
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

    #     print(LdaModeling.top_topics(self._dict_of_topics, TOPICS_NUM))

    @property
    def topics_list(self):
        return self._topics_list


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    print("preprocess_text")
    """This function cleans the input text by
    - removing links
    - removing special chars
    - removing numbers
    - removing stopwords
    - transforming in lower case
    - removing excessive whitespaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): remove stopwords or not
    Returns:
        str: cleaned text
    """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special chars
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. joins all tokens again
        text = " ".join(tokens)
    # returns cleaned text
    text = text.lower().strip()
    return text


def main():
    path_csv = './NIPS_Papers/papers.csv'
    path_json = './json_file/response100_IOT.json'
    num_of_articles = 100
    lda_temp = LdaModeling(path_json, num_of_articles)
    topic_list = lda_temp.topics_list
    print(topic_list)

    # dataset = lda_temp.papers_json
    dataset = lda_temp.processed_papers
    # print('dataset')
    # print(dataset)
    # dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
    #                              remove=('headers', 'footers', 'quotes'))

    # df = pd.DataFrame(dataset)
    # df['cleaned'] = df['corpus'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

    # initialize vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # print('df')
    # print(df)
    # print(dataset)
    # print("df['corpus']")
    # print(df['corpus'])
    # print("df['cleaned']")
    # print(df['cleaned'])
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    # X = vectorizer.fit_transform(df['cleaned'])
    X = vectorizer.fit_transform(dataset)

    # initialize KMeans with 3 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    print('clusters')
    print(clusters)

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass X to the pca
    pca_vecs = pca.fit_transform(X.toarray())
    # save the two dimensions in x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and PCA vectors to columns in the original dataframe
    # df['cluster'] = clusters
    # dataset['cluster'] = clusters
    dataset_temp = pd.DataFrame(dataset)
    dataset_temp['cluster'] = clusters
    # df['x0'] = x0
    dataset_temp['x0'] = x0
    # df['x1'] = x1
    dataset_temp['x1'] = x1

    # print(dataset)

    cluster_map = {0: topic_list[0], 1: topic_list[1], 2: topic_list[2], 3: topic_list[3]}
    # df['cluster'] = df['cluster'].map(cluster_map)
    # print(dataset)
    dataset_temp['cluster'] = dataset_temp['cluster'].map(cluster_map)
    print(dataset_temp)

    plt.figure(figsize=(12, 7))
    plt.title("topic modeling lda + k means", fontdict={"fontsize": 18})
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    #  create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=dataset_temp, x='x0', y='x1', hue='cluster', palette="viridis")
    plt.show()


if __name__ == "__main__":
    main()
