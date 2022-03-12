import pandas as pd
import re
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from pprint import pprint


TEST_PROCESS = 'paper_text_processed'
PAPER_TEXT = 'paper_text'


def loading_and_cleaning_data(path: str, num_of_articles: int):
    papers = pd.read_csv(path)
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(num_of_articles)
    return papers


def remove_punctuation_and_convert_to_lowercase(papers):
    papers[TEST_PROCESS] = papers[PAPER_TEXT].map(lambda x: re.sub('[,\.!?]', '', x))
    papers[TEST_PROCESS] = papers[TEST_PROCESS].map(lambda x: x.lower())


def join_titles_to_list(papers):
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))
    return long_string


# def exploratory_analysis(papers):
    # print('got here1')
    # long_string = join_titles_to_list(papers)
    # wordcloud = WordCloud(background_color="white", max_words=4, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    # wordcloud.generate(long_string)
    # print('Got here2')
    # Visualize the word cloud
    # topics = wordcloud.words_()
    # print(wordcloud)
    # wordcloud.to_image()
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def stopwords_string():
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return stop_words


def remove_stopwords(texts):
    stop_words = stopwords_string()
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def prepare_data_to_lda(papers):
    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    # print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # print(corpus[:1][0][:30])
    lda_model_training(corpus, id2word)


def lda_model_training(corpus, id2word):
    # number of topics
    num_topics = 4
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    # doc_lda = lda_model[corpus]
    dict_of_topics = topics_to_dict(lda_model, num_topics)
    pprint(dict_of_topics)
    print(len(dict_of_topics.keys()))
    print(top_topics(dict_of_topics, num_topics))


def topics_to_dict(lda_model, num_topics):
    topics_dict = {}
    for index in range(num_topics):
        for topic in lda_model.show_topic(index):
            if topic[0] not in topics_dict:
                topics_dict[topic[0]] = topic[1]
            else:
                topics_dict[topic[0]] += topic[1]
    return topics_dict


def top_topics(topics_dict: dict, num_topics: int) -> list:
    return sorted(topics_dict, key=topics_dict.get, reverse=True)[:num_topics]


def main():
    path = './NIPS_Papers/papers.csv'
    num_of_articles = 100
    papers = loading_and_cleaning_data(path, num_of_articles)
    remove_punctuation_and_convert_to_lowercase(papers)
    # exploratory_analysis(papers)
    prepare_data_to_lda(papers)


if __name__ == '__main__':
    main()
