# Importing modules
import glob
import os
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from pprint import pprint
# from pyLDAvis import gensim_models
import pyLDAvis.gensim_models
import pickle
import pyLDAvis


TEST_PROCESS = 'paper_text_processed'
PAPER_TEXT = 'paper_text'


# def load_papers_and_drop_columns(path: str):
def loading_and_cleaning_data(path: str):
    # papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
    papers = pd.read_csv(path)
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
    return papers


def remove_punctuation_and_convert_to_lowercase(papers):
    # papers[TEST_PROCESS] = papers[PAPER_TEXT].str.replace('[{}]'.format(string.punctuation), '')
    papers[TEST_PROCESS] = papers[PAPER_TEXT].map(lambda x: re.sub('[,\.!?]', '', x))
    papers[TEST_PROCESS] = papers[TEST_PROCESS].map(lambda x: x.lower())


def join_titles_to_list(papers):
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))
    return long_string


def exploratory_analysis(papers):
    print('got here1')
    long_string = join_titles_to_list(papers)
    wordcloud = WordCloud(background_color="white", max_words=4, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)
    print('Got here2')
    # Visualize the word cloud
    # topics = wordcloud.words_()
    # print(wordcloud)
    # wordcloud.to_image()
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()


def print_head(papers):
    print(papers.head())
    print(papers[TEST_PROCESS].head())


def print_text_processed_values(papers):
    # print(list(papers['paper_text_processed'].values))
    print(len(list(papers['paper_text_processed'].values)))


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
    print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])
    lda_model_training(corpus, id2word)


def lda_model_training(corpus, id2word):
    # number of topics
    num_topics = 4
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    # print(lda_model.print_topics()[0][1])
    doc_lda = lda_model[corpus]

    analyze_results(num_topics, lda_model, corpus, id2word)


def analyze_results(num_topics: int, lda_model: gensim.models.LdaMulticore, corpus: list, id2word):
    # Visualize the topics
    # pyLDAvis.enable_notebook()
    # pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_' + str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')
    print(LDAvis_prepared)


def main():
    path = './NIPS_Papers/papers.csv'
    papers = loading_and_cleaning_data(path)
    remove_punctuation_and_convert_to_lowercase(papers)
    exploratory_analysis(papers)
    prepare_data_to_lda(papers)


if __name__ == '__main__':
    main()
