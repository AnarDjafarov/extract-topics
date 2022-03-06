# Importing modules
import glob
import os
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


TEST_PROCESS = 'paper_text_processed'
PAPER_TEXT = 'paper_text'


def read_papers_from_csv_and_drop_colums(path: str):
    # papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
    papers = pd.read_csv(path)
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1)
    return papers


def remove_punctuation_and_convert_to_lowercase(papers):
    # papers[TEST_PROCESS] = papers[PAPER_TEXT].str.replace('[{}]'.format(string.punctuation), '')
    papers[TEST_PROCESS] = papers[PAPER_TEXT].map(lambda x: re.sub('[,\\.!?]', '', x))
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
    topics = wordcloud.words_()
    print(topics)
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


def main():
    path = './NIPS_Papers/papers.csv'
    papers = read_papers_from_csv_and_drop_colums(path)
    remove_punctuation_and_convert_to_lowercase(papers)
    # print_head(papers)
    # print_text_processed_values(papers)
    exploratory_analysis(papers)


if __name__ == '__main__':
    main()
