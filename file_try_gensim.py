import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import file_try_gensim
import file_try_gensim.corpora as corpora
from file_try_gensim.utils import simple_preprocess
from file_try_gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

import logging
import warnings
from nltk.corpus import stopwords


# Enable logging for gensim - optional
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

