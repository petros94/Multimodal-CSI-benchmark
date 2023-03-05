import pandas as pd
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import string

from MultimodalDataset import MultimodalDataset
from load_songs import load_lyrics
from utils import get_device


class TextModel:
    def __init__(self, vocabulary=None):
        self.punctuations = string.punctuation
        self.nlp = spacy.load('en_core_web_lg')

        self.tfidf_matrix = None
        self.tfidf_transformer = None
        self.dense_tfidf_matrix = None
        self.vocabulary = vocabulary

    # Creating our tokenizer function
    def spacy_tokenizer(self, sentence):
        mytokens = self.nlp(sentence)

        # Lemmatizing each token and converting each token into lowercase
        filtered = filter(lambda it: it.lemma_.isalpha(), mytokens)
        mytokens = list(map(lambda it: it.lemma_.lower().strip(), filtered))

        # Removing stop words
        mytokens = [word for word in mytokens if word not in self.punctuations]

        # return preprocessed list of tokens
        return mytokens

    def fit(self, text_data):
        tfidf_transformer = TfidfVectorizer(tokenizer=self.spacy_tokenizer,
                                            vocabulary=self.vocabulary,
                                            use_idf=False)
        tfidf_transformer.fit(text_data)
        self.tfidf_transformer = tfidf_transformer
        self.vocabulary = tfidf_transformer.vocabulary_

    def forward(self, text_data):
        device = get_device()
        tfidf_matrix = self.tfidf_transformer.transform(text_data)
        output = torch.Tensor(tfidf_matrix.todense()).to(device)
        return torch.nn.functional.normalize(output)


def make_model():
    model = TextModel()
    train_lyrics = [cover['lyrics']
                    for song_id, covers in
                    load_lyrics(song_dir="/Users/petrosmitseas/Documents/MscAI_text/data/custom_text_features").items()
                    for cover in covers]
    model.fit(train_lyrics)
    return model
