# taken from https://github.com/vineetjohn/linguistic-style-transfer/blob/master/linguistic_style_transfer_model/train_word2vec_model.py
import argparse
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train_word2vec_model(text_file_path, model_file_path):
    # define training data
    # train model
    print("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path), min_count=1, size=300)
    # summarize the loaded model
    print("Model Details: {}".format(model))
    # save model
    model.save(model_file_path)
    print("Model saved")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--model-file-path", type=str, required=True)

    options = vars(parser.parse_args(args=argv))
    
    train_word2vec_model(options['text_file_path'], options['model_file_path'])

    print("Training Complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
