import jsonlines
import preprocessor as p
import nltk
from nltk import word_tokenize, pos_tag, SennaNERTagger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import unicodedata
from sklearn.utils import shuffle
import string
import pandas as pd
import argparse
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.tag.crf import CRFTagger
from sklearn.model_selection import train_test_split
import csv




############# Develop Feature List ###############
def my_features(tokens, idx):
    token = tokens[idx]

    feature_list = []

    if not token:
        return feature_list

    # Capitalization
    if token[0].isupper():
        feature_list.append("CAPITALIZATION")


    if re.search(r"\d", token) is not None:
        feature_list.append("HAS_NUM")

    # Event slot
    if token.lower() in ["test", "positive", "result"]:
        feature_list.append("P")

    if token.lower() in ["die", "dead", "death", "pass", "away"]:
        feature_list.append("D")

    # POS Features of the previous and subsequent two words
    feature_list.append(nltk.tag.pos_tag([token])[0][1])

    # WORD lemma
    feature_list.append("WORD_" + token)

    # Punctuation
    punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
    if all(unicodedata.category(x) in punc_cat for x in token):
        feature_list.append("PUNCTUATION")

    # Stop words
    if token in stopwords.words('english'):
        feature_list.append("STOPWORD")

    return feature_list

def neighbor_features(tokens, idx):

        token = tokens[idx]

        feature_list = []

        if not token:
            return feature_list

        # Capitalization
        if token[0].isupper():
            feature_list.append("CAPITALIZATION")

        # Has number
        if re.search(r"\d", token) is not None:
            feature_list.append("HAS_NUM")

        # Event slot
        if token.lower() in ["test", "positive","result"]:
            feature_list.append("P")

        if token.lower() in ["die","dead","death","pass","away"]:
            feature_list.append("D")

        # POS Features of the previous and subsequent two words
        for num in [-2,-1,0,1,2]:
            n_sent = len(tokens)
            if (idx + num < 0) or (idx + num > n_sent-1):
                feature_list.append("")
            else:
                token_neighbor = tokens[idx + num]
                feature_list.append(nltk.tag.pos_tag([token_neighbor])[0][1])
                feature_list.append("WORD_" + token_neighbor)

        # Punctuation
        punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
        if all(unicodedata.category(x) in punc_cat for x in token):
            feature_list.append("PUNCTUATION")

        # Stop words
        if token in stopwords.words('english'):
            feature_list.append("STOPWORD")

        # NER IOB
        #NERtagger = SennaNERTagger(
        #    '/Users/cynthiasu/PycharmProjects/final_project/extract_COVID19_events_from_Twitter-master/senna-v3.0')
        #feature_list.append(NERtagger.tag([token])[0][1])


        return feature_list


def main(positive, death):
    ############# Compile the dataset ###############
    ## Load the dataset
    text = list()
    response = list()
    file_path = [positive, death]

    for path in file_path:
        input_file = jsonlines.open(path)
        for obj in input_file:
            text.append(obj['text'])
            response.append(obj['annotation']['part1.Response'])

    ## Tweet Preprocessing
    prep_text = list()
    for i in text:
        prep_text.append(p.clean(i))

    ## Tag Keywords and Create Labels
    ### Focus on verbs--therefore, try lemmatization first
    wnl = WordNetLemmatizer()
    n_corpus = len(prep_text)
    token_data = ["test"] * n_corpus

    n = 0
    for sent in prep_text:
        token_data[n] = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for
                         i, j in pos_tag(word_tokenize(sent))]
        n = n + 1


    ### Create labels
    death_list = ["die", "dead", "death", "pass", "away"]

    n = 0
    for sent in token_data:
        for idx, token in enumerate(sent):
            if ((token.lower() in ["test", "positive","result"]) and (response[n] == ["yes"])):
                sent[idx] = [sent[idx], "P-Yes"]
            elif ((token.lower() in ["test", "positive","result"]) and (response[n] == ["no"])):
                sent[idx] = [sent[idx], "P-No"]
            elif ((token.lower() in death_list) and (response[n] == ["yes"])):
                sent[idx] = [sent[idx], "D-Yes"]
            elif ((token.lower() in death_list) and (response[n] == ["no"])):
                sent[idx] = [sent[idx], "D-No"]
            else:
                sent[idx] = [sent[idx], "Irr"]
        n = n + 1

    ## Shuffle and split into train data and dev data
    token_data = shuffle(token_data, random_state=6)
    train_data, dev_data = train_test_split(token_data, test_size=0.3, random_state=616)
    print(f"The number of sentences in training data: {len(train_data)}; The number of sentences in dev data: {len(dev_data)};")

    ############# Fit A CRF Model And Predict ###############
    condition_to_func = {"base": my_features, "include_neighbors": neighbor_features}
    for cond, func in condition_to_func.items():
        # initialize
        crf = CRFTagger(feature_func=func)
        crf.train(train_data, 'model.tagger')
        # Test
        crf._feature_func(prep_text[0].split(), 7)
        crf.tag_sents([['I', 'get', 'covid'], ['he', 'test', 'positive']])

        # Output
        filename = cond + "_final_output.tsv"
        with open(filename, 'w') as pred_file:
            for sent in dev_data:
                sent_words = [item[0] for item in sent]
                gold_tags = [item[1] for item in sent]

                with_tags = crf.tag(sent_words)
                for i, output in enumerate(with_tags):
                    original_word, tag_prediction = output
                    line_as_str = f"{original_word}\t{gold_tags[i]}\t{tag_prediction}\n"
                    pred_file.write(line_as_str)
                # add an empty line after each sentence
                pred_file.write("\n")


    ############# Evaluation ###############
    ## Extract Data with Meaning Labels
    cond_list = ['base', 'include_neighbors']

    for cond in cond_list:
        filename = cond + "_final_output.tsv"

        with open(filename) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            D_data = []
            P_data = []
            for row in rd:
                if len(row) > 1:
                    if row[1] in ['P-Yes', 'P-No']:
                        P_data.append(row)
                    elif row[1] in ['D-Yes', 'D-No']:
                        D_data.append(row)

        column_name = ['token', 'label', 'prediction']
        P_df = pd.DataFrame(P_data, columns=column_name)
        D_df = pd.DataFrame(D_data, columns=column_name)
        Total_df = P_df.append(D_df)

        # Accuracy
        ## Overall Accuracy
        T_a = accuracy_score(Total_df['label'], Total_df['prediction'])

        ## Accuracy, Precision, and Recall for two events
        accuracy = []
        precision = []
        recall = []
        for df in [P_df, D_df]:
            accuracy.append(accuracy_score(df['label'], df['prediction']))
            precision.append(sum(1 for item in range(0, len(df) - 1) if
                                 ('Yes' in df['label'][item] and 'Yes' in df['prediction'][item])) / sum(
                1 for item in range(0, len(df) - 1) if ('Yes' in df['prediction'][item])))
            recall.append(sum(1 for item in range(0, len(df) - 1) if
                              ('Yes' in df['label'][item] and 'Yes' in df['prediction'][item])) / sum(
                1 for item in range(0, len(df) - 1) if ('Yes' in df['label'][item])))

        ## F-1
        f1 = []
        for num in [0, 1]:
            f1.append((2 * precision[num] * recall[num]) / (precision[num] + recall[num]))

        # Report performance
        print("condition: " + cond)
        print(f"Overall Accuracy {T_a:0.03}")
        covid_event = ['Test Positive', 'Death Case']

        num = 0
        for event in covid_event:
            print(f"Scores for {event} : \taccuracy {accuracy[num]:0.03}\tprecision {precision[num]:0.03}\trecall {recall[num]:0.03}\tF1 {f1[num]:0.03}")
            num = num + 1

    ## Basicline Performance / Confusion Matrix
    print("Confusion Matrix:")
    print(pd.crosstab(Total_df['label'], Total_df['prediction']))
    print("Training data:")
    labels = ["P-Yes", "P-No", "D-Yes", "D-No"]
    for label in labels:
        train_data2 = np.concatenate(train_data).flat
        n_label = sum(1 for item in train_data2 if item == label)
        print(f"Number of {label}: {n_label}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='final project')
    parser.add_argument('--test_positive_path', type=str, default="data/positive-add_text.jsonl",
                        help='path to input with one tweet regarding test positive per line')
    parser.add_argument('--death_case_path', type=str, default="data/death-add_text.jsonl",
                        help='path to input with one tweet regarding death case per line')
    args = parser.parse_args()

    main(args.test_positive_path, args.death_case_path)


