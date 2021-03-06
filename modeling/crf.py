import sklearn_crfsuite
from data_loading.loader import DataLoader
from sklearn_crfsuite import metrics
import argparse
from evaluation.span_prediction import get_span_indexes, group_labels, evaluate
# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#evaluation

class CRFTagger:

    @staticmethod
    def word2features(sent, i, mask, valence=None):
        word = sent[i]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:]
        }
        if mask is not None:
            wmask = mask[i]
            features['mask'] = bool(wmask)
        if valence is not None:
            features['valence'] = valence if valence == 0 else 1

        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower()
            })
            if mask is not None:
                mask1 = mask[i - 1]
                features['-1:mask'] = bool(mask1)

        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
            })
            if mask is not None:
                mask1 = mask[i + 1]
                features['+1:mask'] = bool(mask1)

        else:
            features['EOS'] = True

        print(features)
        return features

    @staticmethod
    def sent2features(sent, mask, valence=None):
        return [CRFTagger.word2features(sent, i, mask, valence) for i in range(len(sent))]

    def __init__(self, c1=0.1, c2=0.1, max_iterations=100, use_mask=False, use_valence=False):

        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
        self.use_mask = use_mask
        self.use_valence = use_valence
        self.labels = None

    def featurize(self, training_dataset: DataLoader):
        featurized_data = []
        for tokens, mask, valence in zip(training_dataset.tokens, training_dataset.masks, training_dataset.valences):
            sentence_feature_vector = CRFTagger.sent2features(tokens, mask if self.use_mask else None, valence if self.use_valence else None )
            featurized_data.append(sentence_feature_vector)
        return featurized_data

    def train(self, training_dataset:DataLoader):
        featurized_data = self.featurize(training_dataset)

        self.labels = list(set([element for sublist in training_dataset.iobs for element in sublist]))

        self.crf.fit(featurized_data, training_dataset.iobs)

    def predict(self, test_dataset:DataLoader):
        featurized_data = self.featurize(test_dataset)

        predictions = self.crf.predict(featurized_data)

        return predictions

    def evaluate(self, predictions, reference, remove_o=False):

        # group B and I results
        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )
        if remove_o:
            sorted_labels.remove("O")

        results = metrics.flat_classification_report(
            reference, predictions, labels=sorted_labels, digits=3
        )

        print(results)
        return results


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train','--training-files',  action="store", dest="training_files",nargs='+',
                        help='the training file')
    parser.add_argument('--test','--test-files',  action="store", dest="test_files",nargs='+',
                        help='the test file')

    return parser


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()

    training_data = DataLoader(args.training_files, keep_iob=False)
    test_data = DataLoader(args.test_files, keep_iob=True)

    crf_tagger = CRFTagger(c1=0.30, c2=0.3,use_mask=True, use_valence=True)

    crf_tagger.train(training_data)

    predictions = crf_tagger.predict(test_data)

    span_indexes = get_span_indexes(test_data)

    golden_reference = group_labels(test_data.iobs, span_indexes, iob_tags=True)
    predictions = group_labels(predictions, span_indexes, iob_tags=False)

    evaluate(predictions, golden_reference)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
