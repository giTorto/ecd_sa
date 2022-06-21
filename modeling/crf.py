import sklearn_crfsuite
from data_loading.loader import DataLoader
from sklearn_crfsuite import metrics


class CRFTagger:

    @staticmethod
    def word2features(sent, i, mask):
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

        return features

    @staticmethod
    def sent2features(sent, mask):
        return [CRFTagger.word2features(sent, i, mask) for i in range(len(sent))]

    def __init__(self, c1=0.1, c2=0.1, max_iterations=100, use_mask=False):

        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
        self.use_mask = use_mask
        self.labels = None

    def featurize(self, training_dataset: DataLoader):
        featurized_data = []
        for tokens, mask in zip(training_dataset.tokens, training_dataset.masks):
            sentence_feature_vector = CRFTagger.sent2features(tokens, mask if self.use_mask else None )
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

    def evaluate(self, predictions, reference):

        # group B and I results
        sorted_labels = sorted(
            self.labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            predictions, reference, labels=sorted_labels, digits=3
        ))
