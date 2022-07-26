import warnings
from typing import List, Tuple
from data_loading.loader import DataLoader
from collections import Counter
from sklearn_crfsuite import metrics


def get_span_indexes(input_data:DataLoader):

    span_indexes = []
    for mask in input_data.iobs:

        iob_iterator = enumerate(iter(mask))
        index, iob_value = next(iob_iterator, None)
        start_span = None
        final_spans = []
        while iob_value is not None:

            if iob_value == 'O' and start_span is None:
                pass
            elif iob_value == 'O' and start_span is not None:
                final_spans.append((start_span, index - 1))
                start_span = None
            elif iob_value.startswith('B-') and start_span is not None:
                final_spans.append((start_span, index - 1))
                start_span = index
            elif iob_value.startswith('B-') and start_span is None:
                start_span = index
            else:
                # the tag is a -1
                pass

            index, iob_value = next(iob_iterator, (None, None))

        if start_span is not None:
            final_spans.append((start_span, len(mask)-1))

        span_indexes.append(final_spans)
    return span_indexes


def print_span_indexes(test_data: DataLoader, span_indexes: List[Tuple[int, int]]):
    for span_index, iob, mask in zip(span_indexes, test_data.iobs, test_data.masks):
        print(span_index, iob, mask)


def group_labels(iobs: List[str], span_indexes:List[Tuple[int,int]], iob_tags:bool=False):
    predictions = []
    for iob_list, spans in zip(iobs, span_indexes):
        fu_predictions = []
        for start_span, end_span in spans:
            iobs = iob_list[start_span:end_span+1]
            if iob_tags:
                iobs = [x.replace("B-", "").replace("I-", "") for x in iobs]

            fu_predictions.append(join_multi_tags(iobs))
        predictions.append(fu_predictions)

    return predictions


def join_multi_tags(fu_iob_tags: List[str]):
    if len(set(fu_iob_tags)) == 1:
        return fu_iob_tags[0]
    else:
        print("Majority voting required!!!!!!!!!!!")
        fu_iob_tags_counter = Counter(fu_iob_tags)

        return max(fu_iob_tags_counter, key=fu_iob_tags_counter.get)


def evaluate(predictions, reference):

    results = metrics.flat_classification_report(
        reference, predictions, digits=3
    )

    print(results)
    return results


def unified_evaluation(predictions, test_dataset):

    span_indexes = get_span_indexes(test_dataset)

    golden_reference = group_labels(test_dataset.iobs, span_indexes, iob_tags=True)
    predictions = group_labels(predictions, span_indexes, iob_tags=False)

    evaluate(predictions, golden_reference)


# the 2 types of evaluation: per FU / per span
# per FU = all spans right
# per Span = just span right
