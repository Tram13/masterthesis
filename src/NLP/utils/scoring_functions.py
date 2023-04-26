import numpy as np
from collections import Counter


# input col: (topic, [sentiment_label, sentiment_probability])
def online_bertopic_scoring_func(col, total_amount_topics, use_sentiment=True):
    output = np.zeros(total_amount_topics)

    # add to main topic: sentiment_score
    if use_sentiment:
        np.add.at(output, col[0], col[1] * col[2])
    else:
        # don't use sentiment and normalize later globally, so we know what topics are the most important
        np.add.at(output, col[0], 1)
        return output

    # normalize data per topic
    normalize_values = Counter(col[0])
    for index, count in normalize_values.items():
        output[index] /= count

    # rescale to [0, 1]
    output = (output + 1)/2

    return output


# outdated for offline bertopic
def bertopic_scoring_func(col, total_amount_topics, sentiment=True):
    output = np.zeros(total_amount_topics)

    # sentiment_label * sentiment_probability
    if sentiment:
        sentiment_score = col[1] * col[2]

        # add to main topic: topic_probability * weight * sentiment_score
        np.add.at(output, col[0], sentiment_score)
    else:
        np.add.at(output, col[0], 1)

    # normalize data per topic
    normalize_values = Counter(col[0])
    for index, count in normalize_values.items():
        output[index] /= count

    # rescale to [0, 1]
    output = (output + 1) / 2

    return output


def basic_clustering_scoring_func(col, total_amount_topics):
    output = np.zeros(total_amount_topics)

    # sentiment_label * sentiment_probability
    sentiment_score = col[1] * col[2]

    # add to main topic: sentiment_score
    np.add.at(output, col[0], sentiment_score)

    # normalize output by dividing by amount of sentences
    return output / len(col[0])
