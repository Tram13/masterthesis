import numpy as np
from collections import Counter


def online_bertopic_scoring_func(col, total_amount_topics):
    output = np.zeros(total_amount_topics)

    # add to main topic: sentiment_score
    np.add.at(output, col[0], col[1] * col[2])

    # normalize data per topic
    normalize_values = Counter(col[0])
    for index, count in normalize_values.items():
        output[index] /= count

    # rescale to [0, 1]
    output = (output + 1)/2

    return output


def bertopic_scoring_func(col, total_amount_topics, weight_main_topics=0.75):
    output = np.zeros(total_amount_topics)

    # sentiment_label * sentiment_probability
    sentiment_score = col[3] * col[4]

    # add to main topic: topic_probability * weight * sentiment_score
    np.add.at(output, col[0], col[2] * sentiment_score * weight_main_topics)

    # add to force topic: sentiment_score * (1-weight)
    np.add.at(output, col[1], sentiment_score * (1 - weight_main_topics))

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