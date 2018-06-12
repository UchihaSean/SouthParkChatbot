import Data
from sklearn.model_selection import train_test_split
import numpy as np
import heapq


def generate_idf_dict(word_list):
    """
    Generate word dictionary based on train data
    """
    dict = {}
    for i in range(len(word_list)):
        flag = set()
        for j in range(len(word_list[i])):
            if word_list[i][j] in flag: continue
            if word_list[i][j] not in dict:
                dict[word_list[i][j]] = 1
            else:
                dict[word_list[i][j]] += 1
            flag.add(word_list[i][j])

    return dict


def generate_tf_idf_list(sentences, idf_dict):
    """
    Generate tf-idf for each word in each sentence
    """
    tf_idf = []
    for sentence in sentences:
        dict = {}
        # Get term frequency
        for word in sentence:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1

        # Calculate TF-IDF
        for word in dict:
            if word in idf_dict:
                dict[word] = (1 + np.log(dict[word])) * np.log(len(idf_dict) / (idf_dict[word] + 0.0))
            else:
                dict[word] = 0
        tf_idf.append(dict)

    return tf_idf


def cosine_similarity(dict_x, dict_y):
    """
    Calculate Cosine similarity
    """

    def multiply(dict_u, dict_v):
        """
        Multiply dictionaries
        """
        mul = 0.0
        for word in dict_u:
            if word in dict_v:
                mul += dict_u[word] * dict_v[word]
        return mul

    if len(dict_x) == 0 or len(dict_y) == 0: return 0.0
    return multiply(dict_x, dict_y) / (np.sqrt(multiply(dict_x, dict_x)) * np.sqrt(multiply(dict_y, dict_y)))


if __name__ == "__main__":
    # Read sentences and split train and test sets
    origin_sentences, pred_sentences = Data.read_pred_data("Pred-All-seasons.csv")
    org_sens_train, org_sens_test, pred_sens_train, pred_sens_test = train_test_split(origin_sentences, pred_sentences,
                                                                                      test_size=0.1, shuffle=False)

    # Calculate TF-IDF
    idf_dict = generate_idf_dict(pred_sens_train)
    pred_sens_train_tf_idf = generate_tf_idf_list(pred_sens_train, idf_dict)
    pred_sens_test_tf_idf = generate_tf_idf_list(pred_sens_test, idf_dict)

    # Choose the most similar one
    top_k = 5
    n_gram = 1
    bleu_list = []
    output = open("TFIDF3.txt", 'w')
    for i in range(len(pred_sens_test_tf_idf)):
        top = []
        for j in range(len(pred_sens_train_tf_idf) - 1):
            score = cosine_similarity(pred_sens_test_tf_idf[i], pred_sens_train_tf_idf[j])
            heapq.heappush(top, (-score, str(j)))
        output.write("Sentence:" + org_sens_test[i] + "\n")
        best_bleu = 0.0
        for j in range(top_k):
            # while np.random.random() > 0.2: heapq.heappop(top)
            item = int(heapq.heappop(top)[1])+1

            # BLEU evaluation
            bleu_score = Data.evaluation_bleu(pred_sens_train[item], pred_sens_test[i + 1], n_gram)
            if bleu_score > best_bleu:
                best_bleu = bleu_score
            output.write("Our reply " + str(j + 1) + ": " + org_sens_train[item] + "\n")
        output.write("Ground Truth:" + org_sens_test[i + 1] + "\n\n")
        bleu_list.append(best_bleu)
        # print best_bleu
        if i > 100: break

    output.close()
    print np.mean(bleu_list)
