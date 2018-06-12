from sklearn.model_selection import train_test_split
import Data
import numpy as np
import heapq


def similarity(query, document, doc_len):
    """
    Calculate similarity score for query and document based on LM with Lamplace
    """
    score = 0.0

    # Document dictionary
    doc_dict = {}
    for word in document:
        if word not in doc_dict:
            doc_dict[word] = 1
        else:
            doc_dict[word] += 1

    # Language model
    for word in query:
        if word in doc_dict:
            score += np.log(1 + (doc_dict[word] + 0.0) / doc_len)
    return score


def get_vocab_len(sentences):
    """
    Get vocabulary length
    """
    vocab_set = set()
    for sentence in sentences:
        for word in sentence:
            if word not in vocab_set:
                vocab_set.add(word)

    return len(vocab_set)


if __name__ == "__main__":
    # Read sentences and split train and test sets
    origin_sentences, pred_sentences = Data.read_pred_data("Pred-All-seasons.csv")
    org_sens_train, org_sens_test, pred_sens_train, pred_sens_test = train_test_split(origin_sentences, pred_sentences,
                                                                                      test_size=0.1, shuffle=False)
    # vocab_len = get_vocab_len(pred_sens_train)

    # Choose the most Top K similar one
    top_k = 5
    n_gram = 1
    bleu_list = []
    output = open("CNN.txt", 'w')
    for i in range(len(pred_sens_test) - 1):
        top = []
        for j in range(len(pred_sens_train) - 1):
            score = similarity(pred_sens_test[i], pred_sens_train[j], len(pred_sens_train[j]))
            heapq.heappush(top, (-score, str(j)))

        output.write("Sentence:" + org_sens_test[i] + "\n")
        best_bleu = 0.0
        for j in range(top_k):
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
