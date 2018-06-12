import csv
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def read_origin_data(input_file_name, output_file_name):
    """
    Read CSV and preprocessed data sets
    Output preprocessed data sets
    """

    # Open file and read csv
    conversations = []
    with open(input_file_name, 'r') as csvfile:
        file_info = csv.reader(csvfile)
        # Store the sentences
        for i, line in enumerate(file_info):
            if i == 0: continue
            conversations.append(line[-1])

    pred_conversations = preprocessing(conversations)

    # Output
    with open(output_file_name, 'w') as csvfile:
        fieldnames = ['origin_sentence', 'pred_sentence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(len(conversations)):
            origin_sentence = conversations[i]
            pred_sentence = ""
            for j in range(len(pred_conversations[i])):
                if j < len(pred_conversations[i]) - 1:
                    pred_sentence += pred_conversations[i][j] + " "
                else:
                    pred_sentence += pred_conversations[i][j]
            writer.writerow({'origin_sentence': origin_sentence, 'pred_sentence': pred_sentence})

    return conversations, pred_conversations


def read_pred_data(file_name):
    """
    Read origin and preprocessed data
    """

    origin_sentences = []
    pred_sentences = []

    # Spilt origin and preprocessed sentences
    with open(file_name, 'r') as csvfile:
        file_info = csv.reader(csvfile)
        # Store the sentences
        for i, line in enumerate(file_info):
            if i == 0: continue
            origin_sentences.append(line[0][:-1])
            pred_sentences.append(line[1].split(" "))

    return origin_sentences, pred_sentences


def preprocessing(conversations):
    """
    Word Stemming and Stop words Removal
    """
    pred_conversations = []
    for i in range(len(conversations)):
        # Remove unique chars
        conversation = ""
        for j in range(len(conversations[i])):
            if ord(conversations[i][j]) < 128:
                conversation += conversations[i][j]
        sentence = []
        # word tokenize
        words = word_tokenize(conversation)
        removal = "?!.,( )"
        stop_words = set(stopwords.words('english'))
        stop_words.update(("'s", "n't", "'m", "'ve", "'re", "'d", "'"))
        for word in words:
            # Remove ?!.,
            pred_word = ""
            for j in range(len(word)):
                if word[j] in removal: continue
                pred_word += word[j]
            # Lower, word stemming and stop words removal
            if len(pred_word) != 0:
                pred_word = SnowballStemmer("english").stem(pred_word.lower())
                if pred_word in stop_words: continue
                sentence.append(pred_word)
        pred_conversations.append(sentence)

    return pred_conversations


def evaluation_bleu(eval_sentence, base_sentence, n_gram=2):
    """
    BLEU evaluation with n-gram
    """

    def generate_n_gram_set(sentence, n):
        """
        Generate word set based on n gram
        """
        n_gram_set = set()
        for i in range(len(sentence) - n + 1):
            word = ""
            for j in range(n):
                if j == n - 1:
                    word += sentence[i + j]
                else:
                    word += sentence[i + j] + " "
            n_gram_set.add(word)
        return n_gram_set

    if n_gram > len(eval_sentence) or n_gram > len(base_sentence): return 0.0
    base_n_gram_set = generate_n_gram_set(base_sentence, n_gram)
    eval_n_gram_set = generate_n_gram_set(eval_sentence, n_gram)

    bleu = 0.0
    for word in eval_n_gram_set:
        if word in base_n_gram_set:
            bleu += 1
    return bleu / len(eval_n_gram_set)

# read_origin_data("All-seasons.csv", "Pred-All-seasons.csv")
# read_pred_data("Pred-All-seasons.csv")
# print evaluation_bleu(['this','is','my','dog'],['this','is','dog'],2)
