import os
import pickle
import torch

SPECIAL_WORDS = {'PADDING': '<PAD>'}

def load_data(path):
    '''Load dataset from file'''
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    '''Preprocess text data'''
    text = load_data(dataset_path)

    # Ignore notice, I don't won't the model to use it for training
    text = text[81:]

    # Replace punctuation marks into tokens like ' ||period|| '
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    # Turn text to lowercase and return a list of words divided by spaces
    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(
        text + list(SPECIAL_WORDS.values())
    )
    # Replace every word in text with integer
    int_text = [vocab_to_int[word] for word in text]
    # Write text and dictionaries to a file
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict),
                open('preprocess.p', 'wb'))

def load_preprocess():
    '''Load the preprocessed data'''
    return pickle.load(open('preprocess.p', mode='rb'))

def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)

def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)
