import os
import os.path
import sys
import csv
import pickle
import copy
import gensim
import numpy as np
import torch
from torch.nn.functional import softmax
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  
from tqdm import tqdm

def load_sentences(config, logger):
    sentences = {}
    for part in config.datasets[config.dataset]['parts']:
        file_path = os.path.join(config.data_path, config.dataset, config.datasets[config.dataset]['sentences'][part])
                                    
        sentences[part] = []
        logger.info(f"Reading file: {file_path}")
        with open(file_path) as f_obj:
            for line in f_obj:
                sentences[part].append(line)
        logger.info(f"finished reading {len(sentences[part])} sentences for {part} set")
    
    return sentences


def load_tuples(config, logger):
    contexts = {}
    curr_sents = {}
    next_sents = {}
    for part in config.datasets[config.dataset]['parts']:
        contexts[part] = []
        curr_sents[part] = []
        next_sents[part] = []
        
        file_path = os.path.join(config.data_path, config.dataset, config.datasets[config.dataset]['tuples'][part])
        logger.info(f"Reading file: {file_path}")
        with open(file_path) as f_obj:
            reader = csv.reader(f_obj, delimiter='\t')
            for row in reader:
                assert len(row) == 3, f'Error! row does not contain exactly three items! Count: {len(row)}'
                
                contexts[part].append(row[0])
                curr_sents[part].append(row[1])
                next_sents[part].append(row[2])

    return contexts, curr_sents, next_sents

def load_vectorizers(sentences, config, logger):
    
    all_sentences = []
    for part in config.datasets[config.dataset]['parts']:
        all_sentences.extend(sentences[part])
    
    
    if config.filter_stopwords == True:
        count_vectorizer = CountVectorizer(stop_words='english')
    else:
        count_vectorizer = CountVectorizer()
        
    tfidf_vectorizer = TfidfTransformer()
    
    doc_count_matrix = count_vectorizer.fit_transform(all_sentences)
    
    tfidf_vectorizer.fit(doc_count_matrix)
    
    logger.info(f'Vocabulary size: {len(count_vectorizer.vocabulary_)}')
    
    input_tensors = {}
    
    '''
    filt_sents = {}
    filt_count = {}
    
    f_obj_kws = open(os.path.join(config.save_path, 'keywords.txt'), 'w')
    f_obj_zkw = open(os.path.join(config.save_path, 'zero_keywords.txt'), 'w')
    for part in config.datasets[config.dataset]['parts']:
        input_tensors[part] = []
        filt_sents[part] = []
        filt_count[part] = 0
        for sentence in tqdm(sentences[part]):
            sentence = sentence.strip()
            count_row = count_vectorizer.transform([sentence])
            nnz_kws = count_row.nnz
            
            if nnz_kws == 0:
                f_obj_zkw.write(f'Zero keywords: {sentence}\n')
                filt_count[part] += 1
                continue

            tfidf_row = tfidf_vectorizer.transform(count_row)            
            sorted_indices = np.squeeze(np.argsort(tfidf_row.toarray()))
            input_tensor = torch.zeros(sorted_indices.shape[0])
            
            if nnz_kws > config.sample_count:
                index_num = config.sample_count
            else:
                index_num = nnz_kws
            
            input_tensor[sorted_indices[-index_num:]] = 1.0
            
            f_obj_kws.write(f'{sentence} -> {count_vectorizer.inverse_transform(input_tensor)}\n')
            
            input_tensor = input_tensor.unsqueeze(dim=0)
            
            filt_sents[part].append(sentence)
            input_tensors[part].append(input_tensor)
    
    f_obj_kws.close()        
    f_obj_zkw.close()
                
    for part in input_tensors.keys():
        input_tensors[part] = torch.cat(input_tensors[part], dim=0)
        logger.info(f'Part: {part} Size: {input_tensors[part].size()}')
        logger.info(f'Filtered sentences: {filt_count[part]}')
    '''
    
    return count_vectorizer, tfidf_vectorizer

def load_vocab_tokenizer(sentences, filters, vocab_size, config, logger):
    
    all_sentences = []
    for part in config.datasets[config.dataset]['parts']:
        all_sentences.extend(sentences[part])
        
    """
    Tokenizes a given input sequence of words.
    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary
    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence
    """
    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(all_sentences)

    word_index = dict()
    word_index['<PAD>'] = 0
    word_index['<UNK>'] = 1
    word_index['<SOS>'] = 2
    word_index['<EOS>'] = 3

    if vocab_size == -1:
        for i, word in enumerate(dict(tokenizer.word_index).keys()):
            word_index[word] = i + 4
    else:
        for i, word in enumerate(dict(tokenizer.word_index).keys()):
            if i > vocab_size:
                break
            
            word_index[word] = i + 4
    
    word2index = word_index
    index2word = dict(zip(word2index.values(), word2index.keys()))
    
    logger.info(f'Vocab size: {len(word2index)}')
    
    tokenizer.word_index = word_index

    '''
    input_x = tokenizer.texts_to_sequences(list(sentences))
    
    output_y = []
    seq_lengths = []
    for i, seq in tqdm(enumerate(input_x)):
        seq = seq[:max_num_words-1]
            
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['<UNK>'] for t in seq]
        
        seq_x = copy.deepcopy(seq)
        
        if direction == 'backward':
            seq_x.reverse()
        
        seq_x.insert(0, word_index['<SOS>'])
        input_x[i] = seq_x
        
        seq_lengths.append(len(seq_x))
        
        seq_y = copy.deepcopy(seq)
        seq_y.append(word_index['<EOS>'])
        output_y.append(seq_y)

    input_x = pad_sequences(input_x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['<PAD>'])
    output_y = pad_sequences(output_y, padding='post', truncating='post', maxlen=max_num_words, value=word_index['<PAD>'])
    '''

    return word2index, index2word, tokenizer


def load_embedding_matrix(word2index, emb_dim, emb_type, config, logger):
    
    embedding_matrix = np.random.uniform(-0.05, 0.05, size=(len(word2index), emb_dim))
    
    if emb_type == 'word2vec':
        w2v_model = gensim.models.Word2Vec.load(f'embeddings/word2vec/w2vmodel_{config.dataset}.pkl')
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = w2v_model.wv[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
    elif emb_type == 'glove':
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(f"embeddings/glove/gensim_glove.6B.300d.txt", binary=False)
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = glove_model[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
    
    return embedding_matrix


def load_sentence_batch(sentences, indices, tokenizer, word_index, vocab_size, max_seq_len, config, logger):
    
    input_sents = []
    for index in indices:
        input_sents.append(sentences[index])
    
    input_x = tokenizer.texts_to_sequences(list(input_sents))
    
    output_y = []
    seq_lengths = []
    for i, seq in enumerate(input_x):
        seq = seq[:max_seq_len-1]
        
        if vocab_size != -1:
            if any(t >= vocab_size for t in seq):
                seq = [t if t < vocab_size else word_index['<UNK>'] for t in seq]
        
        seq_x = copy.deepcopy(seq)
        
        seq_x.insert(0, word_index['<SOS>'])
        input_x[i] = seq_x
        
        seq_lengths.append(len(seq_x))
        
        seq_y = copy.deepcopy(seq)
        seq_y.append(word_index['<EOS>'])
        output_y.append(seq_y)

    input_x = pad_sequences(input_x, padding='post', truncating='post', maxlen=max_seq_len, value=word_index['<PAD>'])
    output_y = pad_sequences(output_y, padding='post', truncating='post', maxlen=max_seq_len, value=word_index['<PAD>'])
    
    
    return input_x, output_y, seq_lengths   


def load_bow_batch(sentences, indices, count_vectorizer, tfidf_vectorizer, config, logger, epoch):
    
    if epoch == 1:
        f_obj_kws = open(os.path.join(config.save_path, 'keywords.txt'), 'a')
        f_obj_zkw = open(os.path.join(config.save_path, 'zero_keywords.txt'), 'a')
    
    input_tensors = []
    for index in indices:
        sentence = sentences[index].strip()
        
        count_row = count_vectorizer.transform([sentence])
        nnz_kws = count_row.nnz
            
        if nnz_kws == 0:
            if epoch == 1:
                f_obj_zkw.write(f'Zero keywords: {sentence}\n')
            continue

        tfidf_row = tfidf_vectorizer.transform(count_row)            
        sorted_indices = np.squeeze(np.argsort(tfidf_row.toarray()))
        input_tensor = torch.zeros(sorted_indices.shape[0])
            
        if nnz_kws > config.sample_count:
            index_num = config.sample_count
        else:
            index_num = nnz_kws
            
        input_tensor[sorted_indices[-index_num:]] = 1.0
        
        if epoch == 1:
            f_obj_kws.write(f'{sentence} -> {count_vectorizer.inverse_transform(input_tensor)}\n')
            
        input_tensor = input_tensor.unsqueeze(dim=0)
            
        input_tensors.append(input_tensor)
        
    
    if epoch == 1:
        f_obj_kws.close()        
        f_obj_zkw.close()
    
    if len(input_tensors) > 0:
        return torch.cat(input_tensors, dim=0)
    else:
        return None


def load_gan_batch(contexts, curr_sents, next_sents, indices, count_vectorizer, tfidf_vectorizer, tokenizer, word_index, config, logger):
    
    context_seq = []
    curr_sent_seq = []
    next_sent_seq = []
    
    curr_sent_vecs = []
    next_sent_vecs = []
    for index in indices:
        
        curr_sent = curr_sents[index].strip()
        
        next_sent = next_sents[index].strip()
                
        curr_sent_count_row = count_vectorizer.transform([curr_sent])
        curr_sent_nnz_kws = curr_sent_count_row.nnz
        
        next_sent_count_row = count_vectorizer.transform([next_sent])
        next_sent_nnz_kws = next_sent_count_row.nnz
            
        if curr_sent_nnz_kws == 0 or next_sent_nnz_kws == 0:
            #if epoch == 1:
            #    f_obj_zkw.write(f'Zero keywords: {sentence}\n')
            continue
        
        context_seq.append(contexts[index])
        curr_sent_seq.append(curr_sent)
        next_sent_seq.append(next_sent)

        curr_sent_tfidf_row = tfidf_vectorizer.transform(curr_sent_count_row)            
        curr_sent_sorted_indices = np.squeeze(np.argsort(curr_sent_tfidf_row.toarray()))
        curr_sent_vec = torch.zeros(curr_sent_sorted_indices.shape[0])
            
        if curr_sent_nnz_kws > config.sample_count:
            index_num = config.sample_count
        else:
            index_num = curr_sent_nnz_kws
            
        curr_sent_vec[curr_sent_sorted_indices[-index_num:]] = 1.0
        
        #if epoch == 1:
        #    f_obj_kws.write(f'{sentence} -> {count_vectorizer.inverse_transform(input_tensor)}\n')
            
        curr_sent_vec = curr_sent_vec.unsqueeze(dim=0)
            
        curr_sent_vecs.append(curr_sent_vec)
        
        next_sent_tfidf_row = tfidf_vectorizer.transform(next_sent_count_row)            
        next_sent_sorted_indices = np.squeeze(np.argsort(next_sent_tfidf_row.toarray()))
        next_sent_vec = torch.zeros(next_sent_sorted_indices.shape[0])
            
        if next_sent_nnz_kws > config.sample_count:
            index_num = config.sample_count
        else:
            index_num = next_sent_nnz_kws
            
        next_sent_vec[next_sent_sorted_indices[-index_num:]] = 1.0
        
        #if epoch == 1:
        #    f_obj_kws.write(f'{sentence} -> {count_vectorizer.inverse_transform(input_tensor)}\n')
            
        next_sent_vec = next_sent_vec.unsqueeze(dim=0)
            
        next_sent_vecs.append(next_sent_vec)
    
    
    context_seq = tokenizer.texts_to_sequences(list(context_seq))
    
    context_seq_lens = []
    for i, seq in enumerate(context_seq):
        seq = seq[:config.lm_max_seq_len-1]
        
        if config.lm_vocab_size != -1:
            if any(t >= config.lm_vocab_size for t in seq):
                seq = [t if t < config.lm_vocab_size else word_index['<UNK>'] for t in seq]
        
        seq_x = copy.deepcopy(seq)
        
        seq_x.insert(0, word_index['<SOS>'])
        context_seq[i] = seq_x
        
        context_seq_lens.append(len(seq_x))
        
    context_seq = pad_sequences(context_seq, padding='post', truncating='post', maxlen=config.lm_max_seq_len, value=word_index['<PAD>'])
    
    curr_sent_seq = tokenizer.texts_to_sequences(list(curr_sent_seq))
    
    curr_sent_seq_lens = []
    for i, seq in enumerate(curr_sent_seq):
        seq = seq[:config.lm_max_seq_len-1]
        
        if config.lm_vocab_size != -1:
            if any(t >= config.lm_vocab_size for t in seq):
                seq = [t if t < config.lm_vocab_size else word_index['<UNK>'] for t in seq]
        
        seq_x = copy.deepcopy(seq)
        
        seq_x.insert(0, word_index['<SOS>'])
        curr_sent_seq[i] = seq_x
        
        curr_sent_seq_lens.append(len(seq_x))
        
    curr_sent_seq = pad_sequences(curr_sent_seq, padding='post', truncating='post', maxlen=config.lm_max_seq_len, value=word_index['<PAD>'])
    
    #if epoch == 1:
    #    f_obj_kws.close()        
    #    f_obj_zkw.close()
    
    if len(next_sent_vecs) > 0:
        curr_sent_vecs = torch.cat(curr_sent_vecs, dim=0)
        next_sent_vecs = torch.cat(next_sent_vecs, dim=0)
        
        return context_seq, curr_sent_seq, next_sent_seq, context_seq_lens, curr_sent_seq_lens, curr_sent_vecs, next_sent_vecs
        
    else:
        return None, None, None, None, None, None, None
    

    
def load_lm_dataset(sentences, config, logger):
    
    prep_path = os.path.join(config.data_path, config.dataset, 'prepared')
        
    prep_file_path = os.path.join(prep_path, 'indexed_data.pkl')
    if os.path.isfile(prep_file_path):
        logger.info('Found saved data files. Loading saved data files.')
        logger.info(f'Loading file: {prep_file_path}')
        with open(prep_file_path, 'rb') as f_obj:
            indexed_data = pickle.load(f_obj)
        
        prep_file_path = os.path.join(prep_path, 'word2index.pkl')
        logger.info(f'Loading file: {prep_file_path}')    
        with open(prep_file_path, 'rb') as f_obj:
            word2index = pickle.load(f_obj)
        
        prep_file_path = os.path.join(prep_path, 'index2word.pkl')
        logger.info(f'Loading file: {prep_file_path}')    
        with open(prep_file_path, 'rb') as f_obj:
            index2word = pickle.load(f_obj)

        prep_file_path = os.path.join(prep_path, 'embedding_matrix.pkl')
        logger.info(f'Loading file: {prep_file_path}')    
        with open(prep_file_path, 'rb') as f_obj:
            embedding_matrix = pickle.load(f_obj)
                
        return indexed_data, word2index, index2word, embedding_matrix 
    
    part_indices = []
    start_index = 0
    all_sentences = []
    for part in config.datasets[config.dataset]['parts']:
        part_indices.append((start_index, start_index + len(sentences[part])))
        start_index += len(sentences[part])
        all_sentences.extend(sentences[part])
        
    logger.info(f"Part indices: {part_indices}")
        
    input_x, output_y, seq_lengths, word_index = tokenize_sequence(all_sentences, config.lm_filters, config.lm_max_seq_len, config.lm_vocab_size)
        
    word2index = word_index
    index2word = dict(zip(word2index.values(), word2index.keys()))
    
    logger.info(f'Vocabulary size: {len(word2index)}')
        
    indexed_data = {}
    for index, part in enumerate(config.datasets[config.dataset]['parts']):
        start_idx, end_idx = part_indices[index]
        
        indexed_data[part] = {}
        indexed_data[part]['input_x'] = input_x[start_idx:end_idx]
        indexed_data[part]['output_y'] = output_y[start_idx:end_idx]
        indexed_data[part]['seq_lengths'] = np.asarray(seq_lengths[start_idx:end_idx])
        logger.info(part)
        logger.info(indexed_data[part]['input_x'].shape)
        logger.info(indexed_data[part]['output_y'].shape)
        logger.info(indexed_data[part]['seq_lengths'].shape)
    
    
    '''
    for part in config.datasets[config.dataset]['parts']:
        logger.info(part)
        logger.info(f"input_x: {indices2words(indexed_data[part]['input_x'][0], index2word)}")
        logger.info(f"output_y: {indices2words(indexed_data[part]['output_y'][0], index2word)}")
        logger.info(f"seq_lengths: {indexed_data[part]['seq_lengths'][0]}")
    '''
    
    embedding_matrix = np.random.uniform(-0.05, 0.05, size=(len(word2index), config.lm_embedding_dim))
    
    if config.lm_embedding == 'word2vec':
        w2v_model = gensim.models.Word2Vec.load(f'embeddings/word2vec/w2vmodel_{config.dataset}.pkl')
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = w2v_model.wv[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
    elif config.lm_embedding == 'glove':
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(f"embeddings/glove/gensim_glove.6B.300d.txt", binary=False)
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = glove_model[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
    
    logger.info(f'Missing embedding tokens count: {missing_word_count}')
            
    prep_file_path = os.path.join(prep_path, 'indexed_data.pkl')
    logger.info(f'Saving indexed_data to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(indexed_data, f_obj)
    
    prep_file_path = os.path.join(prep_path, 'word2index.pkl')
    logger.info(f'Saving word2index to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(word2index, f_obj)
    
    prep_file_path = os.path.join(prep_path, 'index2word.pkl')
    logger.info(f'Saving index2word to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(index2word, f_obj)
    
    prep_file_path = os.path.join(prep_path, 'embedding_matrix.pkl')
    logger.info(f'Saving word2index to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(embedding_matrix, f_obj)
            
    return indexed_data, word2index, index2word, embedding_matrix 


def load_partitioned_lm_dataset(contexts, queries, responses, config, logger):
        
    part_indices = {'contexts': [], 'queries': [], 'responses': []}
    start_index = 0
    all_sentences = []
    for part in config.datasets[config.dataset]['parts']:
        part_indices['contexts'].append((start_index, start_index + len(contexts[part])))
        start_index += len(contexts[part])
        all_sentences.extend(contexts[part])
        
        part_indices['queries'].append((start_index, start_index + len(queries[part])))
        start_index += len(queries[part])
        all_sentences.extend(queries[part])
        
        part_indices['responses'].append((start_index, start_index + len(responses[part])))
        start_index += len(responses[part])
        all_sentences.extend(responses[part])
        
        
    logger.info(f"Part indices: {part_indices}")
        
    input_x, output_y, seq_lengths, word_index = tokenize_sequence(all_sentences, config.lm_filters, config.lm_max_seq_len, config.lm_vocab_size)
        
    word2index = word_index
    index2word = dict(zip(word2index.values(), word2index.keys()))
    
    logger.info(f'Vocabulary size: {len(word2index)}')
        
    indexed_data = {'contexts': {}, 'queries': {}, 'responses': {}}
    for index, part in enumerate(config.datasets[config.dataset]['parts']):
        contexts_start_idx, contexts_end_idx = part_indices['contexts'][index]
        queries_start_idx, queries_end_idx = part_indices['queries'][index]
        responses_start_idx, responses_end_idx = part_indices['responses'][index]
        
        indexed_data['contexts'][part] = {}
        indexed_data['queries'][part] = {}
        indexed_data['responses'][part] = {}
        
        indexed_data['contexts'][part]['input_x'] = input_x[contexts_start_idx:contexts_end_idx]
        indexed_data['contexts'][part]['output_y'] = output_y[contexts_start_idx:contexts_end_idx]
        indexed_data['contexts'][part]['seq_lengths'] = np.asarray(seq_lengths[contexts_start_idx:contexts_end_idx])
        
        indexed_data['queries'][part]['input_x'] = input_x[queries_start_idx:queries_end_idx]
        indexed_data['queries'][part]['output_y'] = output_y[queries_start_idx:queries_end_idx]
        indexed_data['queries'][part]['seq_lengths'] = np.asarray(seq_lengths[queries_start_idx:queries_end_idx])
        
        indexed_data['responses'][part]['input_x'] = input_x[responses_start_idx:responses_end_idx]
        indexed_data['responses'][part]['output_y'] = output_y[responses_start_idx:responses_end_idx]
        indexed_data['responses'][part]['seq_lengths'] = np.asarray(seq_lengths[responses_start_idx:responses_end_idx])
        logger.info(part)
        
        logger.info(indexed_data['contexts'][part]['input_x'].shape)
        logger.info(indexed_data['contexts'][part]['output_y'].shape)
        logger.info(indexed_data['contexts'][part]['seq_lengths'].shape)
        logger.info(indexed_data['queries'][part]['input_x'].shape)
        logger.info(indexed_data['queries'][part]['output_y'].shape)
        logger.info(indexed_data['queries'][part]['seq_lengths'].shape)
        logger.info(indexed_data['responses'][part]['input_x'].shape)
        logger.info(indexed_data['responses'][part]['output_y'].shape)
        logger.info(indexed_data['responses'][part]['seq_lengths'].shape)
        

    embedding_matrix = np.random.uniform(-0.05, 0.05, size=(len(word2index), config.lm_embedding_dim))
    
    if config.lm_embedding == 'word2vec':
        w2v_model = gensim.models.Word2Vec.load(f'embeddings/word2vec/w2vmodel_{config.dataset}.pkl')
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = w2v_model.wv[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
    elif config.lm_embedding == 'glove':
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(f"embeddings/glove/gensim_glove.6B.300d.txt", binary=False)
        
        missing_word_count = 0
        for word, index in word2index.items():
            try:
                emb_vector = glove_model[word]
                embedding_matrix[index] = emb_vector
            except KeyError:
                missing_word_count += 1
                
    logger.info(f'Missing embedding tokens count: {missing_word_count}')
            
    return indexed_data, word2index, index2word, embedding_matrix


def load_dialog_dataset(contexts, queries, responses, config, logger):
    
    prep_path = os.path.join(config.data_path, config.dataset, 'prepared')
    prep_file_path = os.path.join(prep_path, 'dialog_sents.pickle')
    if os.path.isfile(prep_file_path):
        logger.info('Found saved data files. Loading saved data files.')
        logger.info(f'Loading file: {prep_file_path}')
        with open(prep_file_path, 'rb') as f_obj:
            dialog_sents = pickle.load(f_obj)
        
        prep_file_path = os.path.join(prep_path, 'count_vectorizer.pickle')
        logger.info(f'Loading file: {prep_file_path}')    
        with open(prep_file_path, 'rb') as f_obj:
            count_vectorizer = pickle.load(f_obj)
        
        prep_file_path = os.path.join(prep_path, 'non_elim_indices.pickle')
        logger.info(f'Loading file: {prep_file_path}')    
        with open(prep_file_path, 'rb') as f_obj:
            non_elim_indices = pickle.load(f_obj)

        dialog_tensors = {'queries': {}, 'responses': {}}
        for part in config.datasets[config.dataset]['parts']:
            queries_path = os.path.join(prep_path, f'{part}_queries.pt')
            responses_path = os.path.join(prep_path, f'{part}_responses.pt')
            
            logger.info(f'Loading file: {queries_path}')
            dialog_tensors["queries"][part] = torch.load(queries_path)
            logger.info(f'Loading file: {responses_path}')
            dialog_tensors["responses"][part] = torch.load(responses_path)
                
        return count_vectorizer, dialog_sents, dialog_tensors, non_elim_indices

    
    sentences = []
    for part in config.datasets[config.dataset]['parts']:
        sentences.extend(queries[part])
        sentences.extend(responses[part])
    
    count_vectorizer = CountVectorizer(stop_words='english')
    #vectorizer = CountVectorizer(binary=True, stop_words='english')
    
    doc_count_matrix = count_vectorizer.fit_transform(sentences)
    tfidf_vectorizer = TfidfTransformer()
    
    tfidf_vectorizer.fit(doc_count_matrix)
    
    logger.info(f'Vocabulary size: {len(count_vectorizer.vocabulary_)}')
    
    dialog_sents = {'contexts': {}, 'queries': {}, 'responses': {}}
    dialog_tensors = {'queries': {}, 'responses': {}}
    filt_count = {}
    
    f_obj_kws = open(os.path.join(config.save_path, 'keywords.txt'), 'w')
    f_obj_zkw = open(os.path.join(config.save_path, 'zero_keywords.txt'), 'w')
    non_elim_indices = {}
    for part in config.datasets[config.dataset]['parts']:
        dialog_sents['contexts'][part] = []
        dialog_sents['queries'][part] = []
        dialog_sents['responses'][part] = []
        dialog_tensors['queries'][part] = []
        dialog_tensors['responses'][part] = []
        non_elim_indices[part] = []
        
        filt_count[part] = 0
        
        for row_index in tqdm(range(len(queries[part]))):
            context_sent = contexts[part][row_index]
            query_sent = queries[part][row_index]
            resp_sent = responses[part][row_index]
            
            query_count_row = count_vectorizer.transform([query_sent])
            resp_count_row = count_vectorizer.transform([resp_sent])
            
            query_kw_count = query_count_row.nnz
            resp_kw_count = resp_count_row.nnz
            
            if query_kw_count == 0 or resp_kw_count == 0:
                f_obj_zkw.write(f'Zero keywords - Query: {query_sent} -> Response: {resp_sent}\n')
                filt_count[part] += 1
                continue
            else:
                non_elim_indices[part].append(row_index)

            query_tfidf_row = tfidf_vectorizer.transform(query_count_row)            
            query_sorted_indices = np.squeeze(np.argsort(query_tfidf_row.toarray()))
            query_input_tensor = torch.zeros(query_sorted_indices.shape[0])
            
            if query_kw_count > config.sample_count:
                index_num = config.sample_count
            else:
                index_num = query_kw_count
                
            dialog_sents['contexts'][part] = context_sent    
            
            query_input_tensor[query_sorted_indices[-index_num:]] = 1.0
            
            query_input_tensor = query_input_tensor.unsqueeze(dim=0)
            
            dialog_sents['queries'][part].append(query_sent)
            dialog_tensors['queries'][part].append(query_input_tensor)
            
            resp_tfidf_row = tfidf_vectorizer.transform(resp_count_row)            
            resp_sorted_indices = np.squeeze(np.argsort(resp_tfidf_row.toarray()))
            resp_input_tensor = torch.zeros(resp_sorted_indices.shape[0])
            
            if resp_kw_count > config.sample_count:
                index_num = config.sample_count
            else:
                index_num = resp_kw_count
            
            resp_input_tensor[resp_sorted_indices[-index_num:]] = 1.0
            
            resp_input_tensor = resp_input_tensor.unsqueeze(dim=0)
            
            dialog_sents['responses'][part].append(resp_sent)
            dialog_tensors['responses'][part].append(resp_input_tensor)
            
            f_obj_kws.write(f'Query --> {query_sent} -> {count_vectorizer.inverse_transform(query_input_tensor)}\n')
            f_obj_kws.write(f'Response --> {resp_sent} -> {count_vectorizer.inverse_transform(resp_input_tensor)}\n')

    f_obj_kws.close()        
    f_obj_zkw.close()    
    
    prep_file_path = os.path.join(prep_path, 'dialog_sents.pickle')
    logger.info(f'Saving dialog_sent to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(dialog_sents, f_obj)
    
    prep_file_path = os.path.join(prep_path, 'count_vectorizer.pickle')
    logger.info(f'Saving count_vectorizer to pickle: {prep_file_path}')
    with open(prep_file_path, 'wb') as f_obj:
        pickle.dump(count_vectorizer, f_obj)
    
    prep_file_path = os.path.join(prep_path, 'non_elim_indices.pickle')
    with open(prep_file_path, 'wb') as f_obj:
        logger.info(f'Saving non_elim_indices to pickle: {prep_file_path}')
        pickle.dump(non_elim_indices, f_obj)
    
    for part in config.datasets[config.dataset]['parts']:
        logger.info(f'Dataset partition: {part}')
        dialog_tensors['queries'][part] = torch.cat(dialog_tensors['queries'][part], dim=0)
        dialog_tensors['responses'][part] = torch.cat(dialog_tensors['responses'][part], dim=0)
        logger.info(f'Queries size: {dialog_tensors["queries"][part].size()}')
        logger.info(f'Responses size: {dialog_tensors["responses"][part].size()}')
        logger.info(f'Filtered items: {filt_count[part]}')
        
        queries_save_path = os.path.join(prep_path, f'{part}_queries.pt')
        logger.info(f'queries_save_path: {queries_save_path}')
        torch.save(dialog_tensors["queries"][part], queries_save_path)
        
        
        responses_save_path = os.path.join(prep_path, f'{part}_responses.pt')
        logger.info(f'responses_save_path: {queries_save_path}')
        torch.save(dialog_tensors["responses"][part], responses_save_path)
        
    return count_vectorizer, dialog_sents, dialog_tensors, non_elim_indices


def load_gan_dataset(config, logger):
    
    contexts, queries, responses = load_dialogs(config, logger)
    
    count_vectorizer, dialog_sents, dialog_tensors, non_elim_indices = load_dialog_dataset(contexts, queries, responses, config, logger)
    
    indexed_data, word2index, index2word, embedding_matrix = load_partitioned_lm_dataset(contexts, queries, responses, config, logger)
    
    context_seqs = {}
    context_seq_lengths = {}
    query_seqs = {}
    query_seq_lengths = {}
    response_seqs = {}
    response_seq_lengths = {}
    query_vectors = {}
    response_vectors = {}
    for part in config.datasets[config.dataset]['parts']:
        context_seqs[part] = torch.tensor(indexed_data['contexts'][part]['input_x'], dtype=torch.long)
        context_seqs[part] = context_seqs[part][non_elim_indices[part]]
        context_seq_lengths[part] = torch.tensor(indexed_data['contexts'][part]['seq_lengths'], dtype=torch.long)
        context_seq_lengths[part] = context_seq_lengths[part][non_elim_indices[part]]
        
        query_seqs[part] = torch.tensor(indexed_data['queries'][part]['input_x'], dtype=torch.long)
        query_seqs[part] = query_seqs[part][non_elim_indices[part]]
        query_seq_lengths[part] = torch.tensor(indexed_data['queries'][part]['seq_lengths'], dtype=torch.long)
        query_seq_lengths[part] = query_seq_lengths[part][non_elim_indices[part]]
        
        response_seqs[part] = torch.tensor(indexed_data['responses'][part]['input_x'], dtype=torch.long)
        response_seqs[part] = response_seqs[part][non_elim_indices[part]]
        response_seq_lengths[part] = torch.tensor(indexed_data['responses'][part]['seq_lengths'], dtype=torch.long)
        response_seq_lengths[part] = response_seq_lengths[part][non_elim_indices[part]]
        
        query_vectors[part] = dialog_tensors['queries'][part]
        response_vectors[part] = dialog_tensors['responses'][part]
        
        
        logger.info(context_seqs[part].size())
        logger.info(context_seq_lengths[part].size())
        logger.info(query_seqs[part].size())
        logger.info(query_seq_lengths[part].size())
        logger.info(response_seqs[part].size())
        logger.info(response_seq_lengths[part].size())
        logger.info(query_vectors[part].size())
        logger.info(response_vectors[part].size())
        
        logger.info(indices2words(context_seqs[part][-2].tolist(), index2word))
        logger.info(context_seq_lengths[part][-2])
        logger.info(indices2words(query_seqs[part][-2].tolist(), index2word))
        logger.info(query_seq_lengths[part][-2])
        logger.info(indices2words(response_seqs[part][-2].tolist(), index2word))
        logger.info(response_seq_lengths[part][-2])
        logger.info(count_vectorizer.inverse_transform(query_vectors[part][-2].tolist()))
        logger.info(count_vectorizer.inverse_transform(response_vectors[part][-2].tolist()))
        

    return context_seqs, context_seq_lengths, query_seqs, query_seq_lengths, response_seqs, response_seq_lengths, query_vectors,  response_vectors, count_vectorizer, index2word


def load_classifier_dataset(config, logger):
    
    contexts, queries, responses = load_dialogs(config, logger)
    
    count_vectorizer, dialog_sents, dialog_tensors, non_elim_indices = load_dialog_dataset(contexts, queries, responses, config, logger)
    
    indexed_data, word2index, index2word, embedding_matrix = load_partitioned_lm_dataset(contexts, queries, responses, config, logger)
    
    context_seqs = {}
    context_seq_lengths = {}
    query_seqs = {}
    query_seq_lengths = {}
    response_vectors = {}
    labels = {}
    for part in config.datasets[config.dataset]['parts']:
        context_seqs_tensor = torch.tensor(indexed_data['contexts'][part]['input_x'], dtype=torch.long)
        context_seqs_tensor = context_seqs_tensor[non_elim_indices[part]]
        context_seq_lengths_tensor = torch.tensor(indexed_data['contexts'][part]['seq_lengths'], dtype=torch.long)
        context_seq_lengths_tensor = context_seq_lengths_tensor[non_elim_indices[part]]
        query_seqs_tensor = torch.tensor(indexed_data['queries'][part]['input_x'], dtype=torch.long)
        query_seqs_tensor = query_seqs_tensor[non_elim_indices[part]]
        response_seqs_tensor = torch.tensor(indexed_data['responses'][part]['input_x'], dtype=torch.long)
        response_seqs_tensor = response_seqs_tensor[non_elim_indices[part]]
        query_seq_lengths_tensor = torch.tensor(indexed_data['queries'][part]['seq_lengths'], dtype=torch.long)
        query_seq_lengths_tensor = query_seq_lengths_tensor[non_elim_indices[part]]
        response_vectors_tensor = dialog_tensors['responses'][part]
        
        logger.info(context_seqs_tensor.size())
        logger.info(context_seq_lengths_tensor.size())
        logger.info(query_seqs_tensor.size())
        logger.info(query_seq_lengths_tensor.size())
        logger.info(response_vectors_tensor.size())
        
        shuffled_indices = torch.randperm(query_seqs_tensor.size(0))
        
        for index in range(shuffled_indices.size(0)):
            if index == shuffled_indices[index]:
                if index == 0:
                    shuffled_indices[index] = torch.randint(low=1, high=shuffled_indices.size(0), size=(1,)).item()
                elif index == (shuffled_indices.size(0)-1):
                    shuffled_indices[index] = torch.randint(shuffled_indices.size(0)-1, (1,)).item()
                else:
                    shuffled_indices[index] = torch.randint(high=index, size=(1,)).item()
        
        logger.info(indices2words(context_seqs_tensor[-2].tolist(), index2word))
        logger.info(context_seq_lengths_tensor[-2])
        logger.info(indices2words(query_seqs_tensor[-2].tolist(), index2word))
        logger.info(query_seq_lengths_tensor[-2])
        logger.info(indices2words(response_seqs_tensor[-2].tolist(), index2word))
        logger.info(count_vectorizer.inverse_transform(response_vectors_tensor[-2].tolist()))
        
        context_seqs[part] = torch.cat((context_seqs_tensor, context_seqs_tensor), dim=0)
        context_seq_lengths[part] = torch.cat((context_seq_lengths_tensor, context_seq_lengths_tensor))
        query_seqs[part] = torch.cat((query_seqs_tensor, query_seqs_tensor), dim=0)
        query_seq_lengths[part] = torch.cat((query_seq_lengths_tensor, query_seq_lengths_tensor))
        response_vectors[part] = torch.cat((response_vectors_tensor, response_vectors_tensor[shuffled_indices]), dim=0)
        labels[part] = torch.cat((torch.ones((query_seqs_tensor.size(0),), dtype=torch.float32), torch.zeros((query_seqs_tensor.size(0),), dtype=torch.float32)))
        
        logger.info(context_seqs[part].size())
        logger.info(context_seq_lengths[part].size())
        logger.info(query_seqs[part].size())
        logger.info(query_seq_lengths[part].size())
        logger.info(response_vectors[part].size())
        logger.info(labels[part].size())
        
        logger.info(indices2words(context_seqs[part][-2].tolist(), index2word))
        logger.info(context_seq_lengths[part][-2])
        logger.info(indices2words(query_seqs[part][-2].tolist(), index2word))
        logger.info(query_seq_lengths[part][-2])
        logger.info(count_vectorizer.inverse_transform(response_vectors[part][-2].tolist()))
        
    return context_seqs, context_seq_lengths, query_seqs, query_seq_lengths, response_vectors, labels


def indices2words(seq, index2word, filter_indices=[0, 2, 3]):
    
    if type(seq[0]) == list:
        return [indices2words(item, index2word) for item in seq]
    else:
        return [index2word[item] for item in seq if item not in filter_indices]
    

def words2indices(seq, word2index):
    if type(seq[0]) == list:
        return [word2indices(item, word2index) for item in seq]
    else:
        return [word2index[item] for item in seq]


def tokenize_sequence(sentences, filters, max_num_words, max_vocab_size):
    """
    Tokenizes a given input sequence of words.
    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary
    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence
    """
    table = str.maketrans({key: None for key in filters})

    new_sentences = []
    for s in tqdm(sentences):
        new_sentences.append(' '.join(word_tokenize(s)[:max_num_words-1]))
    
    sentences = new_sentences

    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(sentences)

    word_index = dict()
    word_index['<PAD>'] = 0
    word_index['<UNK>'] = 1
    word_index['<SOS>'] = 2
    word_index['<EOS>'] = 3

    for i, word in enumerate(dict(tokenizer.word_index).keys()):
        word_index[word] = i + 4

    tokenizer.word_index = word_index
    input_x = tokenizer.texts_to_sequences(list(sentences))
    
    output_y = []
    seq_lengths = []
    for i, seq in tqdm(enumerate(input_x)):
        seq = seq[:max_num_words-1]
            
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['<UNK>'] for t in seq]
        
        seq_x = copy.deepcopy(seq)
        
        if direction == 'backward':
            seq_x.reverse()
        
        seq_x.insert(0, word_index['<SOS>'])
        input_x[i] = seq_x
        
        seq_lengths.append(len(seq_x))
        
        seq_y = copy.deepcopy(seq)
        seq_y.append(word_index['<EOS>'])
        output_y.append(seq_y)

    input_x = pad_sequences(input_x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['<PAD>'])
    output_y = pad_sequences(output_y, padding='post', truncating='post', maxlen=max_num_words, value=word_index['<PAD>'])

    word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}
    
    return input_x, output_y, seq_lengths, word_index


def compute_metrics(logits, labels, sample_count, vectorizer):
    prob_dist = softmax(logits, dim=-1)
    results = torch.topk(prob_dist, sample_count, dim=-1)
    
    gen_bow = torch.zeros(labels.size())
    for index in range(results.indices.size(0)):
        gen_bow[index][results.indices[index]] = 1.0
    
    references = vectorizer.inverse_transform(labels.detach().cpu().numpy())
    hypotheses = vectorizer.inverse_transform(gen_bow.detach().cpu().numpy())
    
    references = [item.tolist() for item in references]
    hypotheses = [item.tolist() for item in hypotheses]
    
    bleu_scores = []
    tokens_ret = []
    total_tokens = []
    for index in range(len(references)):
        if len(references[index]) == 0:
            #print(f'Empty reference at index: {index}')
            continue
        
        #print(references[index])
        #print(hypotheses[index])
        bleu_scores.append(sentence_bleu([references[index]], hypotheses[index], weights=(1.0,)))
        tokens_count = len(set(references[index]).intersection(set(hypotheses[index])))
        tokens_ret.append(tokens_count)
        total_tokens.append(len(references[index]))
    
    return bleu_scores, tokens_ret, total_tokens, references, hypotheses

def compute_keyword_metrics(output):
    bleu_scores = []
    rel_tokens_ret = []
    total_tokens_ret = []
    total_gt_tokens = []
    for o_dict in output:
        bleu_scores.append(sentence_bleu([o_dict['response_bow']], o_dict['t5_predicted_response_bow'], weights=(1.0,)))
        rel_token_count = len(set(o_dict['response_bow']).intersection(o_dict['t5_predicted_response_bow']))
        rel_tokens_ret.append(rel_token_count)
        total_tokens_ret.append(len(o_dict['t5_predicted_response_bow']))
        total_gt_tokens.append(len(o_dict['response_bow']))
        
    bleu_score = np.mean(bleu_scores)    
        
    
    recall = np.sum(rel_tokens_ret) / np.sum(total_gt_tokens)
    precision = np.sum(rel_tokens_ret) / np.sum(total_tokens_ret)
    
    return bleu_score, recall, precision

def multi_compute_metrics(logits, labels, sample_count, vectorizer, num_samples):
    prob_dist = softmax(logits, dim=-1)
    results = torch.topk(prob_dist, sample_count, dim=-1)
    
    gen_bow = torch.zeros((logits.size(0), labels.size(-1)))
    for index in range(results.indices.size(0)):
        gen_bow[index][results.indices[index]] = 1.0
    
    references = vectorizer.inverse_transform(labels.detach().cpu().numpy())
    hypotheses = vectorizer.inverse_transform(gen_bow.detach().cpu().numpy())
    
    references = [item.tolist() for item in references]
    hypotheses = [item.tolist() for item in hypotheses]
    
    recall_bleus = []
    precision_bleus = []
    tokens_ret = []
    total_tokens = []
    for index in range(len(references)):
        if len(references[index]) == 0:
            #print(f'Empty reference at index: {index}')
            continue
        
        #print(references[index])
        #print(hypotheses[index]
        bleu_scores = []
        for h_index in range(num_samples):
            bleu_scores.append(sentence_bleu([references[index]], hypotheses[(index*num_samples)+h_index], weights=(1.0,)))
            
            if h_index == 0:
                tokens_count = len(set(references[index]).intersection(set(hypotheses[index])))
                tokens_ret.append(tokens_count)
                total_tokens.append(len(references[index]))
        
        recall_bleus.append(np.max(bleu_scores))
        precision_bleus.append(np.mean(bleu_scores))
    
    return recall_bleus, precision_bleus, tokens_ret, total_tokens, references, hypotheses

def compute_bleu(outputs):
    
    recall_bleus = []
    precision_bleus = []
    for output in outputs:
        gt_response = output['next sentence']
        bleu_scores = []
        for pred_response in output['generated_sentences']:
            try:
                bleu_scores.append(sentence_bleu([gt_response], word_tokenize(pred_response), smoothing_function=SmoothingFunction().method7, weights=[1./3, 1./3, 1./3]))
            except:
                bleu_scores.append(0.0)
        
        recall_bleus.append(np.max(bleu_scores))
        precision_bleus.append(np.mean(bleu_scores))
    
    return np.mean(recall_bleus), np.mean(precision_bleus)

def compute_sentence_bleu(references, hypotheses):
    
    weights = [[1.], [1./2, 1./2], [1./3, 1./3, 1./3], [1./4, 1./4, 1./4, 1./4]]
    
    bleu_scores = [[], [], [], []]
    combined = []
    for index in range(len(references)):
        combined.append([references[index], hypotheses[index]])
        for n in range(4):
            try:
                bleu_scores[n].append(sentence_bleu([references[index]], hypotheses[index], weights=weights[n]))
            except:
                bleu_scores[n].append(0.0)
    
    avg_bleu_scores = []
    for n in range(4):
        avg_bleu_scores.append(np.mean(bleu_scores[n]))
        
    return avg_bleu_scores, combined

def compute_smoothed_sentence_bleu(queries, references, hypotheses):
    
    weights = [[1.], [1./2, 1./2], [1./3, 1./3, 1./3], [1./4, 1./4, 1./4, 1./4]]
    
    bleu_scores = [[], [], [], []]
    combined = []
    for index in range(len(references)):
        combined.append([queries[index], references[index], hypotheses[index]])
        for n in range(4):
            try:
                bleu_scores[n].append(sentence_bleu([references[index]], hypotheses[index], smoothing_function=SmoothingFunction().method7, weights=weights[n]))
            except:
                bleu_scores[n].append(0.0)
    
    avg_bleu_scores = []
    for n in range(4):
        avg_bleu_scores.append(np.mean(bleu_scores[n]))
        
    return avg_bleu_scores, combined
    


def logits_to_bow(logits, sample_count, vectorizer):
    results = torch.topk(logits, sample_count, dim=-1)
    
    gen_bow = torch.zeros(logits.size())
    for index in range(results.indices.size(0)):
        gen_bow[index][results.indices[index]] = 1.0
    
    return vectorizer.inverse_transform(gen_bow.detach().cpu().numpy())


def one_hot_to_bow(one_hot, vectorizer):
    
    references = vectorizer.inverse_transform(one_hot.detach().cpu().numpy())
    
    references = [item.tolist() for item in references]
    
    return references


def load_model(model_path):
    
    return torch.load(model_path)


def encode_vectors(bow_vectors, vae_model, temperature):
    
    vae_model.eval()
    #with torch.set_grad_enabled(False):
    mean, logv, _ = vae_model.encode(bow_vectors)
    
    eps = torch.randn(mean.size()).cuda()
    z = mean + (temperature * (eps * logv.exp()))
    
    return mean, logv, z


def encode_vectors_with_sampling(bow_vectors, vae_model, temperature_1, temperature_2, num_samples=1):
    
    vae_model.eval()
    #with torch.set_grad_enabled(False):
    mean, logv, _ = vae_model.encode(bow_vectors)
    
    if num_samples == 1:
        eps = torch.randn(mean.size()).cuda()
        z = mean + (temperature_1 * (eps * logv.exp()))
    else:
        latent_codes = []
        for index in range(mean.size(0)):
            map_eps = torch.randn((1, mean.size(-1))).cuda()
            map_z = mean[index].unsqueeze(0) + (temperature_1 * (map_eps * (logv[index].unsqueeze(0).exp())))
            latent_codes.append(map_z)
            
            eps = torch.randn(((num_samples-1), mean.size(-1))).cuda()
            z = torch.cat([mean[index].unsqueeze(0)]*(num_samples-1), dim=0) + (temperature_2 * (eps * torch.cat([logv[index].unsqueeze(0).exp()]*(num_samples-1))))
            latent_codes.append(z)
        z = torch.cat(latent_codes, dim=0)
    
    return mean, logv, z


def decode_vectors(z, vae_model):
    vae_model.decoder.train()
    #with torch.set_grad_enabled(False):
    logits = vae_model.decode(z)
    
    return logits
