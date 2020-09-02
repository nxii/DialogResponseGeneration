import logging
import argparse
import os
import sys
import shutil
import pickle
import json
import torch
import torch.nn.functional as F
from itertools import chain
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.nn.functional import log_softmax, softmax
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

from config import config

from models.Neural_LM import Neural_LM
from models.Sentence_VAE import Sentence_VAE
from models.BOW_VAE import BOW_VAE
from models.Classifier import Classifier
from models.GAN_Generator import GAN_Generator
from models.GAN_Discriminator import GAN_Discriminator

from utils import *


def train_lm(config, logger):
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    with open(os.path.join(config.save_path, 'config.pkl'), 'wb') as f_obj:
        pickle.dump(config, f_obj)
    
    sentences = load_sentences(config, logger)
        
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config, logger)
    
    embedding_matrix = load_embedding_matrix(word2index, config, logger)
    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    logger.info(f'Embedding matrix size: {embedding_matrix.size()}')
    
    vocab_size = embedding_matrix.size(0)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        # torch.tensor(indexed_data[part]['output_y'], dtype=torch.long)
        datasets[part] = TensorDataset(torch.arange(len(sentences[part])))
        
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.lm_batch_size, shuffle=shuffle, pin_memory=True)
        
    model = Neural_LM(config.lm_num_layers, config.lm_hidden_size, config.lm_bidirectional, word2index['<PAD>'], embedding_matrix)
    model.cuda()
    
    logger.info(model)
    
    optimizer = Adam(model.parameters(), lr=config.lm_lr)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'], reduction='none')
    
    logger.info('Begining training.')
    best_epoch = None
    best_loss = None
    for epoch in range(1, config.num_epochs):
        model.train()
        with torch.set_grad_enabled(True):
            train_loss = []
            for batch_index, batch in enumerate(dataloaders['train']):
                indices = batch[0].tolist()
                
                input_x, output_y, seq_lengths = load_lm_batch(sentences['train'], indices, tokenizer, word2index, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                h_n, output = model(input_x, seq_lengths)
                
                output = output.view(-1, vocab_size)
                                
                # Discard useless padding
                target = target[:,:seq_lengths[0]].contiguous().view(-1)
                                
                loss = loss_fn(output, target)
                
                train_loss.extend(loss.detach().cpu().tolist())
                
                loss = torch.mean(loss)
                                
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (batch_index+1) % config.print_every == 0:
                    logger.info(f'Epoch: {epoch} Batch #{batch_index+1}/{len(dataloaders["train"])} Loss: {(loss).item():.2f}')
                    
            
            logger.info(f'Epoch: {epoch}, Training Loss: {np.mean(train_loss):.2f}')
        
        
        logger.info('Performing validation.')
        model.eval()
        with torch.set_grad_enabled(False):
            valid_loss = []
            for batch in dataloaders['valid']:
                indices = batch[0].tolist()
                
                input_x, output_y, seq_lengths = load_lm_batch(sentences['valid'], indices, tokenizer, word2index, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                h_n, output = model(input_x, seq_lengths)
                
                output = output.view(-1, vocab_size)
                                
                # Discard useless padding
                target = target[:,:seq_lengths[0]].contiguous().view(-1)
                                
                loss = loss_fn(output, target)
                
                valid_loss.extend(loss.detach().cpu().tolist())
            
            valid_loss = np.mean(valid_loss)
            logger.info(f'Epoch: {epoch}, Validation Loss: {valid_loss:.2f}')
            
            test_loss = []
            for batch in dataloaders['test']:
                indices = batch[0].tolist()
                
                input_x, output_y, seq_lengths = load_lm_batch(sentences['test'], indices, tokenizer, word2index, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                h_n, output = model(input_x, seq_lengths)
                
                output = output.view(-1, vocab_size)
                                
                # Discard useless padding
                target = target[:,:seq_lengths[0]].contiguous().view(-1)
                                
                loss = loss_fn(output, target)
                
                test_loss.extend(loss.detach().cpu().tolist())
                
            test_loss = np.mean(test_loss)    
            logger.info(f'Epoch: {epoch}, Test Loss: {test_loss:.2f}')
                        
            if best_loss is None or best_loss > valid_loss:
                logger.info('Best validation loss score so far. Saving the model.')
                save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
                torch.save(model, save_path)
                
                best_loss = valid_loss
                best_epoch = epoch
            

    logger.info(f'Best epoch: {best_epoch} Best Validation Loss: {best_loss}')
    
def train_sentence_vae(config, logger, writer):
    
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    with open(os.path.join(config.save_path, 'config.pkl'), 'wb') as f_obj:
        pickle.dump(config, f_obj)
    
    sentences = load_sentences(config, logger)
        
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config.filters, config.sentence_vae_vocab_size, config, logger)
    
    embedding_matrix = load_embedding_matrix(word2index, config.sentence_vae_emb_dim, config.sentence_vae_emb_type, config, logger)
    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    logger.info(f'Embedding matrix size: {embedding_matrix.size()}')
    
    vocab_size = embedding_matrix.size(0)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        # torch.tensor(indexed_data[part]['output_y'], dtype=torch.long)
        datasets[part] = TensorDataset(torch.arange(len(sentences[part])))
        
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.lm_batch_size, shuffle=shuffle, pin_memory=True)
         
    model = Sentence_VAE(config.sentence_vae_encoder_layers, config.sentence_vae_decoder_layers, config.sentence_vae_hidden_size, config.sentence_vae_latent_size, config.sentence_vae_bidirectional, 
                         config.sentence_vae_temperature, config.sentence_vae_max_seq_len, config.sentence_vae_dropout, word2index['<PAD>'], word2index['<SOS>'], word2index['<EOS>'], embedding_matrix)
    model.cuda()
    
    logger.info(model)
    
    optimizer = Adam(model.parameters(), lr=config.sentence_vae_lr)
    
    def kl_anneal_function(anneal_fn, step, slope, margin, max_step):
        if step > max_step:
            step = max_step
            
        if anneal_fn == 'tanh':
            return np.round((np.tanh((step - 4500) / 1000) + 1) / 2, decimals=6)
            #return 0.5*(np.tanh(slope*(step - margin)) + 1)
        elif anneal_fn == 'sigmoid':
            return 1/(1 + np.exp(-slope*(step - margin)))
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'], reduction='none')
    
    logger.info('Begining training.')
    best_epoch = None
    best_bleu = None
    step = 1
    for epoch in range(1, config.num_epochs):
        model.train()
        with torch.set_grad_enabled(True):
            train_rec_loss = []
            train_kl_loss = 0.0
            for batch_index, batch in enumerate(dataloaders['train']):
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                input_x, output_y, seq_lengths = load_sentence_batch(sentences['train'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                hidden, mean, logv, z, logits = model(input_x, seq_lengths)
                
                logits = logits.view(-1, vocab_size)
                
                # Discard useless padding
                target = target[:,:seq_lengths[0]].contiguous().view(-1)
                
                rec_loss = loss_fn(logits, target)
                
                KL_loss = -0.5 * torch.sum(1 + 2 * logv - mean.pow(2) - torch.exp(2 * logv))
                KL_weight = kl_anneal_function('tanh', step, config.sentence_vae_slope, config.sentence_vae_margin, config.sentence_vae_max_step)
                
                batch_loss = torch.mean(rec_loss) + ((KL_weight * KL_loss) / batch_size)
                
                train_rec_loss.extend(rec_loss.detach().cpu().tolist())
                train_kl_loss += KL_loss
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                step += 1
                
                
                if step % config.print_every == 0:
                    logger.info(f'Epoch: {epoch} Step: {step:04} Reconstruction loss: {batch_loss.item():.4f}, KL Weight: {KL_weight:0.4f}, KL loss: {KL_loss.item()/batch_size:.4f}')
                    
            
            train_rec_loss = np.mean(train_rec_loss)
            train_kl_loss = train_kl_loss / len(datasets['train'])
            
            logger.info(f'Epoch: {epoch}, Training -> Reconstruction Loss: {train_rec_loss:.2f}, KL Loss: {train_kl_loss:.2f}')
        
        
        logger.info('Performing validation.')
        model.eval()
        with torch.set_grad_enabled(False):
            references = []
            hypotheses = []
            for batch in dataloaders['valid']:
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                input_x, output_y, seq_lengths = load_sentence_batch(sentences['valid'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                hidden, mean, logv, z, generated_seqs = model(input_x, seq_lengths, inference=True)
                
                refs = indices2words(target.detach().cpu().tolist(), index2word)
                hypos = indices2words(generated_seqs.detach().cpu().tolist(), index2word)
                
                references.extend(refs)
                hypotheses.extend(hypos)
            
             
            val_avg_bleu_scores, combined = compute_sentence_bleu(references, hypotheses)
            
            logger.info(f'Epoch: {epoch}, Valid Bleu-1: {val_avg_bleu_scores[0]:.2f}, Bleu-2: {val_avg_bleu_scores[1]:.2f}, Bleu-3: {val_avg_bleu_scores[2]:.2f}, Bleu-4: {val_avg_bleu_scores[3]:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f"Reference: {' '.join(references[rand_index])}")
            logger.info(f"Reconstruction: {' '.join(hypotheses[rand_index])}")
            logger.info('Sampling from prior.')
            prior_sample = model.inference(samples=1)
            prior_sample = indices2words(prior_sample.detach().cpu().tolist(), index2word)
            logger.info(' '.join(prior_sample[0]))
            
            with open(f'{config.save_path}/valid_output_{epoch}.json', 'w') as f_obj:
                json.dump(combined, f_obj)
                
            with open(f'{config.save_path}/valid_output_{epoch}.txt', 'w') as f_obj:
                for item in combined:
                    f_obj.write(f"Reference: {' '.join(item[0])}\n")
                    f_obj.write(f"Reconstruction: {' '.join(item[1])}\n")
                    f_obj.write("\n")
            
            
            references = []
            hypotheses = []
            for batch in dataloaders['test']:
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                input_x, output_y, seq_lengths = load_sentence_batch(sentences['test'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                input_x = torch.tensor(input_x, dtype=torch.long).cuda()
                seq_lengths = torch.tensor(seq_lengths).cuda()
                target = torch.tensor(output_y, dtype=torch.long).cuda()
                
                sorted_indices = torch.argsort(seq_lengths, descending=True)
                
                input_x = input_x[sorted_indices]
                seq_lengths = seq_lengths[sorted_indices]
                target = target[sorted_indices]
                
                packed_target = nn.utils.rnn.pack_padded_sequence(target, seq_lengths, batch_first=True)
                
                hidden, mean, logv, z, generated_seqs = model(input_x, seq_lengths, inference=True)
                
                refs = indices2words(target.detach().cpu().tolist(), index2word)
                hypos = indices2words(generated_seqs.detach().cpu().tolist(), index2word)
                
                references.extend(refs)
                hypotheses.extend(hypos)
            
             
            test_avg_bleu_scores, combined = compute_sentence_bleu(references, hypotheses)
            
            logger.info(f'Epoch: {epoch}, Test Bleu-1: {test_avg_bleu_scores[0]:.2f}, Bleu-2: {test_avg_bleu_scores[1]:.2f}, Bleu-3: {test_avg_bleu_scores[2]:.2f}, Bleu-4: {test_avg_bleu_scores[3]:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f"Reference: {' '.join(references[rand_index])}")
            logger.info(f"Reconstruction: {' '.join(hypotheses[rand_index])}")
            logger.info('Sampling from prior.')
            prior_sample = model.inference(samples=1)
            prior_sample = indices2words(prior_sample.detach().cpu().tolist(), index2word)
            logger.info(' '.join(prior_sample[0]))
            
            with open(f'{config.save_path}/test_output_{epoch}.json', 'w') as f_obj:
                json.dump(combined, f_obj)
                
            with open(f'{config.save_path}/test_output_{epoch}.txt', 'w') as f_obj:
                for item in combined:
                    f_obj.write(f"Reference: {' '.join(item[0])}\n")
                    f_obj.write(f"Reconstruction: {' '.join(item[1])}\n")
                    f_obj.write("\n")
               
            if best_bleu is None or best_bleu < val_avg_bleu_scores[-1]:
                logger.info(f'Best validation bleu-4 score so far: {val_avg_bleu_scores[-1]}. Saving the model.')
                save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
                torch.save(model, save_path)
                
                best_bleu = val_avg_bleu_scores[-1]
                best_epoch = epoch
            elif epoch % config.save_every == 0:
                save_path = os.path.join(config.save_path, f'model_{epoch}.pt')
                torch.save(model, save_path)
                

    logger.info(f'Best epoch: {best_epoch} Best Validation Bleu-4: {best_bleu}')
    

def train_cross_distillation(config, logger, writer):
    
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    with open(os.path.join(config.save_path, 'config.pkl'), 'wb') as f_obj:
        pickle.dump(config, f_obj)
    
    sentences = load_sentences(config, logger)
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config.filters, config.sentence_vae_vocab_size, config, logger)
    contexts, curr_sents, next_sents = load_tuples(config, logger)
    
    embedding_matrix = load_embedding_matrix(word2index, config.sentence_vae_emb_dim, config.sentence_vae_emb_type, config, logger)
    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    logger.info(f'Embedding matrix size: {embedding_matrix.size()}')
    
    vocab_size = embedding_matrix.size(0)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        datasets[part] = TensorDataset(torch.arange(len(curr_sents[part])))
        
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.lm_batch_size, shuffle=shuffle, pin_memory=True)
        
        
    #logger.info(f'Loading pretrained model from: {config.sentence_vae_saved_model}')    
    #pt_model = torch.load(config.sentence_vae_saved_model)     
         
    model = Sentence_VAE(config.sentence_vae_encoder_layers, config.sentence_vae_decoder_layers, config.sentence_vae_hidden_size, config.sentence_vae_latent_size, config.sentence_vae_bidirectional, 
                         config.sentence_vae_temperature, config.sentence_vae_max_seq_len, config.sentence_vae_dropout, word2index['<PAD>'], word2index['<SOS>'], word2index['<EOS>'], embedding_matrix)
    model.cuda()
    
    logger.info(model)
    
    optimizer = Adam(model.parameters(), lr=config.sentence_vae_lr)
    
    def kl_anneal_function(anneal_fn, step, slope, margin, max_step):
        if step > max_step:
            step = max_step
            
        if anneal_fn == 'tanh':
            return np.round((np.tanh((step - 4500) / 1000) + 1) / 2, decimals=6)
            #return 0.5*(np.tanh(slope*(step - margin)) + 1)
        elif anneal_fn == 'sigmoid':
            return 1/(1 + np.exp(-slope*(step - margin)))
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2index['<PAD>'], reduction='none')
    
    #mse_loss_fn = nn.MSELoss(reduction='sum')
    
    #distill_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    logger.info('Begining training.')
    best_epoch = None
    best_bleu = None
    step = 1
    for epoch in range(1, config.num_epochs):
        #pt_model.train()
        model.train()
        with torch.set_grad_enabled(True):
            train_rec_loss = []
            train_kl_div = []
            train_distill_loss = []
            train_kl_loss = 0.0
            for batch_index, batch in enumerate(dataloaders['train']):
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                # Process Current Sentence
                curr_sent_input, curr_sent_output, curr_sent_seq_lengths = load_sentence_batch(curr_sents['train'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                curr_sent_input = torch.tensor(curr_sent_input, dtype=torch.long).cuda()
                curr_sent_seq_lengths = torch.tensor(curr_sent_seq_lengths).cuda()
                curr_sent_target = torch.tensor(curr_sent_output, dtype=torch.long).cuda()
                
                # Sort the current sentences by seq lengths in descending order
                curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lengths, descending=True)
                
                curr_sent_input = curr_sent_input[curr_sent_sorted_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_sorted_indices]
                curr_sent_target = curr_sent_target[curr_sent_sorted_indices]
                
                curr_sent_input_embedding = model.embedding(curr_sent_input)
        
                packed_curr_sent_input = nn.utils.rnn.pack_padded_sequence(curr_sent_input_embedding, curr_sent_seq_lengths, batch_first=True)
                
                curr_sent_hidden, curr_sent_mean, curr_sent_logv, curr_sent_z = model.encode(packed_curr_sent_input, batch_size, curr_sent_seq_lengths[0])
                
                # Revert to the original order
                curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                curr_sent_input = curr_sent_input[curr_sent_reverse_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_reverse_indices]
                curr_sent_target = curr_sent_target[curr_sent_reverse_indices]
                
                curr_sent_hidden = curr_sent_hidden[:,:,curr_sent_reverse_indices]
                curr_sent_mean = curr_sent_mean[curr_sent_reverse_indices]
                curr_sent_logv = curr_sent_logv[curr_sent_reverse_indices]
                curr_sent_z = curr_sent_z[curr_sent_reverse_indices]
                
                # Process Next Sentence
                next_sent_input, next_sent_output, next_sent_seq_lengths = load_sentence_batch(next_sents['train'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                next_sent_input = torch.tensor(next_sent_input, dtype=torch.long).cuda()
                next_sent_seq_lengths = torch.tensor(next_sent_seq_lengths).cuda()
                next_sent_target = torch.tensor(next_sent_output, dtype=torch.long).cuda()
                
                # Sort the next sentences by seq lengths in descending order
                next_sent_sorted_indices = torch.argsort(next_sent_seq_lengths, descending=True)
                
                next_sent_input = next_sent_input[next_sent_sorted_indices]
                next_sent_seq_lengths = next_sent_seq_lengths[next_sent_sorted_indices]
                next_sent_target = next_sent_target[next_sent_sorted_indices]
                
                next_sent_input_embedding = model.embedding(next_sent_input)
        
                packed_next_sent_input = nn.utils.rnn.pack_padded_sequence(next_sent_input_embedding, next_sent_seq_lengths, batch_first=True)
                
                next_sent_hidden, next_sent_mean, next_sent_logv, next_sent_z = model.encode(packed_next_sent_input, batch_size, next_sent_seq_lengths[0])
                
                # Arrange curr sent hidden, mean, logv, and z using the next sent sorted indices
                curr_sent_input = curr_sent_input[next_sent_sorted_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[next_sent_sorted_indices]
                curr_sent_target = curr_sent_target[next_sent_sorted_indices]
                
                curr_sent_hidden = curr_sent_hidden[:,:,next_sent_sorted_indices]
                curr_sent_mean = curr_sent_mean[next_sent_sorted_indices]
                curr_sent_logv = curr_sent_logv[next_sent_sorted_indices]
                curr_sent_z = curr_sent_z[next_sent_sorted_indices]
                
                # Obtain logits for the next sentence from teacher model
                #decoder_input_embedding = next_sent_input_embedding.clone().detach()
                #stacked_next_sent_z = torch.stack([next_sent_z]*30, dim=1)
                
                #decoder_input = torch.cat((decoder_input_embedding, stacked_next_sent_z), dim=-1)
            
                #packed_decoder_input = nn.utils.rnn.pack_padded_sequence(decoder_input, next_sent_seq_lengths, batch_first=True)
            
                #pt_logits = pt_model.decode(packed_decoder_input)
                
                decoder_input_embedding = next_sent_input_embedding.clone().detach()
                stacked_curr_sent_z = torch.stack([curr_sent_z]*30, dim=1)
                
                decoder_input = torch.cat((decoder_input_embedding, stacked_curr_sent_z), dim=-1)
            
                packed_decoder_input = nn.utils.rnn.pack_padded_sequence(decoder_input, next_sent_seq_lengths, batch_first=True)
            
                logits = model.decode(packed_decoder_input)
                
                flattened_logits = logits.view(-1, vocab_size)
                
                # Reconstruction loss
                next_sent_target = next_sent_target[:,:next_sent_seq_lengths[0]].contiguous().view(-1) # Discard useless padding
                
                rec_loss = loss_fn(flattened_logits, next_sent_target)
                
                # KL Divergence loss
                #source_dists = Normal(curr_sent_mean, (curr_sent_logv.exp())/2.0)
                #target_dists = Normal(next_sent_mean, (next_sent_logv.exp())/2.0)
                
                #KL_divergence = kl_divergence(source_dists, target_dists)
                #KL_divergence = torch.sum(KL_divergence, dim=1)
                
                # KL regularization
                KL_loss = -0.5 * torch.sum(1 + 2 * curr_sent_logv - curr_sent_mean.pow(2) - torch.exp(2 * curr_sent_logv))
                KL_weight = kl_anneal_function('tanh', step, config.sentence_vae_slope, config.sentence_vae_margin, config.sentence_vae_max_step)
                
                # Cross model distillation
                #curr_sent_hidden_norm = torch.norm(curr_sent_hidden, dim=-1).unsqueeze(-1)
                #normalized_curr_sent_hidden = torch.cat([1./curr_sent_hidden_norm]*curr_sent_hidden.size(-1), dim=-1)*curr_sent_hidden
                
                #next_sent_hidden_norm = torch.norm(next_sent_hidden, dim=-1).unsqueeze(-1)
                #normalized_next_sent_hidden = torch.cat([1./next_sent_hidden_norm]*next_sent_hidden.size(-1), dim=-1)*next_sent_hidden
                
                
                #distill_loss = distill_loss_fn(F.log_softmax(logits/config.sentence_vae_distillation_T, dim=0), F.softmax(pt_logits.detach()/config.sentence_vae_distillation_T, dim=0))
                
                                
                #batch_loss = torch.mean(rec_loss) + torch.mean(KL_divergence) + ((KL_weight * KL_loss) / batch_size) + (distill_loss * (config.sentence_vae_distillation_T * config.sentence_vae_distillation_T))
                batch_loss = torch.mean(rec_loss) + ((KL_weight * KL_loss) / batch_size)
                
                train_rec_loss.extend(rec_loss.detach().cpu().tolist())
                #train_kl_div.extend(KL_divergence.detach().cpu().tolist())
                train_kl_loss += KL_loss
                #train_distill_loss.append(distill_loss.item())
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                step += 1
                
                
                if step % config.print_every == 0:
                    #logger.info(f'Epoch: {epoch} Step: {step:04} Distillation loss: {distill_loss.item():.2f}, KL divergence: {torch.mean(KL_divergence).item():.4f}, Reconstruction loss: {torch.mean(rec_loss).item():.4f}, KL Weight: {KL_weight:0.4f}, KL loss: {KL_loss.item()/batch_size:.4f}')
                    logger.info(f'Epoch: {epoch} Step: {step:04} Reconstruction loss: {torch.mean(rec_loss).item():.4f}, KL Weight: {KL_weight:0.4f}, KL loss: {KL_loss.item()/batch_size:.4f}')
                    
            
            #train_kl_div = np.mean(train_kl_div)
            train_rec_loss = np.mean(train_rec_loss)
            #train_distill_loss = np.mean(train_distill_loss)
            train_kl_loss = train_kl_loss / len(datasets['train'])
            
            #logger.info(f'Epoch: {epoch}, Training -> Distillation loss: {distill_loss.item():.2f}, KL divergence: {train_kl_div:.2f}, Reconstruction Loss: {train_rec_loss:.2f}, KL Loss: {train_kl_loss:.2f}')
            logger.info(f'Epoch: {epoch}, Training -> Reconstruction Loss: {train_rec_loss:.2f}, KL Loss: {train_kl_loss:.2f}')
            
        
        logger.info('Performing validation.')
        model.eval()
        with torch.set_grad_enabled(False):
            queries = []
            references = []
            hypotheses = []
            for batch in dataloaders['valid']:
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                # Process Current Sentence
                curr_sent_input, curr_sent_output, curr_sent_seq_lengths = load_sentence_batch(curr_sents['valid'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                curr_sent_input = torch.tensor(curr_sent_input, dtype=torch.long).cuda()
                curr_sent_seq_lengths = torch.tensor(curr_sent_seq_lengths).cuda()
                curr_sent_target = torch.tensor(curr_sent_output, dtype=torch.long).cuda()
                
                # Sort the current sentences by seq lengths in descending order
                curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lengths, descending=True)
                
                curr_sent_input = curr_sent_input[curr_sent_sorted_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_sorted_indices]
                curr_sent_target = curr_sent_target[curr_sent_sorted_indices]
                
                curr_sent_input_embedding = model.embedding(curr_sent_input)
        
                packed_curr_sent_input = nn.utils.rnn.pack_padded_sequence(curr_sent_input_embedding, curr_sent_seq_lengths, batch_first=True)
                
                curr_sent_hidden, curr_sent_mean, curr_sent_logv, curr_sent_z = model.encode(packed_curr_sent_input, batch_size, curr_sent_seq_lengths[0])
                
                # Revert to the original order
                curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                curr_sent_input = curr_sent_input[curr_sent_reverse_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_reverse_indices]
                curr_sent_target = curr_sent_target[curr_sent_reverse_indices]
                
                curr_sent_hidden = curr_sent_hidden[:,:,curr_sent_reverse_indices]
                curr_sent_mean = curr_sent_mean[curr_sent_reverse_indices]
                curr_sent_logv = curr_sent_logv[curr_sent_reverse_indices]
                curr_sent_z = curr_sent_z[curr_sent_reverse_indices]
                
                # Process Next Sentence
                next_sent_input, next_sent_output, next_sent_seq_lengths = load_sentence_batch(next_sents['valid'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                next_sent_input = torch.tensor(next_sent_input, dtype=torch.long).cuda()
                next_sent_seq_lengths = torch.tensor(next_sent_seq_lengths).cuda()
                next_sent_target = torch.tensor(next_sent_output, dtype=torch.long).cuda()
                
                # Generate next sentences
                generated_seqs = model.inference(curr_sent_z)
                                
                contexts = indices2words(curr_sent_target.detach().cpu().tolist(), index2word)                
                refs = indices2words(next_sent_target.detach().cpu().tolist(), index2word)
                hypos = indices2words(generated_seqs.detach().cpu().tolist(), index2word)
                
                queries.extend(contexts)
                references.extend(refs)
                hypotheses.extend(hypos)
            
             
            val_avg_bleu_scores, combined = compute_smoothed_sentence_bleu(queries, references, hypotheses)
            
            logger.info(f'Epoch: {epoch}, Valid Bleu-1: {val_avg_bleu_scores[0]:.2f}, Bleu-2: {val_avg_bleu_scores[1]:.2f}, Bleu-3: {val_avg_bleu_scores[2]:.2f}, Bleu-4: {val_avg_bleu_scores[3]:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f"Query: {' '.join(queries[rand_index])}")
            logger.info(f"Reference: {' '.join(references[rand_index])}")
            logger.info(f"Predicted: {' '.join(hypotheses[rand_index])}")
            
            with open(f'{config.save_path}/valid_output_{epoch}.json', 'w') as f_obj:
                json.dump(combined, f_obj)
                
            with open(f'{config.save_path}/valid_output_{epoch}.txt', 'w') as f_obj:
                for item in combined:
                    f_obj.write(f"Query: {' '.join(item[0])}\n")
                    f_obj.write(f"Reference: {' '.join(item[1])}\n")
                    f_obj.write(f"Predicted: {' '.join(item[2])}\n")
                    f_obj.write("\n")
            
            
            queries = []
            references = []
            hypotheses = []
            for batch in dataloaders['test']:
                indices = batch[0].tolist()
                
                batch_size = batch[0].size(0)
                
                # Process Current Sentence
                curr_sent_input, curr_sent_output, curr_sent_seq_lengths = load_sentence_batch(curr_sents['test'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                curr_sent_input = torch.tensor(curr_sent_input, dtype=torch.long).cuda()
                curr_sent_seq_lengths = torch.tensor(curr_sent_seq_lengths).cuda()
                curr_sent_target = torch.tensor(curr_sent_output, dtype=torch.long).cuda()
                
                # Sort the current sentences by seq lengths in descending order
                curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lengths, descending=True)
                
                curr_sent_input = curr_sent_input[curr_sent_sorted_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_sorted_indices]
                curr_sent_target = curr_sent_target[curr_sent_sorted_indices]
                
                curr_sent_input_embedding = model.embedding(curr_sent_input)
        
                packed_curr_sent_input = nn.utils.rnn.pack_padded_sequence(curr_sent_input_embedding, curr_sent_seq_lengths, batch_first=True)
                
                curr_sent_hidden, curr_sent_mean, curr_sent_logv, curr_sent_z = model.encode(packed_curr_sent_input, batch_size, curr_sent_seq_lengths[0])
                
                # Revert to the original order
                curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                curr_sent_input = curr_sent_input[curr_sent_reverse_indices]
                curr_sent_seq_lengths = curr_sent_seq_lengths[curr_sent_reverse_indices]
                curr_sent_target = curr_sent_target[curr_sent_reverse_indices]
                
                curr_sent_hidden = curr_sent_hidden[:,:,curr_sent_reverse_indices]
                curr_sent_mean = curr_sent_mean[curr_sent_reverse_indices]
                curr_sent_logv = curr_sent_logv[curr_sent_reverse_indices]
                curr_sent_z = curr_sent_z[curr_sent_reverse_indices]
                
                # Process Next Sentence
                next_sent_input, next_sent_output, next_sent_seq_lengths = load_sentence_batch(next_sents['test'], indices, tokenizer, word2index, vocab_size, config.sentence_vae_max_seq_len, config, logger)
                
                next_sent_input = torch.tensor(next_sent_input, dtype=torch.long).cuda()
                next_sent_seq_lengths = torch.tensor(next_sent_seq_lengths).cuda()
                next_sent_target = torch.tensor(next_sent_output, dtype=torch.long).cuda()
                
                # Generate next sentences
                generated_seqs = model.inference(curr_sent_z)
                                
                contexts = indices2words(curr_sent_target.detach().cpu().tolist(), index2word)                
                refs = indices2words(next_sent_target.detach().cpu().tolist(), index2word)
                hypos = indices2words(generated_seqs.detach().cpu().tolist(), index2word)
                
                queries.extend(contexts)
                references.extend(refs)
                hypotheses.extend(hypos)
            
             
            test_avg_bleu_scores, combined = compute_smoothed_sentence_bleu(queries, references, hypotheses)
            
            logger.info(f'Epoch: {epoch}, Test Bleu-1: {test_avg_bleu_scores[0]:.2f}, Bleu-2: {test_avg_bleu_scores[1]:.2f}, Bleu-3: {test_avg_bleu_scores[2]:.2f}, Bleu-4: {test_avg_bleu_scores[3]:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f"Query: {' '.join(queries[rand_index])}")
            logger.info(f"Reference: {' '.join(references[rand_index])}")
            logger.info(f"Predicted: {' '.join(hypotheses[rand_index])}")
            
            with open(f'{config.save_path}/test_output_{epoch}.json', 'w') as f_obj:
                json.dump(combined, f_obj)
                
            with open(f'{config.save_path}/test_output_{epoch}.txt', 'w') as f_obj:
                for item in combined:
                    f_obj.write(f"Query: {' '.join(item[0])}\n")
                    f_obj.write(f"Reference: {' '.join(item[1])}\n")
                    f_obj.write(f"Predicted: {' '.join(item[2])}\n")
                    f_obj.write("\n")
            
             
            if best_bleu is None or best_bleu < val_avg_bleu_scores[-1]:
                logger.info(f'Best validation bleu-4 score so far: {val_avg_bleu_scores[-1]}. Saving the model.')
                save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
                torch.save(model, save_path)
                
                best_bleu = val_avg_bleu_scores[-1]
                best_epoch = epoch
            elif epoch % config.save_every == 0:
                save_path = os.path.join(config.save_path, f'model_{epoch}.pt')
                torch.save(model, save_path)
                

    logger.info(f'Best epoch: {best_epoch} Best Validation Bleu-4: {best_bleu:.2f}')

    
def train_bow_vae(config, logger, writer):
    
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    with open(os.path.join(config.save_path, 'config.pkl'), 'wb') as f_obj:
        pickle.dump(config, f_obj)
        
    config.summary_writer = writer    
    
    sentences = load_sentences(config, logger)
    count_vectorizer, tfidf_vectorizer = load_vectorizers(sentences, config, logger)
    
    config.vae_vocab_size = len(count_vectorizer.vocabulary_)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        datasets[part] = TensorDataset(torch.arange(len(sentences[part])))
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.vae_batch_size, shuffle=shuffle, pin_memory=True)
        
    
    model = BOW_VAE(config.vae_hidden_size, config.latent_size, config.vae_vocab_size, config.vae_temperature) 
    model = model.cuda()
    
    optimizer = Adam(model.parameters(), lr=config.vae_lr)
    
    #loss_fn = nn.BCEWithLogitsLoss()
    
    def loss_fn(logits, x):
        return -torch.sum(logits * x, dim=-1)
    
    def kl_anneal_function(anneal_fn, step, slope, margin, max_step):
        if step > max_step:
            step = max_step
            
        if anneal_fn == 'tanh':
            return np.round((np.tanh((step - 4500) / 1000) + 1) / 2, decimals=6)
            #return 0.5*(np.tanh(slope*(step - margin)) + 1)
        elif anneal_fn == 'sigmoid':
            return 1/(1 + np.exp(-slope*(step - margin)))
    
    logger.info('Begining training.')
    step = 0
    best_epoch = None
    best_bleu = 0.0
    for epoch in range(1, config.num_epochs):
        train_rec_loss = []
        train_kl_loss = 0.0
        train_bleu = []
        train_tokens_ret = []
        train_total_tokens = []
        
        model.train()
        with torch.set_grad_enabled(True):
            for batch in dataloaders['train']:
                indices = batch[0].tolist()
                
                x = load_bow_batch(sentences['train'], indices, count_vectorizer, tfidf_vectorizer, config, logger, epoch)
                
                # Empty batch due to removal of stopwords
                if x == None:
                    continue
                else:
                    x = x.cuda()
                
                batch_size = x.size(0)
                
                mean, logv, z, logits = model(x)
                
                rec_loss = loss_fn(logits, x)
                
                KL_loss = -0.5 * torch.sum(1 + 2 * logv - mean.pow(2) - torch.exp(2 * logv))
                KL_weight = kl_anneal_function('tanh', step, config.vae_slope, config.vae_margin, config.vae_max_step)
                
                batch_loss = torch.mean(rec_loss) + ((KL_weight * KL_loss) / batch_size)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                train_rec_loss.extend(rec_loss.detach().cpu().tolist())
                train_kl_loss += KL_loss
                
                
                bleu_scores, tokens_ret, total_tokens, _, _ = compute_metrics(logits, x, config.sample_count, count_vectorizer)
                train_bleu.extend(bleu_scores)
                train_tokens_ret.extend(tokens_ret)
                train_total_tokens.extend(total_tokens)
                
                if (step+1) % config.print_every == 0:
                    precision = np.sum(tokens_ret) / (len(tokens_ret) * config.sample_count)
                    recall = np.sum(tokens_ret) / np.sum(total_tokens)
                    #logger.info(f'Epoch: {epoch} Step: {step+1:04} KL Weight: {KL_weight:0.4f} KL loss: {KL_loss.item() / batch_size:.4f}, Reconstruction loss: {rec_loss.item()/batch_size:.6f}, Total loss: {batch_loss.item():.6f}, Bleu Score: {np.mean(bleu_scores):.8f}')
                    logger.info(f'Epoch: {epoch} Step: {step+1:04} Reconstruction loss: {batch_loss.item():.4f}, KL Weight: {KL_weight:0.4f}, KL loss: {KL_loss.item()/batch_size:.4f}, Bleu Score: {np.mean(bleu_scores):.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
                
                step += 1
                
            
            train_rec_loss = np.mean(train_rec_loss)
            train_kl_loss = train_kl_loss / len(datasets['train'])
            train_bleu = np.mean(train_bleu)
            train_precision = np.sum(train_tokens_ret) / (len(train_tokens_ret) * config.sample_count)
            train_recall = np.sum(train_tokens_ret) / np.sum(train_total_tokens)
            
            logger.info(f'Epoch: {epoch}, Training -> Reconstruction Loss: {train_rec_loss:.2f}, KL Loss: {train_kl_loss:.2f} Bleu Score: {train_bleu:.2f}, Precsion: {train_precision:.2f}, Recall: {train_recall:.2f}')
            
            config.summary_writer.add_scalar('Reconstruction_Loss/Train', train_rec_loss, epoch)
            config.summary_writer.add_scalar('KL_Loss/Train', train_kl_loss, epoch)
            config.summary_writer.add_scalar('KL_Weight', KL_weight, epoch)
            config.summary_writer.add_scalar('Precision/Train', train_precision, epoch)
            config.summary_writer.add_scalar('Recall/Train', train_recall, epoch)
        
        
        logger.info('Performing validation.')
        model.eval()
        with torch.set_grad_enabled(False):
            references = []
            hypotheses = []
            val_rec_loss = []
            val_bleu = []
            val_tokens_ret = []
            val_total_tokens = []
            for batch in dataloaders['valid']:
                indices = batch[0].tolist()
                
                x = load_bow_batch(sentences['valid'], indices, count_vectorizer, tfidf_vectorizer, config, logger, epoch)
                
                # Empty batch due to removal of stopwords
                if x == None:
                    continue
                else:
                    x = x.cuda()
                    
                batch_size = x.size(0)
                
                mean, logv, z, logits = model(x)
                
                rec_loss = loss_fn(logits, x)
                
                val_rec_loss.extend(rec_loss.detach().cpu().tolist())
                
                bleu_scores, tokens_ret, total_tokens, refs, hypos = compute_metrics(logits, x, config.sample_count, count_vectorizer)
                references.extend(refs)
                hypotheses.extend(hypos)
                val_bleu.extend(bleu_scores)
                val_tokens_ret.extend(tokens_ret)
                val_total_tokens.extend(total_tokens)
            
            
            val_rec_loss = np.mean(val_rec_loss)
            val_bleu = np.mean(val_bleu)
            val_precision = np.sum(val_tokens_ret) / (len(val_tokens_ret) * config.sample_count)
            val_recall = np.sum(val_tokens_ret) / np.sum(val_total_tokens)
            logger.info(f'Validation Set -> Reconstruction Loss: {val_rec_loss:.2f} Bleu Score: {val_bleu:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f'Reference: {references[rand_index]}')
            logger.info(f'Reconstruction: {hypotheses[rand_index]}')
            
            config.summary_writer.add_scalar('Reconstruction_Loss/Valid', val_rec_loss, epoch)
            config.summary_writer.add_scalar('Precision/Valid', val_precision, epoch)
            config.summary_writer.add_scalar('Recall/Valid', val_recall, epoch)
            
            test_rec_loss = []
            test_bleu = []
            test_tokens_ret = []
            test_total_tokens = []
            for batch in dataloaders['test']:
                indices = batch[0].tolist()
                
                x = load_bow_batch(sentences['test'], indices, count_vectorizer, tfidf_vectorizer, config, logger, epoch)
                
                # Empty batch due to removal of stopwords
                if x == None:
                    continue
                else:
                    x = x.cuda()
                    
                batch_size = x.size(0)
                
                mean, logv, z, logits = model(x)
                #hidden, logits = model(x)
                
                rec_loss = loss_fn(logits, x)
                                
                test_rec_loss.extend(rec_loss.detach().cpu().tolist())
                
                bleu_scores, tokens_ret, total_tokens, refs, hypos = compute_metrics(logits, x, config.sample_count, count_vectorizer)
                references.extend(refs)
                hypotheses.extend(hypos)
                test_bleu.extend(bleu_scores)
                test_tokens_ret.extend(tokens_ret)
                test_total_tokens.extend(total_tokens)
            
            
            test_rec_loss = np.mean(test_rec_loss)
            test_bleu = np.mean(test_bleu)
            test_precision = np.sum(test_tokens_ret) / (len(test_tokens_ret) * config.sample_count)
            test_recall = np.sum(test_tokens_ret) / np.sum(test_total_tokens)
            logger.info(f'Test Set -> Reconstruction Loss: {test_rec_loss:.2f} Bleu Score: {test_bleu:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}')
            logger.info(f'Random Item')
            rand_index = np.random.randint(0, len(references))
            logger.info(f'Reference: {references[rand_index]}')
            logger.info(f'Reconstruction: {hypotheses[rand_index]}')
            
            config.summary_writer.add_scalar('Reconstruction_Loss/Test', test_rec_loss, epoch)
            config.summary_writer.add_scalar('Precision/Test', test_precision, epoch)
            config.summary_writer.add_scalar('Recall/Test', test_recall, epoch)
            
            logger.info('Sampling from prior.')
            z = torch.randn(1, config.latent_size).cuda()
            logits = model.decode(z)
            
            logger.info(logits_to_bow(logits, config.sample_count, count_vectorizer)[0].tolist())
            
            if best_bleu <= val_bleu:
                logger.info('Best validation bleu score so far. Saving the model.')
                save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
                torch.save(model, save_path)
                
                best_epoch = epoch
                best_bleu = val_bleu
            elif epoch % config.save_every == 0:
                logger.info('Saving the model.')
                
                save_path = os.path.join(config.save_path, f'model_{epoch}.pt')
                torch.save(model, save_path)

        
    logger.info(f'Best epoch: {best_epoch} Best bleu: {best_bleu}')
    
                    
def train_gan(config, logger):
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    
    sentences = load_sentences(config, logger)
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config, logger)
    count_vectorizer, tfidf_vectorizer = load_vectorizers(sentences, config, logger)
    contexts, curr_sents, next_sents = load_tuples(config, logger)
    
    
    config.vocab_size = len(count_vectorizer.vocabulary_)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        
        datasets[part] = TensorDataset(torch.arange(len(curr_sents[part])))
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.gan_batch_size, shuffle=shuffle, pin_memory=True)
        
    
    logger.info(f'Loading saved language model: {config.lm_path[config.dataset]}')
    lm_model = load_model(config.lm_path[config.dataset])
    logger.info(f'Loading saved vae: {config.vae_path[config.dataset]}')
    vae_model = load_model(config.vae_path[config.dataset])
    
    #generator = GAN_Generator(((4*config.lm_hidden_size) + config.latent_size), config.latent_size)
    generator = GAN_Generator(config.latent_size, config.latent_size)
    generator.cuda()
    
    discriminator = GAN_Discriminator(2*config.latent_size, config.gan_disc_hidden_size)
    discriminator.cuda()
    
    #aux_discriminator = GAN_Discriminator(((4*config.lm_hidden_size) + config.latent_size), config.gan_aux_disc_hidden_size)
    #aux_discriminator.cuda() 
    
    logger.info('Generator')
    logger.info(generator)
    logger.info('Discriminator')
    logger.info(discriminator)
    #logger.info('Auxiliary Discriminator')
    #logger.info(aux_discriminator)
    
    #gen_optimizer = Adam(chain(generator.parameters(), vae_model.parameters(), lm_model.parameters()), lr=config.gan_gen_lr, weight_decay=1e-5)
    gen_optimizer = Adam(chain(generator.parameters(), vae_model.encoder.parameters(), vae_model.hidden2mean.parameters(), vae_model.hidden2logv.parameters()), lr=config.gan_gen_lr, weight_decay=1e-5)
    #gen_optimizer = Adam(chain(generator.parameters(), vae_model.decoder.parameters()), lr=config.gan_gen_lr, weight_decay=1e-5)
    #gen_optimizer = Adam(generator.parameters(), lr=config.gan_gen_lr, weight_decay=1e-5)
    dis_optimizer = SGD(discriminator.parameters(), lr=config.gan_dis_lr, weight_decay=1e-5)
    #aux_dis_optimizer = SGD(aux_discriminator.parameters(), lr=config.gan_dis_lr, weight_decay=1e-5)
    
    gen_ce_loss_fn = nn.BCEWithLogitsLoss()
    dis_ce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    
    def rec_loss_fn(logits, x):
        return -torch.sum(logits * x, dim=-1)
    
       
    logger.info('Begining training.')
    step = 0
    best_epoch = None
    best_bleu = 0.0
    for epoch in range(1, config.num_epochs):
        references = []
        hypotheses = []
        
        train_bleu = []
        train_tokens_ret = []
        train_total_tokens = []
        
        generator.train()
        discriminator.train()
        vae_model.train()
        lm_model.train()
        with torch.set_grad_enabled(True):
            ref_context_seq = []
            ref_curr_sent_seq = []
            ref_next_sent_seq = []
            curr_sent_references = []
            next_sent_references = []
            hypotheses = []
            for batch in dataloaders['train']:
                indices = batch[0].tolist()
                
                context_seq, curr_sent_seq, next_sent_seq, context_seq_lens, curr_sent_seq_lens, curr_sent_vecs, next_sent_vecs = load_gan_batch(contexts['train'], curr_sents['train'], next_sents['train'], indices, count_vectorizer, tfidf_vectorizer, tokenizer, word2index, config, logger)
                
                # empty batch, continue
                if type(context_seq) == type(None):
                    continue
                
                #context_seq = torch.tensor(context_seq, dtype=torch.long).cuda()
                #context_seq_lens = torch.tensor(context_seq_lens).cuda()
                
                #curr_sent_seq = torch.tensor(curr_sent_seq, dtype=torch.long).cuda()
                #curr_sent_seq_lens = torch.tensor(curr_sent_seq_lens).cuda()
                
                curr_sent_vecs = curr_sent_vecs.cuda() 
                next_sent_vecs = next_sent_vecs.cuda() 
                
                # Obtain context representation
                #context_sorted_indices = torch.argsort(context_seq_lens, descending=True)
                
                #context_seq = context_seq[context_sorted_indices]
                #context_seq_lens = context_seq_lens[context_sorted_indices]
                
                #context_h_n, _ = lm_model(context_seq, context_seq_lens)
                
                #context_reverse_indices = torch.argsort(context_sorted_indices)
                
                #context_h_n = context_h_n[context_reverse_indices]
                #context_seq = context_seq[context_reverse_indices]
                
                # Obtain current sentence representation
                #curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lens, descending=True)
                
                #curr_sent_seq = curr_sent_seq[curr_sent_sorted_indices]
                #curr_sent_seq_lens = curr_sent_seq_lens[curr_sent_sorted_indices]
                
                #curr_sent_h_n, _ = lm_model(curr_sent_seq, curr_sent_seq_lens)
                
                #curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                #curr_sent_h_n = curr_sent_h_n[curr_sent_reverse_indices]
                #curr_sent_seq = curr_sent_seq[curr_sent_reverse_indices]
                
                # Obtain current and next sentence latent codes
                _, _, curr_sent_z = encode_vectors(curr_sent_vecs, vae_model, config.temperature)
                _, _, next_sent_z = encode_vectors(next_sent_vecs, vae_model, config.temperature)
                
                gen_optimizer.zero_grad()
                
                #gen_input = torch.cat((context_h_n.detach(), curr_sent_h_n.detach(), curr_sent_z), dim=-1)
                
                #predicted_z = generator(gen_input)
                predicted_z = generator(curr_sent_z)
                
                gen_pair = torch.cat((curr_sent_z, predicted_z), dim=-1)
                
                d_logits = discriminator(gen_pair)
                
                #aux_input = torch.cat((context_h_n.detach(), curr_sent_h_n.detach(), predicted_z.detach()), dim=-1)
                
                #aux_d_logits = aux_discriminator(aux_input)
                
                valid = torch.ones_like(d_logits).cuda()
                
                vae_logits = decode_vectors(predicted_z, vae_model)
                
                ce_loss = gen_ce_loss_fn(d_logits, valid)
                #aux_ce_loss = gen_ce_loss_fn(aux_d_logits, valid)
                mse_loss = mse_loss_fn(predicted_z, next_sent_z)
                rec_loss = torch.mean(rec_loss_fn(vae_logits, next_sent_vecs))
                
                #gen_loss = ce_loss
                #gen_loss = ce_loss + config.mse_weight * mse_loss
                #gen_loss = ce_loss + config.rec_weight * rec_loss
                gen_loss = ce_loss + config.mse_weight * mse_loss + config.rec_weight * rec_loss
                #gen_loss = ce_loss + aux_ce_loss + config.mse_weight * mse_loss + config.rec_weight * rec_loss
                
                gen_acc = torch.sigmoid(d_logits).round().mean() 
                
                #gen_aux_acc = torch.sigmoid(aux_d_logits).round().mean() 
                
                gen_loss.backward(retain_graph=True)
                
                #nn.utils.clip_grad_norm_(chain(generator.parameters(), vae_model.decoder.parameters(), discriminator.parameters(), aux_discriminator.parameters()), 5)
                nn.utils.clip_grad_norm_(chain(generator.parameters(), vae_model.decoder.parameters(), discriminator.parameters()), 5)
                nn.utils.clip_grad_value_(generator.parameters(), 1)
                
                gen_optimizer.step()
                
                dis_optimizer.zero_grad()
                #aux_dis_optimizer.zero_grad()
                
                true_pair = torch.cat((curr_sent_z, next_sent_z), dim=-1)
                
                real_pred_logits = discriminator(true_pair)
                real_d_loss = dis_ce_loss_fn(real_pred_logits, valid)
                
                fake = torch.zeros_like(d_logits).cuda()
                
                fake_pred_logits = discriminator(gen_pair.detach())
                fake_d_loss = dis_ce_loss_fn(fake_pred_logits, fake)
                
                disc_real_acc = torch.sigmoid(real_pred_logits).round().mean()
                disc_fake_acc = (fake_pred_logits.size(0) - torch.sigmoid(fake_pred_logits).round().sum()) / fake_pred_logits.size(0)
                
                disc_loss = (real_d_loss + fake_d_loss) / 2
                
                #true_aux_input = torch.cat((context_h_n.detach(), curr_sent_h_n.detach(), next_sent_z), dim=-1)
                
                #real_aux_pred_logits = aux_discriminator(true_aux_input)
                #real_aux_d_loss = dis_ce_loss_fn(real_aux_pred_logits, valid)
                
                #fake = torch.zeros_like(d_logits).cuda()
                
                #fake_aux_pred_logits = aux_discriminator(aux_input.detach())
                #fake_aux_d_loss = dis_ce_loss_fn(fake_aux_pred_logits, fake)
                
                #aux_disc_real_acc = torch.sigmoid(real_aux_pred_logits).round().mean()
                #aux_disc_fake_acc = (fake_aux_pred_logits.size(0) - torch.sigmoid(fake_aux_pred_logits).round().sum()) / fake_aux_pred_logits.size(0)
                
                #aux_disc_loss = (real_aux_d_loss + fake_aux_d_loss) / 2
                
                disc_loss.backward(retain_graph=True)
                dis_optimizer.step()
                
                #aux_disc_loss.backward()
                #aux_dis_optimizer.step()
                
                bleu_scores, tokens_ret, total_tokens, refs, hypos = compute_metrics(vae_logits, next_sent_vecs, config.sample_count, count_vectorizer)
                #ref_context_seq.extend(indices2words(context_seq.detach().cpu().tolist(), index2word))
                ref_context_seq.extend(indices2words(context_seq.tolist(), index2word))
                #ref_curr_sent_seq.extend(indices2words(curr_sent_seq.detach().cpu().tolist(), index2word))
                ref_curr_sent_seq.extend(indices2words(curr_sent_seq.tolist(), index2word))
                ref_next_sent_seq.extend(next_sent_seq)
                curr_sent_references.extend(one_hot_to_bow(curr_sent_vecs, count_vectorizer))
                next_sent_references.extend(refs)
                hypotheses.extend(hypos)
                train_bleu.extend(bleu_scores)
                train_tokens_ret.extend(tokens_ret)
                train_total_tokens.extend(total_tokens)
                
                if (step+1) % config.print_every == 0:
                    precision = np.sum(tokens_ret) / (len(tokens_ret) * config.sample_count)
                    recall = np.sum(tokens_ret) / np.sum(total_tokens)
                    
                    '''
                    logger.info(f'Epoch: {epoch} Step: {step+1:04} Generator -> Accuracy: {gen_acc.item():.2f} gen_acc CE Loss: {ce_loss.item():.2f}, Total Loss: {gen_loss.item():.2f} ' +
                           f'Discriminator -> Real Accuracy: {disc_real_acc.item():.2f} Fake Accuracy: {disc_fake_acc.item():.2f} Real CE Loss: {real_d_loss.item():.2f}, Fake CE Loss: {fake_d_loss.item():.2f} ' + 
                           f'Aux Discriminator -> Real Accuracy: {aux_disc_real_acc.item():.2f} Fake Accuracy: {aux_disc_fake_acc.item():.2f} Real CE Loss: {real_aux_d_loss.item():.2f}, Fake CE Loss: {fake_aux_d_loss.item():.2f}, Total Loss: {((real_aux_d_loss + fake_aux_d_loss)/2).item():.2f}')
                    '''
                    logger.info(f'Epoch: {epoch} Step: {step+1:04} Generator -> Accuracy: {gen_acc.item():.2f} gen_acc CE Loss: {ce_loss.item():.2f}, Total Loss: {gen_loss.item():.2f} ' +
                           f'Discriminator -> Real Accuracy: {disc_real_acc.item():.2f} Fake Accuracy: {disc_fake_acc.item():.2f} Real CE Loss: {real_d_loss.item():.2f}, Fake CE Loss: {fake_d_loss.item():.2f}, Total Loss: {((real_d_loss + fake_d_loss)/2).item():.2f}')
                    logger.info(f'Epoch: {epoch} Step: {step+1:04} Bleu Score: {np.mean(bleu_scores):.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')
                    
                step += 1
                
            train_bleu = np.mean(train_bleu)
            train_precision = np.sum(train_tokens_ret) / (len(train_tokens_ret) * config.sample_count)
            train_recall = np.sum(train_tokens_ret) / np.sum(train_total_tokens)
            logger.info(f'Epoch: {epoch}, Training -> Bleu Score: {train_bleu:.2f}, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}')
            
            output = []
            for index in range(len(ref_context_seq)):
                o_dict = {'context': ref_context_seq[index], 'current sentence': ref_curr_sent_seq[index], 'next sentence': ref_next_sent_seq[index], 
                          'curr_sent_bow': curr_sent_references[index], 'next_sent_bow': next_sent_references[index], 'predicted_sent_bow': hypotheses[index]}
                output.append(o_dict)
            
            with open(f'{config.save_path}/train_output_{epoch}.json', 'w') as f_obj:
                json.dump(output, f_obj)
                
            with open(f'{config.save_path}/train_output_{epoch}.txt', 'w') as f_obj:
                for o_dict in output:
                    for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'predicted_sent_bow']:
                        if key in ['context', 'current sentence', 'next sentence']:
                            f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                        elif key == 'next sentence':
                            f_obj.write(f"{key}: {o_dict[key]}\n")
                        else:
                            f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                    f_obj.write('\n')
        
        logger.info('Performing validation.')
        generator.eval()
        vae_model.eval()
        lm_model.eval()
        with torch.set_grad_enabled(False):
            ref_context_seq = []
            ref_curr_sent_seq = []
            ref_next_sent_seq = []
            curr_sent_references = []
            next_sent_references = []
            hypotheses = []
            val_recall_bleu = []
            val_precision_bleu = []
            '''
            val_bleu = []
            '''
            val_tokens_ret = []
            val_total_tokens = []
            for batch in dataloaders['valid']:
                indices = batch[0].tolist()
                
                context_seq, curr_sent_seq, next_sent_seq, context_seq_lens, curr_sent_seq_lens, curr_sent_vecs, next_sent_vecs = load_gan_batch(contexts['valid'], curr_sents['valid'], next_sents['valid'], indices, count_vectorizer, tfidf_vectorizer, tokenizer, word2index, config, logger)
                
                '''
                # empty batch, continue
                if type(context_seq) == type(None):
                    continue
                
                context_seq = torch.tensor(context_seq, dtype=torch.long).cuda()
                context_seq_lens = torch.tensor(context_seq_lens).cuda()
                
                curr_sent_seq = torch.tensor(curr_sent_seq, dtype=torch.long).cuda()
                curr_sent_seq_lens = torch.tensor(curr_sent_seq_lens).cuda()
                
                # Obtain context representation
                context_sorted_indices = torch.argsort(context_seq_lens, descending=True)
                
                context_seq = context_seq[context_sorted_indices]
                context_seq_lens = context_seq_lens[context_sorted_indices]
                
                context_h_n, _ = lm_model(context_seq, context_seq_lens)
                
                context_reverse_indices = torch.argsort(context_sorted_indices)
                
                context_h_n = context_h_n[context_reverse_indices]
                context_seq = context_seq[context_reverse_indices]
                
                # Obtain current sentence representation
                curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lens, descending=True)
                
                curr_sent_seq = curr_sent_seq[curr_sent_sorted_indices]
                curr_sent_seq_lens = curr_sent_seq_lens[curr_sent_sorted_indices]
                
                curr_sent_h_n, _ = lm_model(curr_sent_seq, curr_sent_seq_lens)
                
                curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                curr_sent_h_n = curr_sent_h_n[curr_sent_reverse_indices]
                curr_sent_seq = curr_sent_seq[curr_sent_reverse_indices]
                '''
                
                curr_sent_vecs = curr_sent_vecs.cuda() 
                next_sent_vecs = next_sent_vecs.cuda() 
                
                # Obtain current sentence latent code
                #_, _, curr_sent_z = encode_vectors(curr_sent_vecs, vae_model, config.temperature)
                _, _, curr_sent_z = encode_vectors_with_sampling(curr_sent_vecs, vae_model, config.temperature, 1.0, config.latent_samples)
                
                #context_h_n = torch.repeat_interleave(context_h_n, config.latent_samples, dim=0)
                #curr_sent_h_n = torch.repeat_interleave(curr_sent_h_n, config.latent_samples, dim=0)
                
                gen_optimizer.zero_grad()
                
                #gen_input = torch.cat((context_h_n.detach(), curr_sent_h_n.detach(), curr_sent_z), dim=-1)
                
                #predicted_z = generator(gen_input)
                predicted_z = generator(curr_sent_z)
                
                vae_logits = decode_vectors(predicted_z, vae_model)
                
                recall_bleus, precision_bleus, tokens_ret, total_tokens, refs, hypos = multi_compute_metrics(vae_logits, next_sent_vecs, config.sample_count, count_vectorizer, config.latent_samples)
                #bleu_scores, tokens_ret, total_tokens, refs, hypos = compute_metrics(vae_logits, next_sent_vecs, config.sample_count, count_vectorizer)
                #ref_context_seq.extend(indices2words(context_seq.detach().cpu().tolist(), index2word))
                ref_context_seq.extend(indices2words(context_seq.tolist(), index2word))
                #ref_curr_sent_seq.extend(indices2words(curr_sent_seq.detach().cpu().tolist(), index2word))
                ref_curr_sent_seq.extend(indices2words(curr_sent_seq.tolist(), index2word))
                ref_next_sent_seq.extend(next_sent_seq)
                curr_sent_references.extend(one_hot_to_bow(curr_sent_vecs, count_vectorizer))
                next_sent_references.extend(refs)
                hypotheses.extend(hypos)
                val_recall_bleu.extend(recall_bleus)
                val_precision_bleu.extend(precision_bleus)
                '''
                val_bleu.extend(bleu_scores)
                '''
                val_tokens_ret.extend(tokens_ret)
                val_total_tokens.extend(total_tokens)
                            
            
            val_precision = np.sum(val_tokens_ret) / (len(val_tokens_ret) * config.sample_count)
            val_recall = np.sum(val_tokens_ret) / np.sum(val_total_tokens)
            val_recall_bleu = np.mean(val_recall_bleu)
            val_precision_bleu = np.mean(val_precision_bleu)
            '''
            val_bleu = np.mean(val_bleu)
            '''
            logger.info(f'Validation Set -> Avg Recall Bleu Score: {val_recall_bleu:.2f}, Avg Precision Bleu Score: {val_precision_bleu:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}')
            #logger.info(f'Validation Set -> Bleu Score: {val_bleu:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}')
            logger.info(f'Random Validations Items')
            rand_index = np.random.randint(0, len(curr_sent_references))
            logger.info('-'*60)
            logger.info(f'Ground Current Sentence Query: {curr_sent_references[rand_index]}')
            logger.info(f'Ground Next Sentence Response: {next_sent_references[rand_index]}')
            for index in range(config.latent_samples):
                logger.info(f'Predicted Response #{index+1}: {hypotheses[(rand_index*config.latent_samples)+index]}')
            #logger.info(f'Predicted Sentence: {hypotheses[rand_index]}')
            logger.info('-'*60)
            
            output = []
            for index in range(len(ref_context_seq)):
                o_dict = {'context': ref_context_seq[index], 'current sentence': ref_curr_sent_seq[index], 'next sentence': ref_next_sent_seq[index], 
                          'curr_sent_bow': curr_sent_references[index], 'next_sent_bow': next_sent_references[index], 'predicted_sent_bow': hypotheses[(index*config.latent_samples):(index*config.latent_samples)+config.latent_samples]}
                '''
                o_dict = {'context': ref_context_seq[index], 'current sentence': ref_curr_sent_seq[index], 'next sentence': ref_next_sent_seq[index], 
                          'curr_sent_bow': curr_sent_references[index], 'next_sent_bow': next_sent_references[index], 'predicted_sent_bow': hypotheses[index]}
                '''
                output.append(o_dict)
            
            with open(f'{config.save_path}/valid_output_{epoch}.json', 'w') as f_obj:
                json.dump(output, f_obj)
                
            with open(f'{config.save_path}/valid_output_{epoch}.txt', 'w') as f_obj:
                for o_dict in output:
                    for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'predicted_sent_bow']:
                        if key in ['context', 'current sentence']:
                            f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                        elif key == 'next sentence':
                            f_obj.write(f"{key}: {o_dict[key]}\n")
                        elif key == 'predicted_sent_bow':
                            for index, pred_bow in enumerate(o_dict[key]):
                                f_obj.write(f"{key} #{index+1}: {', '.join(o_dict[key][index])}\n")
                        else:
                            f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                    f_obj.write('\n')
            
            ref_context_seq = []
            ref_curr_sent_seq = []
            ref_next_sent_seq = []
            curr_sent_references = []
            next_sent_references = []
            hypotheses = []

            test_recall_bleu = []
            test_precision_bleu = []
            '''
            test_bleu = []
            '''
            test_tokens_ret = []
            test_total_tokens = []
            for batch in dataloaders['test']:
                indices = batch[0].tolist()
                
                context_seq, curr_sent_seq, next_sent_seq, context_seq_lens, curr_sent_seq_lens, curr_sent_vecs, next_sent_vecs = load_gan_batch(contexts['test'], curr_sents['test'], next_sents['test'], indices, count_vectorizer, tfidf_vectorizer, tokenizer, word2index, config, logger)
                
                '''
                # empty batch, continue
                if type(context_seq) == type(None):
                    continue
                
                context_seq = torch.tensor(context_seq, dtype=torch.long).cuda()
                context_seq_lens = torch.tensor(context_seq_lens).cuda()
                
                curr_sent_seq = torch.tensor(curr_sent_seq, dtype=torch.long).cuda()
                curr_sent_seq_lens = torch.tensor(curr_sent_seq_lens).cuda()
                
                # Obtain context representation
                context_sorted_indices = torch.argsort(context_seq_lens, descending=True)
                
                context_seq = context_seq[context_sorted_indices]
                context_seq_lens = context_seq_lens[context_sorted_indices]
                
                context_h_n, _ = lm_model(context_seq, context_seq_lens)
                
                context_reverse_indices = torch.argsort(context_sorted_indices)
                
                context_h_n = context_h_n[context_reverse_indices]
                context_seq = context_seq[context_reverse_indices]
                
                # Obtain current sentence representation
                curr_sent_sorted_indices = torch.argsort(curr_sent_seq_lens, descending=True)
                
                curr_sent_seq = curr_sent_seq[curr_sent_sorted_indices]
                curr_sent_seq_lens = curr_sent_seq_lens[curr_sent_sorted_indices]
                
                curr_sent_h_n, _ = lm_model(curr_sent_seq, curr_sent_seq_lens)
                
                curr_sent_reverse_indices = torch.argsort(curr_sent_sorted_indices)
                
                curr_sent_h_n = curr_sent_h_n[curr_sent_reverse_indices]
                curr_sent_seq = curr_sent_seq[curr_sent_reverse_indices]
                '''
                
                curr_sent_vecs = curr_sent_vecs.cuda() 
                next_sent_vecs = next_sent_vecs.cuda() 
                
                # Obtain current sentence latent code
                #_, _, curr_sent_z = encode_vectors(curr_sent_vecs, vae_model, config.temperature)
                _, _, curr_sent_z = encode_vectors_with_sampling(curr_sent_vecs, vae_model, config.temperature, 1.0, config.latent_samples)
                
                #context_h_n = torch.repeat_interleave(context_h_n, config.latent_samples, dim=0)
                #curr_sent_h_n = torch.repeat_interleave(curr_sent_h_n, config.latent_samples, dim=0)
                
                gen_optimizer.zero_grad()
                
                #gen_input = torch.cat((context_h_n.detach(), curr_sent_h_n.detach(), curr_sent_z), dim=-1)
                
                #predicted_z = generator(gen_input)
                predicted_z = generator(curr_sent_z)
                
                vae_logits = decode_vectors(predicted_z, vae_model)
                
                recall_bleus, precision_bleus, tokens_ret, total_tokens, refs, hypos = multi_compute_metrics(vae_logits, next_sent_vecs, config.sample_count, count_vectorizer, config.latent_samples)
                #bleu_scores, tokens_ret, total_tokens, refs, hypos = compute_metrics(vae_logits, next_sent_vecs, config.sample_count, count_vectorizer)
                #ref_context_seq.extend(indices2words(context_seq.detach().cpu().tolist(), index2word))
                ref_context_seq.extend(indices2words(context_seq.tolist(), index2word))
                #ref_curr_sent_seq.extend(indices2words(curr_sent_seq.detach().cpu().tolist(), index2word))
                ref_curr_sent_seq.extend(indices2words(curr_sent_seq.tolist(), index2word))
                ref_next_sent_seq.extend(next_sent_seq)
                curr_sent_references.extend(one_hot_to_bow(curr_sent_vecs, count_vectorizer))
                next_sent_references.extend(refs)
                hypotheses.extend(hypos)
                test_recall_bleu.extend(recall_bleus)
                test_precision_bleu.extend(precision_bleus)
                '''
                test_bleu.extend(bleu_scores)
                '''
                test_tokens_ret.extend(tokens_ret)
                test_total_tokens.extend(total_tokens)
                            
            
            test_precision = np.sum(test_tokens_ret) / (len(test_tokens_ret) * config.sample_count)
            test_recall = np.sum(test_tokens_ret) / np.sum(test_total_tokens)
            test_recall_bleu = np.mean(test_recall_bleu)
            test_precision_bleu = np.mean(test_precision_bleu)
            '''
            test_bleu = np.mean(test_bleu)
            '''
            logger.info(f'Validation Set -> Avg Recall Bleu Score: {val_recall_bleu:.2f}, Avg Precision Bleu Score: {val_precision_bleu:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}')
            #logger.info(f'Test Set -> Bleu Score: {test_bleu:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}')
            logger.info(f'Random Test Items')
            rand_index = np.random.randint(0, len(curr_sent_references))
            logger.info('-'*60)
            logger.info(f'Ground Truth Current Sentence: {curr_sent_references[rand_index]}')
            logger.info(f'Ground Truth Next Sentence: {next_sent_references[rand_index]}')
            for index in range(config.latent_samples):
                logger.info(f'Predicted Response #{index+1}: {hypotheses[(rand_index*config.latent_samples)+index]}')
            #logger.info(f'Predicted Sentence: {hypotheses[rand_index]}')
            logger.info('-'*60)
            
            output = []
            output = []
            for index in range(len(ref_context_seq)):
                o_dict = {'context': ref_context_seq[index], 'current sentence': ref_curr_sent_seq[index], 'next sentence': ref_next_sent_seq[index], 
                          'curr_sent_bow': curr_sent_references[index], 'next_sent_bow': next_sent_references[index], 'predicted_sent_bow': hypotheses[(index*config.latent_samples):(index*config.latent_samples)+config.latent_samples]}
                '''
                o_dict = {'context': ref_context_seq[index], 'current sentence': ref_curr_sent_seq[index], 'next sentence': ref_next_sent_seq[index], 
                          'curr_sent_bow': curr_sent_references[index], 'next_sent_bow': next_sent_references[index], 'predicted_sent_bow': hypotheses[index]}
                '''
                output.append(o_dict)
            
            with open(f'{config.save_path}/test_output_{epoch}.json', 'w') as f_obj:
                json.dump(output, f_obj)
                
            with open(f'{config.save_path}/test_output_{epoch}.txt', 'w') as f_obj:
                for o_dict in output:
                    for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'predicted_sent_bow']:
                        if key in ['context', 'current sentence']:
                            f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                        elif key == 'next sentence':
                            f_obj.write(f"{key}: {o_dict[key]}\n")
                        elif key == 'predicted_sent_bow':
                            for index, pred_bow in enumerate(o_dict[key]):
                                f_obj.write(f"{key} #{index+1}: {', '.join(o_dict[key][index])}\n")
                        else:
                            f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                    f_obj.write('\n')
            
            if best_bleu <= val_recall_bleu:
            #if best_bleu <= val_bleu:
                logger.info('Best validation recall bleu score so far. Saving the model.')
                save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
                torch.save(generator, save_path)
                
                best_bleu = val_recall_bleu
                #best_bleu = val_bleu
                best_epoch = epoch
            '''
            elif epoch % config.save_every == 0:
                logger.info('Saving the model.')
                
                save_path = os.path.join(config.save_path, f'model_{epoch}.pt')
                torch.save(generator, save_path)
            '''
    
    logger.info(f'Best epoch: {best_epoch} Best bleu: {best_bleu}')

def train_t5(config, logger):
    
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    
    sentences = load_sentences(config, logger)
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config, logger)
    count_vectorizer, tfidf_vectorizer = load_vectorizers(sentences, config, logger)
    contexts, curr_sents, next_sents = load_tuples(config, logger)
    
    tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrained_model)
    model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrained_model)
    
    t5_input = {}
    t5_target = {}
    
    with open(config.t5_output_path[config.dataset]['valid']) as f_obj:
        valid_output = json.load(f_obj)
    
    with open(config.t5_output_path[config.dataset]['test']) as f_obj:
        test_output = json.load(f_obj)
    
    t5_input['train'] = []
    t5_target['train'] = []
    for index in tqdm(range(len(curr_sents['train']))):
        if config.t5_use_keywords == True:

            response_vec = load_bow_batch([next_sents['train'][index]], [0], count_vectorizer, tfidf_vectorizer, config, logger, -1)
            if response_vec == None:
                continue
                
            input_seq = f"generate response query: {curr_sents['train'][index]}" +  ' keywords: ' + ', '.join(one_hot_to_bow(response_vec, count_vectorizer)[0]) + ' </s>'
        else:
            input_seq = f"generate response query: {curr_sents['train'][index]} </s>"
            
        target_seq = next_sents['train'][index] + ' </s>'
        
        if len(tokenizer.encode(input_seq)) > config.t5_max_seq_len:
            continue
        
        logger.info(f'input_seq: {input_seq}')
        logger.info(f'target_seq: {target_seq}')
        
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
        encoded_tar_seq = tokenizer.encode(target_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                    
        t5_input['train'].append(encoded_in_seq)
        t5_target['train'].append(encoded_tar_seq)

    logger.info(f'Max training seq length: {np.max(lengths)}')
        
    t5_input['valid'] = []
    t5_target['valid'] = []
    for o_dict in tqdm(valid_output):
        if config.t5_use_keywords == True:
            for kw_list in o_dict['predicted_sent_bow']:
                input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ', '.join(kw_list)
                
                target_seq = o_dict['next sentence']
                
                logger.info(f'input_seq: {input_seq}')
                
                encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
                t5_input['valid'].append(encoded_in_seq)
                t5_target['valid'].append(target_seq)

            #input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ' '.join(o_dict['predicted_sent_bow']) + ' </s>'
            #lengths.append(len(tokenizer.encode(input_seq)))
            #target_seq = o_dict['next sentence']
            
           
                            
            #encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
            #t5_input['valid'].append(encoded_in_seq)
            #t5_target['valid'].append(target_seq)
        else:
            input_seq = f"generate response query: {' '.join(o_dict['current sentence'])}"
            
            logger.info(f'input_seq: {input_seq}')
            
            lengths.append(len(tokenizer.encode(input_seq)))
            
            target_seq = o_dict['next sentence']
        
            encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
            t5_input['valid'].append(encoded_in_seq)
            t5_target['valid'].append(target_seq)

    logger.info(f'Max valid seq length: {np.max(lengths)}')

    t5_input['test'] = []
    t5_target['test'] = []
    for o_dict in tqdm(test_output):
        if config.t5_use_keywords == True:
            for kw_list in o_dict['predicted_sent_bow']:
                input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ', '.join(kw_list)
                
                target_seq = o_dict['next sentence']
                
                logger.info(f'input_seq: {input_seq}')
                
                encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
                t5_input['test'].append(encoded_in_seq)
                t5_target['test'].append(target_seq)
            
            #input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ' '.join(o_dict['predicted_sent_bow']) + ' </s>'
            #lengths.append(len(tokenizer.encode(input_seq)))
            #target_seq = o_dict['next sentence']
            
           
                            
            #encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
            #t5_input['test'].append(encoded_in_seq)
            #t5_target['test'].append(target_seq)
        else:
            input_seq = f"generate response query: {' '.join(o_dict['current sentence'])}"
            
            logger.info(f'input_seq: {input_seq}')
            
            target_seq = o_dict['next sentence']
            
            encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
            
            t5_input['test'].append(encoded_in_seq)
            t5_target['test'].append(target_seq)
       

    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        
        if part == 'train':
            datasets[part] = TensorDataset(torch.cat(t5_input[part]), torch.cat(t5_target[part]))
            dataloaders[part] = DataLoader(datasets[part], batch_size=config.t5_batch_size, shuffle=True, pin_memory=True)            
        else:
            datasets[part] = TensorDataset(torch.cat(t5_input[part]))
            if config.t5_use_keywords == True:
                dataloaders[part] = DataLoader(datasets[part], batch_size=config.t5_batch_size, shuffle=False, pin_memory=True)
            else:
                dataloaders[part] = DataLoader(datasets[part], batch_size=config.t5_batch_size_evaluation, shuffle=False, pin_memory=True)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.t5_lr)
    
    model.cuda()
    
    best_bleu = 0.0
    for epoch in range(1, config.num_epochs):
        model.train()
        train_loss = []
        for batch_num, batch in enumerate(dataloaders['train']):
            input_ids = batch[0].cuda()
            lm_labels = batch[1].cuda()
            
            outputs = model(input_ids=input_ids, lm_labels=lm_labels)
            
            loss = outputs[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            if (batch_num+1) % config.print_every == 0:
                logger.info(f"Batch #{batch_num+1}/{len(dataloaders['train'])} Loss: {loss.item():.2f}")
            
        logger.info(f'Epoch: {epoch} Train loss: {np.mean(train_loss):.2f}')
    
        logger.info('Performing validation.')
        model.eval()
        valid_responses = []
        for batch_num, batch in enumerate(dataloaders['valid']):
            input_ids = batch[0].cuda()
            
            if config.t5_use_keywords == True:
                outputs = model.generate(input_ids)
                #outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
            else:
                outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
                #outputs = model.generate(input_ids)
                
            gen_responses = []
            for index in range(len(outputs)):
                gen_responses.append(tokenizer.decode(outputs[index]))
                #logger.info(tokenizer.decode(outputs[index]))
            
            valid_responses.extend(gen_responses)
            
        
        valid_gen_output = []
        for index, o_dict in enumerate(valid_output):
            o_dict['generated_sentences'] = valid_responses[(index*config.t5_samples):((index*config.t5_samples)+config.t5_samples)]
            valid_gen_output.append(o_dict)

        
        val_avg_recall, val_avg_precision = compute_bleu(valid_gen_output)
        
        logger.info(f'Epoch: {epoch} Valid avg. recall bleu: {val_avg_recall:.2f} avg. precision bleu: {val_avg_precision:.2f}')
        
        with open(f'{config.save_path}/valid_output_{epoch}.json', 'w') as f_obj:
            json.dump(valid_gen_output, f_obj)
                
        with open(f'{config.save_path}/valid_output_{epoch}.txt', 'w') as f_obj:
            for o_dict in valid_gen_output:
                for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'current sentence']:
                    if key in ['context', 'current sentence']:
                        f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                    elif key == 'next sentence':
                        f_obj.write(f"{key}: {o_dict[key]}\n")
                    else:
                        f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                f_obj.write(f"Predicted Response BOW: {o_dict['predicted_sent_bow']}\n")
                #f_obj.write(f"Generated Response: {o_dict['generated_sentences'][0]}\n")
                for index, pred_response in enumerate(o_dict['generated_sentences']):
                    #f_obj.write(f"Predicted Response BOW #{index+1}: {o_dict['predicted_response_bow'][index]}\n")
                    f_obj.write(f'Generated Response #{index+1}: {pred_response}\n')
                f_obj.write('\n')
        
        test_responses = []
        for batch_num, batch in enumerate(dataloaders['test']):
            input_ids = batch[0].cuda()
            
            if config.t5_use_keywords == True:
                outputs = model.generate(input_ids)
                #outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
            else:
                outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
                #outputs = model.generate(input_ids)
            
            gen_responses = []
            for index in range(len(outputs)):
                gen_responses.append(tokenizer.decode(outputs[index]))
                #logger.info(tokenizer.decode(outputs[index]))
            
            test_responses.extend(gen_responses)
            
        test_gen_output = []
        for index, o_dict in enumerate(test_output):
            o_dict['generated_sentences'] = test_responses[(index*config.t5_samples):((index*config.t5_samples)+config.t5_samples)]
            test_gen_output.append(o_dict)
        
        
        test_avg_recall, test_avg_precision = compute_bleu(test_gen_output)
        
        logger.info(f'Epoch: {epoch} Test avg. recall bleu: {test_avg_recall:.2f} avg. precision bleu: {test_avg_precision:.2f}')
        
        with open(f'{config.save_path}/test_output_{epoch}.json', 'w') as f_obj:
            json.dump(test_gen_output, f_obj)
                
        with open(f'{config.save_path}/test_output_{epoch}.txt', 'w') as f_obj:
            for o_dict in valid_gen_output:
                for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'current sentence']:
                    if key in ['context', 'current sentence']:
                        f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                    elif key == 'next sentence':
                        f_obj.write(f"{key}: {o_dict[key]}\n")
                    else:
                        f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                f_obj.write(f"Predicted Response BOW: {o_dict['predicted_sent_bow']}\n")
                #f_obj.write(f"Generated Response: {o_dict['generated_sentences'][0]}\n")
                for index, pred_response in enumerate(o_dict['generated_sentences']):
                    f_obj.write(f'Generated Response #{index+1}: {pred_response}\n')
                    
                f_obj.write('\n')

        if best_bleu <= val_avg_recall:
            logger.info('Best validation recall bleu score so far. Saving the model.')
            save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
            torch.save(model, save_path)
                
            best_bleu = val_avg_recall
            best_epoch = epoch
        
    logger.info(f'Best epoch: {best_epoch} validation recall bleu: {best_bleu}')
    

def evaluate_t5(config, logger):

    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    
    sentences = load_sentences(config, logger)
    word2index, index2word, tokenizer = load_vocab_tokenizer(sentences, config, logger)
    count_vectorizer, tfidf_vectorizer = load_vectorizers(sentences, config, logger)
    contexts, curr_sents, next_sents = load_tuples(config, logger)
    
    tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrained_model)
    model = torch.load(config.finetuned_t5_path[config.dataset])
    
    t5_input = {}
    t5_target = {}
    
    with open(config.t5_output_path[config.dataset]['valid']) as f_obj:
        valid_output = json.load(f_obj)
    
    with open(config.t5_output_path[config.dataset]['test']) as f_obj:
        test_output = json.load(f_obj)
    
    t5_input['valid'] = []
    t5_target['valid'] = []
    for o_dict in tqdm(valid_output):
        input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ', '.join(o_dict['next_sent_bow'])
            
        target_seq = o_dict['next sentence']
        
        logger.info(f'input_seq: {input_seq}')
        
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len_evaluation, pad_to_max_length=True, return_tensors='pt')
        
        t5_input['valid'].append(encoded_in_seq)
        t5_target['valid'].append(target_seq)

        #input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ' '.join(o_dict['predicted_sent_bow']) + ' </s>'
        #lengths.append(len(tokenizer.encode(input_seq)))
        #target_seq = o_dict['next sentence']
        
        
                        
        #encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
            
        #t5_input['valid'].append(encoded_in_seq)
        #t5_target['valid'].append(target_seq)
 

 
    t5_input['test'] = []
    t5_target['test'] = []
    for o_dict in tqdm(test_output):
            
        input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ', '.join(o_dict['next_sent_bow'])
        
        target_seq = o_dict['next sentence']
        
        logger.info(f'input_seq: {input_seq}')
        
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len_evaluation, pad_to_max_length=True, return_tensors='pt')
        
        t5_input['test'].append(encoded_in_seq)
        t5_target['test'].append(target_seq)
        
        #input_seq = 'generate response query: ' +  ' '.join(o_dict['current sentence']) + ' keywords: ' + ' '.join(o_dict['predicted_sent_bow']) + ' </s>'
        #lengths.append(len(tokenizer.encode(input_seq)))
        #target_seq = o_dict['next sentence']
                            
        
       

    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        
        if part == 'train':
            continue
        else:
            datasets[part] = TensorDataset(torch.cat(t5_input[part]))
            dataloaders[part] = DataLoader(datasets[part], batch_size=config.t5_batch_size_evaluation, shuffle=False, pin_memory=True)
    
    
    model.cuda()
    
     
    
    logger.info('Performing validation.')
    model.eval()
    valid_responses = []
    for batch_num, batch in enumerate(dataloaders['valid']):
        input_ids = batch[0].cuda()
        
        #outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
        outputs = model.generate(input_ids)
            
        gen_responses = []
        for index in range(len(outputs)):
            gen_responses.append(tokenizer.decode(outputs[index]))
        
        valid_responses.extend(gen_responses)
        
    
    valid_gen_output = []
    for index, o_dict in enumerate(valid_output):
        #o_dict['generated_sentences'] = valid_responses[(index*config.t5_samples):((index*config.t5_samples)+config.t5_samples)]
        o_dict['generated_sentences'] = valid_responses[index:index+1]
        valid_gen_output.append(o_dict)

    
    valid_avg_recall, valid_avg_precision = compute_bleu(valid_gen_output)
    
    logger.info(f'Valid avg. recall bleu: {valid_avg_recall:.2f} avg. precision bleu: {valid_avg_precision:.2f}')
    
    with open(f'{config.save_path}/{config.evaluate_type}_valid_output.json', 'w') as f_obj:
        json.dump(valid_gen_output, f_obj)
            
    with open(f'{config.save_path}/{config.evaluate_type}_valid_output.txt', 'w') as f_obj:
        for o_dict in valid_gen_output:
            for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'current sentence']:
                if key in ['context', 'current sentence']:
                    f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                elif key == 'next sentence':
                    f_obj.write(f"{key}: {o_dict[key]}\n")
                else:
                    f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
            f_obj.write(f"Predicted Response BOW: {o_dict['predicted_sent_bow']}\n")
            #f_obj.write(f"Generated Response: {o_dict['generated_sentences'][0]}\n")
            for index, pred_response in enumerate(o_dict['generated_sentences']):
                #f_obj.write(f"Predicted Response BOW #{index+1}: {o_dict['predicted_response_bow'][index]}\n")
                f_obj.write(f'Generated Response #{index+1}: {pred_response}\n')
            f_obj.write('\n')
    
    test_responses = []
    for batch_num, batch in enumerate(dataloaders['test']):
        input_ids = batch[0].cuda()
        
        #outputs = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=config.t5_samples)
        outputs = model.generate(input_ids)
        
        gen_responses = []
        for index in range(len(outputs)):
            gen_responses.append(tokenizer.decode(outputs[index]))
        
        test_responses.extend(gen_responses)
        
    test_gen_output = []
    for index, o_dict in enumerate(test_output):
        #o_dict['generated_sentences'] = test_responses[(index*config.t5_samples):((index*config.t5_samples)+config.t5_samples)]
        o_dict['generated_sentences'] = valid_responses[index:index+1]
        test_gen_output.append(o_dict)
    
    
    test_avg_recall, test_avg_precision = compute_bleu(test_gen_output)
    
    logger.info(f'Test avg. recall bleu: {test_avg_recall:.2f} avg. precision bleu: {test_avg_precision:.2f}')
    
    with open(f'{config.save_path}/{config.evaluate_type}_test_output.json', 'w') as f_obj:
        json.dump(test_gen_output, f_obj)
            
    with open(f'{config.save_path}/{config.evaluate_type}_test_output.txt', 'w') as f_obj:
        for o_dict in valid_gen_output:
            for key in ['context', 'current sentence', 'next sentence', 'curr_sent_bow', 'next_sent_bow', 'current sentence']:
                if key in ['context', 'current sentence']:
                    f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                elif key == 'next sentence':
                    f_obj.write(f"{key}: {o_dict[key]}\n")
                else:
                    f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
            f_obj.write(f"Predicted Response BOW: {o_dict['predicted_sent_bow']}\n")
            #f_obj.write(f"Generated Response: {o_dict['generated_sentences'][0]}\n")
            for index, pred_response in enumerate(o_dict['generated_sentences']):
                f_obj.write(f'Generated Response #{index+1}: {pred_response}\n')
                
            f_obj.write('\n')

            

def train_t5_keywords(config, logger):
    
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
    
    context_seqs, context_seq_lengths, query_seqs, query_seq_lengths, response_seqs, response_seq_lengths, query_vectors,  response_vectors, vectorizer, index2word = load_gan_dataset(config, logger)
    
    tokenizer = T5Tokenizer.from_pretrained(config.t5_pretrained_model)
    model = T5ForConditionalGeneration.from_pretrained(config.t5_pretrained_model)
    
    t5_input = {}
    t5_target = {}
    
    with open(config.t5_train_output_path) as f_obj:
        train_output = json.load(f_obj)
        
    with open(config.t5_valid_output_path) as f_obj:
        valid_output = json.load(f_obj)
    
    with open(config.t5_test_output_path) as f_obj:
        test_output = json.load(f_obj)
    
    t5_input['train'] = []
    t5_target['train'] = []
    lengths = []
    for index in tqdm(range(len(query_seqs['train']))):
            
        input_seq = 'generate response keywords. query: ' +  ' '.join(indices2words(query_seqs['train'][index].tolist(), index2word)) + ' </s>'
            
        target_seq = ' ' + ' '.join(one_hot_to_bow(response_vectors['train'][index], vectorizer)[0]) + ' </s>'
        
        logger.info(f'training input_seq: {input_seq}')
        logger.info(f'training target_seq: {target_seq}')
        
        lengths.append(len(tokenizer.encode(input_seq)))
        lengths.append(len(tokenizer.encode(target_seq)))
        
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
        encoded_tar_seq = tokenizer.encode(target_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                    
        t5_input['train'].append(encoded_in_seq)
        t5_target['train'].append(encoded_tar_seq)

    logger.info(f'Max training seq length: {np.max(lengths)}')
        
    t5_input['valid'] = []
    lengths = []
    for o_dict in tqdm(valid_output):
        
        input_seq = 'generate response keywords. query: ' +  ' '.join(o_dict['query']) + ' </s>'
        
        logger.info(f'valid input_seq: {input_seq}')
        
        lengths.append(len(tokenizer.encode(input_seq)))
            
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
                
        t5_input['valid'].append(encoded_in_seq)

    logger.info(f'Max valid seq length: {np.max(lengths)}')

    t5_input['test'] = []
    lengths = []
    for o_dict in tqdm(test_output):

        input_seq = 'generate response keywords. query: ' +  ' '.join(o_dict['query']) + ' </s>'
        
        logger.info(f'test input_seq: {input_seq}')
            
        lengths.append(len(tokenizer.encode(input_seq)))
            
        encoded_in_seq = tokenizer.encode(input_seq, max_length=config.t5_max_seq_len, pad_to_max_length=True, return_tensors='pt')
            
        t5_input['test'].append(encoded_in_seq)
       
    
    logger.info(f'Max test seq length: {np.max(lengths)}')

    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        
        if part == 'train':
            datasets[part] = TensorDataset(torch.cat(t5_input[part]), torch.cat(t5_target[part]))
            dataloaders[part] = DataLoader(datasets[part], batch_size=config.t5_batch_size, shuffle=True, pin_memory=True)            
        else:
            datasets[part] = TensorDataset(torch.cat(t5_input[part]))
            if config.t5_use_keywords == True:
                dataloaders[part] = DataLoader(datasets[part], batch_size=10, shuffle=False, pin_memory=True)
            else:
                dataloaders[part] = DataLoader(datasets[part], batch_size=30, shuffle=False, pin_memory=True)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.t5_lr)
    
    model.cuda()
    
    best_bleu = 0.0
    for epoch in range(1, config.num_epochs):
        model.train()
        train_loss = []
        for batch_num, batch in enumerate(dataloaders['train']):
            break
            input_ids = batch[0].cuda()
            lm_labels = batch[1].cuda()
            
            outputs = model(input_ids=input_ids, lm_labels=lm_labels)
            
            loss = outputs[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            if (batch_num+1) % config.print_every == 0:
                logger.info(f"Batch #{batch_num+1}/{len(dataloaders['train'])} Loss: {loss.item():.2f}")
            
        logger.info(f'Epoch: {epoch} Train loss: {np.mean(train_loss):.2f}')
    
        logger.info('Performing validation.')
        model.eval()
        valid_response_kws = []
        for batch_num, batch in enumerate(dataloaders['valid']):
            input_ids = batch[0].cuda()
            
            outputs = model.generate(input_ids)
                            
            gen_response_kws = []
            for index in range(len(outputs)):
                gen_response_kws.append(tokenizer.decode(outputs[index]))
                #logger.info(tokenizer.decode(outputs[index]))
            
            valid_response_kws.extend(gen_response_kws)
            
        
        valid_gen_output = []
        for index, o_dict in enumerate(valid_output):
            o_dict['t5_predicted_response_bow'] = valid_response_kws[index].split()
            valid_gen_output.append(o_dict)

        
        valid_bleu, valid_recall, valid_precision = compute_keyword_metrics(valid_gen_output)
        print(valid_bleu)
        print(valid_recall)
        print(valid_precision)
        
        logger.info(f'Epoch: {epoch} Valid bleu: {valid_bleu:.2f}  recall: {valid_recall:.2f} avg. precision: {valid_precision:.2f}')
        
        with open(f'{config.save_path}/valid_output_{epoch}.json', 'w') as f_obj:
            json.dump(valid_gen_output, f_obj)
                
        with open(f'{config.save_path}/valid_output_{epoch}.txt', 'w') as f_obj:
            for o_dict in valid_gen_output:
                print(o_dict)
                for key in ['context', 'query', 'response', 'query_bow', 'response_bow', 'query']:
                    print(key)
                    if key in ['context', 'query', 'response']:
                        f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                    else:
                        f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                f_obj.write(f"T5 Predicted Response BOW: {o_dict['t5_predicted_response_bow']}\n")
                '''        
                for index, pred_response in enumerate(o_dict['generated_responses']):
                    f_obj.write(f"Predicted Response BOW #{index+1}: {o_dict['predicted_response_bow'][index]}\n")
                    f_obj.write(f'Response #{index+1}: {pred_response}\n')
                '''
                    
                f_obj.write('\n')
        
        test_response_kws = []
        for batch_num, batch in enumerate(dataloaders['test']):
            input_ids = batch[0].cuda()
            
            outputs = model.generate(input_ids)
                            
            gen_response_kws = []
            for index in range(len(outputs)):
                gen_response_kws.append(tokenizer.decode(outputs[index]))
                #logger.info(tokenizer.decode(outputs[index]))
            
            test_response_kws.extend(gen_response_kws)
            
        
        test_gen_output = []
        for index, o_dict in enumerate(test_output):
            o_dict['t5_predicted_response_bow'] = test_response_kws[index].split()
            test_gen_output.append(o_dict)

        
        test_bleu, test_recall, test_precision = compute_keyword_metrics(test_gen_output)
        
        logger.info(f'Epoch: {epoch} Test bleu: {test_bleu:.2f}  recall: {test_recall:.2f} avg. precision: {test_precision:.2f}')
        
        with open(f'{config.save_path}/test_output_{epoch}.json', 'w') as f_obj:
            json.dump(test_gen_output, f_obj)
                
        with open(f'{config.save_path}/test_output_{epoch}.txt', 'w') as f_obj:
            for o_dict in valid_gen_output:
                for key in ['context', 'query', 'response', 'query_bow', 'response_bow', 'query']:
                    if key in ['context', 'query', 'response']:
                        f_obj.write(f"{key}: {' '.join(o_dict[key])}\n")
                    else:
                        f_obj.write(f"{key}: {', '.join(o_dict[key])}\n")
                f_obj.write(f"T5 Predicted Response BOW: {o_dict['t5_predicted_response_bow']}\n")
                '''        
                for index, pred_response in enumerate(o_dict['generated_responses']):
                    f_obj.write(f"Predicted Response BOW #{index+1}: {o_dict['predicted_response_bow'][index]}\n")
                    f_obj.write(f'Response #{index+1}: {pred_response}\n')
                '''
                    
                f_obj.write('\n')
        sys.exit(0)
        '''
        if best_bleu <= valid_bleu:
            logger.info('Best validation recall bleu score so far. Saving the model.')
            save_path = os.path.join(config.save_path, f'best_model_{epoch}.pt')
            torch.save(model, save_path)
                
            best_bleu = val_avg_recall
            best_epoch = epoch
        '''
    #logger.info(f'Best epoch: {best_epoch} validation recall bleu: {best_bleu}')            
    
def train_classifier(config, logger):
    logger.info(f'Config: {vars(config)}')
    logger.info('Saving config file.')
        
    context_seqs, context_seq_lengths, query_seqs, query_seq_lengths, response_vectors, labels = load_classifier_dataset(config, logger)
    
    
    logger.info(f'Loading saved language model: {config.lm_path}')
    lm_model = load_model(config.lm_path)
    logger.info(f'Loading saved vae: {config.vae_path}')
    vae_model = load_model(config.vae_path)
    
    datasets = {}
    dataloaders = {}
    for part in config.datasets[config.dataset]['parts']:
        datasets[part] = TensorDataset(context_seqs[part], context_seq_lengths[part], query_seqs[part], query_seq_lengths[part], response_vectors[part], labels[part])
        
        if part == 'train':
            shuffle = True
        else:
            shuffle = False
            
        dataloaders[part] = DataLoader(datasets[part], batch_size=config.classifier_batch_size, shuffle=shuffle, pin_memory=True)
    
        
    model = Classifier(4*config.lm_hidden_size + config.latent_size, config.classifier_hidden_size)
    model.cuda()
    
    logger.info(model)
    
    optimizer = Adam(model.parameters(), lr=config.lm_lr)
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    logger.info('Begining training.')
    for epoch in range(1, config.num_epochs):
        model.train()
        vae_model.train()
        lm_model.train()
        with torch.set_grad_enabled(True):
            train_loss = []
            train_acc = []
            for batch_index, batch in enumerate(dataloaders['train']):
                context_x = batch[0].cuda()
                context_seq_lengths = batch[1].cuda()
                query_x = batch[2].cuda()
                query_seq_lengths = batch[3].cuda()
                response_vectors = batch[4].cuda()
                labels = torch.unsqueeze(batch[5].cuda(), dim=-1) 
                
                context_sorted_indices = torch.argsort(context_seq_lengths, descending=True)
                
                context_x = context_x[context_sorted_indices]
                context_seq_lengths = context_seq_lengths[context_sorted_indices]
                
                context_h_n, _ = lm_model(context_x, context_seq_lengths)
                
                context_reverse_indices = torch.argsort(context_sorted_indices)
                
                context_h_n = context_h_n[context_reverse_indices]
                
                query_sorted_indices = torch.argsort(query_seq_lengths, descending=True)
                
                query_x = query_x[query_sorted_indices]
                query_seq_lengths = query_seq_lengths[query_sorted_indices]
                
                query_h_n, _ = lm_model(query_x, query_seq_lengths)
                
                query_reverse_indices = torch.argsort(query_sorted_indices)
                
                query_h_n = query_h_n[query_reverse_indices]
                
                _, _, z = encode_vectors(response_vectors, vae_model, config.temperature)
                
                c_input = torch.cat((context_h_n.detach(), query_h_n.detach(), z.detach()), dim=-1)
                
                logits = model(c_input)
                
                loss = loss_fn(logits, labels)
                
                train_loss.extend(loss.detach().cpu().tolist())
                
                loss = torch.mean(loss)
                
                batch_acc = (torch.sigmoid(logits).round() == labels).float()
                train_acc.extend(batch_acc.detach().cpu().tolist())
                batch_acc = torch.mean(batch_acc)
                                
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (batch_index+1) % config.print_every == 0:
                    logger.info(f'Epoch: {epoch} Batch #{batch_index+1} Loss: {(loss).item():.2f} Accuracy: {(batch_acc).item():.2f}')
                    
            logger.info(f'Epoch: {epoch}, Training Loss: {np.mean(train_loss):.2f} Accuracy: {np.mean(train_acc):.2f}')
        
        logger.info('Performing validation.')
        model.eval()
        vae_model.eval()
        lm_model.eval()
        with torch.set_grad_enabled(False):
            valid_loss = []
            valid_acc = []
            for batch in dataloaders['valid']:
                context_x = batch[0].cuda()
                context_seq_lengths = batch[1].cuda()
                query_x = batch[2].cuda()
                query_seq_lengths = batch[3].cuda()
                response_vectors = batch[4].cuda()
                labels = torch.unsqueeze(batch[5].cuda(), dim=-1) 
                
                context_sorted_indices = torch.argsort(context_seq_lengths, descending=True)
                
                context_x = context_x[context_sorted_indices]
                context_seq_lengths = context_seq_lengths[context_sorted_indices]
                
                
                context_h_n, _ = lm_model(context_x, context_seq_lengths)
                
                context_reverse_indices = torch.argsort(context_sorted_indices)
                
                context_h_n = context_h_n[context_reverse_indices]
                
                query_sorted_indices = torch.argsort(query_seq_lengths, descending=True)
                
                query_x = query_x[query_sorted_indices]
                query_seq_lengths = query_seq_lengths[query_sorted_indices]
                
                query_h_n, _ = lm_model(query_x, query_seq_lengths)
                
                query_reverse_indices = torch.argsort(query_sorted_indices)
                
                query_h_n = query_h_n[query_reverse_indices]
                
                _, _, z = encode_vectors(response_vectors, vae_model, config.temperature)
                
                c_input = torch.cat((context_h_n.detach(), query_h_n.detach(), z.detach()), dim=-1)
                
                logits = model(c_input)
                
                loss = loss_fn(logits, labels)
                
                valid_loss.extend(loss.detach().cpu().tolist())
                
                batch_acc = (torch.sigmoid(logits).round() == labels).float()
                valid_acc.extend(batch_acc.detach().cpu().tolist())

            
            valid_loss = np.mean(valid_loss)
            valid_acc = np.mean(valid_acc)
            logger.info(f'Epoch: {epoch}, Validation Loss: {valid_loss:.2f} Accuracy: {valid_acc:.2f}')
            
            test_loss = []
            test_acc = []
            for batch in dataloaders['test']:
                context_x = batch[0].cuda()
                context_seq_lengths = batch[1].cuda()
                query_x = batch[2].cuda()
                query_seq_lengths = batch[3].cuda()
                response_vectors = batch[4].cuda()
                labels = torch.unsqueeze(batch[5].cuda(), dim=-1) 
                
                context_sorted_indices = torch.argsort(context_seq_lengths, descending=True)
                
                context_x = context_x[context_sorted_indices]
                context_seq_lengths = context_seq_lengths[context_sorted_indices]
                
                
                context_h_n, _ = lm_model(context_x, context_seq_lengths)
                
                context_reverse_indices = torch.argsort(context_sorted_indices)
                
                context_h_n = context_h_n[context_reverse_indices]
                
                query_sorted_indices = torch.argsort(query_seq_lengths, descending=True)
                
                query_x = query_x[query_sorted_indices]
                query_seq_lengths = query_seq_lengths[query_sorted_indices]
                
                query_h_n, _ = lm_model(query_x, query_seq_lengths)
                
                query_reverse_indices = torch.argsort(query_sorted_indices)
                
                query_h_n = query_h_n[query_reverse_indices]
                
                _, _, z = encode_vectors(response_vectors, vae_model, config.temperature)
                
                c_input = torch.cat((context_h_n.detach(), query_h_n.detach(), z.detach()), dim=-1)
                
                logits = model(c_input)
                
                loss = loss_fn(logits, labels)
                
                test_loss.extend(loss.detach().cpu().tolist())
                
                batch_acc = (torch.sigmoid(logits).round() == labels).float()
                test_acc.extend(batch_acc.detach().cpu().tolist())

            
            test_loss = np.mean(test_loss)
            test_acc = np.mean(test_acc)
            logger.info(f'Epoch: {epoch}, Test Loss: {test_loss:.2f} Accuracy: {test_acc:.2f}')    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, choices=['DailyDialog', 'SWDA', 'ROCStories', 'Taskmaster-2'], default='DailyDialog')
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--exp_path', type=str, default="experiments")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--direction', type=str, default="forward")
    parser.add_argument(dest='action', type=str, choices=['train', 'evaluate'])
        
    config = parser.parse_args(namespace=config)
    
    config.save_path = os.path.join(config.exp_path, config.exp_name)
    if os.path.isdir(config.save_path):
        print(f'Error! Experiment folder: {config.save_path} already exists!')
        sys.exit(1)
    else:
        print(f"Using experiment save path: {config.save_path}")
        os.makedirs(config.save_path)
    
    log_file = os.path.join(config.save_path, 'process.log')
    open(log_file, 'a').close()
    
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s %(name)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
        
    logger = logging.getLogger(f"{config.model} : {config.action.capitalize()} ->")
    
    writer = SummaryWriter(os.path.join(config.save_path, 'runs'))
    
    if config.model == 'lm' and config.action == 'train':
        train_lm(config, logger)
    elif config.model == 'bow_vae' and config.action == 'train':   
        train_bow_vae(config, logger, writer)
    elif config.model == 'sentence_vae' and config.action == 'train':
        train_sentence_vae(config, logger, writer)
    elif config.model == 'cross_distillation' and config.action == 'train':
        train_cross_distillation(config, logger, writer)
    elif config.model == 'classifier' and config.action == 'train':
        train_classifier(config, logger)
    elif config.model == 'gan' and config.action == 'train':
        train_gan(config, logger)
    elif config.model == 't5' and config.action == 'train':
        train_t5(config, logger)
    elif config.model == 't5' and config.action == 'evaluate':
        evaluate_t5(config, logger)
    elif config.model == 't5_keywords' and config.action == 'train':
        train_t5_keywords(config, logger)    
    else:
        logger.info(f'Error! Invalid model and action combination: {config.model}, {config.action}')
        sys.exit(1)
