class Config(object):
    
    def __init__(self):
       
        self.data_path = 'data'
        self.datasets = {
                'DailyDialog': {
                        'parts': ['train', 'valid', 'test'],
                        'sentences': {'train': 'processed/train_sentences.tsv', 'valid': 'processed/valid_sentences.tsv', 'test': 'processed/test_sentences.tsv'},
                        'tuples': {'train': 'processed/train_tuples.tsv', 'valid': 'processed/valid_tuples.tsv', 'test': 'processed/test_tuples.tsv'},
                    },
                'SWDA': {
                        'parts': ['train', 'valid', 'test'],
                        'sentences': {'train': 'processed/train_sentences.tsv', 'valid': 'processed/valid_sentences.tsv', 'test': 'processed/test_sentences.tsv'},
                        'tuples': {'train': 'processed/train_tuples.tsv', 'valid': 'processed/valid_tuples.tsv', 'test': 'processed/test_tuples.tsv'},
                    },
                'Taskmaster-2': {
                        'parts': ['train', 'valid', 'test'],
                        'sentences': {'train': 'processed/train_sentences.tsv', 'valid': 'processed/valid_sentences.tsv', 'test': 'processed/test_sentences.tsv'},
                        'tuples': {'train': 'processed/train_tuples.tsv', 'valid': 'processed/valid_tuples.tsv', 'test': 'processed/test_tuples.tsv'},
                    },
                'ROCStories': {
                        'parts': ['train', 'valid', 'test'],
                        'sentences': {'train': 'processed/train_sentences.tsv', 'valid': 'processed/valid_sentences.tsv', 'test': 'processed/test_sentences.tsv'},
                        'first_sents': {'train': 'processed/train_first_sents.tsv', 'valid': 'processed/valid_first_sents.tsv', 'test': 'processed/test_first_sents.tsv'},
                        'tuples': {'train': 'processed/train_tuples.tsv', 'valid': 'processed/valid_tuples.tsv', 'test': 'processed/test_tuples.tsv'},
                    }
                
            }
        
        # Special symbols to filter
        self.filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
        
        self.vae_vocab_size = -1
        self.vae_batch_size = 256
        self.vae_hidden_size = 256
        self.latent_size = 128
        self.vae_temperature = 1.0
        self.vae_lr = 1e-3
        self.vae_slope = 0.0025
        self.vae_margin = 2500
        self.vae_max_step = 3600
        self.sample_count = 6
        
        self.sentence_vae_vocab_size = -1
        self.sentence_vae_emb_dim = 300
        self.sentence_vae_emb_type = 'word2vec'
        self.sentence_vae_batch_size = 128
        self.sentence_vae_hidden_size = 256
        self.sentence_vae_latent_size = 128
        self.sentence_vae_encoder_layers = 3 
        self.sentence_vae_decoder_layers = 3
        self.sentence_vae_bidirectional = True
        self.sentence_vae_temperature = 1e-5
        self.sentence_vae_lr = 1e-3
        self.sentence_vae_slope = 0.0025
        self.sentence_vae_margin = 2500
        self.sentence_vae_max_step = 3000
        self.sentence_vae_max_seq_len = 30
        self.sentence_vae_dropout = 0.2
        self.sentence_vae_saved_model = 'experiments/sentence_vae/best_model_97.pt'
        self.sentence_vae_distillation_T = 10

        
        self.filter_stopwords = False
        
        self.num_epochs = 100
        self.print_every = 100
        self.save_every = 5
        
        self.vae_path = {
            'DailyDialog': 'experiments/S6_DailyDialog_BOW_VAE/model_35.pt',
            'SWDA': 'experiments/S6_SWDA_BOW_VAE/model_30.pt',
        }
            
        self.lm_path = {
            'DailyDialog': 'experiments/DailyDialog_LM/best_model_18.pt',
            'SWDA': 'experiments/SWDA_LM/best_model_20.pt',
        }
        
        
        self.gan_disc_hidden_size = 512
        self.gan_aux_disc_hidden_size = 768
        
        self.gan_gen_lr = 5e-4
        self.gan_dis_lr = 5e-5
        self.gan_batch_size = 128
        
        self.dist_lr = 1e-3
        
        self.mse_weight = 2.0
        self.rec_weight = 1.0
        self.temperature = 1e-5
        
        self.action = 'train'
        
        self.generator_path = 'saved/best_model_13.pt'
        
        self.latent_samples = 10
        
        # LM settings
        self.lm_max_seq_len = 128
        self.lm_vocab_size = -1
        self.lm_embedding_dim = 300
        self.lm_num_layers = 3
        self.lm_bidirectional = True
        self.lm_hidden_size = 64
        self.lm_embedding = 'word2vec'
        self.lm_lr = 1e-3
        self.lm_batch_size = 128
        
        
        self.classifier_batch_size = 128
        self.classifier_hidden_size = 384
        self.classifier_lr = 1e-3
        
        
        self.t5_pretrained_model = 't5-large'
        self.t5_output_path = {
            'DailyDialog': {
                    'valid': 'experiments/S6_DailyDialog_AC-GAN_EndtoEnd/valid_output_100.json',
                    'test': 'experiments/S6_DailyDialog_AC-GAN_EndtoEnd/test_output_100.json',
                },
            'SWDA': {
                    'valid': 'experiments/S6_SWDA_AC-GAN_EndtoEnd/valid_output_35.json',
                    'test': 'experiments/S6_SWDA_AC-GAN_EndtoEnd/test_output_35.json',
                },
            }
                
        self.t5_max_seq_len = 190
        self.t5_max_seq_len_evaluation = 390
        self.t5_batch_size = 12
        self.t5_batch_size_evaluation = 24
        self.t5_lr = 5e-5
        self.t5_use_keywords = True
        self.t5_samples = 10
        
        self.evaluate_type = 'ground_truth'
        
        self.finetuned_t5_path = {
            'DailyDialog': 'experiments/S6_T5_DailyDialog_w_Keywords/best_model_7.pt',
            'SWDA': 'experiments/S6_T5_SWDA_w_Keywords/best_model_4.pt'
        }
    
    
config = Config()
 
