Code for https://uwspace.uwaterloo.ca/handle/10012/16188

To run the code, place tab separated data files in data/{dataset}/processed/{train,valid,test}_{sentences,tuples}.tsv. Embedding files in embeddings/{glove,word2vec} folder. Provide appropriate arguments to the main file:

usage: python3 main.py [-h] [--device DEVICE]
               [--dataset {DailyDialog,SWDA,ROCStories,Taskmaster-2}]
               [--data_path DATA_PATH] [--exp_path EXP_PATH] --model MODEL
               --exp_name EXP_NAME [--direction DIRECTION]
               {train,evaluate}
