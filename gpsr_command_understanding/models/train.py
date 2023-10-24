# finetune T5 using huggingface on commands dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")


#TODO: Setup the trainer to use the T5 fine tuner and code loss function myself (simple cross entropy loss)

# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 512 


# encode the inputs
task_prefix = "translate utterance to logical form: "
input_sequences = [] # input sequences from hf dataset we are loading 
output_sequences = []# output sequences from dataset

encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequences],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    [], # add list of output sequences 
    padding="longest",
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.item()