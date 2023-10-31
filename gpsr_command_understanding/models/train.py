import transformers
from datasets import load_dataset
import evaluate 
from transformers import Seq2SeqTrainer
# import wandb

# load our robot commands dataset 
dataset = load_dataset('ezishiri/robot_commands')

print(dataset)


# import nltk
# nltk.download('punkt')
import string
from transformers import AutoTokenizer

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)





# BELOW IS CODE TAKEN FROM https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

# NEED TO FIX AND CHANGE FOR MY USECASE (less data cleaning, only need to remove \n at some point, but for now should be fine


prefix = "translate command utterance to logical form: " # need prefix for good performance with t5 
max_input_length = 512
max_target_length = 512

# def clean_text(text):
#   sentences = nltk.sent_tokenize(text.strip())
#   sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
#   sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
#                                  if len(sent) > 0 and
#                                  sent[-1] in string.punctuation]
#   text_cleaned = "\n".join(sentences_cleaned_no_titles)
#   return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [text for text in examples["utterance"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["logical_form"], max_length=max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs




tokenized_datasets = dataset.map(preprocess_data,
                                                 batched=True)


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


batch_size = 8
model_name = "t5-base-robot-command-understanding"
model_dir = f"./{model_name}" # save the model in the models directory we are currently working in 

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=150,
    predict_with_generate=True,
    # fp16=True, # no fp16 for now while testing on my machine 
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
    push_to_hub=True,
    # report_to="wandb"  no reporting for now, need to set up personal wandb account 
)



data_collator = DataCollatorForSeq2Seq(tokenizer)


metric = evaluate.load("exact_match")


import numpy as np

def compute_metrics(eval_pred): # probably not going to work for exact match calculations 
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}




# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# trainer.train()
# trainer.push_to_hub









