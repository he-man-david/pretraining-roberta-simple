import os
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import torch

device = torch.device("cpu")
print("Device >>> ", device)
token_dir = './KantaiBERT'

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Read the content from the files, ignoring or replacing invalid characters
file_contents = []
for path in paths:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            file_contents.append(file.read())
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Join the contents into a single string
text = "\n".join(file_contents)

# Initialize a Byte Pair Encoding tokenizer, which is more efficient than word wise tokenizer
tokenizer = ByteLevelBPETokenizer()

# We are essentially using BPE tokenizer to clean and add more tokens like start,end,padding... to improve training quality
tokenizer.train_from_iterator([text], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>", # start token
    "<pad>", #padding token
    "</s>", # end token
    "<unk>", # unknown token
    "<mask>", # mask token
])

# Same the byte pair mappings tokens into a dir
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model(token_dir)

# using pretrained transformer model RoBERTa with small 12 heads, 6 layers, 52k 
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
print(config)

tokenizer = RobertaTokenizer.from_pretrained("./KantaiBERT", max_length=512)

model = RobertaForMaskedLM(config=config)
print(model)
print(model.num_parameters())
# Move the model to the correct device
model.to(device)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./kant.txt",
    block_size=128,
)

# batch of masked collated samples from dataset, 15% will be masked for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
# at this point the data has been prepared, tokenized, and loaded.

training_args = TrainingArguments(
    output_dir="./KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=1, #can be increased
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    use_cpu=True, # need this for MPS device M1 Mac
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./KantaiBERT")

fill_mask = pipeline(
    "fill-mask",
    model="./KantaiBERT",
    tokenizer="./KantaiBERT"
)

fill_mask("Hello <mask>");