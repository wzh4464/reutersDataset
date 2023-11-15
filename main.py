'''
File: /main.py
Created Date: Wednesday November 15th 2023
Author: Zihan
-----
Last Modified: Wednesday, 15th November 2023 11:33:05 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from datasets import load_dataset
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

dataset = load_dataset('reuters21578', 'ModHayes')

# vectorizer the dataset using the tokenizer
def vectorize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

dataVectorized = dataset.map(vectorize, batched=True)

print(dataVectorized['train'][0])

