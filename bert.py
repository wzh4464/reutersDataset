'''
File: /main.py
Created Date: Tuesday November 14th 2023
Author: Zihan
-----
Last Modified: Tuesday, 14th November 2023 1:27:12 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Hi, I am Zihan."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
