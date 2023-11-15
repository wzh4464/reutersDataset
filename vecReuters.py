'''
File: /vecReuters.py
Created Date: Thursday November 16th 2023
Author: Zihan
-----
Last Modified: Monday, 20th November 2023 7:30:43 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import torch
from torch.nn import DataParallel
from transformers import BertModel, BertTokenizer
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np
from functools import partial


def vectorize_and_save(examples, tokenizer, model, resultPath, firstLayerName, lastLayerName):
    """
    Vectorizes the given examples using the provided tokenizer and saves the resulting vectors to disk.

    Args:
        examples: The examples to be vectorized.
        tokenizer: The tokenizer object.
        model: The model object.
        resultPath: The path to save the resulting vectors.
        firstLayerName: The name of the file to save the vectors from the first layer.
        lastLayerName: The name of the file to save the vectors from the last layer.

    Returns:
        A dictionary containing the vectors from the first layer and the last layer.
    """
    # 对文本进行向量化
    inputs = tokenizer(
        examples['text'], padding='max_length', truncation=True, return_tensors="pt")
    # 把每个tensor单独转移到GPU上
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # print('inputs transformed to cuda')

    with torch.no_grad():
        # <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>
        outputs = model(**inputs)
        # print('outputs generated')

    # print('len(outputs.hidden_states):', len(outputs.hidden_states)) # tuple of 13 elements

    first_layer = outputs.hidden_states[1].cpu().numpy()
    last_layer = outputs.hidden_states[-1].cpu().numpy()

    # check path
    import os
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    np.save(f'{resultPath}/{firstLayerName}', first_layer)
    np.save(f'{resultPath}/{lastLayerName}', last_layer)

    return {'first_layer': first_layer, 'last_layer': last_layer}


def genVecMain():
    # 检测是否有多个GPU
    if torch.cuda.device_count() > 1:
        print(f"发现 {torch.cuda.device_count()} GPUs.")

    # 初始化模型和tokenizer
    model = BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 使用DataParallel将模型放到多个GPU上
    model = DataParallel(model).cuda()

    dataset = load_dataset('reuters21578', 'ModHayes')

    import time
    curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # 创建一个部分应用函数
    vectorize_and_save_part = partial(
        vectorize_and_save,
        tokenizer=tokenizer,
        model=model,
        resultPath=f'result{curTime}',
        firstLayerName='first_layer_features_gpu.npy',
        lastLayerName='last_layer_features_gpu.npy',
    )

    # 使用部分应用函数和 map
    dataset.map(vectorize_and_save_part, batched=True, batch_size=64)

def showVecMain():
    first_layer = np.load("../dataset/first_layer_features_gpu.npy")
    last_layer = np.load("../dataset/last_layer_features_gpu.npy")

    print('first_layer.shape:', first_layer.shape)  # (56, 512, 768)
    print('last_layer.shape:', last_layer.shape)

    print('first_layer[0][0]:', first_layer[0][0])


if __name__ == '__main__':
    genVecMain()
    # showVecMain()
