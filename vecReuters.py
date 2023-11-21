'''
File: /vecReuters.py
Created Date: Thursday November 16th 2023
Author: Zihan
-----
Last Modified: Tuesday, 21st November 2023 5:51:28 pm
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
import tqdm


def vectorize_and_save(examples, tokenizer, model, resultPath) -> None:
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
    import pyarrow  # 用于处理parquet文件

    if isinstance(examples['text'], pyarrow.lib.ChunkedArray):
        examples['text'] = examples['text'].to_pylist()  # 或.to_numpy()

    if isinstance(examples['new_id'], pyarrow.lib.ChunkedArray):
        examples['new_id'] = examples['new_id'].to_pylist()  # 或.to_numpy()

    # 对文本进行向量化
    inputs = tokenizer(
        examples['text'], padding='max_length', truncation=True, return_tensors="pt")
    # 把每个tensor单独转移到GPU上
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():  # 不计算梯度
        # 使用模型进行预测
        outputs = model(**inputs)

    first_layer = outputs.hidden_states[1].cpu().numpy()  # 第一层的输出
    last_layer = outputs.hidden_states[-1].cpu().numpy()  # 最后一层的输出

    # reshape to (batch_size, seq_len * hidden_size)
    first_layer = first_layer.reshape(first_layer.shape[0], -1)
    last_layer = last_layer.reshape(last_layer.shape[0], -1)

    print('first_layer.shape:', first_layer.shape)

    # concat with new_id (new_id, layer_features)
    new_id = np.array(examples['new_id']).reshape(-1, 1)

    first_layer = np.concatenate((new_id, first_layer), axis=1)
    last_layer = np.concatenate((new_id, last_layer), axis=1)

    print('first_layer.shape:', first_layer.shape)

    # check path
    import os
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
        print(f'created {resultPath}')
    else:
        print(f'{resultPath} already exists')

    batch_index = examples['new_id'][0].split('"')[1]
    # 保存每个批次的结果
    batch_first_layer_name = f'{resultPath}/batch_{batch_index}_first_layer.npy'
    batch_last_layer_name = f'{resultPath}/batch_{batch_index}_last_layer.npy'
    np.save(batch_first_layer_name, first_layer)
    np.save(batch_last_layer_name, last_layer)

    print(f'batch {batch_index} saved at {batch_first_layer_name}')
    print(f'batch {batch_index} saved at {batch_last_layer_name}')

    # return {'first_layer': first_layer, 'last_layer': last_layer}


def genVecMain(batch_size=64, resultPath='/media/zihan/Ventoy/vecReutersResult'):
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

    dataset.map(partial(vectorize_and_save, tokenizer=tokenizer, model=model,
                        resultPath=resultPath), batched=True, batch_size=batch_size)

    # merged_first_layer, merged_last_layer = merge_batches(
    #     resultPath, num_batches)
    # 保存合并后的结果
    # np.save(f'{resultPath}/merged_first_layer.npy', merged_first_layer)
    # np.save(f'{resultPath}/merged_last_layer.npy', merged_last_layer)


def showVecMain():
    first_layer = np.load(
        "result2023-11-20-13-05-44/first_layer_features_gpu.npy")
    last_layer = np.load(
        "result2023-11-20-13-05-44/last_layer_features_gpu.npy")

    # (56, 512, 768) batch_size, seq_len, hidden_size
    print('first_layer.shape:', first_layer.shape)
    print('last_layer.shape:', last_layer.shape)

    # print('first_layer[0][0]:', first_layer[0][0])


def merge_batches(resultPath):
    # for each .npy file in resultPath, load it and concatenate it to merged_first_layer and merged_last_layer
    # first merge first layer
    
    import glob

    # 获取所有.npy文件的路径
    batch_first_layer_paths = glob.glob(f'{resultPath}/*_first_layer.npy')
    merge_batche_num = 30

    # for j in range(len(batch_first_layer_paths)//merge_batche_num):
    # # 用于保存合并后的结果
    #     saveBatch(resultPath, batch_first_layer_paths,
    #           merge_batche_num, j)

    # use tqdm to show progress
    for j in tqdm.tqdm(range(len(batch_first_layer_paths)//merge_batche_num)):
        # 用于保存合并后的结果
        saveBatch(resultPath, batch_first_layer_paths,
                  merge_batche_num, j)

def saveBatch(resultPath, batch_first_layer_paths, merge_batche_num, j):
    import os
    merge_batches = []
    tmp_npy = None
    for i in batch_first_layer_paths[j*merge_batche_num:(j+1)*merge_batche_num]:
        tmp_npy = np.concatenate(
            (tmp_npy, np.load(i)), axis=0) if tmp_npy is not None else np.load(i)
        # delete the merged batch
        os.remove(i)
        merge_batches.append(i)

    np.savez_compressed(f'{resultPath}/merged_first_layer_{j}.npz', tmp_npy)

    # print out the merged batches list
    with open(f'{resultPath}/merged_batches_{j}.txt', 'w') as f:
        for i in merge_batches:
            f.write(i+'\n')


if __name__ == '__main__':
    # genVecMain(batch_size=64)
    merge_batches('/media/zihan/Ventoy/vecReutersResult')
    # showVecMain()
