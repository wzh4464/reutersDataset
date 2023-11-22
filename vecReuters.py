'''
File: /vecReuters.py
Created Date: Thursday November 16th 2023
Author: Zihan
-----
Last Modified: Wednesday, 22nd November 2023 3:39:22 pm
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


def vectorize_and_save(examples, tokenizer, model, resultPath, firstBool, lastBool) -> None:
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

    # if isinstance(examples['text'], pyarrow.lib.ChunkedArray):
    #     examples['text'] = examples['text'].to_pylist()  # 或.to_numpy()

    # if isinstance(examples['new_id'], pyarrow.lib.ChunkedArray):
    #     examples['new_id'] = examples['new_id'].to_pylist()  # 或.to_numpy()

    # 对文本进行向量化
    inputs = tokenizer(
        examples['text'], padding='max_length', truncation=True, return_tensors="pt")
    # 把每个tensor单独转移到GPU上
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():  # 不计算梯度
        # 使用模型进行预测
        outputs = model(**inputs)

    if firstBool:
        save_layer(examples, resultPath, outputs, 1)

    if lastBool:
        save_layer(examples, resultPath, outputs, -1)


def save_layer(examples, resultPath, outputs, layer_num):
    layer = outputs.hidden_states[layer_num].cpu().numpy()  # 第一层的输出

    # reshape to (batch_size, seq_len * hidden_size)
    layer = layer.reshape(layer.shape[0], -1)

    print('layer.shape:', layer.shape)

    # concat with new_id (new_id, layer_features)
    new_id = np.array(examples['new_id']).reshape(-1, 1)

    layer = np.concatenate((new_id, layer), axis=1)

    print('layer.shape:', layer.shape)

    # check path
    import os
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
        print(f'created {resultPath}')
    else:
        print(f'{resultPath} already exists')

    batch_index = examples['new_id'][0].split('"')[1]
    # 保存每个批次的结果
    batch_layer_name = f'{resultPath}/batch_{batch_index}_first_layer.npy'

    np.savez_compressed(batch_layer_name, layer)

    print(f'batch {batch_index} saved at {batch_layer_name}')


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

    dataset.map(partial(vectorize_and_save, tokenizer=tokenizer, model=model, lastBool=True, firstBool=False,
                        resultPath=resultPath), batched=True, batch_size=batch_size)

    # merged_first_layer, merged_last_layer = merge_batches(
    #     resultPath, num_batches)
    # 保存合并后的结果
    # np.save(f'{resultPath}/merged_first_layer.npy', merged_first_layer)
    # np.save(f'{resultPath}/merged_last_layer.npy', merged_last_layer)

def unionToOne(path):
    '''read collection all files named *.npz and concat them to one npz in path
    
    return True or false
    '''
    resultMat = None
    
    import os
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npz'):
                print(file)
                mat = np.load(os.path.join(root, file))['arr_0']
                if resultMat is None:
                    resultMat = mat
                else:
                    resultMat = np.concatenate((resultMat, mat), axis=0)
                    
    if resultMat is None:
        return False
    
    np.savez_compressed(os.path.join(path, 'result.npz'), resultMat)
    return True

if __name__ == '__main__':
    # genVecMain(batch_size=96,
    #            resultPath='/media/zihan/LinuxBackup/vecReutersResult')
    unionToOne('/media/zihan/LinuxBackup/vecReutersResult')
