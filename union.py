'''
File: /union.py
Created Date: Wednesday November 22nd 2023
Author: Zihan
-----
Last Modified: Thursday, 23rd November 2023 12:14:23 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import numpy as np
import os
from tqdm import tqdm

def unionToOne(path):
    npzList = [os.path.join(root, file) for root, dirs, files in os.walk(
        path) for file in files if file.endswith('.npz')]
    if not npzList:
        print(f'No npz files found at {path}')
        return False

    print(f'{len(npzList)} npz files found at {path}')

    # 顺序合并文件，并使用 tqdm 显示进度
    for _ in tqdm(range(len(npzList) // 2), desc="Merging files"):
        file1 = npzList.pop(0)
        file2 = npzList.pop(0)

        resultPath = os.path.join(path, f'{len(npzList)}.npz')
        if concatResult := concatTwoFiles(file1, file2, resultPath):
            npzList.append(concatResult)

    # 处理剩余文件
    if npzList:
        final_file = npzList[0]
        os.rename(final_file, os.path.join(path, 'merged_result.npz'))
        return True
    else:
        return False

def concatTwoFiles(file1, file2, resultPath):
    print(f"Concatenating {file1} and {file2} to {resultPath}")
    try:
        return concatHelper(file1, file2, resultPath)
    except Exception as e:
        print(f"Error during processing {file1} and {file2}: {e}")
        raise e


# TODO Rename this here and in `concatTwoFiles`
def concatHelper(file1, file2, resultPath):
    mat1 = np.load(file1)['arr_0']
    mat2 = np.load(file2)['arr_0']
    mat = np.concatenate((mat1, mat2), axis=0)
    np.savez_compressed(resultPath, arr_0=mat)

    # 合并成功后删除原始文件
    os.remove(file1)
    os.remove(file2)
    return resultPath


if __name__ == '__main__':
    unionToOne('/media/zihan/LinuxBackup/vecReutersResult/result_test')
