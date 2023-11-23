'''
File: /union.py
Created Date: Wednesday November 22nd 2023
Author: Zihan
-----
Last Modified: Thursday, 23rd November 2023 1:26:55 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

def unionToOne(path):
    npzList = [os.path.join(root, file) for root, dirs, files in os.walk(
        path) for file in files if file.endswith('.npz')]
    if not npzList:
        print(f'No npz files found at {path}')
        return False

    print(f'{len(npzList)} npz files found at {path}')

    total_cpus = os.cpu_count()
    reserved_cpus = 4  # 保留的 CPU 核心数
    pool_size = max(1, total_cpus - reserved_cpus)

    while len(npzList) > 1:
        with Pool(pool_size) as p:
            results = []
            for _ in tqdm(range(len(npzList) // 2), desc="Merging files"):
                file1 = npzList.pop(0)
                file2 = npzList.pop(0)
                result_path = os.path.join(path, f'{len(npzList)}.npz')
                async_result = p.apply_async(concatTwoFiles, args=(file1, file2, result_path))
                results.append(async_result)

            # 获取所有异步任务的结果
            npzList.extend(async_result.get() for async_result in results)
    if not npzList:
        return False
    final_file = npzList[0]
    os.rename(final_file, os.path.join(path, 'merged_result.npz'))
    return True

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
    print(f"Concatenation of {file1} and {file2} to {resultPath} finished")
    return resultPath


if __name__ == '__main__':
    unionToOne('/media/zihan/LinuxBackup/vecReutersResult/result_test')
