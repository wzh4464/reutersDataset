'''
File: /union.py
Created Date: Wednesday November 22nd 2023
Author: Zihan
-----
Last Modified: Wednesday, 22nd November 2023 10:06:00 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import numpy as np
import os


def unionToOne(path):
    npzList = [os.path.join(root, file) for root, dirs, files in os.walk(
        path) for file in files if file.endswith('.npz')]
    if not npzList:
        print('No npz files found')
        return False

    print(f'{len(npzList)} npz files found')

    # 顺序合并文件
    while len(npzList) > 1:
        file1 = npzList.pop(0)  # 获取并移除列表中的第一个文件
        file2 = npzList.pop(0)  # 获取并移除列表中的现在的第一个文件（原来的第二个文件）

        resultPath = os.path.join(path, f'{len(npzList)}.npz')
        if concatResult := concatTwoFiles(file1, file2, resultPath):
            npzList.append(concatResult)  # 将结果文件添加到列表中

    # 最后的文件是合并后的结果
    if npzList:
        final_file = npzList[0]
        os.rename(final_file, os.path.join(path, 'merged_result.npz'))
        return True
    else:
        return False


def concatTwoFiles(file1, file2, resultPath):
    try:
        mat1 = np.load(file1)['arr_0']
        mat2 = np.load(file2)['arr_0']
        mat = np.concatenate((mat1, mat2), axis=0)
        np.savez_compressed(resultPath, arr_0=mat)

        # 合并成功后删除原始文件
        os.remove(file1)
        os.remove(file2)
        return resultPath
    except Exception as e:
        print(f"Error during processing {file1} and {file2}: {e}")
        raise e


if __name__ == '__main__':
    unionToOne('/media/zihan/LinuxBackup/vecReutersResult')
