import os

def rename_files_in_directory(directory, old_word, new_word):
    # 遍历指定目录中的文件
    for filename in os.listdir(directory):
        # 检查文件名中是否包含要替换的单词
        if old_word in filename:
            # 新的文件名，将旧单词替换为新单词
            new_filename = filename.replace(old_word, new_word)
            # 获取文件的完整路径
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')
        else:
            print(f'No change for: {filename}')

# 使用示例
directory_path = r'D:\时间序列\LTSF-Linear\logs\LongForecasting\univariate\Etth1'  # 替换为你的txt文件所在的目录
old_word = ('DLinear_maxpool')  # 替换为你要替换的单词
new_word = 'MLinear'  # 替换为新的单词

rename_files_in_directory(directory_path, old_word, new_word)
