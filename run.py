import subprocess
#
# with open('README_1.md', 'r') as f:
#     # 逐行读取文件内容
#     for line in f:
#         # 检查每行是否以 'bash' 开头
#         if line.startswith('python'):
#             # 提取命令并执行
#             command = line
#             subprocess.run(command, shell=True)
with open('Finally.md', 'r') as f:
# with open('Com_Pho_CS.md', 'r') as f:
    # 逐行读取文件内容
    for line in f:
        # 检查每行是否以 'bash' 开头
        if line.startswith('python'):
            # 提取命令并执行
            command = line
            subprocess.run(command, shell=True)
