import wfdb
import os

# 设置下载目录（可根据需要修改）
download_dir = os.path.join(r'd:\智愈派', 'data', 'mit-bih')
os.makedirs(download_dir, exist_ok=True)

# 精选高质量记录（避免重复和低质量数据）
selected_records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']

# 下载精选记录并跳过已存在文件
wfdb.dl_database(
    'mitdb',
    download_dir,
    records=selected_records,
    overwrite=False  # 跳过已下载文件
)