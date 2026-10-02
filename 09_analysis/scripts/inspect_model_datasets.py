import os
import glob
import re
import numpy as np

save_dir = '/Users/hikaru/Code/2026_Project/04_Savemodel/'
handson_dir = '/Users/hikaru/Code/2026_Project/02_HandsOn/'

models = [
    '20260816_01_z_noise16_bs128',
    '20260816_01_z_noise16_bs512',
    '20260816_z_noise16_bs128',
    '20260816_z_noise16_bs512',
    '20260817_z_noise10_bs128_beta100',
    '20260817pdb_multi_v2_bs128_beta100',
    '20260825_01_psi_noise_seed421',
    '20260825_psi_noise_seed420',
    '20260826_04_template_v2_per1_seed420',
    '20260826_04_template_v2_psi_seed420',
    '20260826_05_template_v2_per1_seed720',
    '20260826_05_template_v2_psi_seed421',
    '20260826_06_template_v2_rot90_seed720',
    '20260826_07_template_v2_rot90_seed420'
]

print("=== 14モデルのデータセット検証レポート ===")
for m in models:
    s_path = os.path.join(save_dir, m)
    data_info = "Unknown"
    
    kills_file = os.path.join(s_path, 'kills_test.npy')
    if os.path.exists(kills_file):
        try:
            kills = np.load(kills_file, allow_pickle=True)
            if len(kills) > 0:
                sample_p = str(kills[0])
                if 'pdb_PSI_noise' in sample_p:
                    data_info = "pdb_PSI_noise"
                elif 'PDBSidefix_rot90' in sample_p:
                    data_info = "PDBSidefix_rot90"
                elif 'PDBSidefix_per1' in sample_p:
                    data_info = "PDBSidefix_per1"
                elif 'PDBMulti' in sample_p:
                    data_info = "PDBMulti (or PDBMulti_v2)"
                else:
                    data_info = sample_p
        except Exception as e:
            data_info = f"Error reading kills: {e}"

    log_files = glob.glob(os.path.join(s_path, "*.log")) + glob.glob(os.path.join(s_path, "logs", "*.log"))
    for h in os.listdir(handson_dir):
        if m in h or h in m:
            log_files += glob.glob(os.path.join(handson_dir, h, "*.log"))
    
    found_log = ""
    for lf in log_files:
        try:
            with open(lf, 'r', errors='ignore') as f:
                content = f.read()
                m_data = re.search(r'/Users/hikaru/Code/2026_Project/05_Data/[a-zA-Z0-9_]+', content)
                if m_data:
                    found_log = m_data.group(0)
                    break
        except Exception:
            pass

    print(f"- {m} --> Data: {data_info} (Path: {found_log})")
