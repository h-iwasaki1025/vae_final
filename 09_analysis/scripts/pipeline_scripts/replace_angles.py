import os

SCRIPT_DIR = "/Users/hikarui./Code/pdb_vae_project/06_Analysis/pipeline_scripts"

def replace_in_file(filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, 'r') as f:
        content = f.read()

    # パースのインデックスを変更 (元の y(3) を x(3), z(4) はそのまま)
    # 実際には変数名自体を y から x に変えれば、インデックスはそのままでも自然
    # ただし、元のスクリプトがどうなっているか
    #
    # 例： run_pipe_classifier.py
    # return int(parts[3]), int(parts[4])
    # 
    # 文字列の置換
    content = content.replace("Angle Y", "Angle X")
    content = content.replace("AngleY", "AngleX")
    content = content.replace("angleY", "angleX")
    content = content.replace("y_ang", "x_ang")
    content = content.replace("y_angle", "x_angle")
    content = content.replace("angle_y", "angle_x")
    
    with open(filepath, 'w') as f:
        f.write(content)

scripts = [
    "run_pipe_scatter.py",
    "run_pipe_individual.py",
    "run_pipe_classifier.py",
    "run_pipe_angle_maps.py",
]

for s in scripts:
    replace_in_file(s)
    print(f"Replaced in {s}")

