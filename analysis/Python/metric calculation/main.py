import cv2
import os
from metrics import *
from tqdm import tqdm

eval_funcs = {
    "AG": ag,
    "CE": cross_entropy,
    "EI": edge_intensity,
    "EN": entropy,
    "MI": mutinf,
    "MSE": mse,
    "PSNR": psnr,
    "SD": sd,
    "SF": sf,
    "SSIM": ssim,
    "Qabf": qabf,
    "Qcb": qcb,
    "Qcv": qcv,
    "VIF": vif,
}


def main():
    path_fusimgs, path_irimgs, path_viimgs = [], [], []
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    fus_root = os.path.join(base_dir, "fusion_results_ycbcr")
    vi_root = os.path.join(base_dir, "dataset/val/visible")
    ir_root = os.path.join(base_dir, "dataset/val/infrared")
    metrics = eval_funcs.keys()

    print("Base directory:", base_dir)
    print("Fusion results directory:", fus_root)
    print("Visible images directory:", vi_root)
    print("Infrared images directory:", ir_root)

    # 获取可用的原始图像列表
    vi_files = {f: os.path.join(vi_root, f) for f in os.listdir(vi_root) if f.endswith('.png')}
    ir_files = {f: os.path.join(ir_root, f) for f in os.listdir(ir_root) if f.endswith('.png')}

    print("\nNumber of visible images:", len(vi_files))
    print("Number of infrared images:", len(ir_files))
    print("\nFirst few visible images:", list(vi_files.keys())[:5])
    print("First few infrared images:", list(ir_files.keys())[:5])

    img_list = os.listdir(fus_root)
    print("\nNumber of fusion results:", len(img_list))
    print("First few fusion results:", [img for img in img_list[:5] if img.endswith('.png')])

    for img in img_list:
        if not img.endswith('.png'):
            continue
            
        # 从融合结果文件名中提取原始图像名
        base_name = img.replace('fused_', '')
        
        # 在验证集中查找对应的图像
        found_vi = None
        found_ir = None
        
        # 提取融合结果的数字编号
        fused_num = int(base_name.split('.')[0])
        
        # 在验证集中查找最接近的图像
        min_diff = float('inf')
        for vi_name in vi_files:
            vi_num = int(vi_name.split('.')[0])
            diff = abs(vi_num - fused_num)
            if diff < min_diff:
                min_diff = diff
                found_vi = vi_name
                found_ir = found_vi  # 因为红外图像和可见光图像使用相同的编号
        
        if found_vi and found_ir and found_ir in ir_files:
            print(f"\nMatched {img} with:")
            print(f"Visible: {found_vi}")
            print(f"Infrared: {found_ir}")
            path_fusimgs.append(os.path.join(fus_root, img))
            path_viimgs.append(vi_files[found_vi])
            path_irimgs.append(ir_files[found_ir])

    if not path_fusimgs:
        print("\nNo matching image pairs found!")
        return

    if len(path_viimgs) != len(path_irimgs):
        print("\nThe number of vi_imgs and ir_imgs are different!")
        return

    print(f"\nFound {len(path_fusimgs)} matching image pairs.")

    res = {}
    for key in metrics:
        res[key] = [None] * len(path_fusimgs)

    pbar = iter(tqdm(range(len(path_fusimgs))))
    for i in range(len(path_fusimgs)):
        next(pbar)
        print("Now calculate the {}th img".format(i + 1))

        img_fus = cv2.imread(path_fusimgs[i], 0)
        img_vi = cv2.imread(path_viimgs[i], 0)
        img_ir = cv2.imread(path_irimgs[i], 0)

        if img_fus is None or img_vi is None or img_ir is None:
            print(f"Warning: Could not read one of the images:")
            print(f"Fusion: {path_fusimgs[i]}")
            print(f"Visible: {path_viimgs[i]}")
            print(f"Infrared: {path_irimgs[i]}")
            continue

        max_h = img_fus.shape[0]
        max_w = img_fus.shape[1]
        img_fus = cv2.resize(img_fus, (max_w, max_h))
        img_vi = cv2.resize(img_vi, (max_w, max_h))
        img_ir = cv2.resize(img_ir, (max_w, max_h))

        for metric in metrics:
            try:
                res[metric][i] = eval_funcs[metric](img_fus)
            except:
                res[metric][i] = eval_funcs[metric](img_fus, img_vi, img_ir)

    N = len(path_fusimgs)
    print("\nEvaluation Results:")
    print("-" * 30)
    for k, v in res.items():
        # 过滤掉None值
        valid_values = [x for x in v if x is not None]
        if valid_values:
            mean_value = sum(valid_values) / len(valid_values)
            print(f"{k}: {mean_value:.4f}")
        else:
            print(f"{k}: No valid results")
    print("-" * 30)


if __name__ == "__main__":
    main()
