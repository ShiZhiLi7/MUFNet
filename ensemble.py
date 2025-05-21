import argparse
import pickle
import numpy as np
from tqdm import tqdm

def ensemble(items,ds):


    num_class=60
    if 'xsub' in ds:
        with open(r'E:\data\HAR\multi_model_HAR\ntu60\xsub\val_label.pkl','rb') as f:
            sample_name, label = pickle.load(f)

    elif 'xview' in ds:
        with open(r'E:\data\HAR\multi_model_HAR\ntu60\xview\val_label.pkl','rb') as f:
            sample_name, label = pickle.load(f)



    ckpt_dirs, alphas = list(zip(*items))

    ckpts = []
    for ckpt_dir in ckpt_dirs:
        with open(ckpt_dir, 'rb') as f:
            ckpts.append(list(pickle.load(f).items()))

    right_num = total_num = right_num_5 = 0
    
    classnum = np.zeros(num_class)
    classacc = np.zeros(num_class)
    for i in tqdm(range(len(label))):
        l = label[i]
        r = np.zeros(num_class)
        for alpha, ckpt in zip(alphas, ckpts):
            _, r11 = ckpt[i]
            r += r11 * alpha

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
        
        classnum[int(l)] += 1
        classacc[int(l)] += int(r != int(l))
    
    classacc = 100 * classacc / classnum
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    





    alphas = [1.0, 1.0, 1.0, 1.0]


    # ckpts = [
    #     # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\joint\best_acc_result.pkl', alphas[0]],
    #     # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\bone\best_acc_result.pkl', alphas[1]],
    #
    #     [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\joint_bone_u\best_acc_result.pkl', alphas[1]],
    #     # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\j2d\best_acc_result.pkl', alphas[1]],
    #     [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\rgb_mmn_j2d\best_acc_result.pkl', alphas[3]],
    # ]
    ckpts = [
        # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\joint\best_acc_result.pkl', alphas[0]],
        # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\bone\best_acc_result.pkl', alphas[1]],

        [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\joint_bone_u\best_acc_result.pkl', alphas[1]],
        # [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\j2d\best_acc_result.pkl', alphas[1]],
        [r'E:\Desktop\MMN_TGC\NTU_60\xview\CTR-TCN-U\rgb_mmn_j2d\best_acc_result.pkl', alphas[3]],
    ]
    print('alphas:', alphas)
    ensemble(ckpts,'xview')

    # Top1
    # Acc: 89.3007 %