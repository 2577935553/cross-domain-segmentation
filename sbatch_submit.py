import subprocess

# 生成不同超参数的脚本内容
def generate_script_content(train_set,validation_set,pred_title,log_path,lr=0.0001,batch_size=8,crop=192,loss_edge='boundary',attn_features=16,time_h=72,epochs=100):
    script_content = f"""#!/bin/bash
#SBATCH --account=def-gawright                  # using graham's account
#SBATCH --gpus-per-node=v100l:1            	    # request a p100 GPU 
#SBATCH --mem=64G                  				# memory per node
#SBATCH --time=00-{time_h}:00                         # reserve a gpu for 15 hours.
#SBATCH --output=model_mine_run_all_singleMT.out               # log output

source /home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/zhsy/bin/activate

python model_mine_run_direct_adv_all_singleMT.py \
    --train_set /home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/{train_set} \\
    --validation_set /home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/{validation_set} \\
    --pred_title {pred_title} \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --lr {lr} \\
    --use_gpu True \\
    --crop_w {crop} \\
    --crop_h {crop} \\
    --log_path {log_path} \\
    --attn_features {attn_features} \\
    --augment True \\
    --loss_edge {loss_edge} &&\\
    
python calculate_metrics_bk.py \\
    --validation_set /home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/{validation_set} \\
    --data_path /home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/ \\
    --log_path {log_path} \\
    --pred_title {pred_title}
    """
    return script_content



# 生成并提交多个脚本

train_name_ls=['GE','Canon','Philips','Siemens','center_1','center_2','center_3','center_4','center_5','ARV','HCM','DCM','HHD','NOR']
# train_name_ls=['center_2','center_3','center_4','center_5','ARV','HCM','DCM','HHD','NOR']

train_set_ls=[f'test_txt_MMM/test_{path_train}' for path_train in train_name_ls]
validation_set='test_txt_MMM/MMM_labeled_subj'
pred_title_ls=[f'train_on_{path_train}' for path_train in train_name_ls]
log_path='./log_model_mine_all_singleMT'
epochs=300

for train_set,pred_title in zip(train_set_ls,pred_title_ls):
    script_content=generate_script_content(train_set,validation_set,pred_title,log_path,epochs=epochs)
    with open('script.sh','w') as f:
        f.write(script_content)
    subprocess.run(['sbatch','script.sh'])


# for i in range(1, num_scripts + 1):
#     script_content = generate_script_content()
#     with open(f"script{i}.sh", "w") as f:
#         f.write(script_content)
#     subprocess.run(["sbatch", f"script{i}.sh"])

