import subprocess
import itertools

# 生成不同超参数的脚本内容
def generate_script_content(validation_set,pred_title,log_path,name_csv):
    script_content = f"""python calculate_metrics_bk.py --validation_set /home/fguo/projects/def-gawright/fguo/GuoLab_students/szhang/advchain-master/{validation_set} --data_path /home/fguo/projects/def-gawright/fguo/WHSeg/MMM/Converted/all_labeled/ --log_path {log_path} --pred_title {pred_title} --name_csv {name_csv}
    """
    return script_content



# 生成并提交多个脚本

# train_name_ls=['GE','Canon','Philips','Siemens','center_1','center_2','center_3','center_4','center_5','ARV','HCM','DCM','HHD','NOR']
# train_set_ls=[f'test_txt_MMM/test_{path_train}' for path_train in train_name_ls]
validation_name_ls=['GE','Canon','Philips','Siemens']
validation_set=[f'test_txt_MMM/test_{path_train}' for path_train in validation_name_ls]
pred_name_ls=['GE','Canon','Philips','Siemens']
pred_title_ls=[f'train_on_{path_train}' for path_train in pred_name_ls]
pred_validation_ls=list(itertools.product(pred_title_ls,validation_set))
names_tp_ls=list(itertools.product(pred_name_ls,validation_name_ls))
log_path='./log_model_mine_all_nattn'

for (pred_name,val_name),(pred_title,validation_set) in zip(names_tp_ls,pred_validation_ls):
    names=f'{pred_name}2{val_name}'
    print(names)
    script_content=generate_script_content(validation_set,pred_title,log_path,names)
    # with open('script.sh','w') as f:
    #     f.write(script_content)
    subprocess.run(script_content,shell=True)


# for i in range(1, num_scripts + 1):
#     script_content = generate_script_content()
#     with open(f"script{i}.sh", "w") as f:
#         f.write(script_content)
#     subprocess.run(["sbatch", f"script{i}.sh"])

