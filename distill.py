import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\algorithms\RTDETR-mg\runs\prune\M-speedup=1.6-group_taylor -prune\weights\prune.pt',
        'data':r'D:\algorithms\RTDETR-mg\dataset\bladeThird.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 4,
        'workers': 2,
        'cache': False,
        'device': '0', # 指定显卡或者多卡参考正常训练的教程.
        'project':'runs/distill',
        # 'name':'M-原始best权重-mimic-0.08',
        'name':'M-原始best权重-mgd-0.02',

        # distill
        'prune_model': True, #设为True则使用剪枝后的pt文件作为学生模型
        'teacher_weights': r'D:\algorithms\RTDETR-mg\runs\Third\(rtdetr-CSP-PMSFA)+(rtdetr-HyperCompute-MFM)+(rtdetr-Pola-FMFFN)\weights\best.pt',
        'teacher_cfg': r'D:\algorithms\RTDETR-mg\ultralytics\cfg\models\second-消融\消融1\(rtdetr-CSP-PMSFA)+(rtdetr-HyperCompute-MFM)+(rtdetr-Pola-FMFFN).yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        'kd_loss_epoch': 1.0,#设为1.0则表示全部epoch都用上蒸馏
        
        'logical_loss_type': 'logical', # logical、mutillogical
        'logical_loss_ratio': 0.25,

        'teacher_kd_layers': '2,4,6,8,20,22,25,29,33,36,39',
        'student_kd_layers': '2,4,6,8,20,22,25,29,33,36,39',
        'feature_loss_type': 'chsim', # mimic、mgd、cwd、chsim、sp
        'feature_loss_ratio': 0.08
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()