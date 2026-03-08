import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.compress import RTDETRCompressor, RTDETRFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = RTDETRCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = RTDETRFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\algorithms\RTDETR-mg\runs\Third\(rtdetr-CSP-PMSFA)+(rtdetr-HyperCompute-MFM)+(rtdetr-Pola-FMFFN)\weights\best.pt',
        'data':r'D:\algorithms\RTDETR-mg\dataset\bladeThird.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 4,
        'workers': 2,
        'cache': False,
        'device': '0',
        'project':'runs/prune',
        'name':'M-speedup=1.6-slim ',
        
        # prune
        'prune_method':'growing_reg',#l1,lamp，slim，group_slim，group_norm，group_sl，growing_reg，group_hessian，group_taylor
        'global_pruning': True, # 全局剪枝参数
        'speed_up': 1.6, # 剪枝前计算量/剪枝后计算量
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
        'iterative_steps': 50
    }
    
    # prune_model_path = compress(copy.deepcopy(param_dict))
    prune_model_path = r'D:\algorithms\RTDETR-mg\runs\prune\M-speedup=1.6-slim -prune2\weights\best.pt'
    finetune(copy.deepcopy(param_dict), prune_model_path)