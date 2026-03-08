import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR(r'D:\algorithms\RTDETR-mg\ultralytics\cfg\models\second-消融\消融2\LHICNet-r50.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'D:\algorithms\RTDETR-mg\dataset\bladeThird.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # resume='', # last.pt path
                patience=0, # 设置0代表不早停，设置30代表精度持续30epoch没有比之前最高的高就早停
                project='runs/train',
                name='LHICNet-r50',
                # python train.py > logs/baseline.txt 2>&1
                # nohup python train.py > logs/baseline.txt 2>&1 &
                # nohup python train.py > logs/baseline.txt 2>&1 & tail -f baseline.txt
                )