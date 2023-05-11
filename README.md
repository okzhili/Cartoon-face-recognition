# 简介
方法大体参考了re-ID的方法，原始代码[AICITY2020_DMT_VehicleReID](https://github.com/heshuting555/AICITY2020_DMT_VehicleReID)
使用resnest网络作为backbone, 使用triplet loss和softmax loss进行多任务训练，同时在为triplet loss组建三元组的时候，当anchor为真人的时候，neg和pos都强制设为卡通，当anchor为卡通的时候，neg和pos都设为真人。使用这种方式组建三元组计算triplet loss
最终得到初始相似度矩阵，并进行检索的rerank操作，得到最终矩阵。
最后进行三模型集成得到最终结果
# 环境要求
Tesla V100 32GB*1

pytorch 1.6

apex
# 数据
将train和test解压到images文件夹，并把两个txt文件也放进去
# 训练
```bash
python train.py --config_file=configs/resnest269_16.yml
```
```bash
python train.py --config_file=configs/esnest269_16_320.yml
```
```bash
python train.py --config_file=configs/resnest101_16.yml
```

# 推理
```bash
python test.py --config_file=configs/resnest269_16.yml
```
```bash
python test.py --config_file=configs/resnest269_16_320.yml
```
```bash
python test.py --config_file=configs/resnest101_16.yml
```
推理完成后后在submit文件夹里会生成单模的提交文件，在dis文件夹里保存了距离矩阵用于后面的多模型集成
# 集成
python ensemble.py
最后在根目录生成submit.csv提交文件
