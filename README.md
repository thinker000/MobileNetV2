# MobileNetV2


### 重点内容
#### 加密算法的改进
- 密钥生成阶段 预生成r,h 
- 加密阶段 在加密阶段直接使用预生成的h进行计算，减少计算量
- 解密阶段 使用中国剩余定理简化模运算

#### 

### 每个服务器实验文件的作用与区别
- myserver_diff.py    未引入加密，引入差分更新
- myserver_diff_v2.py   引入模型参数加密，引入差分更新
- myserver_diff_v3.py   引入模型参数加密，引入差分更新，做了优化（只对有差异的参数进行加密，即模型参数量化后大于零的参数进行加密）


### 每个客户端实验文件的作用和区别
- myclients.py
- myclient_diff.py    返回模型参数的差分更新
- myclient_diff_v2.py  返回模型参数的差分更行，同时评估全局模型在本地数据集上的效果，来决定是否更新本地模型



