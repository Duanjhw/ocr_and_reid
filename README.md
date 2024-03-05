这是CPU版本

## 第一步 安装依赖

```bash
pip install -r requirements.txt
```

## 第二步 安装yolov5(如果主机和github通信有困难的话）

```
git clone https://gitcode.com/mirrors/ultralytics/yolov5.git
```

修改round_detect/person_detect.py 中的troch hub.load 为：

```
yolov5 = torch.hub.load('/gitclone/path','custom',path='path/to/yolov5.pt',source='local')
```

## 运行game cut

```bash
cd ./Yolov5-Deepsort-Fastreid
python game_cut.py
```

## paddle paddle 安装

### 首先在Mac上安装paddlepaddle

```
python -m pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 接着安装paddle ocr

```
pip install "paddleocr>=2.0.1" # 推荐使用2.0.1+版本
```

### 验证安装是否成功

```
import paddle
paddle.utils.run_check()
```

### 运行round cut

```
 python /home/user/tongchonggang/Yolov5-Deepsort-Fastreid/round_cut.py
```
