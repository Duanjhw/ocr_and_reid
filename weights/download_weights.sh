#!/bin/bash
# Download common models

python -c "
from round_utils.google_utils import *;
attempt_download('weights/yolov5s.pt');
attempt_download('weights/yolov5m.pt');
attempt_download('weights/yolov5l.pt');
attempt_download('weights/yolov5x.pt')
"
