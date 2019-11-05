
IF EXIST "backup/yolov3-tiny_last.weights" (
  darknet.exe detector train data/obj.data cfg/yolov3-tiny.cfg backup/yolov3-tiny_last.weights yolov3-tiny.conv.15 -dont_show
) ELSE (
  darknet.exe detector train data/obj.data cfg/yolov3-tiny.cfg yolov3-tiny.conv.15 -dont_show 
)

pause