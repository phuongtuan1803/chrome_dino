
REM darknet.exe detector train data/chrome_dino.data cfg/chrome_dino.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map
darknet.exe detector train data/obj.data cfg/chrome_dino.cfg darknet53.conv.74 -dont_show 
pause