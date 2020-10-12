#!/usr/bin/python3
import clientdemo.Conf as Conf
from clientdemo.DataModel import *
import clientdemo.HttpHelper as HttpHelper
import time

# 获取或设置本机IP地址信息
local_ip = '192.168.1.60'

ages = {}  # 老人字典


# 此函数需要你去完成，我只模拟实现要实现的功能项
def pose_detect_with_video(video_url, aged_id):
    use_aged = ages[aged_id]
    use_aged.timedown += 5
    use_aged.timelie += 5
    use_aged.timesit += 5
    use_aged.timestand += 5
    use_aged.timeother += 5
    use_aged.timein = '19:34:00'
    use_aged.status = PoseStatus.Lie.value
    # 报警条件判读
    if use_aged.status == PoseStatus.Down.value:
        use_aged.isalarm = True
    else:
        use_aged.isalarm = False


# 拼接url，参考接口文档
get_current_server_url = Conf.Urls.ServerInfoUrl + "/GetServerInfo?ip=" + local_ip
print(f'get {get_current_server_url}')

current_server = HttpHelper.get_items(get_current_server_url)
# print(current_server)

for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
    print(f'do_with camera video url：{camera.videoAddress}')

    for aged in camera.roomInfo.agesInfos:  # 遍历本摄像头所在房间的老人信息
        print(f'do_with aged name：{aged.name}')

        ages[aged.id] = PoseInfo(agesInfoId=aged.id, date=time.strftime('%Y-%m-%dT00:00:00', time.localtime()),
                                 timeStand=0, timeSit=0, timeLie=0, timeDown=0, timeOther=0)
        # 读取摄像头视频内容，进行pose识别，然后更新各种状态的时间值
        pose_detect_with_video(camera.videoAddress, aged.id)

        # 创建或更新PoseInfo数据库记录
        pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'
        HttpHelper.create_item(pose_url, ages[aged.id])
