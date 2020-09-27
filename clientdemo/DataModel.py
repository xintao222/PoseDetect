#!/usr/bin/python3
from enum import Enum

'''此文件提供数据访问过程中使用到的数据模型'''


class AgesInfo:
    def __init__(self, id, name, roomInfoId, contacterName=None, contacterPhone=None, nurseName=None, address=None, roomInfo=None, poseInfos=None):
        self.id = id
        self.name = name
        self.contactername = contacterName
        self.contacterphone = contacterPhone
        self.nursename = nurseName
        self.address = address
        self.roominfoid = roomInfoId
        self.roominfo = roomInfo
        self.poseinfos = poseInfos


class CameraInfo:
    def __init__(self, id, ipAddress, videoAddress, serverInfoId, roomInfoId, serverInfo=None, factoryInfo=None, roomInfo=None):
        self.id = id
        self.factoryinfo = factoryInfo
        self.ipaddress = ipAddress
        self.videoaddress = videoAddress
        self.serverinfoid = serverInfoId
        self.serverinfo = serverInfo
        self.roominfoid = roomInfoId
        self.roominfo = roomInfo


class PoseInfo:
    def __init__(self, agesInfoId, date, timeStand=0, timeSit=0, timeLie=0, timeDown=0, timeOther=0, timeIn=None, isAlarm=False,
                 status=None, agesInfo=None,):
        self.agesinfoid = agesInfoId
        self.agesinfo = agesInfo
        self.date = date  # formart is "2020_08_05T00:00:00"
        self.timestand = timeStand
        self.timesit = timeSit
        self.timelie = timeLie
        self.timedown = timeDown
        self.timeother = timeOther
        self.timein = timeIn
        self.isalarm = isAlarm
        self.status = status


class PoseStatus(Enum):
    Stand = 0
    Sit = 1
    Lie = 2
    Down = 3
    Other = 4


class RoomInfo:
    def __init__(self, id, name, roomSize=0, agesInfos=None, cameraInfos=None):
        self.id = id
        self.name = name
        self.roomsize = roomSize
        self.agesinfos = agesInfos
        self.camerainfos = cameraInfos


class ServerInfo:
    def __init__(self, id, name, ip, factoryInfo=None, maxCameraCount=5, cameraInfos=None):
        self.id = id
        self.name = name
        self.factoryinfo = factoryInfo
        self.maxcameracount = maxCameraCount
        self.ip = ip
        self.camerainfos = cameraInfos
