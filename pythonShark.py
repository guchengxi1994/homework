'''
lanhuage: python
Descripttion:
version: beta
Author: xiaoshuyui
Date: 2021-01-18 09:07:12
LastEditors: xiaoshuyui
LastEditTime: 2021-01-18 15:37:23
'''

import copy
import socket
import subprocess
import threading
import time

import psutil
import pyshark
import traceback
import os
import yaml
import logging.config

from pyshark.capture.capture import TSharkCrashException

from utils.propertiesUtils import getText

__connections__ = []
ts = []


class IpPortLength:
    def __init__(self) -> None:
        super().__init__()


def singleNetCardInfo(number: int, tshark_path=''):
    capture = pyshark.LiveCapture(interface=str(number),
                                  tshark_path=tshark_path)
    try:
        capture.sniff(timeout=3)
        if len(capture) > 0:
            return number
        else:
            return 0
    except Exception:
        # print(e)
        return 0
    finally:
        capture.close()


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()

    return ip


def getNetcardInfo(wiresharkPath: str):
    cmd = '"{}"'.format(wiresharkPath) + ' -D'
    print(cmd)
    res = subprocess.Popen(cmd,
                           shell=True,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)  # 使用管道
    result = res.stdout.read()
    res.wait()  # 等待命令执行完成
    res.stdout.close()  # 关闭标准输出
    logging.debug("当前网卡情况:"+result.decode('utf-8'))
    return result.decode('utf-8')

# cyclethreshhold = int(getText(path='..', sector='network', key='cyclethreshhold'))
# print(cyclethreshhold)
cyclethreshhold = 100000
localIp = get_host_ip()
def yehaiboCallBack(packet):
    try:
        if packet is not None:
            # 进行文件的记录，避免后续全面的线程拉起的进程阻塞（性能优化）
            with open("netcard.jerry", "a") as f1:
                huamn_readable_size = (os.stat("netcard.jerry").st_size) / 1024
                if (huamn_readable_size < 10):
                    f1.write(str(netCardNumber) + "\n")

            with open("socket.jerry", "a") as f2:
                f2.write(str(packet.length) + "\n")

            # # 如果打开软件后，已经监控很多轮了，那么就需要终止监控，不要总监控了
            with open("socket.jerry", "r") as f3:
                lines = f3.readlines()
                if (len(lines) > cyclethreshhold):
                    # 这个数值来控制数据包监控轮数
                    stopFlag = True
                    return

            protocol = packet.transport_layer
            src_addr = packet.ip.src
            dst_addr = packet.ip.dst
            src_port = packet[packet.transport_layer].srcport
            dst_port = packet[packet.transport_layer].dstport
            pkgLength = int(packet.length)

            ipl = IpPortLength()
            setattr(ipl, 'src_addr', src_addr)
            setattr(ipl, 'dst_addr', dst_addr)
            setattr(ipl, 'pkgLength', pkgLength)
            setattr(ipl, 'src_port', src_port)
            setattr(ipl, 'dst_port', dst_port)
            setattr(ipl, 'protocol', protocol)
            # (src_addr,src_port,dst_addr,dst_port,protocol,pkgLength)
            if dst_addr == localIp:
                tmp = list(i for i, x in enumerate(__connections__)
                           if x.src_addr == src_addr
                           and x.dst_addr == dst_addr)

                if len(tmp) == 0:
                    __connections__.append(ipl)
                else:
                    __connections__[
                        tmp[0]].pkgLength = __connections__[
                                                tmp[0]].pkgLength + int(pkgLength)
            logging.debug("成功抓取数据包一次!")
            # time.sleep(1)
        return
    except TSharkCrashException as e:
        logging.debug("in TSharkCrashException")
        return
    except AttributeError as e:
        logging.debug("in AttributeError")
        return
    except Exception as e:
        logging.debug("in Exception")
        traceback.print_exc()
        return

netCardNumber = 0
def getIpPortPairs(number: int,
                   tshark_path: str = '',
                   localIp='192.168.1.108'):
    netCardNumber = number
    capture = pyshark.LiveCapture(interface=str(netCardNumber), tshark_path=tshark_path)
    # yehaiboInterface = []
    # for iii in range(1, number + 1):
    #     yehaiboInterface.append(str(iii))
    # capture.interfaces = yehaiboInterface
    global __connections__
    logging.debug("in getIpPortPairs, 准备回调函数了")
    capture.apply_on_packets(yehaiboCallBack)


def port2pid2pname(port: int = 49677):
    cmd_port2pid = 'netstat -ano| findstr "{}"'.format(port)
    res = subprocess.Popen(cmd_port2pid,
                           shell=True,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)  # 使用管道
    result = res.stdout.read()
    res.wait()  # 等待命令执行完成
    res.stdout.close()  # 关闭标准输出
    tmp = result.decode('utf-8')

    pids = list()
    for i in tmp.split('\n'):
        if len(i) > 0 and i.strip() != '':
            pid = str(i).split(" ")
            pids.append(pid[-1])

    if len(pids) > 0:
        proc = psutil.Process(int(pids[0]))
        proc_name = proc.name()
        return proc_name
    else:
        return '未查询到进程名'


def grepNetcardInfo(thres: int = 25, timesleep: int = 10):
    while 1 == 1:
        time.sleep(timesleep)
        global __connections__
        tmp = copy.deepcopy(__connections__)
        __connections__.clear()
        for i in tmp:
            if i.pkgLength > thres:
                # ★★★★★★★★★★★★★★★★★★★★★抓到异常流量之后的处理（服务器远端就是需要这个异常流量的进程名| 或者立刻终止考试）★★★★★★★★★★★★★★★★★★
                pname = port2pid2pname(i.dst_port)
                checkstr = str(pname) + ':::::::::::::远端端口' + str(i.src_port) + '★★★★★★★本机端口:' + i.dst_port
                logging.debug(checkstr)
                with open("alert.jerry", "a") as f4:
                    f4.write(checkstr + "\n")
                # print("error,ip:{}, port:{} ".format(str(i.src_addr),
                #                                      str(i.src_port)))
            time.sleep(1)
        logging.debug("this round (conn length:) {}.".format(len(tmp)))
        # print("thres:" + str(thres))
        logging.debug('=' * 100)


def getInfo(lists: list,
            tshark_path: str = '',
            localIp='192.168.1.108'):
    # ts = []
    for i in lists:
        cardId = i
        t = threading.Thread(target=getIpPortPairs,
                             args=(cardId, tshark_path, localIp))
        ts.append(t)

    # 【★☆getText的路径，单独运行和打包的路径不一样 单独:path='..', ★☆】
    t1 = threading.Thread(target=grepNetcardInfo,
                          args=(int(getText(sector='network', key='socckerthreshhold')),))
    ts.append(t1)

    for i in ts:
        i.start()

def setup_logging(default_path="logging.yaml", default_level=logging.INFO, env_key="LOG_CFG"):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "r") as f:
            config = yaml.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

if __name__ == "__main__":
    setup_logging(default_path="../logging.yaml")
    localIp = get_host_ip()
    print(localIp)
    res = getNetcardInfo(wiresharkPath=getText(path='..', sector='network', key='wiresharkPath'))
    print(res)
    resList = res.split('\n')
    resList.remove('')
    print(len(resList))
    getInfo([i for i in range(1, 13)], getText(path='..', sector='network', key='wiresharkPath'), localIp)
