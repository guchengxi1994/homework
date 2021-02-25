'''
@lanhuage: python
@Descripttion: 
@version: beta
@Author: xiaoshuyui
@Date: 2020-06-11 16:20:27
@LastEditors: xiaoshuyui
@LastEditTime: 2020-06-12 11:24:54
'''
import sys
sys.path.append('..')
from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtGui import 
from PyQt5.QtWidgets import QLabel,QPushButton,QFileDialog,QWidget,QListView,QDialog
from PyQt5.QtCore import QStringListModel,QRect,pyqtSignal,Qt
# from processor.compare import P_P_S
from skimage.io import imread
import numpy as np
import os
import pyperclip 


class P_P_S(object):
    def __init__(self,p1:str,p2:str,s:float):
        self.p1 = p1
        self.p2 = p2
        self.s = s 
    
    def __str__(self):
        return self.p1 +">>>>>>>>>>"+self.p2+">>>>>>>>>>"+str(int(self.s))

    def __eq__(self, other):
        if not isinstance(other,P_P_S):
            return False
        elif self.p1 == other.p2 and self.p2 == other.p1:
            return True
        else:
            return self.p1 == other.p1 and self.p2 == other.p2 and self.s == self.s
    
    def __hash__(self):
        return hash(self.p1) + hash(self.p2) + hash(self.s)


class MyQLabel(QLabel):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self,QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()



class GetResults(QDialog):
    def __init__(self):
        super(GetResults,self).__init__()

        self.setFixedSize(1000,800)
        self.setWindowTitle('展示结果')

        self.label_img1 = MyQLabel(self)
        self.label_img1.setText("显示图片1")
        self.label_img1.setFixedSize(200, 300)
        self.label_img1.move(200, 100)

        self.label_img1_path = QLabel(self)
        self.label_img1_path.move(50,100)
        self.label_img1_path.setText("1")

        self.label_img1.setStyleSheet("QLabel{background:white;}"
                    "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                    )

        
        self.label_img2 = MyQLabel(self)
        self.label_img2.setText("显示图片2")
        self.label_img2.setFixedSize(200, 300)
        self.label_img2.move(560, 100)

        self.label_img1.clicked.connect(self._imgShow1)
        self.label_img2.clicked.connect(self._imgShow2)


        self.label_img2_path = QLabel(self)
        self.label_img2_path.move(410,100)
        self.label_img2_path.setText("2")


        self.label_sim = QLabel(self)
        self.label_sim.move(770,100)
        self.label_sim.setText("相似等级：")
        self.label_sim.setFixedWidth(300)

        self.label_img2.setStyleSheet("QLabel{background:white;}"
                    "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                    )

        btn = QPushButton(self)
        btn.setText("打开文件")
        btn.move(30, 30)
        btn.clicked.connect(self._openfile)

        self.slm=QStringListModel()
        self.listview=QListView(self)
        self.listview.move(30,450)
        self.listview.setFixedWidth(950)
        self.listview.setFixedHeight(300)

        # self.label_img1_path.setWordWrap(True)
        self.label_img1_path.setGeometry(QRect(50, 100, 100, 27*4))
        self.label_img1_path.setWordWrap(True)
        self.label_img1_path.setAlignment(QtCore.Qt.AlignTop)

        self.label_img2_path.setGeometry(QRect(410, 100, 100, 27*4))
        self.label_img2_path.setWordWrap(True)
        self.label_img2_path.setAlignment(QtCore.Qt.AlignTop)

        self.resultList = []

        self.listview.clicked.connect(self._listclick)

    
    def _imgShow1(self):
        print(self.label_img1_path.text())
        if os.path.exists(self.label_img1_path.text()):
            os.system('start  {}'.format(self.label_img1_path.text()))
            pyperclip.copy(self.label_img1_path.text().split(os.sep)[-1])

    def _imgShow2(self):
        print(self.label_img2_path.text())
        if os.path.exists(self.label_img2_path.text()):
            os.system('start  {}'.format(self.label_img2_path.text()))
            pyperclip.copy(self.label_img2_path.text().split(os.sep)[-1])

    
    def _openfile(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开文件", "", "*.txt;;*.log;;All Files(*)")

        # print(imgType)
        
        try:
            res = set()
            with open(imgName,'r',encoding='utf-8') as f:
                lines = f.readlines()

                # lines.replace('*')
                for i in lines :
                    if i.strip()!='':
                        # tmp = i.decode('utf-8')
                        tmp = i.replace('*','')

                        tmp = tmp.split('>>>>>>>>>>')
                        p1 = tmp[0]
                        p2 = tmp[1]
                        s = float(tmp[2])

                        res.add(str(P_P_S(p1,p2,s)))
            
            # print(list(res))
            self.resultList = list(res)
            self.slm.setStringList(self.resultList)
            row = self.slm.rowCount()


            self.slm.insertRow(row)

            self.listview.setModel(self.slm)

        except Exception as e :
            print(e)
    

    def _listclick(self,qModelIndex):
        # print(self.resultList[qModelIndex.row()])
        try:
            line = self.resultList[qModelIndex.row()]
            tmp = line.split('>>>>>>>>>>')
            p1 = tmp[0]
            p2 = tmp[1]
            s = tmp[2]

            self.label_img1_path.setText(p1)
            self.label_img2_path.setText(p2)
            # print(float(s[:6])*100)
            s = s
            # s = float(s)*100
            self.label_sim.setText('相似等级：'+str(s))
            # self.label_sim.setText('相似度：'+str(s)[:5]+'%')

            jpg1 = QtGui.QPixmap(p1)
            # jpg1 = imread(p1)
            jpg2 = QtGui.QPixmap(p2)
            # jpg2 = imread(p2)

            # jpg1 = self.img2pixmap(jpg1)
            # jpg2 = self.img2pixmap(jpg2)

            jpg1 = self.m_resize(self.label_img1.width(),self.label_img1.height(),jpg1)
            jpg2 = self.m_resize(self.label_img2.width(),self.label_img2.height(),jpg2)

            self.label_img1.setPixmap(jpg1)
            self.label_img2.setPixmap(jpg2)
        except Exception as e:
            # pass
            print(e)

    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def m_resize(self,w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片
        w, h = pil_image.width(), pil_image.height() # 获取图像的原始大小

        f1 = 1.0*w_box/w
        f2 = 1.0 * h_box / h

        factor = min([f1, f2])

        width = int(w * factor)

        height = int(h * factor)
        #return pil_image.resize(width, height)
        return pil_image.scaled(width, height)         

                

                





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = GetResults()
    my.show()
    sys.exit(app.exec_())