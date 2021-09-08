'''
Descripttion: 
version: 
Author: xiaoshuyui
email: guchengxi1994@qq.com
Date: 2021-09-08 16:54:52
LastEditors: xiaoshuyui
LastEditTime: 2021-09-08 17:25:30
'''
import sys

from PyQt5.QtWidgets import *

class Tabel(QWidget):
    def __init__(self, names:list) -> None:
        super().__init__()
        self.names = names
        self.cbList = []
        self.cbItemList = []
        self.getCbItems()
        self.initUI()
    
    def getCbItems(self):
        for i in range(len(self.names)):
            self.cbItemList.append("第{}张表".format(i+1))
    
    def initUI(self):
        layout = QVBoxLayout()
        tabelWidget = QTableWidget(len(self.names),2)
        tabelWidget.setHorizontalHeaderLabels(['项目名','分组'])
        for i in range(len(self.names)):
            tabelWidget.setItem(i,0,QTableWidgetItem(self.names[i]))
            cb = QComboBox(self)
            cb.addItems(self.cbItemList)
            tabelWidget.setCellWidget(i,1,cb)
        layout.addWidget(tabelWidget)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Tabel(['aaa','bbb','ccc'])
    win.show()
    sys.exit(app.exec_())
