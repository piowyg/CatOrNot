from PyQt5 import QtGui, QtCore, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QHBoxLayout
import sys
from threading import Thread
import queue
import time
import os
from imageai.Detection import ObjectDetection

from PyQt5.QtGui import QPixmap, QPen, QFont


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "Is Cat Or Not"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.imagePathGen = ""
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()
        #ustawiamy przyciski w gui oraz pusty label ktory bedzie naszym obrazkiem oraz 2 pusty label ktory bedzie resultem czy jest kot
        self.btn1 = QPushButton("Open Image")
        self.btn1.clicked.connect(self.getImage)
        self.btn2 = QPushButton("Check Image")

        self.btn2.clicked.connect(self.checkImage)
        self.btn3 = QPushButton("Advantage Check Image")
        self.btn3.clicked.connect(self.advantageCheck)
        hbox = QHBoxLayout()
        vbox.addWidget(self.btn1)
        hbox.addWidget(self.btn2, alignment=QtCore.Qt.AlignRight)
        hbox.addWidget(self.btn3, alignment=QtCore.Qt.AlignLeft)
        vbox.addLayout(hbox)
        self.newLabel = QLabel("", alignment=QtCore.Qt.AlignCenter)
        font = QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(12)
        self.newLabel.setFont(font)
        self.newLabel.setStyleSheet("color: red;")
        vbox.addWidget(self.newLabel)
        self.label = QLabel("")
        vbox.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(vbox)

        self.show()

    def getImage(self):
        self.newLabel.setText("")
        fname = QFileDialog.getOpenFileName()
        imagePath = fname[0]
        #zapisujemy w klasie patha do zdjecia, aby isCat mogl z niej korzystac i pobrac odpowiednie zdjecie na calym kompie
        self.imagePathGen = imagePath
        # tutaj wyswietlamy obraz w labelu i zmieniami rozmiary zeby to ladnie wygladalo
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

    def checkImage(self):
        self.newLabel.setText("")
        start_time = time.time()
        que = queue.Queue()
        # otwieram liste 3 watkow, aby poszly wszystkie maszyny
        threads_list = list()
        # startuje watek z retina
        t2 = Thread(target=lambda q, arg1: q.put(self.isCat(arg1)), args=(que, 'Retina'))
        t2.start()
        threads_list.append(t2)

        for t in threads_list:
            t.join()

        checker = False
        # jesli ktores zwrocilo cata to checker jest true i wykona sie if else ktory pokazuje napisy
        while not que.empty():
            result = que.get()
            if (result == "cat"):
                checker = True
                break

        if (checker == True):
            print("Na zdjeciu jest kot")
            self.newLabel.setText("Na zdjeciu jest kot!")
        else:
            print("Na zdjeciu nie ma kota")
            self.newLabel.setText("Na zdjeciu nie ma kota\n wyprobuj 'Advantage Check Image' dla pewnosci")
        print("--- %s seconds ---" % (time.time() - start_time))

    def advantageCheck(self):
        self.newLabel.setText("")
        start_time = time.time()
        # otwieram liste 3 watkow, aby poszly wszystkie maszyny
        que = queue.Queue()
        threads_list = list()

        t = Thread(target=lambda q, arg1: q.put(self.isCat(arg1)), args=(que, 'YOLO'))
        t.start()
        threads_list.append(t)

        t3 = Thread(target=lambda q, arg1: q.put(self.isCat(arg1)), args=(que, 'tinyYOLO'))
        t3.start()
        threads_list.append(t3)

        for t in threads_list:
            t.join()

        checker = False
        # jesli ktores zwrocilo cata to checker jest true i wykona sie if else ktory pokazuje napisy
        while not que.empty():
            result = que.get()
            if (result == "cat"):
                checker = True
                break

        if (checker == True):
            print("Na zdjeciu jest kot")
            self.newLabel.setText("Na zdjeciu jest kot")
        else:
            print("Na zdjeciu nie ma kota")
            self.newLabel.setText("Na zdjeciu nie ma kota")
        print("--- %s seconds ---" % (time.time() - start_time))

    def isCat(self, name):
        execution_path = os.getcwd()  # os.getcwd() zwraca bieżący katalog roboczy
        detector = ObjectDetection()
        if (name == "YOLO"):
            detector.setModelTypeAsYOLOv3()
            detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
        elif (name == "Retina"):
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
        elif (name == "tinyYOLO"):
            detector.setModelTypeAsTinyYOLOv3()
            detector.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(
            input_image=os.path.join(execution_path, self.imagePathGen),
            output_image_path=os.path.join(execution_path,
                                           "new.jpg"))  # wczytuje obrazek, dla którego chcemy wykryć znajdujące się na nim obiekty, tworzę nowy obrazek z zaznaczonymi na nim wykrytymi obiektami
        for eachObject in detections:  # wyświetlam nazwy wykrytych obiektów oraz pewność ich wystąpienia (prawdopodobieńswto)
            # print(eachObject["name"], " : ", eachObject["percentage_probability"])
            if (eachObject["name"] == "cat" and eachObject["percentage_probability"] > 75):
                return 'cat'


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
