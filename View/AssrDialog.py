# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets


class Ui_AssrDialog(object):
    def setupUi(self, AssrDialog):
        AssrDialog.setObjectName("AssrDialog")
        AssrDialog.resize(520, 420)
        self.verticalLayout = QtWidgets.QVBoxLayout(AssrDialog)
        self.verticalLayout.setContentsMargins(12, 12, 12, 12)
        self.verticalLayout.setSpacing(10)

        self.label_hint = QtWidgets.QLabel(AssrDialog)
        self.label_hint.setWordWrap(True)
        self.label_hint.setObjectName("label_hint")
        self.verticalLayout.addWidget(self.label_hint)

        self.grid = QtWidgets.QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(8)

        self.label_stimulus = QtWidgets.QLabel(AssrDialog)
        self.comboBox_stimulus = QtWidgets.QComboBox(AssrDialog)
        self.comboBox_stimulus.addItem("am")
        self.comboBox_stimulus.addItem("click")
        self.grid.addWidget(self.label_stimulus, 0, 0, 1, 1)
        self.grid.addWidget(self.comboBox_stimulus, 0, 1, 1, 1)

        self.label_ear = QtWidgets.QLabel(AssrDialog)
        self.comboBox_ear = QtWidgets.QComboBox(AssrDialog)
        self.comboBox_ear.addItem("stereo")
        self.comboBox_ear.addItem("left")
        self.comboBox_ear.addItem("right")
        self.grid.addWidget(self.label_ear, 0, 2, 1, 1)
        self.grid.addWidget(self.comboBox_ear, 0, 3, 1, 1)

        self.label_carrier = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_carrier = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_carrier.setText("1000")
        self.grid.addWidget(self.label_carrier, 1, 0, 1, 1)
        self.grid.addWidget(self.lineEdit_carrier, 1, 1, 1, 1)

        self.label_modulation = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_modulation = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_modulation.setText("40")
        self.grid.addWidget(self.label_modulation, 1, 2, 1, 1)
        self.grid.addWidget(self.lineEdit_modulation, 1, 3, 1, 1)

        self.label_depth = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_depth = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_depth.setText("1.0")
        self.grid.addWidget(self.label_depth, 2, 0, 1, 1)
        self.grid.addWidget(self.lineEdit_depth, 2, 1, 1, 1)

        self.label_amplitude = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_amplitude = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_amplitude.setText("0.18")
        self.grid.addWidget(self.label_amplitude, 2, 2, 1, 1)
        self.grid.addWidget(self.lineEdit_amplitude, 2, 3, 1, 1)

        self.label_duration = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_duration = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_duration.setText("60")
        self.grid.addWidget(self.label_duration, 3, 0, 1, 1)
        self.grid.addWidget(self.lineEdit_duration, 3, 1, 1, 1)

        self.label_discard = QtWidgets.QLabel(AssrDialog)
        self.lineEdit_discard = QtWidgets.QLineEdit(AssrDialog)
        self.lineEdit_discard.setText("2")
        self.grid.addWidget(self.label_discard, 3, 2, 1, 1)
        self.grid.addWidget(self.lineEdit_discard, 3, 3, 1, 1)

        self.verticalLayout.addLayout(self.grid)

        self.label_status = QtWidgets.QLabel(AssrDialog)
        self.label_status.setObjectName("label_status")
        self.verticalLayout.addWidget(self.label_status)

        self.horizontalLayout_buttons = QtWidgets.QHBoxLayout()
        self.pushButton_start = QtWidgets.QPushButton(AssrDialog)
        self.pushButton_stop = QtWidgets.QPushButton(AssrDialog)
        self.pushButton_analyze = QtWidgets.QPushButton(AssrDialog)
        self.horizontalLayout_buttons.addWidget(self.pushButton_start)
        self.horizontalLayout_buttons.addWidget(self.pushButton_stop)
        self.horizontalLayout_buttons.addWidget(self.pushButton_analyze)
        self.verticalLayout.addLayout(self.horizontalLayout_buttons)

        spacer = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacer)

        self.retranslateUi(AssrDialog)
        QtCore.QMetaObject.connectSlotsByName(AssrDialog)

    def retranslateUi(self, AssrDialog):
        _translate = QtCore.QCoreApplication.translate
        AssrDialog.setWindowTitle(_translate("AssrDialog", "ASSR 听觉稳态反应"))
        self.label_hint.setText(
            _translate(
                "AssrDialog",
                "请先开始 EEG 采集，再播放刺激。默认 1000 Hz 载波 + 40 Hz 调幅。"
                "音量请从小幅度试起。结束后自动在调制频率处计算 SNR。",
            )
        )
        self.label_stimulus.setText(_translate("AssrDialog", "刺激类型"))
        self.comboBox_stimulus.setItemText(0, _translate("AssrDialog", "am"))
        self.comboBox_stimulus.setItemText(1, _translate("AssrDialog", "click"))
        self.label_ear.setText(_translate("AssrDialog", "声道"))
        self.comboBox_ear.setItemText(0, _translate("AssrDialog", "stereo"))
        self.comboBox_ear.setItemText(1, _translate("AssrDialog", "left"))
        self.comboBox_ear.setItemText(2, _translate("AssrDialog", "right"))
        self.label_carrier.setText(_translate("AssrDialog", "载波(Hz)"))
        self.label_modulation.setText(_translate("AssrDialog", "调制(Hz)"))
        self.label_depth.setText(_translate("AssrDialog", "调幅度 0-1"))
        self.label_amplitude.setText(_translate("AssrDialog", "音量 0-1"))
        self.label_duration.setText(_translate("AssrDialog", "时长(s)"))
        self.label_discard.setText(_translate("AssrDialog", "去起始(s)"))
        self.label_status.setText(_translate("AssrDialog", "未开始"))
        self.pushButton_start.setText(_translate("AssrDialog", "开始刺激"))
        self.pushButton_stop.setText(_translate("AssrDialog", "停止"))
        self.pushButton_analyze.setText(_translate("AssrDialog", "分析最近记录"))
