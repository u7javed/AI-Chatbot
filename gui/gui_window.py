import sys
from PySide6.QtWidgets import (QMainWindow, QApplication, 
        QPushButton, QLineEdit, QMessageBox, QTextEdit)
from PySide6.QtGui import QFont, QIcon, QTextCursor, QColor
import utilities

import gui.chatbot

class Window(QMainWindow):

    def __init__(self, chatbot):
        super().__init__()
        self.title = 'AI ChatBot'
        self.left = 200
        self.top = 200
        self.width = 600
        self.height = 460
        self.chatbot = chatbot
        self.setup()
    
    def setup(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)
        self.setWindowIcon(QIcon("icon.png"))
        self.setStyleSheet("background: #21252B;")
        
        # Create label box
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        self.log.move(20, 20)
        
        self.log.setFixedSize(560, 300)
        
        self.log.setStyleSheet(
            """
            QScrollBar::handle {
                background-color: #282C34;
                border-radius:8px;
                border: 2px solid white;
                min-width:25px;
                min-height:25px;
            }
            """
        )
        
        
        font = QFont()
        font.setFamily("Courier")
        font.setPixelSize(16)
        
        self.log.moveCursor(QTextCursor.End)
        self.log.setCurrentFont(font)
    
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.returnPressed.connect(self.on_submit)
        self.textbox.setFont(font)
        self.textbox.move(20, 340)
        self.textbox.resize(560, 40)
        self.textbox.setStyleSheet(
            """
            color: white; 
            background: #282C34;
            padding-left: 10px;
            padding-right: 10px;
            """
        )
        
        # Create a button in the window
        self.button = QPushButton('Enter Text', self)
        self.button.setMinimumSize(120, 40)
        self.button.setStyleSheet(
            """
            color: white; 
            background: #282C34;
            padding-bottom: 5px;
            """
        )
        self.button.setFont(font)
        self.button.move(20,400)
        
        # connect button to function on_click
        self.button.clicked.connect(self.on_submit)
    
    def on_submit(self):
        text_value = self.textbox.text()
        if text_value:
            try:
                self.log.setTextColor(QColor(255, 255, 255))
                self.log.insertPlainText("You: " + text_value + "\n\n")
                self.textbox.clear()
                
                # preprocess text
                preprocessed_sentence = utilities.preprocess(text_value)
                
                # get chatbot response
                self.log.setTextColor(QColor(0, 200, 202))
                
                response = self.chatbot.response_to_input(preprocessed_sentence)
                self.log.insertPlainText("Bot: " + response + "\n\n")
            except KeyError:
                self.log.setTextColor(QColor(0, 200, 202))
                self.log.insertPlainText("Bot: Error! Unknown word encounter!\n\n")

def execute_gui(encoder, decoder, seq2seq, dictionary, device, max_length):
    
    chatbot = gui.chatbot.Chatbot(encoder, decoder, seq2seq, dictionary, device, max_length)
    
    app = QApplication(sys.argv)
    window = Window(chatbot)
    window.show()
    sys.exit(app.exec())