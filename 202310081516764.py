import sys, time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtGui import QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from bigdl.llm.langchain.llms import TransformersLLM


class BackgroundThread(QThread):
    message_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        # 在这里模拟生成回复的流程，可以替换为实际的回复生成逻辑
        # for i in range(5):
        #     time.sleep(1)  # 模拟生成每部分回复的延迟
        #     partial_message = f"这是回复的第 {i+1} 部分。\n"
        #     self.message_ready.emit(partial_message)

        self.message_ready.emit("")


class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.init_chat()

    def init_ui(self):
        self.setWindowTitle("Chatbot")
        self.setGeometry(100, 100, 800, 600)

        # 主窗口中的中心部分
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建垂直布局
        layout = QVBoxLayout()

        background_color = QColor(255, 255, 240)
        central_widget.setStyleSheet(f"background-color: {background_color.name()};")

        # 对话框区域
        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setFont(QFont("Arial", 12))  # 设置合适的字体和字体大小
        layout.addWidget(self.chat_box)

        # 输入框和发送按钮
        input_layout = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("在这里输入您的问题...")
        self.input_box.setFont(QFont("Arial", 12))  # 设置合适的字体和字体大小

        # 自定义输入框边框
        input_box_style = """
            QTextEdit {
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
        """
        self.input_box.setStyleSheet(input_box_style)

        input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet(
            "background-color: #4CAF50; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;")
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        layout.addSpacing(20)  # 增加垂直间距

        central_widget.setLayout(layout)

        # 初始化对话
        self.add_message("ChatGPT:", "您好！有什么我可以帮助您的？")
        self.add_message("Tip:", "Enter（或Alt+Enter）手动换行\tCtrl+Enter快捷发送")

    def init_chat(self):
        self.llm = TransformersLLM.from_model_id(
            model_id=r"D:\2023fudan\vicuna-7b-v1.5",
            model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
        )

        self.template = "中文回答.\
                      \nCurrent conversation:\n{history}\nHuman: {input}\nAI Asistant:"

        self.prompt = PromptTemplate(template=self.template, input_variables=["history", "input"])
        self.conversation_chain = ConversationChain(
            verbose=True,
            prompt=self.prompt,
            llm=self.llm,
            memory=ConversationBufferMemory(),
            llm_kwargs={"max_new_tokens": 128},
        )

    def send_message(self):
        user_message = self.input_box.toPlainText()
        if user_message:
            self.add_message("User:", user_message)

            # 在这里添加与问答模型的交互代码
            # 示例：answer = your_question_answering_model(user_message)
            # answer = "这里是模型的回答。"  # 替换为实际的答案

            # answer = self.llm_chain.run(user_message)
            # answer = self.conversation_chain.run(user_message)
            # # prompt = user_message
            # # VICUNA_PROMPT_TEMPLATE = "USER: {prompt}\nASSISTANT:"
            # # answer = self.llm(prompt=VICUNA_PROMPT_TEMPLATE.format(prompt=prompt), max_new_tokens=128)
            #
            # # print(type(answer))

            # self.conversation_history += f"Human: {user_message}\n"
            answer = self.conversation_chain.run( user_message)
            answer = str(answer).split("AI Asistant:")[-1]
            # self.conversation_history += f"AI Assistant: {answer}\n"
            self.add_message("ChatGPT:", answer)
            # self.add_message("ChatGPT:", "")
            # self.message_to_display = answer  # 设置要显示的消息
            # self.character_index = 0  # 重置字符索引
            # self.timer.start(50)  # 启动定时器，每50毫秒显示一个字符
            # 启动后台任务来生成回复

        # 清空输入框
        self.input_box.clear()
        self.input_box.setFocus()

    def add_message(self, sender, message):
        formatted_message = f"<b>{sender}</b>: {message}<br>"
        # current_text = self.chat_box.toHtml()
        # self.chat_box.setHtml(current_text + "<br>" + formatted_message)
        cursor = self.chat_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_box.setTextCursor(cursor)
        self.chat_box.insertHtml(formatted_message)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ControlModifier:
                self.send_message()
            elif event.modifiers() & Qt.AltModifier:
                cursor = self.input_box.textCursor()
                cursor.insertText('\n')
        else:
            super().keyPressEvent(event)


def run_app():
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
