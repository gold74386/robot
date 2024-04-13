import speech_recognition as sr

class OpenaiASRg:
    def __init__(self):
        self.r = sr.Recognizer()
# 从麦克风收集音频并写入文件
    def h_record(self, if_cmu=False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            print('请说出你的问题或需要的帮助')
            #edgetts = EdgeTTS()
            #asyncio.run(edgetts.text_to_speech_and_play("请说出你的问题或需要的帮助"))
            audio = self.r.listen(source, timeout=20, phrase_time_limit=6)

        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())
    

        

    # 从本地文件中加载音频 作为后续百度语音服务的输入
    

    def speech_to_text(self):
        self.h_record(if_cmu=False, rate=16000)
        with sr.AudioFile("./speech.wav") as source:
            audio = self.r.record(source)
        transcript = self.r.recognize_google(audio, language='yue-CN')
        #print(transcript)
        return transcript
if __name__ == '__main__':
    openaiasr = OpenaiASRg()
    print(openaiasr.speech_to_text())