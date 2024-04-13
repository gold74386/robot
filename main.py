import time
import struct
import asyncio
import requests
import json
import hashlib
import random
import difflib
from langchain.memory import ConversationBufferMemory
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.agent import AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.conversational_chat.prompt import PREFIX
from langchain.agents import load_tools
import openai
import os
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
from edge_tts import Communicate
import pygame # 导入pygame，playsound报错或运行不稳定时直接使用
import asyncio
import opencc
import threading
from zhipuai import ZhipuAI
#from zhipi import ChatBot

PICOVOICE_API_KEY = "wY0VT8DVVUP4K21nZQRsDRGBBbhAy+66zu6SxOxw5WMbpKWT0Vwcxw=="  # 你的picovoice key
keyword_path = './Hey--Venus_en_windows_v2_1_0.ppn'  # 你的唤醒词检测离线文件地址
os.environ["WOLFRAM_ALPHA_APPI"] = "P578EU-3YH6JXT7UE"
os.environ["SERPER_API_KEY"] = "5481e88e323b8e28aa150f79f89916447ca5ca0f"
openai_api_key = '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'
MYPREFIX = PREFIX + "\n\n Remember Your name is Venus!,When you give the Final Answer,you MUST speak in Chinese"

PICOVOICE_API_KEY = "wY0VT8DVVUP4K21nZQRsDRGBBbhAy+66zu6SxOxw5WMbpKWT0Vwcxw=="  # 你的picovoice key
#keyword_path = './speechmodules/Hey--Venus_en_windows_v2_1_0.ppn'  # 你的唤醒词检测离线文件地址
openai_api_key = "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"

api_key = "eb96b3d8daba32c54cbffe8e350096ba.Ot4TZHhR33zfdBMR"  # 请填写您自己的APIKey

class PicoWakeWord:
    def __init__(self, PICOVOICE_API_KEY, keyword_path):
        self.PICOVOICE_API_KEY = PICOVOICE_API_KEY
        self.keyword_path = keyword_path
        self.porcupine = pvporcupine.create(
            access_key=self.PICOVOICE_API_KEY,
            keyword_paths=[self.keyword_path]
        )
        self.myaudio = pyaudio.PyAudio()
        self.stream = self.myaudio.open(
            input_device_index=0,
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def detect_wake_word(self):
        audio_obj = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
        audio_obj_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, audio_obj)
        keyword_idx = self.porcupine.process(audio_obj_unpacked)
        return keyword_idx

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
    
class EdgeTTS:
    def __init__(self, voice: str = "zh-HK-HiuGaaiNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume

    async def text_to_speech_and_play(self, text):
        
        communicate = Communicate(text, self.voice)
        await communicate.save('./audio.mp3')
        # playsound('./audio.wav') # playsound无法运行时删去此行改用pygame，若正常运行择一即可
        self.play_audio_with_pygame('audio.mp3')  # 注意pygame只能识别mp3格式
    def play_audio_with_pygame(self, audio_file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

class ChineseConverter:
    def __init__(self):
        self.converter = opencc.OpenCC('t2s.json')
        
    def convert_tw_to_cn(self, text):
        return self.converter.convert(text)
class GetTransLator:
    def __init__(self):
        self.name = "Get TransLate"
        self.description = "Useful for when you need to answer questions about translate."

    def run(self, text: str) -> str:
        from_lang = 'auto'
        to_lang = 'zh'
        api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
        app_id = '20220614001248307'
        app_key = 'd3kw204gkRpej7YyMM6o'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        salt = str(random.randint(32768, 65536))
        sign = app_id + text + salt + app_key
        sign = hashlib.md5(sign.encode()).hexdigest()
        data = {
            'q': text,
            'from': from_lang,
            'to': to_lang,
            'appid': app_id,
            'salt': salt,
            'sign': sign
        }

        response = requests.post(api_url, headers=headers, data=data)
        result = json.loads(response.text)
        if result == None:
            return "没有内容需要翻译"
        else:
            #return result['trans_result'][0]['dst']
            return ', '.join([item['dst'] for item in result['trans_result']])
class GetTransLatores:
    def __init__(self):
        self.name = "Get TransLate"
        self.description = "Useful for when you need to answer questions about translate."

    def run(self, text: str) -> str:
        from_lang = 'auto'
        to_lang = 'en'
        api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
        app_id = '20220614001248307'
        app_key = 'd3kw204gkRpej7YyMM6o'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        salt = str(random.randint(32768, 65536))
        sign = app_id + text + salt + app_key
        sign = hashlib.md5(sign.encode()).hexdigest()
        data = {
            'q': text,
            'from': from_lang,
            'to': to_lang,
            'appid': app_id,
            'salt': salt,
            'sign': sign
        }

        response = requests.post(api_url, headers=headers, data=data)
        result = json.loads(response.text)
        if result == None:
            return "没有内容需要翻译"
        else:
            #return result['trans_result'][0]['dst']
            return ', '.join([item['dst'] for item in result['trans_result']])
        
class ChatBot:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def chat(self, user_input):
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "我是人工智能助手"},
            ],
        )
        
        return ( response.choices[0].message.content)
class OpenaiAgentModule:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.tools = load_tools(['ddg-search','wikipedia','wolfram-alpha'])
        self.tool_names = [tool.name for tool in self.tools]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm=ChatOpenAI(temperature=0.2, openai_api_key=self.openai_api_key,model="gpt-3.5-turbo",)
        self.agent_cls = ConversationalChatAgent
        self.agent_obj = self.agent_cls.from_llm_and_tools(self.llm, self.tools, callback_manager=None, system_message=MYPREFIX)
        self.agent = AgentExecutor.from_agent_and_tools(agent=self.agent_obj, tools=self.tools, callback_manager=None, verbose=True, memory=self.memory)

    def chat_with_agent(self, text):
        openai.api_key = self.openai_api_key
        text = text.replace('\n', ' ').replace('\r', '').strip()
        if len(text) == 0:
            return
        print(f'chatGPT Q:{text}')
        reply = self.agent.run(input=text)
        return reply

def similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()
def run(picowakeword, asr, tts ,bot):
    while True:  # 需要始终保持对唤醒词的监听
        audio_obj = picowakeword.stream.read(picowakeword.porcupine.frame_length, exception_on_overflow=False)
        audio_obj_unpacked = struct.unpack_from("h" * picowakeword.porcupine.frame_length, audio_obj)
        keyword_idx = picowakeword.porcupine.process(audio_obj_unpacked)
        if keyword_idx >= 0:
            #picowakeword.porcupine.delete()
            picowakeword.stream.close()
            picowakeword.myaudio.terminate()  # 需要对取消对麦克风的占用!

            print("嗯,我在请讲！")
            asyncio.run(tts.text_to_speech_and_play("嗯,我在请讲！"))
            while True:  # 进入一次对话session
                
                l = asr.speech_to_text()
                converter = ChineseConverter()
                m = converter.convert_tw_to_cn(l)
                print('识别结果:'+m)
                asjk=GetTransLatores()
                w = asjk.run(m)
                q = w.rstrip("。").rstrip("？")
                
                print(f'recognize_from_microphone, text={q}')
                if "quit" in q:  # 检测到关键词“退出”时退出对话
                    break
                
                else:
                    #ras = openai_chat_module.chat_with_agent(q)
                    #bot = ChatBot(api_key)
                    user_input=q
                    ras = bot.chat(user_input)
                    translator = GetTransLator()
                    res = translator.run(ras)
                    print(res)
                    asyncio.run(tts.text_to_speech_and_play('嗯'+res))
                    
            #print('本轮对话结束')
            #asyncio.run(tts.text_to_speech_and_play('嗯'+'主人，我退下啦！'))
PICOVOICE_API_KEY = "wY0VT8DVVUP4K21nZQRsDRGBBbhAy+66zu6SxOxw5WMbpKWT0Vwcxw=="  # 你的picovoice key
keyword_path = './Hey--Venus_en_windows_v2_1_0.ppn'  # 你的唤醒词检测离线文件地址
openai_api_key = "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"
api_key = "eb96b3d8daba32c54cbffe8e350096ba.Ot4TZHhR33zfdBMR"  # 请填写您自己的APIKey

def Orator():
    picowakeword = PicoWakeWord(PICOVOICE_API_KEY, keyword_path)
    asr=OpenaiASRg()
    tts = EdgeTTS()
    #openai_chat_module = OpenaiAgentModule(openai_api_key)
    bot = ChatBot(api_key)
    
    try:
        run(picowakeword, asr, tts, bot)
    except KeyboardInterrupt:
        if picowakeword.porcupine is not None:
            picowakeword.porcupine.delete()
            print("Deleting porc")
        if picowakeword.stream is not None:
            picowakeword.stream.close()
            print("Closing stream")
        if picowakeword.myaudio is not None:
            picowakeword.myaudio.terminate()
            print("Terminating pa")
            exit(0)
    finally:
        print('本轮对话结束')
        print('主人，我退下啦！')
        asyncio.run(tts.text_to_speech_and_play('嗯'+'主人，我退下啦！'))
        if picowakeword.porcupine is not None:
            picowakeword.porcupine.delete()
            print("Deleting porc")
        if picowakeword.stream is not None:
            picowakeword.stream.close()
            print("Closing stream")
        if picowakeword.myaudio is not None:
            picowakeword.myaudio.terminate()
            print("Terminating pa")
            Orator()

if __name__ == '__main__':
    Orator()
