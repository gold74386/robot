import requests
from aip import AipSpeech
import speech_recognition as sr
# pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk

import requests
import subprocess
#import speech_recognition as sr
import multiprocessing
from edge_tts import Communicate
import asyncio
#from speechmodules.text2speech import BaiduTTS, Pyttsx3TTS, AzureTTS,EdgeTTS
import asyncio
import aiohttp
#import whisper
import librosa
import speech_recognition as sr
import multiprocessing
#import whisper
#import logging
import torch
import argparse
import functools
import platform


class BaiduASR:
    def __init__(self, APP_ID, API_KEY, SECRET_KEY):
        self.APP_ID = APP_ID
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        self.r = sr.Recognizer()

    # 从麦克风收集音频并写入文件
    def _record(self, if_cmu: bool = False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            # 校准环境噪声水平的energy threshold
            # duration:用于指定计算环境噪声的持续时间（秒）。默认值为1秒。函数将等待指定时间来计算环境噪声水平，并相应地调整麦克风增益，以提高语音识别的准确性。如果噪声水平很高，则可以增加此值以获得更准确的噪声估计。
            # self.r.adjust_for_ambient_noise(source, duration=1)
            print('您可以开始说话了')
            # timeout 用于指定等待语音输入的最长时间（秒），如果没有检测到语音输入，则函数将返回None。默认值为 None，表示等待无限长的时间。如果指定了超时时间，则函数将在等待指定时间后自动返回。
            # phrase_time_limit：用于指定允许单次语音输入的最长时间（秒），如果超过这个时间，函数将自动停止录制，并返回None.默认值为 None，表示允许单次语音输入的时间没有限制。
            audio = self.r.listen(source, timeout=25, phrase_time_limit=4)

        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())

        if if_cmu:
            return audio
        else:
            return self._get_file_content(file_name)

    # 从本地文件中加载音频 作为后续百度语音服务的输入
    def _get_file_content(self, file_name):
        with open(file_name, 'rb') as f:
            audio_data = f.read()
        return audio_data

    def speech_to_text(self, audio_path: str = "test.wav", if_microphone: bool = True):
        # 麦克风输入 采样频率必须为8的倍数 我们使用16000和上面保持一致
        if if_microphone:
            result = self.client.asr(self._record(), 'wav', 16000, {
                'dev_pid': 1537  # 识别中文普通话
            })
        # 从文件中读取
        else:
            result = self.client.asr(self._get_file_content(audio_path), 'wav', 16000, {
                'dev_pid': 1537  # 识别中文普通话
            })
        if result["err_msg"] != "success.":
            return "语音识别失败：" + result["err_msg"]
        else:
            return result['result'][0]


class OpenaiASR:
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY
        self.r = sr.Recognizer()

    # 从麦克风收集音频并写入文件
    def _record(self, if_cmu: bool = False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            print('您可以开始说话了')
            audio = self.r.listen(source, timeout=20, phrase_time_limit=5)

        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())

        if if_cmu:
            return audio
        else:
            return self._get_file_content(file_name)

    # 从本地文件中加载音频 作为后续百度语音服务的输入
    def _get_file_content(self, file_name):
        with open(file_name, 'rb') as f:
            audio_data = f.read()
        return audio_data

    def _get_speech_text(self, audio_file):
        #print('调用用语音识别')
        url = 'https://api.openai.com/v1/audio/transcriptions'
        headers = {
            'Authorization': 'Bearer ' + self.API_KEY
        }
        files = {
            'file': ('./speech.wav', audio_file),
        }
        data = {
            'model': 'whisper-1',
        }
        response = requests.post(url, headers=headers, data=data, files=files)
        result = response.json()['text']
        # print(result)
        return result

    def speech_to_text(self, audio_path: str = "test.wav", if_microphone: bool = True):
        if if_microphone:
            result = self._get_speech_text(self._record())
        else:
            result = self._get_speech_text(audio_path)
        return result


class AzureASR:
    def __init__(self, AZURE_API_KEY, AZURE_REGION):
        self.AZURE_API_KEY = AZURE_API_KEY
        self.AZURE_REGION = AZURE_REGION
        self.speech_config = speechsdk.SpeechConfig(subscription=AZURE_API_KEY, region=AZURE_REGION)

    def speech_to_text(self, audio_path: str = "test.wav", if_microphone: bool = True):
        self.speech_config.speech_recognition_language = "zh-CN"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        print("Speak into your microphone.")
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized:{}".format(speech_recognition_result.text))
            return speech_recognition_result.text
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized :{}".format(speech_recognition_result.no_match_details))
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled:{}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details:{}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")
        return None
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


import torch

import speech_recognition as sr
import logging
import torch
#import whisper

class bensiASR:
    def __init__(self, num_threads=5):
        self.r = sr.Recognizer()
        logging.basicConfig(level=logging.WARNING)
        self.num_threads = num_threads
        self.device = torch.device("cpu")
        
    def h_record(self, if_cmu=False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            self.r.adjust_for_ambient_noise(source, duration=0.1)
            print('您可以开始说话了')
            audio = self.r.listen(source, timeout=20, phrase_time_limit=5)

        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())

    #def speech_to_text(self, audio_path=None, model_name="small", language='Chinese'):
    def speech_to_text(self, audio_path=None, model_name="base", language='Chinese'):
    #def speech_to_text(self, audio_path=None, model_name="tiny", language='Chinese'):
        self.h_record(if_cmu=False, rate=16000)
        torch.set_num_threads(self.num_threads)
        model = whisper.load_model(model_name)
        model.to(self.device)
        result = model.transcribe("./speech.wav", beam_size=5, language="zh",initial_prompt="Convert to Simplified Chinese")
        return result["text"]
    
class benASR:
    def __init__(self):
        self.r = sr.Recognizer()

    def h_record(self, if_cmu=False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            print('请说出你的问题或需要的帮助')
            audio = self.r.listen(source, timeout=20, phrase_time_limit=8)
        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())

    def speech_to_text(self):
        try:
            self.h_record(if_cmu=False, rate=16000)
            response = requests.post(url="http://localhost:5500/recognition", 
                                 files=[("audio", ("test.wav", open("./speech.wav", 'rb'), 'audio/wav'))],
                                 json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"}, timeout=30)
            result_json = response.json()
            if "results" in result_json and len(result_json["results"]) > 0:
                return result_json["results"][0]["result"]
            else:
                return "No transcription result available"
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"
class SpeechToText:
    def __init__(self):
        self.r = sr.Recognizer()

    def h_record(self, if_cmu=False, rate=16000):
        with sr.Microphone(sample_rate=rate) as source:
            print('请说出你的问题或需要的帮助')
            audio = self.r.listen(source, timeout=20, phrase_time_limit=8)
        file_name = "./speech.wav"
        with open(file_name, "wb") as f:
            f.write(audio.get_wav_data())

    def speech_to_text(self, audio_path, model_path, use_gpu=True, language="chinese", num_beams=1, batch_size=16, use_compile=False, task="transcribe", assistant_model_path=None, local_files_only=True, use_flash_attention_2=False, use_bettertransformer=False):
        device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() and use_gpu else torch.float32

        processor = AutoProcessor.from_pretrained(model_path)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
            use_flash_attention_2=use_flash_attention_2
        )
        if use_bettertransformer and not use_flash_attention_2:
            model = model.to_bettertransformer()
        if use_compile:
            if torch.__version__ >= "2" and platform.system().lower() != 'windows':
                model = torch.compile(model)
        model.to(device)

        generate_kwargs_pipeline = None
        if assistant_model_path is not None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            assistant_model.to(device)
            generate_kwargs_pipeline = {"assistant_model": assistant_model}

        infer_pipe = pipeline("automatic-speech-recognition",
                              model=model,
                              tokenizer=processor.tokenizer,
                              feature_extractor=processor.feature_extractor,
                              max_new_tokens=128,
                              chunk_length_s=30,
                              batch_size=batch_size,
                              torch_dtype=torch_dtype,
                              generate_kwargs=generate_kwargs_pipeline,
                              device=device)

        generate_kwargs = {"task": task, "num_beams": num_beams}
        if language is not None:
            generate_kwargs["language"] = language

        result = infer_pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

        for chunk in result["chunks"]:
           print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")

    def run(self):
        parser = argparse.ArgumentParser(description=__doc__)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg("audio_path",  type=str,  default="./speech.wav", help="预测的音频路径")
        add_arg("model_path",  type=str,  default="models/whisper-tiny", help="合并模型的路径，或者是huggingface上模型的名称")
        add_arg("use_gpu",     type=bool, default=True,      help="是否使用gpu进行预测")
        add_arg("language",    type=str,  default="chinese", help="设置语言，如果为None则预测的是多语言")
        add_arg("num_beams",   type=int,  default=1,         help="解码搜索大小")
        add_arg("batch_size",  type=int,  default=16,        help="预测batch_size大小")
        add_arg("use_compile", type=bool, default=False,     help="是否使用Pytorch2.0的编译器")
        add_arg("task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
        add_arg("assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
        add_arg("local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
        add_arg("use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
        add_arg("use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
        args = parser.parse_args()
        self.h_record(if_cmu=False, rate=16000)

        self.speech_to_text(args.audio_path, args.model_path, args.use_gpu, args.language, args.num_beams, args.batch_size, args.use_compile, args.task, args.assistant_model_path, args.local_files_only, args.use_flash_attention_2, args.use_bettertransformer)


if __name__ == '__main__':
    #APP_ID = '32353381'
    #API_KEY = 'dlh6gQDLkpKWZKhgjxRGHo0p'
    #SECRET_KEY = 'j4IVX3UalUCOG6E6oyCIsgcHaser1xI6'
    # baiduasr = BaiduASR(APP_ID, API_KEY, SECRET_KEY)
    # result = baiduasr.speech_to_text()
    # print(result)
    
    # AZURE_API_KEY = ""
    # AZURE_REGION = ""
    # azureasr = AzureASR(AZURE_API_KEY, AZURE_REGION)
    # azureasr.speech_to_text()
    #openai_api_key = "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"
    #openaiasr = OpenaiASR(openai_api_key)
    #print(openaiasr.speech_to_text())
    openaiasr = OpenaiASRg()
    print(openaiasr.speech_to_text())
    #openaiasr = OpenaiASRg()
    #print(openaiasr.transcribe_audio("audio_data"))
    #openaiasr = bensiASR()
    #print(openaiasr.speech_to_text())
    #openaiasr = benASR()
    #print(openaiasr.speech_to_text())
    #stt = SpeechToText()
    #stt.run()
