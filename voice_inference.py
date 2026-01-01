version = "v2ProPlus"
model_version = "v2ProPlus" 

import voice_management
# import psutil
# import os

# def set_high_priority():
#     """把当前 Python 进程设为 HIGH_PRIORITY_CLASS"""
#     if os.name != "nt":
#         return # 仅 Windows 有效
#     p = psutil.Process(os.getpid())
#     try:
#         p.nice(psutil.HIGH_PRIORITY_CLASS)
#         print("已将进程优先级设为 High")
#     except psutil.AccessDenied:
#         print("权限不足，无法修改优先级（请用管理员运行）")
# set_high_priority()
import json
import logging
import os
import re
import sys
import traceback
import warnings

import torch
import torchaudio
from text.LangSegmenter import LangSegmenter

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

# 하드코딩 경로 (voice_management에서 캐릭터별로 관리)
cnhubert_base_path = './pretrained_models/chinese-hubert-base'
bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'

# Zero-shot용 pretrained 모델 경로
PRETRAINED_SOVITS_PATH = "./pretrained_models/v2Pro/s2Gv2ProPlus.pth"
PRETRAINED_GPT_PATH = "./pretrained_models/s1v3.ckpt"

# 기본 설정
is_half = True
device = 'cuda'
dtype = torch.float16 if is_half else torch.float32

# 구분자 집합
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

# GPT 관련 전역변수
hz = 50
max_sec = 30  # 기본값, change_gpt_weights에서 업데이트됨
cache = {}
tts_idx = 0  # 파일 충돌 방지용 인덱스

# LRU Cache Configuration
MAX_CACHED_ACTORS = 10

class LRUModelCache:
    """LRU(Least Recently Used) 기반 모델 캐시 관리 클래스"""
    def __init__(self, max_size=10, cache_type="unknown"):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_type = cache_type
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, model):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = model
            return
        if len(self.cache) >= self.max_size:
            # pretrained 모델은 eviction 대상에서 제외
            for oldest_key in list(self.cache.keys()):
                if oldest_key != 'pretrained':
                    oldest_model = self.cache.pop(oldest_key)
                    print(f"[LRU] Evicting {self.cache_type} model: {oldest_key}")
                    del oldest_model
                    torch.cuda.empty_cache()
                    break
        self.cache[key] = model
    
    def keys(self):
        return list(self.cache.keys())
    
    def clear(self):
        for key, model in self.cache.items():
            del model
        self.cache.clear()
        torch.cuda.empty_cache()
    
    def __contains__(self, key):
        return key in self.cache

# pretrained + 10 actors = 11 max size
vq_models = LRUModelCache(max_size=MAX_CACHED_ACTORS + 1, cache_type="SoVITS")
t2s_models = LRUModelCache(max_size=MAX_CACHED_ACTORS + 1, cache_type="GPT")

# BERT Lazy Loading
tokenizer = None
bert_model = None

# cnhubert 초기화
from feature_extractor import cnhubert
cnhubert.cnhubert_base_path = cnhubert_base_path

# 필요한 모듈 import
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch

# ssl_model (CNHubert) 초기화
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

class DictToAttrRecursive(dict):
    """dict를 attribute로 접근 가능하게 변환"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# v2Pro 모델 로더 (특별한 헤더 처리)
from io import BytesIO

def load_sovits_new(sovits_path):
    """v2Pro/v2ProPlus 모델은 헤더가 다름 (05/06 -> PK로 복원 필요)"""
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":  # 일반 zip 형식이 아니면 (v2Pro/v2ProPlus)
        data = b"PK" + f.read()  # "PK" 헤더 복원
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

# 전역 hps (마지막 로드된 SoVITS config)
hps = None

# def change_sovits_weights(actor, sovits_path):
#     """v2Pro SoVITS 모델 로딩 (LRU 캐시)"""
#     global hps, version
    
#     cached_model = vq_models.get(actor)
#     if cached_model is not None:
#         return cached_model
    
#     dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
#     hps = dict_s2["config"]
#     hps = DictToAttrRecursive(hps)
#     hps.model.semantic_frame_rate = "25hz"
    
#     # v2Pro/v2ProPlus 버전 설정
#     hps.model.version = model_version  # "v2ProPlus"
#     version = "v2"  # symbol version for text processing
    
#     model = SynthesizerTrn(
#         hps.data.filter_length // 2 + 1,
#         hps.train.segment_size // hps.data.hop_length,
#         n_speakers=hps.data.n_speakers,
#         **hps.model
#     )
#     if "pretrained" not in sovits_path:
#         try:
#             del model.enc_q
#         except:
#             pass
#     if is_half:
#         model = model.half().to(device)
#     else:
#         model = model.to(device)
#     model.eval()
#     print(f"[v2Pro] Loading SoVITS: {sovits_path}")
#     print(model.load_state_dict(dict_s2["weight"], strict=False))
    
#     vq_models.put(actor, model)
#     return model


def change_sovits_weights(actor, sovits_path):
    """v2Pro SoVITS 모델 로딩 (LRU 캐시)"""
    global vq_model, hps, version, model_version
    
    # LRU 캐시 체크
    cached_model = vq_models.get(actor)
    if cached_model is not None:
        return cached_model
    
    # ========== WebUI용 로직 (사용 안함) ==========
    # if "！" in sovits_path or "!" in sovits_path:
    #     sovits_path = name2sovits_path[sovits_path]
    # version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    # is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    # path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    # if if_lora_v3 == True and is_exist == False:
    #     ...
    # dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    # if prompt_language is not None and text_language is not None:
    #     yield (...)  # WebUI 업데이트용
    
    # 모델 로드 (load_sovits_new: v2Pro 헤더 처리)
    dict_s2 = load_sovits_new(sovits_path)
    
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    
    # 버전 감지 (v2Pro: n_speakers=300)
    if hps.data.n_speakers == 300:
        hps.model.version = "v2ProPlus"
        model_version = "v2ProPlus"
    elif "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"
        model_version = "v2"
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
        model_version = "v1"
    else:
        hps.model.version = "v2"
        model_version = "v2"
    
    version = "v2"  # text processing용
    print(f"[SoVITS] {sovits_path} -> version={version}, model_version={model_version}")
    
    # ========== v3/v4 분기 (사용 안함) ==========
    # if model_version not in v3v4set:
    #     ...
    # else:
    #     vq_model = SynthesizerTrnV3(...)
    
    # v2/v2Pro 모델 생성
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    # ========== LoRA 로딩 (v2Pro는 LoRA 없음) ==========
    # if if_lora_v3 == False:
    #     ...
    # else:
    #     lora_rank = dict_s2["lora_rank"]
    #     ...
    
    print(f"loading sovits_{model_version}", vq_model.load_state_dict(dict_s2["weight"], strict=False))
    
    # ========== WebUI yield / weight.json (사용 안함) ==========
    # yield (...)
    # with open("./weight.json") as f:
    #     ...
    
    # LRU 캐시에 저장
    vq_models.put(actor, vq_model)
    return vq_model

# ========== WebUI 초기화 (사용 안함) ==========
# try:
#     next(change_sovits_weights(sovits_path))
# except:
#     pass



def change_gpt_weights(actor, gpt_path):
    """GPT 모델 로딩 (LRU 캐시)"""
    global hz, max_sec, config
    
    cached_model = t2s_models.get(actor)
    if cached_model is not None:
        return cached_model
    
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(dict_s1["weight"])
    if is_half:
        model = model.half()
    model = model.to(device)
    model.eval()
    print(f"[v2Pro] Loading GPT: {gpt_path}")
    
    t2s_models.put(actor, model)
    return model

# resample 캐시
resample_transform_dict = {}

def resample(audio_tensor, sr0, sr1, device):
    """오디오 리샘플링 (캐시 사용)"""
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    """스펙트로그램 추출 (v2Pro: audio_tensor도 반환)"""
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio

# SV (Speaker Verification) 모델 - v2Pro 전용
from sv import SV

sv_cn_model = None

def init_sv_cn():
    """SV 모델 초기화 (v2Pro 전용)"""
    global sv_cn_model
    sv_cn_model = SV(device, is_half)
    print("[v2Pro] SV model initialized")

# v2Pro/v2ProPlus일 때 SV 모델 사전 로딩
if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

# BERT Lazy Loading 함수
def load_bert_model():
    """중국어 처리를 위한 BERT 모델을 Lazy Loading"""
    global tokenizer, bert_model
    if bert_model is not None:
        return
    print("🔄 Loading BERT model for Chinese language support...")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    print("✅ BERT model loaded successfully!")

def get_bert_feature(text, word2ph):
    """중국어 BERT feature 추출"""
    load_bert_model()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    """언어별 BERT feature 생성 - 중국어만 실제 BERT 사용"""
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    
    # ========== 중국어/월어 (사용 안함) ==========
    # if language == "all_zh":
    #     for tmp in LangSegmenter.getTexts(text,"zh"):
    #         langlist.append(tmp["lang"])
    #         textlist.append(tmp["text"])
    # elif language == "all_yue":
    #     for tmp in LangSegmenter.getTexts(text,"zh"):
    #         if tmp["lang"] == "zh":
    #             tmp["lang"] = "yue"
    #         langlist.append(tmp["lang"])
    #         textlist.append(tmp["text"])
    
    # ========== 일본어/한국어/영어 ==========
    if language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    # elif language == "auto_yue":  # 월어 혼합 (사용 안함)
    #     for tmp in LangSegmenter.getTexts(text):
    #         if tmp["lang"] == "zh":
    #             tmp["lang"] = "yue"
    #         langlist.append(tmp["lang"])
    #         textlist.append(tmp["text"])
    else:
        # ja, ko 등 혼합 처리
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                langlist.append(language)
            textlist.append(tmp["text"])
    
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text

def merge_short_text_in_array(texts, threshold):
    if len(texts) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError('###process_text valueerror')
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def synthesize_char(char_name, audio_text, audio_language='ja', speed=1):
    import os
    
    voice_info = voice_management.get_voice_info_from_name(char_name)
    prompt_info = voice_management.get_prompt_info_from_name(char_name)
    
    if prompt_info is None:
        print(f'[synthesize_char] ERROR: No prompt info for "{char_name}"')
        return 'no info'
    if not os.path.exists(prompt_info.get('wav_path', '')):
        print(f'[synthesize_char] ERROR: Reference audio not found for "{char_name}"')
        return 'no info'
    
    use_zeroshot = False
    if voice_info is None:
        use_zeroshot = True
        print(f'[synthesize_char] No trained model for "{char_name}", using zero-shot...')
    elif not os.path.exists(voice_info.get('gpt_path', '')) or not os.path.exists(voice_info.get('sovits_path', '')):
        use_zeroshot = True
        print(f'[synthesize_char] Model files missing for "{char_name}", using zero-shot...')
    
    prompt_language = prompt_info['language']
    ref_wav_path = prompt_info['wav_path']
    prompt_text = prompt_info['text']
    
    if use_zeroshot:
        result = synthesize_cloning_voice(char_name, audio_text, audio_language, speed)
    else:
        result = get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor=char_name, speed=speed)
    
    return result

# ========== Zero-Shot Voice Cloning ==========
# pretrained 모델용 전역 변수
pretrained_hps = None
PRETRAINED_ACTOR = "pretrained"  # pretrained 모델용 actor 이름 (LRU eviction 대상 제외)

def load_pretrained_models():
    """Zero-shot용 pretrained 모델 로딩 (LRU 캐시에 저장)"""
    global pretrained_hps, hps, hz, max_sec, config, model_version, version
    
    # 이미 캐시에 있으면 스킵
    if PRETRAINED_ACTOR in vq_models.cache:
        return
    
    print("[Zero-Shot] Loading pretrained models...")
    
    # SoVITS 모델 로드
    dict_s2 = load_sovits_new(PRETRAINED_SOVITS_PATH)
    pretrained_hps = DictToAttrRecursive(dict_s2["config"])
    pretrained_hps.model.semantic_frame_rate = "25hz"
    pretrained_hps.model.version = "v2ProPlus"
    model_version = "v2ProPlus"
    version = "v2"
    hps = pretrained_hps  # 전역 hps 설정
    
    pretrained_vq_model = SynthesizerTrn(
        pretrained_hps.data.filter_length // 2 + 1,
        pretrained_hps.train.segment_size // pretrained_hps.data.hop_length,
        n_speakers=pretrained_hps.data.n_speakers,
        **pretrained_hps.model,
    )
    if is_half:
        pretrained_vq_model = pretrained_vq_model.half().to(device)
    else:
        pretrained_vq_model = pretrained_vq_model.to(device)
    pretrained_vq_model.eval()
    print("[Zero-Shot] SoVITS loaded:", pretrained_vq_model.load_state_dict(dict_s2["weight"], strict=False))
    
    # GPT 모델 로드
    hz = 50
    dict_s1 = torch.load(PRETRAINED_GPT_PATH, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    pretrained_t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    pretrained_t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        pretrained_t2s_model = pretrained_t2s_model.half()
    pretrained_t2s_model = pretrained_t2s_model.to(device)
    pretrained_t2s_model.eval()
    print("[Zero-Shot] GPT loaded")
    
    # LRU 캐시에 저장 (특별 actor 이름으로)
    vq_models.put(PRETRAINED_ACTOR, pretrained_vq_model)
    t2s_models.put(PRETRAINED_ACTOR, pretrained_t2s_model)
    print(f"[Zero-Shot] Pretrained models cached as '{PRETRAINED_ACTOR}'")

def synthesize_cloning_voice(char_name, audio_text, audio_language='ja', speed=1):
    """
    Zero-shot 음성 클로닝 (pretrained 모델 사용)
    
    Parameters:
        char_name: voices 폴더의 캐릭터 이름 (wav/text 참조용)
        audio_text: 생성할 텍스트
        audio_language: 생성할 언어 ('ja', 'ko', 'en')
        speed: 속도 배율 (default: 1.0)
    
    Returns:
        str: 생성된 wav 파일 경로
    
    Note:
        - actor별 학습된 .pth 모델 대신 pretrained 모델 사용
        - voices/{char_name}/에서 참조 wav와 text만 사용
        - 새 캐릭터 즉시 테스트 가능 (학습 불필요)
    """
    # pretrained 모델 로딩 (LRU 캐시에 저장)
    load_pretrained_models()
    
    # 참조 정보 가져오기 (wav/text만 사용, pth 불필요)
    prompt_info = voice_management.get_prompt_info_from_name(char_name)
    print(f"[Zero-Shot] {char_name}:", prompt_info)
    
    prompt_language = prompt_info['language']
    ref_wav_path = prompt_info['wav_path']
    prompt_text = prompt_info['text']
    
    # 기존 get_tts_wav 함수 재사용 (actor를 PRETRAINED_ACTOR로 설정)
    result = get_tts_wav(
        ref_wav_path, prompt_text, prompt_language, 
        audio_text, audio_language, 
        actor=PRETRAINED_ACTOR, speed=speed
    )
    return result

'''
Parameters:
    ref_wav_path   : 참조 오디오 파일 경로 (3~10초 권장)
    prompt_text    : 참조 오디오의 텍스트 내용
    prompt_language: 참조 오디오 언어 # "ja", "ko", "en", "zh"
    text           : 생성할 텍스트
    text_language  : 생성할 텍스트 언어 # "ja", "ko", "en", "zh"
    how_to_cut     : 텍스트 분할 방식 # "No cut", "Cut every 4 sentences", "Cut every 50 characters", "Cut by Chinese period", "Cut by English period", "Cut by punctuation"
    top_k          : GPT 샘플링 top-k (default: 15)
    top_p          : GPT 샘플링 top-p (default: 1)
    temperature    : GPT 샘플링 온도 (default: 1) - 높을수록 다양성 증가
    ref_free       : 참조 없이 생성 여부 (default: False) - v2Pro는 미지원
    speed          : 생성 속도 배율 (default: 1.0)
    if_freeze      : 캐시 사용 여부 (default: False)
    inp_refs       : 추가 참조 오디오 리스트 (default: None)
    sample_steps   : v3/v4 전용 샘플 스텝 (default: 8) - v2Pro는 미사용
    if_sr          : 오디오 초고해상도 적용 (default: False) - v3 전용
    pause_second   : 문장 간 휴지 시간 초 (default: 0.3)
'''
# GitHub 기본값: top_k=20, top_p=0.6, temperature=0.6
# WebUI 및 기존 v2 경험을 통해 top_k=15, top_p=1, temperature=1로 조절
# (zero-shot에서 EOS 토큰 누락 방지 및 안정적인 생성을 위함)
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="No cut", top_k=15, top_p=1, temperature=1, ref_free=False, speed=1, if_freeze=False, inp_refs=None, sample_steps=8, if_sr=False, pause_second=0.3, actor='arona'):
    global cache, hps
    
    # LRU 캐시에서 모델 로딩 (Reference v2 방식)
    if actor not in vq_models:
        voice_info = voice_management.get_voice_info_from_name(actor)
        sovits_path = voice_info['sovits_path']
        print('### Actor loading', actor, ':', sovits_path)
        change_sovits_weights(actor, sovits_path)
    if actor not in t2s_models:
        voice_info = voice_management.get_voice_info_from_name(actor)
        gpt_path = voice_info['gpt_path']
        change_gpt_weights(actor, gpt_path)
    
    # 캐시에서 모델 가져오기
    vq_model = vq_models.get(actor)
    t2s_model = t2s_models.get(actor)
    
    t = []
    # if prompt_text is None or len(prompt_text) == 0:
    #     ref_free = True
    
    # if model_version in v3v4set:  # {"v3", "v4"}
    #     ref_free = False
    # else:
    #     if_sr = False
        
    ref_free = False
    if_sr = False
    
    # if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
    #     clean_bigvgan_model()
    #     clean_hifigan_model()
    #     clean_sv_cn_model()
    
    t0 = ttime()
    # v2 Reference처럼 직접 언어 코드 사용 (dict_language 변환 불필요)
    # prompt_language = dict_language[prompt_language]
    # text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print("Actual input reference text:", prompt_text)
    
    text = text.strip("\n")
    print("Actual input target text:", text)
    
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half == True else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half == True:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                print("Warning: Reference audio must be between 3-10 seconds")
                return None
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device)
            else:
                wav16k = wav16k.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    # if how_to_cut == "Cut every 4 sentences":
    #     text = cut1(text)
    # elif how_to_cut == "Cut every 50 characters":
    #     text = cut2(text)
    # elif how_to_cut == "Cut by Chinese period":
    #     text = cut3(text)
    # elif how_to_cut == "Cut by English period":
    #     text = cut4(text)
    # elif how_to_cut == "Cut by punctuation":
    #     text = cut5(text)
    
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    
    print("Actual input target text (after sentence cutting):", text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print("Actual input target text (per sentence):", text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print("Frontend processed text (per sentence):", norm_text2)
        
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        
        if i_text in cache and if_freeze == True:
            pred_semantic = cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                # Early stop 안전장치: 생성 실패 시 즉시 종료
                if idx == 0:
                    print("[WARNING] Early stop triggered: idx==0, synthesis failed")
                    return 'early stop'
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic
        
        t3 = ttime()
        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        
        if True: # model_version not in v3v4set:
            refers = []
            if is_v2pro:
                sv_emb = []
                if sv_cn_model == None:
                    init_sv_cn()
            if inp_refs:
                for path in inp_refs:
                    try:
                        refer, audio_tensor = get_spepc(hps, path.name, dtype, device, is_v2pro)
                        refers.append(refer)
                        if is_v2pro:
                            sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                    except:
                        traceback.print_exc()
            if len(refers) == 0:
                refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
                refers = [refers]
                if is_v2pro:
                    sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
            if is_v2pro:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
                )[0][0]
            else:
                audio = vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
                )[0][0]
        # ========== v3/v4 분기 (v2Pro에서는 사용 안함) ==========
        # else:
        #     refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
        #     phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
        #     phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
        #     fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
        #     ref_audio, sr = torchaudio.load(ref_wav_path)
        #     ref_audio = ref_audio.to(device).float()
        #     if ref_audio.shape[0] == 2:
        #         ref_audio = ref_audio.mean(0).unsqueeze(0)
        #     tgt_sr = 24000 if model_version == "v3" else 32000
        #     if sr != tgt_sr:
        #         ref_audio = resample(ref_audio, sr, tgt_sr, device)
        #     
        #     mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
        #     mel2 = norm_spec(mel2)
        #     T_min = min(mel2.shape[2], fea_ref.shape[2])
        #     mel2 = mel2[:, :, :T_min]
        #     fea_ref = fea_ref[:, :, :T_min]
        #     Tref = 468 if model_version == "v3" else 500
        #     Tchunk = 934 if model_version == "v3" else 1000
        #     if T_min > Tref:
        #         mel2 = mel2[:, :, -Tref:]
        #         fea_ref = fea_ref[:, :, -Tref:]
        #         T_min = Tref
        #     chunk_len = Tchunk - T_min
        #     mel2 = mel2.to(dtype)
        #     fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
        #     cfm_resss = []
        #     idx = 0
        #     while 1:
        #         fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        #         if fea_todo_chunk.shape[-1] == 0:
        #             break
        #         idx += chunk_len
        #         fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        #         cfm_res = vq_model.cfm.inference(
        #             fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
        #         )
        #         cfm_res = cfm_res[:, :, mel2.shape[2] :]
        #         mel2 = cfm_res[:, :, -T_min:]
        #         fea_ref = fea_todo_chunk[:, :, -T_min:]
        #         cfm_resss.append(cfm_res)
        #     cfm_res = torch.cat(cfm_resss, 2)
        #     cfm_res = denorm_spec(cfm_res)
        #     if model_version == "v3":
        #         if bigvgan_model == None:
        #             init_bigvgan()
        #     else:
        #         if hifigan_model == None:
        #             init_hifigan()
        #     vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
        #     with torch.inference_mode():
        #         wav_gen = vocoder_model(cfm_res)
        #         audio = wav_gen[0][0]
        
        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav_torch)
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
    audio_opt = torch.cat(audio_opt, 0)
    
    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        opt_sr = 32000
    elif model_version == "v3":
        opt_sr = 24000
    else:
        opt_sr = 48000
    
    if if_sr == True and opt_sr == 24000:
        pass
        # v3 전용 audio super resolution (v2Pro는 미사용)
    else:
        audio_opt = audio_opt.cpu().detach().numpy()
    
    # 파일 저장 (Reference v2 방식)
    global tts_idx
    tts_idx = (tts_idx + 1) % 20
    result = 'output' + str(tts_idx) + '.wav'
    
    if os.path.exists('./files_server/'):
        sf.write("./files_server/" + result, (audio_opt * 32768).astype(np.int16), opt_sr)
    # elif os.path.exists('./test/voice_inference/'):
    #     result = 'output' + actor + str(tts_idx) + '.wav'
    #     sf.write("./test/voice_inference/" + result, (audio_opt * 32768).astype(np.int16), opt_sr)
    else:
        sf.write("./" + result, (audio_opt * 32768).astype(np.int16), opt_sr)
    
    # Test용 파일저장
    try:
        voice_file_name = actor + "_voice_" + str(datetime.now().strftime("%y%m%d_%H%M%S")) + "_" + text + ".wav"
        voice_audio_path = os.path.join('./test/voice', voice_file_name)
        os.makedirs('./test/voice', exist_ok=True)
        sf.write(voice_audio_path, (audio_opt * 32768).astype(np.int16), opt_sr)
    except:
        print('fail saving get_tts_wav')
    
    return result


if __name__ == '__main__':    
    # TODO : 로컬화할 경우, 영향도 파악 (현재 패키징은 가능)
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    cnhubert_base_path = './pretrained_models/chinese-hubert-base'
    bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'
    _CUDA_VISIBLE_DEVICES = 0
    # is_half = False  # Float32
    is_half = True  # Float32
    
    # audio_text = '안녕? 난 민구라고 해'
    # audio_text = '테스트중! 메리크리스마스!'
    # audio_text = 'API 사용 가능한가요?'
    # audio_text = 'Only English Spokened'
    # audio_text = '오케이!'
    # audio_text = 'python can be spoken'
    # audio_text = 'get some rest sensei! 안녕하세요?'
    # audio_language = 'ko'
    audio_text = '待っておったぞ、先生。'
    audio_text = 'そなたはイタズラが好きなのじゃな。'
    # audio_text = '新しきを知るのは良いことじゃ。そうじゃろ?'
    # audio_text = 'じゃが、ゆるそう。'
    # audio_text = 'ほれ、カボチャとナツメの料理じゃ。そなたと一緒に食べたいと思ってな。'
    audio_text = '右クリックでメニューを開き、設定を変更することができます。'
    # audio_text = 'ふぅえ…'
    # audio_text = 'ひえええっ！'
    # audio_text = '先生を信用しているつもりです。'  # miyako idx = 0 오류
    audio_language = 'ja'
    # print('error?')
    
    audio_text = audio_text.replace('AI', 'えいあい')
    audio_text = audio_text.replace('MY-Little-JARVIS-3D', 'マイリトル・ジャービス スリーでぃ')
    audio_text = audio_text.replace('MY-Little-JARVIS', 'マイリトル・ジャービス')
    audio_text = audio_text.replace('Android', 'アンドロイド')
    audio_text = audio_text.replace('Windows', 'ウィンドウズ')
    audio_text = audio_text.replace('方', 'かた')
    audio_text = audio_text.replace('.exe', 'ドット exe')
    
    print(audio_text)
    
    actor = 'arona'

    prompt_info = voice_management.get_prompt_info_from_name(actor)  # Todo : 없을때의 Try Catch
    prompt_language = prompt_info['language'] # 'ja'
    ref_wav_path = prompt_info['wav_path'] #'./voices/noa.wav'
    prompt_text = prompt_info['text'] # 'さすがです、先生。勉強になりました。'
       
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor=actor)
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')

    result = synthesize_char(actor, audio_text, audio_language='ja', speed=1)
    print('save at ' + result)
    
    # ========== Arona 30개 문장 연속 테스트 ==========
    if False:
        print("\n" + "="*60)
        print("[Arona Multi-Sentence Test] Testing 30 sentences...")
        print("="*60 + "\n")
        
        audio_text_list = []
        # 일본어 테스트 문장 30개
        audio_text_list.append("おはようございます、先生。今日も良い一日になりますように。")
        audio_text_list.append("先生、お疲れ様です。少し休憩しませんか？")
        audio_text_list.append("このデータを分析してみましたが、興味深い結果が出ました。")
        audio_text_list.append("シャーレの生徒たちは、みんな元気にしています。")
        audio_text_list.append("今日の授業、とても楽しかったですね。")
        audio_text_list.append("先生、明日の予定は確認されましたか？")
        audio_text_list.append("新しいプロジェクトが始まりますよ。")
        audio_text_list.append("みんなで協力すれば、必ず成功できます。")
        audio_text_list.append("データベースの更新が完了しました。")
        audio_text_list.append("セキュリティシステムに異常はありません。")
        audio_text_list.append("先生のおかげで、問題が解決しました。")
        audio_text_list.append("この資料、とても参考になると思います。")
        audio_text_list.append("次の会議は午後2時からです。")
        audio_text_list.append("お昼ご飯、何を食べますか？")
        audio_text_list.append("天気が良いですね。散歩しませんか？")
        audio_text_list.append("新しい機能を追加してみました。")
        audio_text_list.append("システムのメンテナンスは終わりました。")
        audio_text_list.append("レポートの提出期限は明日までです。")
        audio_text_list.append("みんな、頑張っていますね。")
        audio_text_list.append("先生、質問があります。聞いてもいいですか？")
        audio_text_list.append("計画通りに進んでいます。")
        audio_text_list.append("もうすぐゴールが見えてきました。")
        audio_text_list.append("素晴らしい成果ですね。おめでとうございます。")
        audio_text_list.append("次のステップに進みましょう。")
        audio_text_list.append("困ったことがあれば、いつでも相談してください。")
        audio_text_list.append("先生は本当に頼りになります。")
        audio_text_list.append("今日も一日、お疲れ様でした。")
        audio_text_list.append("明日も頑張りましょうね。")
        audio_text_list.append("良い夢を見てください。おやすみなさい。")
        audio_text_list.append("また明日お会いしましょう。さようなら。")
        
        for idx, text in enumerate(audio_text_list, 1):
            try:
                print(f"[{idx}/30] Synthesizing: {text[:30]}...")
                result = synthesize_char('arona', text, audio_language='ja', speed=1)
                result_path = os.path.abspath(result)
                print(f"[{idx}/30] -> {result_path} ✓")
            except Exception as e:
                print(f"[{idx}/30] -> FAILED: {e}")
        
        print("\n" + "="*60)
        print("[Arona Multi-Sentence Test] Completed!")
        print("="*60)
    
    # ========== Zero-Shot Voice Cloning 전체 캐릭터 테스트 ==========
    if True:
        print("\n" + "="*60)
        print("[Zero-Shot] Testing all characters...")
        print("="*60 + "\n")
        
        # info.json의 모든 캐릭터 목록
        all_characters = [
            'arona', 'plana', 'mika', 'yuuka', 'noa', 'koyuki',
            'nagisa', 'mari', 'kisaki', 'miyako', 'ui', 'seia', 'prana'
        ]
        
        test_text = "先生、お疲れ様でした。今日も頑張りましたね。"  # 테스트용 일본어 텍스트
        
        for char in all_characters:
            try:
                print(f"\n[Zero-Shot Test] {char}...")
                result = synthesize_cloning_voice(char, test_text, audio_language='ja', speed=1)
                # Ctrl+클릭 가능한 절대 경로로 출력
                result_path = os.path.abspath(result)
                print(f"[Zero-Shot Test] {char} -> {result_path} ✓")
            except Exception as e:
                print(f"[Zero-Shot Test] {char} -> FAILED: {e}")
        
        print("\n" + "="*60)
        print("[Zero-Shot] All tests completed!")
        print("="*60)