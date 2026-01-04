version = "v2ProPlus"
model_version = "v2ProPlus" 

import voice_management
import json
import logging
import os
import re
import sys
import traceback
import warnings
from io import BytesIO
from time import time as ttime
from datetime import datetime

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoModelForMaskedLM, AutoTokenizer

from feature_extractor import cnhubert
from text.LangSegmenter import LangSegmenter
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from sv import SV

# 로깅 설정
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

# 경로 설정
cnhubert_base_path = './pretrained_models/chinese-hubert-base'
bert_path = './pretrained_models/chinese-roberta-wwm-ext-large'
PRETRAINED_SOVITS_PATH = "./pretrained_models/v2Pro/s2Gv2ProPlus.pth"
PRETRAINED_GPT_PATH = "./pretrained_models/s1v3.ckpt"

# Device 설정
is_half = True
device = 'cuda'
dtype = torch.float16 if is_half else torch.float32

# 전역 변수
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
hz = 50
max_sec = 30
cache = {}
tts_idx = 0
hps = None
pretrained_hps = None  # Zero-shot용 hps 캐시

# LRU Cache
MAX_CACHED_ACTORS = 10

class LRUModelCache:
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
            for oldest_key in list(self.cache.keys()):
                if oldest_key != 'pretrained':
                    oldest_model = self.cache.pop(oldest_key)
                    print(f"[LRU] Evicting {self.cache_type} model: {oldest_key}")
                    del oldest_model
                    torch.cuda.empty_cache()
                    break
        self.cache[key] = model
    
    def __contains__(self, key):
        return key in self.cache

vq_models = LRUModelCache(max_size=MAX_CACHED_ACTORS + 1, cache_type="SoVITS")
t2s_models = LRUModelCache(max_size=MAX_CACHED_ACTORS + 1, cache_type="GPT")

# Initialize models
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

tokenizer = None
bert_model = None
sv_cn_model = None

class DictToAttrRecursive(dict):
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

def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)

def change_sovits_weights(actor, sovits_path):
    global vq_model, hps, version, model_version
    
    cached_model = vq_models.get(actor)
    if cached_model is not None:
        # 캐시된 모델이 있다면, hps가 그에 맞는지 확인 필요하지만
        # 현재 구조상 매번 로드하지 않으므로, 사용자 주의 필요.
        # (일반적으로 같은 v2Pro 모델끼리는 hps 호환됨)
        return cached_model
    
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    
    if hps.data.n_speakers == 300:
        hps.model.version = "v2ProPlus"
        model_version = "v2ProPlus"
    else:
        hps.model.version = "v2"
        model_version = "v2"
    
    version = "v2"
    print(f"[SoVITS] Loading {actor}: {sovits_path}")
    
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
    
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    vq_models.put(actor, vq_model)
    return vq_model

def change_gpt_weights(actor, gpt_path):
    global hz, max_sec, config
    
    cached_model = t2s_models.get(actor)
    if cached_model is not None:
        return cached_model
    
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(dict_s1["weight"])
    if is_half:
        model = model.half()
    model = model.to(device)
    model.eval()
    print(f"[GPT] Loading {actor}: {gpt_path}")
    
    t2s_models.put(actor, model)
    return model

# SV Model
def init_sv_cn():
    global sv_cn_model
    if sv_cn_model is None:
        sv_cn_model = SV(device, is_half)
        print("[v2Pro] SV model initialized")

if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

# BERT
def load_bert_model():
    global tokenizer, bert_model
    if bert_model is not None:
        return
    print("🔄 Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    print("✅ BERT model loaded!")

def get_bert_feature(text, word2ph):
    load_bert_model()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
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
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert

def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    
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
    else:
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
        return []
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

# Zero-Shot / Pretrained Model Logic
PRETRAINED_ACTOR = "pretrained"

def load_pretrained_models():
    global pretrained_hps, hps, hz, max_sec, config, model_version, version
    
    # [수정] 이미 캐시된 hps가 있다면 복원 (전역 hps가 중요함)
    if pretrained_hps is not None:
        hps = pretrained_hps
    
    if PRETRAINED_ACTOR in vq_models.cache:
        return
    
    print("[Zero-Shot] Loading pretrained models...")
    
    dict_s2 = load_sovits_new(PRETRAINED_SOVITS_PATH)
    pretrained_hps = DictToAttrRecursive(dict_s2["config"])
    pretrained_hps.model.semantic_frame_rate = "25hz"
    pretrained_hps.model.version = "v2ProPlus"
    model_version = "v2ProPlus"
    version = "v2"
    hps = pretrained_hps # 전역 hps 갱신
    
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
    pretrained_vq_model.load_state_dict(dict_s2["weight"], strict=False)
    
    dict_s1 = torch.load(PRETRAINED_GPT_PATH, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    pretrained_t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    pretrained_t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        pretrained_t2s_model = pretrained_t2s_model.half()
    pretrained_t2s_model = pretrained_t2s_model.to(device)
    pretrained_t2s_model.eval()
    
    vq_models.put(PRETRAINED_ACTOR, pretrained_vq_model)
    t2s_models.put(PRETRAINED_ACTOR, pretrained_t2s_model)
    print(f"[Zero-Shot] Pretrained models cached as '{PRETRAINED_ACTOR}'")


resample_transform_dict = {}
def resample(audio_tensor, sr0, sr1, device):
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    # hps safety check
    if hps is None:
        raise RuntimeError("hps is None in get_spepc. Model might not be loaded correctly.")
        
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


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="No cut", top_k=15, top_p=1, temperature=1, ref_free=False, speed=1, if_freeze=False, inp_refs=None, sample_steps=8, if_sr=False, pause_second=0.3, actor='arona'):
    global cache, hps, tts_idx
    
    # 1. Fallback / Zero-Shot Logic
    target_actor = actor
    use_pretrained = (actor == PRETRAINED_ACTOR)

    if not use_pretrained:
        voice_info = voice_management.get_voice_info_from_name(actor)
        print('###voiceinfo', voice_info)
        has_custom = False
        if voice_info:
            if os.path.exists(voice_info.get('sovits_path', '')) and os.path.exists(voice_info.get('gpt_path', '')):
                has_custom = True
        
        if has_custom:
            if actor not in vq_models:
                change_sovits_weights(actor, voice_info['sovits_path'])
            if actor not in t2s_models:
                change_gpt_weights(actor, voice_info['gpt_path'])
        else:
            print(f"[get_tts_wav] Model for '{actor}' not found. Using Pretrained model.")
            target_actor = PRETRAINED_ACTOR
            use_pretrained = True
            
    if use_pretrained:
        load_pretrained_models()

    vq_model = vq_models.get(target_actor)
    t2s_model = t2s_models.get(target_actor)
    
    if vq_model is None or t2s_model is None:
        return "no info"
    
    # hps Safety Check
    if hps is None:
        # Fallback: if pretrained loaded, force it
        if pretrained_hps is not None:
            hps = pretrained_hps
        else:
            return "no info (hps is None)"

    t = []
    t0 = ttime()

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print("Actual input reference text:", prompt_text)
    
    text = text.strip("\n")
    print("Actual input target text:", text)
    
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * pause_second),
        dtype=np.float16 if is_half else np.float32,
    )
    zero_wav_torch = torch.from_numpy(zero_wav)
    if is_half:
        zero_wav_torch = zero_wav_torch.half().to(device)
    else:
        zero_wav_torch = zero_wav_torch.to(device)
    
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            if is_half:
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
    
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    
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
        
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        
        if i_text in cache and if_freeze:
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
                if idx == 0:
                    return 'early stop'
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic
        
        t3 = ttime()
        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        
        refers = []
        if is_v2pro:
            sv_emb = []
            if sv_cn_model is None:
                init_sv_cn()
                
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
    
    if if_sr and model_version == "v3":
        pass
    else:
        audio_opt = audio_opt.cpu().detach().numpy()
    
    tts_idx = (tts_idx + 1) % 20
    result = 'output' + str(tts_idx) + '.wav'
    
    if os.path.exists('./files_server/'):
        sf.write("./files_server/" + result, (audio_opt * 32768).astype(np.int16), 32000)
    else:
        sf.write("./" + result, (audio_opt * 32768).astype(np.int16), 32000)
    
    # Test용 파일 저장
    try:
        safe_actor = actor.replace('/', '_').replace('\\', '_')
        voice_file_name = f"{safe_actor}_{datetime.now().strftime('%y%m%d_%H%M%S')}.wav"
        voice_audio_path = os.path.join('./test/voice', voice_file_name)
        os.makedirs('./test/voice', exist_ok=True)
        sf.write(voice_audio_path, (audio_opt * 32768).astype(np.int16), 32000)
    except:
        pass
    
    return result

def synthesize_cloning_voice(char_name, audio_text, audio_language='ja', speed=1):
    load_pretrained_models()
    
    prompt_info = voice_management.get_prompt_info_from_name(char_name)
    if prompt_info is None:
        return "no info"
    
    ref_wav_path = prompt_info['wav_path'] 
    prompt_text = prompt_info['text']
    prompt_language = prompt_info['language']
    
    result = get_tts_wav(
        ref_wav_path, prompt_text, prompt_language, 
        audio_text, audio_language, 
        actor=PRETRAINED_ACTOR, speed=speed
    )
    return result

def synthesize_char(char_name, audio_text, audio_language='ja', speed=1):
    ref_wav_path = ''
    prompt_text = ''
    prompt_language = ''
    
    try:
        prompt_info = voice_management.get_prompt_info_from_name(char_name)
        if prompt_info is None:
            return "no info"
        
        ref_wav_path = prompt_info['wav_path'] 
        prompt_text = prompt_info['text']
        prompt_language = prompt_info['language']
    except:
        pass
    
    return get_tts_wav(
        ref_wav_path=ref_wav_path, prompt_text=prompt_text, prompt_language=prompt_language, 
        text=audio_text, text_language=audio_language, 
        actor=char_name, speed=speed
    )

if __name__ == '__main__':
    # voice_inference_test에서 만든 voices/info.json에 들어간 내용 테스트
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
    
    actor = 'Shigure_(Hot_Spring)'
    actor = 'Yuzu_(Maid)'  # 현재 normal이 없으면 오류나는 대표적 예시
    actor = 'Yuzu'  
    
    # prompt_info = voice_management.get_prompt_info_from_name(actor)  # Todo : 없을때의 Try Catch
    # prompt_language = prompt_info['language'] # 'ja'
    # ref_wav_path = prompt_info['wav_path'] #'./voices/noa.wav'
    # prompt_text = prompt_info['text'] # 'さすがです、先生。勉強になりました。'
       
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor=actor)
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    # get_tts_wav(ref_wav_path, prompt_text, prompt_language, audio_text, audio_language, actor='arona')
    
    # result = synthesize_char(actor, audio_text, audio_language='ja', speed=1)
    # print('save at ' + result)
    
    # ========== Actor 30개 문장 연속 테스트 ==========
    if True:
        print("\n" + "="*60)
        print("[Multi-Sentence Test] Testing 30 sentences...")
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
                result = synthesize_char(actor, text, audio_language='ja', speed=1)
                result_path = os.path.abspath(result)
                print(f"[{idx}/30] -> {result_path} ✓")
            except Exception as e:
                print(f"[{idx}/30] -> FAILED: {e}")
        
        print("\n" + "="*60)
        print("[Arona Multi-Sentence Test] Completed!")
        print("="*60)
    
    # ========== Zero-Shot Voice Cloning 전체 캐릭터 테스트 ==========
    if False:
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