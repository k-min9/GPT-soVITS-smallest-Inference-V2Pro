"""
v2Pro TTS Backend Server
Reference: GPT-soVITS-smallest-Inference/tts_backend.py

API Endpoints:
- GET  /alive                    - Health check
- POST /getSound                 - TTS 음성 합성 (학습 모델, 없으면 자동으로 zero-shot fallback)
- POST /getSoundZeroShot         - TTS Zero-shot 음성 클로닝 (pretrained 모델 강제 사용)
- POST /stt                      - STT (음성 인식)
- POST /speech_diarization       - 화자 분석 필터링
- POST /cache/clear_all          - 캐시 전체 삭제
- POST /cache/remain             - N개 actor만 유지
- POST /cache/load               - actor 미리 로딩
- GET  /cache/status             - 캐시 상태 조회
"""

import voice_inference
# import util_pyngrok  # 필요시 활성화
# import util_silerovad  # 필요시 활성화
# import util_speech_diarization  # 필요시 활성화

import os
import shutil
import uuid

# Server-Flask
from flask import Flask, Response, request, jsonify, send_file, abort
from waitress import serve
app = Flask(__name__)


# ===== Health Check =====
@app.route('/alive', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'tts-voice-service-v2pro',
        'timestamp': int(__import__('time').time())
    }), 200


# ===== TTS (음성 합성) =====
@app.route('/getSound/jp', methods=['POST'])  # legacy
@app.route('/getSound/ko', methods=['POST'])  # legacy
@app.route('/getSound', methods=['POST'])
def synthesize_sound():
    def get_sound_text_ja(text):
        text = text.replace('RABBIT', 'ラビット')
        text = text.replace('SCHALE', 'シャーレ')
        return text
    
    print('###getSound request', request.json)
    text = request.json.get('text', '안녕하십니까.')
    char = request.json.get('char', 'arona')
    lang = request.json.get('lang', 'ko')
    speed = request.json.get('speed', 100)  # % 50~100
    speed = float(speed) / 100
    chat_idx = request.json.get('chatIdx', '-1')
    
    if lang == 'ja' or lang == 'jp':
        lang = 'ja'
        text = get_sound_text_ja(text)
    
    result = voice_inference.synthesize_char(char, text, audio_language=lang, speed=speed)
    if result == 'early stop':
        abort(500, description="Synthesis process stopped early.")
    
    response = send_file(result, mimetype="audio/wav")
    response.headers['Chat-Idx'] = chat_idx
    return response


# ===== TTS Zero-Shot (음성 클로닝) =====
@app.route('/getSoundZeroShot', methods=['POST'])
def synthesize_sound_zeroshot():
    """
    Zero-shot 음성 클로닝 API
    - 학습된 모델 없이 참조 음성만으로 합성
    - pretrained 모델 사용
    """
    def get_sound_text_ja(text):
        text = text.replace('RABBIT', 'ラビット')
        text = text.replace('SCHALE', 'シャーレ')
        return text
    
    print('###getSoundZeroShot request', request.json)
    text = request.json.get('text', '안녕하십니까.')
    char = request.json.get('char', 'arona')
    lang = request.json.get('lang', 'ko')
    speed = request.json.get('speed', 100)
    speed = float(speed) / 100
    chat_idx = request.json.get('chatIdx', '-1')
    
    if lang == 'ja' or lang == 'jp':
        lang = 'ja'
        text = get_sound_text_ja(text)
    
    result = voice_inference.synthesize_cloning_voice(char, text, audio_language=lang, speed=speed)
    if result == 'early stop':
        abort(500, description="Synthesis process stopped early.")
    
    response = send_file(result, mimetype="audio/wav")
    response.headers['Chat-Idx'] = chat_idx
    return response


# ===== STT (음성 인식) =====
@app.route('/stt', methods=['POST'])
def main_stream_stt():
    def transcribe_audio_to_text(audio_path, expected_stt_lang='en', model_name="small") -> str:
        from faster_whisper import WhisperModel
        try:
            print(f"Loading Whisper model: {model_name}...")
            model = WhisperModel(model_name, device="cpu", download_root='./model')
            
            print(f"Transcribing {audio_path}...")
            segments, info = model.transcribe(audio_path)
            text = ""
            for segment in segments:
                text = text + segment.text
            
            return text.lower(), info.language
        except Exception as e:
            print(f"Error occurred during transcription: {e}")
            return ""
    
    try:
        stt_lang = request.form.get('lang', 'ko')
        stt_level = request.form.get('level', 'small')
        stt_chat_idx = request.form.get('chatIdx', '-1')
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        
        audio_path = os.path.join('./files', f"{uuid.uuid4()}.wav")
        os.makedirs('./files', exist_ok=True)
        file.save(audio_path)
        
        # STT 수행
        trans_text, trans_lang = transcribe_audio_to_text(audio_path, stt_lang, stt_level)
        
        os.remove(audio_path)
        
        response = {"text": trans_text, "lang": trans_lang, "chatIdx": stt_chat_idx}
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /stt endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ===== Speech Diarization (화자 분석) =====
@app.route('/speech_diarization', methods=['POST'])
def speech_diarization_filter():
    """음성 화자 분석을 통한 음성 필터링 (v2Pro는 간소화)"""
    try:
        player_name = request.form.get('player', 'sensei')
        char_name = request.form.get('char', 'arona')
        ai_voice_filter_idx = request.form.get('ai_voice_filter_idx', '0')
        
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "should_ignore": True,
                "similarity": 0.0,
                "character": char_name
            }), 400
        
        file = request.files['file']
        audio_path = os.path.join('./files', f"speech_check_{uuid.uuid4()}.wav")
        os.makedirs('./files', exist_ok=True)
        file.save(audio_path)
        
        # 모드에 따른 처리 (v2Pro는 간소화: 항상 False)
        result = {
            "should_ignore": False,
            "similarity": 0.0,
            "character": char_name,
            "mode": "disabled"
        }
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print('### /speech_diarization\n', result)
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] /speech_diarization: {e}")
        return jsonify({
            "error": "Internal server error",
            "should_ignore": False,
            "similarity": 0.0,
            "character": char_name if 'char_name' in locals() else 'unknown'
        }), 500


# ===== Cache Management =====
@app.route('/cache/clear_all', methods=['POST'])
def cache_clear_all():
    """모든 캐시된 actor 모델 제거"""
    try:
        voice_inference.vq_models.clear()
        voice_inference.t2s_models.clear()
        return jsonify({
            "status": "success",
            "message": "All cached models cleared"
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/clear_all: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cache/remain', methods=['POST'])
def cache_remain():
    """최근 사용된 N개 actor만 남기고 나머지 제거"""
    try:
        count = request.json.get('count', 2)
        count = int(count)
        if count < 0:
            return jsonify({"status": "error", "message": "count must be non-negative"}), 400
        
        # LRU 캐시에서 오래된 것들 제거
        while len(voice_inference.vq_models.cache) > count:
            voice_inference.vq_models.cache.popitem(last=False)
        while len(voice_inference.t2s_models.cache) > count:
            voice_inference.t2s_models.cache.popitem(last=False)
        
        return jsonify({
            "status": "success",
            "message": f"Kept only {count} most recent actors",
            "count": count
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/remain: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cache/remain/<int:n>', methods=['GET'])
def cache_remain_get(n):
    """최근 사용된 N개 actor만 남기고 나머지 제거 (테스트용)"""
    try:
        if n < 0:
            return jsonify({"status": "error", "message": "n must be non-negative"}), 400
        
        while len(voice_inference.vq_models.cache) > n:
            voice_inference.vq_models.cache.popitem(last=False)
        while len(voice_inference.t2s_models.cache) > n:
            voice_inference.t2s_models.cache.popitem(last=False)
        
        return jsonify({
            "status": "success",
            "message": f"Kept only {n} most recent actors",
            "count": n
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/remain/{n}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cache/load', methods=['POST'])
def cache_load():
    """특정 actor 모델을 미리 로딩"""
    try:
        actor = request.json.get('actor', None)
        if not actor:
            return jsonify({"status": "error", "message": "actor parameter is required"}), 400
        
        voice_inference.synthesize_char(actor, 'テスト', audio_language='ja')
        return jsonify({
            "status": "success",
            "message": f"Actor {actor} loaded",
            "actor": actor
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/load: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cache/load/<actor>', methods=['GET'])
def cache_load_get(actor):
    """특정 actor 모델을 미리 로딩 (테스트용)"""
    try:
        voice_inference.synthesize_char(actor, 'テスト', audio_language='ja')
        return jsonify({
            "status": "success",
            "message": f"Actor {actor} loaded",
            "actor": actor
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/load/{actor}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cache/status', methods=['GET'])
def cache_status():
    """현재 캐시 상태 조회"""
    try:
        cached_vq = voice_inference.vq_models.keys()
        cached_t2s = voice_inference.t2s_models.keys()
        
        return jsonify({
            "status": "success",
            "cached_actors": cached_vq,
            "count": len(cached_vq),
            "max_cache_size": voice_inference.MAX_CACHED_ACTORS
        }), 200
    except Exception as e:
        print(f"[ERROR] /cache/status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    
    # preloading (optional)
    print("[v2Pro] Starting TTS Backend...")
    # voice_inference.synthesize_char('arona', '안녕하세요!', audio_language='ja')
    
    # Ngrok (optional)
    # util_pyngrok.start_ngrok(id='dev_voice')
    
    # Server run
    tts_port = 5010
    print(f"[v2Pro] TTS Server running on port {tts_port}")
    # app.run(host='0.0.0.0', port=tts_port)
    serve(app, host="0.0.0.0", port=tts_port)
