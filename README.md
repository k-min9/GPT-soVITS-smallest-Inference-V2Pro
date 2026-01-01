# AI - GPT-soVITS-smallest-Inference-V2Pro

## 개요

- GPT-soVITS v2Pro를 실행하기 위한 최소 라이브러리 및 함수 세팅
- 백엔드 통신을 통한 음성합성 기능
- v2 -> v2ProPlus로 버전업을 통한 제로샷 기능 획득

## 환경 세팅

- venv, library 세팅

    ``` bash
        py -3.10 -m venv venv

        pip install transformers==4.57.1
        pip install librosa==0.9.2
        pip install soundfile
        pip install split-lang  # fast-langdetect가 덤으로 추가
        pip install soxr  # transformers 버전 이슈
        pip install einops
        pip install pytorch-lightning
        pip install matplotlib
        pip install nltk
        pip install pynvml
        pip install pyngrok
        pip install supabase

          # Backserver 추가(설치시 torch 재설치 필요.)
        pip install silero-vad
        pip install faster-whisper
        pip install pyannote-audio

        pip install flask
        pip install waitress  # WSGI for production

        pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        pip install torchaudio==2.5.1
        pip install torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

        # Conflict Solving
        pip install numpy==1.23

        # pyopenjtalk-0.3.4.dist-info 이동

    ```

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)의 해당항목 이동
  - 최상단에 ffmpeg.exe, ffprobe.exe 세팅
  - voices 이동
    - GPT_weights_v2/로 ckpt 이동
    - SoVITS_weights_v2/로 pth 이동
  - venv의 LangSegment, pyopenjtalk 이동
  - pretrained_models 이동

## 빌드

- pyinstaller --onedir main.py -n main --noconsole --contents-directory=files --noconfirm # 메인 프로그램
- pyinstaller --onedir tts_backend.py -n server --contents-directory=files_server --noconfirm # 서버 인터페이스
- --icon=./icon_plana.ico
- 몇몇 라이브러리 이동 필요

## 트러블 슈팅

- torch jit (torch\jit\_script.py) 이슈
  - AR 폴더 변경사항
  - @torch.jit.script > @torch.jit._script_if_tracing
- text 의 cleaner에 import 추가하여 강제 로딩
  - import text.japanese  
    import text.korean  
    import text.english 추가  
- transformers\utils\import_utils.py", line 1647, in check_torch_load_is_safe 버전 이슈
  
    ```python
    def check_torch_load_is_safe() -> None:
        if not is_torch_greater_or_equal("2.6"):
            pass
            # raise ValueError(
            #     "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            #     "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            #     "when loading files with safetensors."
            #     "\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
            # )
    ```
