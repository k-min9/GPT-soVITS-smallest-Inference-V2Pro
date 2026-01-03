import os
import re
import shutil
import argparse
from datetime import datetime

from voice_inference import get_tts_wav
import voice_management


def sanitize_filename_keep_jp_punc(name: str, max_len: int) -> str:
    s = (name or "").replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r'[\\/:*?"<>|]', "_", s)  # Windows forbidden
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s if s else "noname"


def resolve_result_path(result: str) -> str:
    if not result:
        return ""
    c1 = os.path.join("./files_server", result)
    c2 = os.path.join(".", result)
    if os.path.exists(c1):
        return os.path.abspath(c1)
    if os.path.exists(c2):
        return os.path.abspath(c2)
    if os.path.exists(result):
        return os.path.abspath(result)
    return ""


def move_with_collision_suffix(src_abs: str, dst_abs: str) -> str:
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)

    if not os.path.exists(dst_abs):
        shutil.move(src_abs, dst_abs)
        return dst_abs

    base, ext = os.path.splitext(dst_abs)
    n = 2
    while True:
        cand = f"{base}_{n}{ext}"
        if not os.path.exists(cand):
            shutil.move(src_abs, cand)
            return cand
        n += 1


def infer_prompt_language_from_text(text: str) -> str:
    if not text:
        return "ja"
    has_hangul = re.search(r"[가-힣]", text) is not None
    has_kana = re.search(r"[ぁ-ゟ゠-ヿ]", text) is not None
    has_cjk = re.search(r"[\u4e00-\u9fff]", text) is not None
    has_latin = re.search(r"[A-Za-z]", text) is not None
    if has_hangul:
        return "ko"
    if has_kana:
        return "ja"
    if has_cjk and not has_latin:
        return "zh"
    if has_latin and not has_cjk:
        return "en"
    return "ja"


def build_audio_text_list_ja_30() -> list:
    audio_text_list = []
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
    return audio_text_list


def build_test_default_cases(default_actors=None, default_emotions=None) -> list:
    """
    voice_management.get_prompt_info_from_name에서 가져와서 케이스를 만드는 기본 빌더입니다.
    결과 튜플 형식: (wav_path, wav_ref_text, actor, emotion_key)
    """
    cases = []

    if default_actors is None:
        try:
            default_actors = voice_management.get_voice_name()
        except Exception:
            default_actors = []

    if default_emotions is None:
        default_emotions = ["normal"]

    for actor in default_actors:
        for emo_raw in default_emotions:
            emo_key = voice_management.normalize_emotion(emo_raw)

            # 실제로 emotion prompt가 없으면 normal로 강제 fallback
            if emo_key != "normal":
                if not voice_management.has_prompt_emotion(actor, emo_key):
                    emo_key = "normal"

            prompt_info = voice_management.get_prompt_info_from_name(actor, emo_key)
            if not prompt_info:
                continue

            wav_path = prompt_info.get("wav_path", "")
            wav_ref_text = prompt_info.get("text", "")

            if not wav_path or not wav_ref_text:
                continue
            if not os.path.exists(wav_path):
                continue

            cases.append((wav_path, wav_ref_text, actor, emo_key))

    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_language", default="ja", choices=["ja", "ko", "en", "zh", "auto"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out_root", default="./test/actor")
    parser.add_argument("--snippet_len", type=int, default=18)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    audio_text_list = build_audio_text_list_ja_30()

    # 1) 기본 케이스 생성
    test_cases = []
    # test_cases.extend(build_test_default_cases(
    #     default_actors=None,
    #     default_emotions=["normal", "question", "surprise"]
    # ))

    # 2) 여기부터는 수동으로 하드코딩 append
    # 튜플 형식: (wav_path, wav_ref_text, actor, emotion)
    # emotion은 normal, surprise, question 같은 key를 권장하지만,
    # "..." "!?" 같은 값도 넣으면 EMOTION_MAP으로 변환됩니다.
    #
    # 수동 ref 지정 예시
    # test_cases.append(("./voices/custom_ref.wav", "参照音声のテキストです。", "arona", "normal"))
    # test_cases.append(("./voices/custom_ref.wav", "参照音声のテキストです。", "arona", "...!"))
    # test_cases.append(("./voices/custom_ref.wav", "参照音声のテキストです。", "arona", "?"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Cafe_monolog_4~ちょっとお腹空いてきちゃった……。.ogg", "ちょっとお腹空いてきちゃった……。", "Airi_(Band)", "calm"))
    # test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Lobby_5_1~正直、最初は心配でした。余計なことをしたんじゃないのかなって……。.ogg", "正直、最初は心配でした。余計なことをしたんじゃないのかなって……。", "Airi_(Band)", "calm"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Minigame_Mission_2~わ、私じゃなくてあちらを見てください…….ogg", "わ、私じゃなくてあちらを見てください……", "Airi_(Band)", "calm"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_EventLogin_1~おかえりなさい、先生！お待ちしておりました！.ogg", "おかえりなさい、先生！お待ちしておりました！", "Airi_(Band)", "surprise"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_EventLogin_2~先生！よかったら練習の成果を見てください！.ogg", "先生！よかったら練習の成果を見てください！", "Airi_(Band)", "surprise"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_EventLogin_Season_2~最後まで頑張ります！見ていてください、先生！.ogg", "最後まで頑張ります！見ていてください、先生！", "Airi_(Band)", "surprise"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Battle_Victory_1~お祝いの曲でも演奏しましょうか？.ogg", "お祝いの曲でも演奏しましょうか？", "Airi_(Band)", "question"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_EventLobby_4~もう～！私もやり返しちゃいますよ？.ogg", "もう～！私もやり返しちゃいますよ？", "Airi_(Band)", "question"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_LogIn_1~おかえりなさい、先生！弾いてほしい曲とかありますか？.ogg", "おかえりなさい、先生！弾いてほしい曲とかありますか？", "Airi_(Band)", "question"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Minigame_Failed~大丈夫です。まだ時間はあります。.ogg", "大丈夫です。まだ時間はあります。", "Airi_(Band)", "normal"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Minigame_MissionDaily_end~本日のチェックはこれで終わりですね。おつかれさまでした。.ogg", "本日のチェックはこれで終わりですね。おつかれさまでした。", "Airi_(Band)", "normal"))
    test_cases.append(("voices/Airi_(Band)/Airi_(Band)_Tactic_Defeat_2~も、もう一回。もう一回やればいけます。.ogg", "も、もう一回。もう一回やればいけます。", "Airi_(Band)", "normal"))


    print("\n" + "=" * 60)
    print("[Voice Inference Test] Start")
    print("=" * 60)
    print(f"run_id: {run_id}")
    print(f"sentences: {len(audio_text_list)}")
    print(f"cases: {len(test_cases)}")
    print(f"audio_language: {args.audio_language}, speed: {args.speed}")
    print(f"out_root: {os.path.abspath(args.out_root)}")
    print("=" * 60 + "\n")

    ok = 0
    fail = 0

    for case_idx, (wav_path, wav_ref_text, actor, emotion_raw) in enumerate(test_cases, 1):
        actor = (actor or "").strip()
        wav_path = (wav_path or "").strip()
        wav_ref_text = (wav_ref_text or "").strip()

        emotion_key = voice_management.normalize_emotion(emotion_raw)

        # emotion prompt가 실제로 없으면 normal로 fallback
        if emotion_key != "normal":
            if not voice_management.has_prompt_emotion(actor, emotion_key):
                emotion_key = "normal"

        if not actor:
            print(f"[Case {case_idx}] SKIP: actor empty")
            continue

        actor_dir = os.path.join(args.out_root, sanitize_filename_keep_jp_punc(actor, 32))
        os.makedirs(actor_dir, exist_ok=True)

        # wav_path/ref_text가 비어있으면 info.json에서 채움
        if (not wav_path) or (not wav_ref_text) or (not os.path.exists(wav_path)):
            prompt_info = voice_management.get_prompt_info_from_name(actor, emotion_key)
            if not prompt_info:
                print(f"[Case {case_idx}] SKIP: no prompt_info actor={actor}, emotion={emotion_key}")
                fail += len(audio_text_list)
                continue

            wav_path = prompt_info.get("wav_path", "")
            wav_ref_text = prompt_info.get("text", "")

        if (not wav_path) or (not wav_ref_text) or (not os.path.exists(wav_path)):
            print(f"[Case {case_idx}] SKIP: invalid wav_path/text actor={actor}, emotion={emotion_key}")
            fail += len(audio_text_list)
            continue

        prompt_language = infer_prompt_language_from_text(wav_ref_text)

        # 수동으로 넣은 케이스라면(즉, wav_path/ref_text를 직접 지정했다면) prompts에 저장 시도
        # 기본 케이스는 이미 info.json에 있으니 중복 저장 필요 없습니다.
        if wav_path and wav_ref_text:
            voice_management.append_voice_info_data(
                name=actor,
                wav_path=wav_path,
                text=wav_ref_text,
                emotion=emotion_key,
                language=prompt_language,
                update_flag=False
            )

        print("\n" + "-" * 60)
        print(f"[Case {case_idx}/{len(test_cases)}] actor={actor}, emotion={emotion_key}")
        print(f"ref_wav_path: {wav_path}")
        print(f"prompt_language: {prompt_language}")
        print(f"out_dir: {os.path.abspath(actor_dir)}")
        print("-" * 60)

        for sent_idx, text in enumerate(audio_text_list, 1):
            try:
                print(f"[{sent_idx:02d}/{len(audio_text_list)}] Synth: {text[:30]} ...")

                result = get_tts_wav(
                    wav_path,
                    wav_ref_text,
                    prompt_language,
                    text,
                    args.audio_language,
                    actor=actor,
                    speed=args.speed
                )

                if not result or result in ("no info", "early stop"):
                    raise RuntimeError(f"get_tts_wav returned '{result}'")

                src_abs = resolve_result_path(result)
                if not src_abs:
                    raise RuntimeError(f"cannot resolve result path from '{result}'")

                snippet = sanitize_filename_keep_jp_punc(text, args.snippet_len)
                wav_name = f"{snippet}_{run_id}.wav"
                dst_abs = os.path.abspath(os.path.join(actor_dir, wav_name))

                final_wav = move_with_collision_suffix(src_abs, dst_abs)
                print(f"[{sent_idx:02d}] OK -> {final_wav}")
                ok += 1

            except Exception as e:
                fail += 1
                print(f"[{sent_idx:02d}] FAILED: {e}")

    print("\n" + "=" * 60)
    print("[Voice Inference Test] Completed")
    print("=" * 60)
    print(f"OK: {ok}")
    print(f"FAILED: {fail}")
    print("=" * 60)
