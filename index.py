from pathlib import Path
import modal
from fastapi.responses import Response
from voicevox_core import VoicevoxCore

app = modal.App("voicevox-serverless-test")
@app.function(
        image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
            .pip_install(
                "modal",
                "pathlib",
                "https://github.com/VOICEVOX/voicevox_core/releases/download/0.15.4/voicevox_core-0.15.4+cuda-cp38-abi3-linux_x86_64.whl",
                "onnxruntime",
            ).copy_local_dir("./lib","/root/"),
        gpu="T4",
)

@modal.web_endpoint()
def get_audio(text: str):
    speaker_id = 1
    core = VoicevoxCore(open_jtalk_dict_dir=Path("/root/open_jtalk_dic_utf_8-1.11"))
    if not core.is_model_loaded(speaker_id):
        core.load_model(speaker_id)
    wave_bytes = core.tts(text, speaker_id)
    return Response(content=wave_bytes, media_type="audio/wav")
