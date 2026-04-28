import os, sys, torch, numpy as np, soundfile as sf, tempfile, base64, io, traceback, threading, yt_dlp, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

if sys.platform == 'win32':
    os.environ['USER'] = os.environ.get('USERNAME', 'user')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(os.environ.get('TEMP', '.'), 'torch_cache')

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2(rope=True, rms_norm=True, swiglu=True, cross_entropy=False)
except Exception: pass
    
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError: HAS_PYDUB = False

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, COMMON_AUDIO_EXTS
import gradio as gr

COMMON_AUDIO_EXTS.extend(['.mp4', '.mov', '.avi', '.mkv', '.webm'])

class VibeVoiceASRInference:
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "sdpa"):
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_path, dtype=dtype, device_map="cuda", load_in_4bit=True, attn_implementation=attn_implementation, trust_remote_code=True)
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()
    
    def transcribe(self, audio_path=None, max_new_tokens=512, temperature=0.0, top_p=1.0, do_sample=False, num_beams=1, repetition_penalty=1.0, context_info=None, streamer=None, **kwargs) -> dict:
        inputs = self.processor(audio=audio_path, return_tensors="pt", add_generation_prompt=True, context_info=context_info)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        gen_config = {"max_new_tokens": max_new_tokens, "temperature": temperature if temperature > 0 else None, "top_p": top_p if do_sample else None, "do_sample": do_sample, "num_beams": num_beams, "repetition_penalty": repetition_penalty, "pad_token_id": self.processor.pad_id, "eos_token_id": self.processor.tokenizer.eos_token_id}
        if streamer: gen_config["streamer"] = streamer
        gen_config["stopping_criteria"] = StoppingCriteriaList([StopOnFlag()])
        gen_config = {k: v for k, v in gen_config.items() if v is not None}
        
        start_time = time.time()
        with torch.no_grad(): output_ids = self.model.generate(**inputs, **gen_config)
        generation_time = time.time() - start_time
        
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        try: segments = self.processor.post_process_transcription(generated_text)
        except Exception: segments = []
        return {"raw_text": generated_text, "segments": segments, "generation_time": generation_time}

asr_model, stop_generation_flag = None, False
class StopOnFlag(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs): global stop_generation_flag; return stop_generation_flag

def initialize_model(model_path: str, device: str = "cuda", attn_implementation: str = "sdpa"):
    global asr_model
    try:
        asr_model = VibeVoiceASRInference(model_path=model_path, device=device, dtype=torch.bfloat16 if device != "cpu" else torch.float32, attn_implementation=attn_implementation)
        return "✅ Hệ thống sẵn sàng"
    except Exception as e: return f"❌ Lỗi: {str(e)}"

def transcribe_audio(audio_input, audio_path_input, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, context_info) -> iter:
    if asr_model is None: yield "❌ Vui lòng tải model!", ""; return
    if not audio_path_input and audio_input is None: yield "❌ Thiếu audio!", ""; return
    
    try:
        audio_path = audio_path_input if audio_path_input else audio_input
        streamer = TextIteratorStreamer(asr_model.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        res_box = {"res": None, "err": None}
        
        def run_t():
            try: res_box["res"] = asr_model.transcribe(audio_path=audio_path, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample, repetition_penalty=repetition_penalty, context_info=context_info if context_info and context_info.strip() else None, streamer=streamer)
            except Exception as e: res_box["err"] = str(e)
            
        t = threading.Thread(target=run_t); t.start()
        
        gen_text, st = "", time.time()
        for text in streamer:
            gen_text += text
            yield f"--- Đang phân tích ({time.time() - st:.1f}s) ---\n{gen_text.replace('},', '},\\n')}", "<div style='color:#777; text-align:center;'>Đang xử lý...</div>"
        t.join()
        
        if res_box["err"]: yield f"❌ Lỗi: {res_box['err']}", ""; return
        res = res_box["res"]
        
        raw = f"--- KẾT QUẢ ---\nThời gian: {res['generation_time']:.2f}s\n---\n{res['raw_text'].replace('},', '},\\n')}"
        html = f"<div style='padding:10px;'><h4>Đã trích xuất {len(res['segments'])} đoạn thoại</h4>"
        for i, s in enumerate(res['segments']):
            html += f"<div style='margin-bottom:10px; padding:10px; border:1px solid #ddd; border-radius:5px;'><b>Đoạn {i+1} [{s.get('start_time',0):.1f}s - {s.get('end_time',0):.1f}s] Người nói {s.get('speaker_id','?')}:</b><br>{s.get('text','')}</div>"
        html += "</div>"
        yield raw, html
    except Exception as e: yield f"❌ Lỗi hệ thống: {e}", ""

def create_gradio_interface(model_path: str, default_max_tokens: int = 8192, attn_implementation: str = "sdpa"):
    initialize_model(model_path, "cuda", attn_implementation)
    with gr.Blocks(title="Bóc băng AI") as demo:
        gr.Markdown("<h2 style='text-align:center;'>HỆ THỐNG TRÍCH XUẤT VĂN BẢN</h2>")
        
        # ĐÃ KHÓA CHẶT MINIMUM/MAXIMUM CHO TẤT CẢ SLIDER ĐỂ TRÁNH LỖI GRADIO
        with gr.Row(visible=False):
            m_tok = gr.Slider(minimum=1024, maximum=65536, value=default_max_tokens, step=1024)
            samp = gr.Checkbox(value=False)
            temp = gr.Slider(minimum=0.0, maximum=2.0, value=0.0)
            tp = gr.Slider(minimum=0.0, maximum=1.0, value=1.0)
            rp = gr.Slider(minimum=0.1, maximum=5.0, value=1.0)

        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(label="Dán Link YouTube/TikTok", lines=1)
                gr.Markdown("<center>HOẶC</center>")
                file_input = gr.File(label="Tải Video/Audio (MP4, MP3...)", file_types=["audio", "video"], type="filepath")
                ctx_input = gr.Textbox(label="Từ khóa tham chiếu (Tùy chọn)", lines=2)
                run_btn = gr.Button("Khởi chạy", variant="primary")
                stop_btn = gr.Button("Hủy", variant="secondary")
            with gr.Column(scale=2):
                raw_out = gr.Textbox(label="Văn bản", lines=15, show_copy_button=True)
                seg_out = gr.HTML(label="Phân rã")

        def process(url, fp, ctx, mt, t, p, s, r):
            global stop_generation_flag; stop_generation_flag = False
            f_path = ""
            if url and url.strip():
                yield "Đang tải dữ liệu từ mạng...", ""
                try:
                    with yt_dlp.YoutubeDL({'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'outtmpl': '/content/temp_audio.%(ext)s', 'quiet': True}) as ydl: ydl.extract_info(url, download=True)
                    f_path = "/content/temp_audio.mp3"
                except Exception as e: yield f"Lỗi tải link: {e}", ""; return
            elif fp: f_path = fp; yield "Đang phân tích file...", ""
            else: yield "Vui lòng dán link hoặc tải file!", ""; return
            yield from transcribe_audio(None, f_path, mt, t, p, s, r, ctx)

        run_btn.click(fn=process, inputs=[url_input, file_input, ctx_input, m_tok, temp, tp, samp, rp], outputs=[raw_out, seg_out])
        stop_btn.click(fn=lambda: "Đang hủy...", outputs=[raw_out])
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1")
    parser.add_argument("--port", type=int, default=7888) # ĐỔI CỔNG ĐỂ NÉ TIẾN TRÌNH ẢO
    args = parser.parse_args()
    demo = create_gradio_interface(args.model_path)
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=True)

if __name__ == "__main__": main()
