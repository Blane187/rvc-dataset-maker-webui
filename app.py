import gradio as gr
import yt_dlp
import numpy as np
import librosa
import soundfile as sf
import os
import zipfile


# Function to download audio from YouTube and save it as a WAV file
def download_youtube_audio(url, audio_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        "outtmpl": f'youtubeaudio/{audio_name}',  # Output template
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f'youtubeaudio/{audio_name}.wav'

# Function to calculate RMS
def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

# Slicer class
class Slicer:
    def __init__(self, sr, threshold=-40., min_length=5000, min_interval=300, hop_size=20, max_sil_kept=5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

# Function to slice and save audio chunks
def slice_audio(file_path, audio_name):
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    os.makedirs(f'dataset/{audio_name}', exist_ok=True)
    slicer = Slicer(sr=sr, threshold=-40, min_length=5000, min_interval=500, hop_size=10, max_sil_kept=500)
    chunks = slicer.slice(audio)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        sf.write(f'dataset/{audio_name}/split_{i}.wav', chunk, sr)
    return f"dataset/{audio_name}"

# Function to zip the dataset directory
def zip_directory(directory_path, audio_name):
    zip_file = f"dataset/{audio_name}.zip"
    os.makedirs(os.path.dirname(zip_file), exist_ok=True)  # Ensure the directory exists
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=directory_path)
                zipf.write(file_path, arcname)
    return zip_file


# Gradio interface
def process_audio(url, audio_name):
    file_path = download_youtube_audio(url, audio_name)
    dataset_path = slice_audio(file_path, audio_name)
    zip_file = zip_directory(dataset_path, audio_name)
    return zip_file

with gr.Blocks(title="rvc dataset maker webui") as demo:
    gr.Markdown("# <div style='text-align: center;'> RVC DATASET MAKER</div>")
    with gr.Tabs():
        with gr.TabItem("make dataset"):
            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL", value="https://youtu.be/ZdoiTX1f1tU?si=hZ96tryqaPIYfwg1")
            with gr.Row():
                audio_name_input = gr.Textbox(label="Audio Name", value="Amy")
            with gr.Row():
                result_output = gr.File(label="Download Sliced Audio Zip")
            with gr.Row():
                run_button = gr.Button("Slice 👾 Audio")
                run_button.click(fn=process_audio, inputs=[url_input, audio_name_input], outputs=[result_output])
                push_hub.click(fn=push_to_hub, inputs=[result_output, hf_token_input, repo_id_input], outputs=[upload_output])
        with gr.TabItem("note"):
            with gr.Row():
                gr.Markdown(
                    f"""
                    
                   # huggingface version:
                  [click here](https://huggingface.co/spaces/Hev832/RVC-Dataset-Maker)
                    """
                )

    with gr.Tabs():
        gr.Markdown(f"### <div style='text-align: center;'>made with ❤ by <a href='https://huggingface.co/Hev832'>Hev832</a></div>")



demo.launch(
    debug=True,
    share=True,
)
