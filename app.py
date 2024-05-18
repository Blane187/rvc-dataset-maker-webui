import gradio as gr
#from __future__ import unicode_literals
import yt_dlp
import ffmpeg
import subprocess
import numpy as np
import librosa
import soundfile

# Function to download audio from YouTube
def download_audio(url, audio_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        "outtmpl": f'youtubeaudio/{audio_name}',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Function to separate vocals using demucs
def separate_vocals(audio_path, audio_name):
    command = f"demucs --two-stems=vocals {audio_path}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode())
    subprocess.run(f"!mkdir -p /content/audio/{audio_name}", shell=True)
    subprocess.run(f"!cp -r /content/separated/htdemucs/{audio_name}/* /content/audio/{audio_name}", shell=True)
    subprocess.run(f"!cp -r /content/youtubeaudio/{audio_name}.wav /content/audio/{audio_name}", shell=True)

# RMS function from librosa
def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    target_axis = axis + 1 if axis >= 0 else axis - 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

# Slicer class to split audio
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

def process_audio(mode, dataset, url, drive_path, audio_name):
    if dataset == "Drive":
        print("Dataset is set to Drive. Skipping this section")
    elif dataset == "Youtube":
        download_audio(url, audio_name)
    
    audio_input = f"/content/youtubeaudio/{audio_name}.wav"
    
    if dataset == "Drive":
        command = f"demucs --two-stems=vocals {drive_path}"
    elif dataset == "Youtube":
        command = f"demucs --two-stems=vocals {audio_input}"
    
    subprocess.run(command.split(), stdout=subprocess.PIPE)
    
    if mode == "Splitting":
        audio, sr = librosa.load(f'/content/separated/htdemucs/{audio_name}/vocals.wav', sr=None, mono=False)
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=500,
            hop_size=10,
            max_sil_kept=500
        )
        chunks = slicer.slice(audio)
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            soundfile.write(f'/content/dataset/{audio_name}/split_{i}.wav', chunk, sr)
    
    return f"Processing complete for {audio_name}"

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Dataset Maker")
        mode = gr.Dropdown(choices=["Splitting", "Separate"], label="Mode")
        dataset = gr.Dropdown(choices=["Youtube", "Drive"], label="Dataset")
        url = gr.Textbox(label="URL")
        drive_path = gr.Textbox(label="Drive Path")
        audio_name = gr.Textbox(label="Audio Name")
        output = gr.Textbox(label="Output")
        process_button = gr.Button("Process")
    
    process_button.click(
        process_audio,
        inputs=[mode, dataset, url, drive_path, audio_name],
        outputs=[output]
    )

demo.launch(share=True)