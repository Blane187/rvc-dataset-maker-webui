import gradio as gr
import yt_dlp
import ffmpeg
import subprocess
import os
import numpy as np
import librosa
import soundfile as sf


def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
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


class Slicer:
    def __init__(
        self,
        sr,
        threshold=-40.0,
        min_length=5000,
        min_interval=300,
        hop_size=20,
        max_sil_kept=5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
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
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks


def download_and_process(dataset, mode, url, drive_path, audio_name):
    result_text = ""

    if dataset == "Drive":
        result_text += "Dataset is set to Drive. Skipping YouTube download.\n"
    elif dataset == "Youtube":
        # Install yt-dlp and ffmpeg
        os.system("pip install yt_dlp")
        os.system("pip install ffmpeg")

        if not os.path.exists("youtubeaudio"):
            os.mkdir("youtubeaudio")

        # Download YouTube WAV
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "outtmpl": f"youtubeaudio/{audio_name}.wav",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        result_text += (
            f"Downloaded audio from YouTube to youtubeaudio/{audio_name}.wav\n"
        )

    # Install Demucs
    os.system("python3 -m pip install -U demucs")

    # Separate Audio using Demucs
    if dataset == "Drive":
        command = f"demucs --two-stems=vocals {drive_path}"
    elif dataset == "Youtube":
        command = f"demucs --two-stems=vocals youtubeaudio/{audio_name}.wav"

    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    result_text += result.stdout.decode() + "\n"

    # Copy results to Google Drive (simulated local directory for this example)
    if not os.path.exists(f"/content/drive/MyDrive/audio/{audio_name}"):
        os.makedirs(f"/content/drive/MyDrive/audio/{audio_name}")

    if dataset == "Youtube":
        os.system(
            f"cp -r /content/separated/htdemucs/{audio_name}/* /content/drive/MyDrive/audio/{audio_name}"
        )
        os.system(
            f"cp -r youtubeaudio/{audio_name}.wav /content/drive/MyDrive/audio/{audio_name}"
        )

    result_text += f"Separated audio files have been copied to /content/drive/MyDrive/audio/{audio_name}\n"

    # Split the Audio if required
    if mode == "Separate":
        result_text += "Mode is set to Separate. Skipping splitting.\n"
    elif mode == "Splitting":
        os.system("pip install numpy")
        os.system("pip install librosa")
        os.system("pip install soundfile")

        if not os.path.exists(f"dataset/{audio_name}"):
            os.mkdir(f"dataset/{audio_name}")

        audio, sr = librosa.load(
            f"/content/separated/htdemucs/{audio_name}/vocals.wav", sr=None, mono=False
        )
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=500,
            hop_size=10,
            max_sil_kept=500,
        )
        chunks = slicer.slice(audio)
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            sf.write(f"dataset/{audio_name}/split_{i}.wav", chunk, sr)

        os.system(f"mkdir -p /content/drive/MyDrive/dataset/{audio_name}")
        os.system(
            f"cp -r dataset/{audio_name}/* /content/drive/MyDrive/dataset/{audio_name}"
        )

        result_text += (
            f"Audio files have been split and saved to dataset/{audio_name}\n"
        )

    return result_text


with gr.Blocks(title="RVCDMFGC") as demo:
    gr.Markdown(" RVC DATASET MAKER FOR GOOGLE COLAB")
    with gr.Row():
        dataset = gr.Dropdown(choices=["Youtube", "Drive"], label="Dataset")
        mode = gr.Dropdown(choices=["Separate", "Splitting"], label="Mode")
    url = gr.Textbox(label="YouTube URL (if YouTube selected)")
    drive_path = gr.Textbox(label="Drive Path (if Drive selected)")
    audio_name = gr.Textbox(label="Audio Name")

    output = gr.Textbox(label="Output")

    def process(dataset, mode, url, drive_path, audio_name):
        return download_and_process(dataset, mode, url, drive_path, audio_name)

    submit = gr.Button("Submit")
    submit.click(
        process, inputs=[dataset, mode, url, drive_path, audio_name], outputs=output
    )

demo.launch(share=True,debug=True)
