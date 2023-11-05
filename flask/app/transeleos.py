import concurrent.futures
import logging
import os
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import boto3
import torch
import whisper
import youtube_dl
from TTS.api import TTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence

from .config import Config

s3 = boto3.client('s3',
                  aws_access_key_id=Config.S3_ACCESS_KEY,
                  aws_secret_access_key=Config.S3_SECRET_ACCESS_KEY)


def transeleos(video_url, output_lang):
    # Parse the URL to extract the video ID
    parsed_url = urlparse(video_url)
    query_parameters = parse_qs(parsed_url.query)
    video_id = query_parameters.get('v', [''])[0]

    if not video_id:
        logging.error("Video ID not found in the URL.")
        return

    # check if mp3 of the translated audio already exists
    s3_object_key = f'translated_audio/{video_id}/{output_lang}/{video_id}.mp3'
    try:
        print("checking if final file already exists")
        s3.head_object(Bucket=Config.S3_BUCKET_NAME, Key=s3_object_key)
        s3_url = f'https://{Config.S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_object_key}'
        print("returning s3 url", s3_url)
    # if not then translate and then upload to s3
    except Exception as e:
        print("final file desn't exist, extracting audio")

        # extract audio from youtube video. if s3 already exists, return s3_url
        _, yt_audio_file = extract_audio(video_url=video_url, video_id=video_id)

        print("translating")
        final_trans = translate_to_output_lang(yt_audio_file, output_lang)
        print("final_trans", final_trans)

        print("tts-ing")
        mp3_path, s3_url = coqui_tts(video_id=video_id, input_audio_file=yt_audio_file,
                                     final_translation=final_trans, output_lang=output_lang,
                                     s3_object_key=s3_object_key)
        try:
            print("deleting local files")
            delete_file(yt_audio_file)
            delete_file(mp3_path)
        except Exception as e:
            logging.error("error deleting files: ", str(e))
    return s3_url


def delete_file(file_to_be_deleted):
    try:
        if os.path.exists(file_to_be_deleted):
            os.remove(file_to_be_deleted)
    except Exception as e:
        logging.error("error deleting files: ", str(e))


def extract_audio(video_url, video_id):
    s3_url = ''
    audio_path = ''

    s3_object_key = f'translated_audio/{video_id}/{video_id}.wav'  # Object key in S3

    # if audio for video id already in s3, fetch to local
    try:
        print("checking if downloaded file already exists or not")
        s3.head_object(Bucket=Config.S3_BUCKET_NAME, Key=s3_object_key)

        audio_path = os.path.join('output', f'{video_id}.wav')
        s3.download_file(Config.S3_BUCKET_NAME, s3_object_key, audio_path)

        s3_url = f'https://{Config.S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_object_key}'
    except Exception as e:
        print("doesnt exist, extracting and downloading")
        # Extract youtube audio from url
        is_audio_extracted = extract_audio_yt(video_url=video_url, video_id=video_id)

        # if audio successfully extracted, upload to s3
        if is_audio_extracted:
            audio_path = os.path.join('output', f'{video_id}.wav')
            try:
                s3.upload_file(audio_path, Config.S3_BUCKET_NAME, s3_object_key)
                s3_url = f'https://{Config.S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_object_key}'
            except Exception as e:
                s3_url = ''
                logging.error(f'Failed to upload audio to S3: {str(e)}')

    return s3_url, audio_path


def download_video(video_url, output_file):
    # Use subprocess to run youtube-dl to download the video
    cmd = [
        'youtube-dl',
        '--format', 'bestaudio/best',  # Download the best quality video
        '-o', output_file,  # Output file path
        video_url,  # Video URL
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        return False


def extract_audio_yt(video_url, video_id, output_dir='output'):
    # Ensure that the output directory exists and is deterministic
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Set the output file path and filename
    output_filename = f'{video_id}.wav'
    output_file = os.path.join(output_dir, output_filename)

    # Use subprocess to run ffmpeg with the desired options
    # Download the best quality audio to a temporary file
    temp_output_file = os.path.join(output_dir, f'{video_id}_temp.wav')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Change to WAV format
            'preferredquality': '192',  # You can adjust the quality as needed
        }],
        'outtmpl': temp_output_file,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Use FFmpeg to convert the temporary audio file
    cmd = [
        'ffmpeg',
        '-i', temp_output_file,
        '-vn', '-ac', '2', '-ar', '44100', '-b:a', '192k',
        output_file,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False
    finally:
        # Remove the temporary file
        os.remove(temp_output_file)


def translate_to_output_lang(audio_file, output_lang):
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model("base", device=device)
    # TODO: uncomment for prod
    # model = whisper.load_model("large-v2", device=device)
    model = model.to(device)

    # Split audio into non-silent chunks
    non_silent_chunks = split_on_silence(audio_file,
                                         seek_step=5,  # ms
                                         min_silence_len=1250,  # ms
                                         silence_thresh=-25,  # dB
                                         keep_silence=True)

    # Initialize an empty list to store futures for parallel processing
    chunk_futures = []

    # Process each chunk in parallel and store the futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for chunk in non_silent_chunks:
            future = executor.submit(process_chunk, model, chunk, output_lang)
            chunk_futures.append(future)

    # Wait for all futures to complete and get the translated texts
    translated_chunks = [future.result() for future in chunk_futures]
    final_translation = " ".join(translated_chunks)

    return final_translation


# Function to process each chunk
def process_chunk(model, chunk, output_lang):
    audio = AudioSegment.from_wav(chunk)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_name = temp.name
    audio.export(temp_name, format="wav")

    # TODO: uncomment for prod
    result = model.transcribe(temp_name, fp16=False)
    # result = model.transcribe(temp_name, fp16=True, initial_prompt=prompt)
    text = result['text']

    # Translate the text
    translated_text = GoogleTranslator(source='auto', target=output_lang).translate(text=text)

    # Clean up the temporary file
    os.remove(temp_name)

    return translated_text


def coqui_tts(video_id, input_audio_file, final_translation, output_lang, s3_object_key):
    mp3_path = os.path.join('output', f'{video_id}_{output_lang}.mp3')

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_models = {
        "bg": "tts_models/bg/cv/vits",
        "cs": "tts_models/cs/cv/vits",
        "da": "tts_models/da/cv/vits",
        "et": "tts_models/et/cv/vits",
        "en": "tts_models/en/ek1/tacotron2",
        "es": "tts_models/es/mai/tacotron2-DDC",
        "fr": "tts_models/fr/mai/tacotron2-DDC",
        "zh-CN": "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        "nl": "tts_models/nl/mai/tacotron2-DDC",
        "de": "tts_models/de/thorsten/tacotron2-DDC",
        "ja": "tts_models/ja/kokoro/tacotron2-DDC",
        "tr": "tts_models/tr/common-voice/glow-tts",
        "it": "tts_models/it/mai_male/glow-tts",
        "hu": "tts_models/hu/css10/vits",
        "el": "tts_models/el/cv/vits",
        "fi": "tts_models/fi/css10/vits",
        "hr": "tts_models/hr/cv/vits",
        "lt": "tts_models/lt/cv/vits",
        "lv": "tts_models/lv/cv/vits",
        "mt": "tts_models/mt/cv/vits",
        "ro": "tts_models/ro/cv/vits",
        "sk": "tts_models/sk/cv/vits",
        "sl": "tts_models/sl/cv/vits",
        "sv": "tts_models/sv/cv/vits",
        "fa": "tts_models/fa/custom/glow-tts"
    }
    lang_model = tts_models[output_lang]

    tts = TTS(lang_model).to(device)

    try:
        # voice cloning
        tts.tts_with_vc_to_file(
            final_translation,
            speaker_wav=input_audio_file,
            file_path=mp3_path
        )

        s3.upload_file(mp3_path, Config.S3_BUCKET_NAME, s3_object_key)
        s3_url = f'https://{Config.S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_object_key}'
        return mp3_path, s3_url
    except Exception as e:
        logging.error(f'Failed to upload audio to S3: {str(e)}')
