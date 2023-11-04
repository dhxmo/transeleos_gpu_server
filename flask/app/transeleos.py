import concurrent.futures
import io
import logging
import os
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import boto3
import nltk
import numpy as np
import scipy
import torch
import whisper
import youtube_dl
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoProcessor, BarkModel

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
        _, audio_file = extract_audio(video_url=video_url, video_id=video_id)

        print("translating")
        final_trans = translate_to_output_lang(audio_file, output_lang)

        print("tts-ing")
        mp3_path, s3_url = bark_stt(video_id=video_id, input_text=final_trans,
                                    output_lang=output_lang, s3_object_key=s3_object_key)

        try:
            print("deleting local files")
            delete_file(audio_file)
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

    model = whisper.load_model("v2-large", device=device)
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

    prompt = """Transcribe the audio and include the following for higher accuracy:
            if a man is speaking add '[MAN]' at the beginning of the sentence.
            if a woman is speaking add '[WOMAN]' at the beginning of the sentence.
            add '[laughter]' if there is laughter in the track.
            add '[laughs]' if the main speaker laughs in the track.
            add '[sighs]' if there is a sigh in the track.
            add '[music]' if there is a melody in the track
            add '[gasps]' if someone gasps in the track.
            add '[clears throat]' if someone clears throat in the track.
            add '—' or '...' if someone hesitates in the track.
            add '♪'  if there are any song lyrics in the track.
            capitalize a word if a word has been emphasized in the track.
            """

    result = model.transcribe(temp_name, fp16=True, initial_prompt=prompt)
    text = result['text']

    # Translate the text
    translated_text = GoogleTranslator(source='auto', target=output_lang).translate(text=text)

    # Clean up the temporary file
    os.remove(temp_name)

    return translated_text


def bark_stt(video_id, input_text, output_lang, s3_object_key):
    # Initialize the processor and model
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Define the script to be processed
    script = input_text.replace("\n", " ").strip()

    # Tokenize the script into sentences using nltk
    sentences = nltk.sent_tokenize(script)

    # Initialize an empty list to store audio arrays
    audio_arrays = []

    voices_map = {
        "en": "v2/en_speaker_6",
        "zh": "v2/zh_speaker_0",
        "fr": "v2/fr_speaker_3",
        "de": "v2/de_speaker_4",
        "hi": "v2/hi_speaker_6",
        "it": "v2/it_speaker_8",
        "ja": "v2/ja_speaker_2",
        "po": "v2/pl_speaker_2",
        "pt": "v2/pt_speaker_2",
        "ru": "v2/ru_speaker_0",
        "es": "v2/es_speaker_0",
        "tr": "v2/tr_speaker_3"
    }

    voice = voices_map[output_lang]

    # Process each sentence in parallel
    def process_sentence(sentence):
        inputs = processor(sentence, voice_preset=voice)
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array

    with concurrent.futures.ThreadPoolExecutor() as executor:
        audio_arrays = list(executor.map(process_sentence, sentences))

    # Concatenate the audio arrays
    final_audio_array = np.concatenate(audio_arrays)

    # Get the sample rate
    sample_rate = model.generation_config.sample_rate

    try:
        # Create an in-memory WAV file
        wav_io = io.BytesIO()
        scipy.io.wavfile.write(wav_io, rate=sample_rate, data=final_audio_array)

        # Convert the WAV audio to MP3 using pydub
        audio_segment = AudioSegment.from_wav(wav_io)
        mp3_path = os.path.join('output', f'{video_id}_{output_lang}.mp3')
        audio_segment.export(mp3_path, format="mp3")

        # Close and delete the in-memory WAV file
        wav_io.close()

        try:
            s3.upload_file(mp3_path, Config.S3_BUCKET_NAME, s3_object_key)
            s3_url = f'https://{Config.S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_object_key}'
            return mp3_path, s3_url
        except Exception as e:
            logging.error(f'Failed to upload audio to S3: {str(e)}')
    except Exception as e:
        logging.error("error writing file:", str(e))
