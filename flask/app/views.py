import logging

from flask import request, jsonify
from ratelimit import limits

from . import app
from .config import Config
from .transeleos import transeleos


@limits(calls=1, period=1)
@app.route('/gpu_translate_to_output_lang', methods=['GET'])
def gpu_translate_to_output_lang():
    client_ip = request.remote_addr

    # TODO: uncomment for prod
    # if client_ip == Config.ALLOWED_IP:
    #     # Get the video URL from the 'url' query parameter
    #     video_url = request.args.get('url')
    #     output_lang = request.args.get('language')
    #
    #     if not video_url and not output_lang:
    #         return jsonify({'error': 'Missing "url" parameter'}), 400
    #
    #     s3_url = transeleos(video_url, output_lang)
    #
    #     return s3_url
    # else:
    #     logging.error("invalid request")

    # Get the video URL from the 'url' query parameter
    video_url = request.args.get('url')
    output_lang = request.args.get('language')

    if not video_url and not output_lang:
        return jsonify({'error': 'Missing "url" parameter'}), 400

    s3_url = transeleos(video_url, output_lang)

    return s3_url
