from backoff import on_exception, expo
from flask import request, jsonify
from ratelimit import limits, RateLimitException

from . import app
from .transeleos import transeleos


@on_exception(expo, RateLimitException, max_tries=5)
@limits(calls=1, period=10)
@app.route('/store_translated_audio', methods=['GET'])
def gpu_translate_to_output_lang():
    # Get the video URL from the 'url' query parameter
    video_url = request.args.get('url')
    output_lang = request.args.get('language')

    if not video_url and not output_lang:
        return jsonify({'error': 'Missing "url" parameter'}), 400

    s3_url = transeleos(video_url, output_lang)

    return jsonify({"message": "success", "s3Url": s3_url}), 200