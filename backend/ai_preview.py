import os
import base64
import tempfile
import logging
import requests
import json

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '')
REPLICATE_MODEL = os.environ.get(
    'REPLICATE_PREVIEW_MODEL',
    'lucataco/face-to-styles'
)
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY', '')

HAIRSTYLE_PROMPTS = {
    "Quiff": "modern quiff hairstyle with volume on top and tapered sides",
    "Pompadour": "classic pompadour hairstyle with swept back volume",
    "Crew Cut": "short crew cut hairstyle, clean and tapered",
    "Buzz Cut": "very short buzz cut hairstyle, uniform length all over",
    "Faux Hawk": "edgy faux hawk hairstyle with raised center strip",
    "Slick Back": "slicked back hairstyle with polished finish",
    "Textured Crop": "short textured crop hairstyle with choppy top",
    "Side Part": "clean side part hairstyle, professional look",
    "Undercut": "undercut hairstyle with long top and shaved sides",
    "Fade": "fade haircut with gradual taper on sides",
    "Long Flow": "long flowing hair swept back",
    "Curly": "natural curly hairstyle, well-defined curls",
    "Wavy": "wavy hairstyle with natural texture",
    "Dreadlocks": "dreadlocks hairstyle, neat and clean",
    "Straight": "straight hairstyle, sleek and smooth",
    "Short": "short clean haircut, professional style",
    "Medium": "medium length hairstyle, versatile and styled",
    "Long": "long hairstyle, flowing and natural",
}

DEFAULT_PROMPT = "stylish modern hairstyle, professional portrait, high quality"


def generate_preview_replicate(image_base64, hairstyle, face_shape=None):
    if not REPLICATE_API_TOKEN:
        return None, "REPLICATE_API_TOKEN not configured. Set it in the environment."

    try:
        import replicate
    except ImportError:
        return None, "replicate package not installed. Run: pip install replicate"

    try:
        image_bytes = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        prompt = HAIRSTYLE_PROMPTS.get(hairstyle, DEFAULT_PROMPT)
        if face_shape and face_shape.lower() != 'oval':
            prompt += f", suited for {face_shape} face shape"

        model = REPLICATE_MODEL

        input_data = {
            "image": open(temp_path, "rb"),
            "prompt": prompt,
            "num_outputs": 1,
        }

        output = replicate.run(model, input=input_data)

        os.unlink(temp_path)

        result_url = None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str) and item.startswith('http'):
                    result_url = item
                    break
        elif isinstance(output, str) and output.startswith('http'):
            result_url = output
        elif hasattr(output, 'url'):
            result_url = output.url

        if isinstance(output, str) and not result_url:
            result_url = output

        if not result_url:
            if isinstance(output, list):
                result_url = str(output[0]) if output else None
            else:
                result_url = str(output)

        if result_url:
            img_resp = requests.get(result_url, timeout=30)
            if img_resp.status_code == 200:
                img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
                mime = img_resp.headers.get('content-type', 'image/png')
                return f"data:{mime};base64,{img_b64}", None

        return None, f"Could not get generated image URL from model output: {str(output)[:200]}"

    except Exception as e:
        logger.exception("Replicate preview failed")
        return None, str(e)


def generate_preview_stability(image_base64, hairstyle, face_shape=None):
    if not STABILITY_API_KEY:
        return None, "STABILITY_API_KEY not configured."

    try:
        image_bytes = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)

        prompt = HAIRSTYLE_PROMPTS.get(hairstyle, DEFAULT_PROMPT)
        if face_shape and face_shape.lower() != 'oval':
            prompt += f", suited for {face_shape} face shape"

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace",
            headers={
                "authorization": f"Bearer {STABILITY_API_KEY}",
                "accept": "image/*"
            },
            files={
                "image": ("image.jpg", image_bytes, "image/jpeg"),
            },
            data={
                "prompt": prompt,
                "search_prompt": "hair hairstyle",
                "output_format": "png",
            },
            timeout=60,
        )

        if response.status_code == 200:
            img_b64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/png;base64,{img_b64}", None
        else:
            return None, f"Stability AI error {response.status_code}: {response.text[:200]}"

    except Exception as e:
        logger.exception("Stability AI preview failed")
        return None, str(e)


def generate_preview(image_base64, hairstyle, face_shape=None):
    if REPLICATE_API_TOKEN:
        result, error = generate_preview_replicate(image_base64, hairstyle, face_shape)
        if result:
            return result, None
        logger.warning("Replicate failed, trying Stability: %s", error)

    if STABILITY_API_KEY:
        result, error = generate_preview_stability(image_base64, hairstyle, face_shape)
        if result:
            return result, None
        logger.warning("Stability failed: %s", error)

    return None, "No AI provider configured. Set REPLICATE_API_TOKEN or STABILITY_API_KEY."
