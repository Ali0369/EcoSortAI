"""Upload validation and decoding. Pure Pillow/NumPy (no web framework, no TensorFlow)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..core.config import Settings


class ImageValidationError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code, self.code, self.message = status_code, code, message
        super().__init__(message)


def decode_upload(raw: bytes, content_type: str | None, settings: Settings) -> np.ndarray:
    """Validate an uploaded image and return a float32 RGB array (H, W, 3) with values 0..255.

    Order of checks: empty -> size -> declared type -> real decode -> format -> pixel count.
    """
    if not raw:
        raise ImageValidationError(400, "empty_file", "The uploaded file is empty.")
    if len(raw) > settings.max_upload_bytes:
        raise ImageValidationError(413, "file_too_large", f"The image is larger than {settings.max_upload_mb:g} MB.")
    if content_type and content_type.split(";")[0].strip().lower() not in settings.allowed_content_types:
        raise ImageValidationError(415, "unsupported_type", "Use a JPG, PNG or WebP image.")
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_pixels
    try:
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                probe.verify()                       # cheap integrity check
            im = Image.open(io.BytesIO(raw))
            fmt = im.format
            if fmt not in settings.allowed_formats:
                raise ImageValidationError(415, "unsupported_type", "Use a JPG, PNG or WebP image.")
            if im.width * im.height > settings.max_pixels:
                raise ImageValidationError(413, "image_too_large", "The image has too many pixels.")
            im = ImageOps.exif_transpose(im)         # respect phone-camera rotation
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            longest = max(im.size)
            if longest > settings.max_side_before_downscale:
                scale = settings.max_side_before_downscale / longest
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            if min(im.size) < 8:
                raise ImageValidationError(400, "image_too_small", "The image is too small to classify.")
            return np.asarray(im, dtype="float32")
        except ImageValidationError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError) as exc:
            raise ImageValidationError(400, "invalid_image", "This file is not a valid image.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
