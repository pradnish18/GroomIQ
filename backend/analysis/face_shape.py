def detect_face_shape():
    return "Oval"

"""
Face Shape Detection Module
GroomIQ v1

Currently uses deterministic analysis placeholders.
Later this file can be upgraded to MediaPipe Face Mesh.
"""

from typing import Dict


FACE_SHAPES = [
    "Oval",
    "Round",
    "Square",
    "Heart",
    "Diamond",
    "Rectangle"
]


BEARD_MAPPING = {
    "Oval": "Full Beard",
    "Round": "Goatee",
    "Square": "Short Stubble",
    "Heart": "Van Dyke",
    "Diamond": "Boxed Beard",
    "Rectangle": "Balbo"
}


STYLE_MAPPING = {
    "Oval": [
        "Pompadour",
        "Quiff",
        "Textured Crop"
    ],
    "Round": [
        "High Fade",
        "Side Part",
        "Spiky Hair"
    ],
    "Square": [
        "Crew Cut",
        "Buzz Cut",
        "French Crop"
    ],
    "Heart": [
        "Fringe",
        "Textured Quiff",
        "Side Sweep"
    ],
    "Diamond": [
        "Messy Fringe",
        "Classic Taper",
        "Textured Top"
    ],
    "Rectangle": [
        "Side Part",
        "Low Fade",
        "Brush Up"
    ]
}


def detect_face_shape(image=None) -> str:
    """
    GroomIQ v1 placeholder.

    Future versions should:
    1. Use MediaPipe Face Mesh
    2. Calculate facial ratios
    3. Classify face shape dynamically

    For now returns a stable value so the
    recommendation pipeline can run.
    """

    return "Oval"


def get_face_profile(face_shape: str) -> Dict:
    """
    Returns additional metadata for UI rendering.
    """

    return {
        "face_shape": face_shape,
        "recommended_beard": BEARD_MAPPING.get(
            face_shape,
            "Short Stubble"
        ),
        "recommended_styles": STYLE_MAPPING.get(
            face_shape,
            ["Classic Cut"]
        )
    }