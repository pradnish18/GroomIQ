BEARD_RULES = {
    "Oval": "Full Beard",
    "Round": "Goatee",
    "Square": "Stubble",
    "Heart": "Van Dyke",
    "Diamond": "Short Boxed Beard"
}

def recommend_beard(face_shape):
    return BEARD_RULES.get(face_shape, "Stubble")
"""
Beard Recommendation Engine
GroomIQ v1

Uses face-shape-based recommendations.
Future versions can incorporate:
- Face landmarks
- Beard density detection
- Age-based recommendations
- User preferences
"""

from typing import Dict, List


BEARD_RULES = {
    "Oval": {
        "primary": "Full Beard",
        "alternatives": [
            "Short Boxed Beard",
            "Heavy Stubble",
            "Corporate Beard"
        ]
    },

    "Round": {
        "primary": "Goatee",
        "alternatives": [
            "Van Dyke",
            "Extended Goatee",
            "Anchor Beard"
        ]
    },

    "Square": {
        "primary": "Short Stubble",
        "alternatives": [
            "Circle Beard",
            "Balbo",
            "Short Boxed Beard"
        ]
    },

    "Heart": {
        "primary": "Van Dyke",
        "alternatives": [
            "Goatee",
            "Anchor Beard",
            "Light Stubble"
        ]
    },

    "Diamond": {
        "primary": "Short Boxed Beard",
        "alternatives": [
            "Corporate Beard",
            "Balbo",
            "Heavy Stubble"
        ]
    },

    "Rectangle": {
        "primary": "Balbo",
        "alternatives": [
            "Circle Beard",
            "Short Beard",
            "Heavy Stubble"
        ]
    }
}


def recommend_beard(face_shape: str) -> str:
    """
    Returns the best beard style.
    Used directly by app.py.
    """

    return BEARD_RULES.get(
        face_shape,
        {"primary": "Short Stubble"}
    )["primary"]



def get_beard_profile(face_shape: str) -> Dict:
    """
    Returns detailed beard recommendations.
    Useful for dashboard rendering.
    """

    profile = BEARD_RULES.get(
        face_shape,
        {
            "primary": "Short Stubble",
            "alternatives": []
        }
    )

    return {
        "face_shape": face_shape,
        "recommended_beard": profile["primary"],
        "alternative_styles": profile["alternatives"]
    }



def get_all_recommendations(face_shape: str) -> List[str]:
    """
    Returns all beard recommendations.
    """

    profile = BEARD_RULES.get(
        face_shape,
        {
            "primary": "Short Stubble",
            "alternatives": []
        }
    )

    return [profile["primary"]] + profile["alternatives"]