import bcrypt
import jwt
import os
from datetime import datetime, timedelta, UTC

SECRET_KEY = os.environ.get(
    "JWT_SECRET_KEY",
    "d9d2b2b875caf593ea705e03529f105a1f6065c6019547a542cd67e44a40b29e"
)


def hash_password(password):
    """
    Hash plain text password
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password, password_hash):
    """
    Verify password against stored hash
    """
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )


def generate_token(user_id):
    """
    Generate JWT token
    """

    payload = {
        "user_id": user_id,
        "exp": datetime.now(UTC) + timedelta(days=7),
        "iat": datetime.now(UTC)
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm="HS256"
    )


def verify_token(token):
    """
    Verify JWT token
    """

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"]
        )

        return payload

    except jwt.ExpiredSignatureError:
        return None

    except jwt.InvalidTokenError:
        return None