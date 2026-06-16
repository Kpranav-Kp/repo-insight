import logging

from django.conf import settings
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

logger = logging.getLogger(__name__)


class CookieJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        header = self.get_header(request)
        if header is not None:
            try:
                return super().authenticate(request)
            except (InvalidToken, TokenError) as e:
                logger.debug(f"Header token validation failed: {e}")
                return None
        access_token = request.COOKIES.get(settings.SIMPLE_JWT["AUTH_COOKIE"])
        if not access_token:
            return None
        try:
            validated_token = self.get_validated_token(access_token)
        except (InvalidToken, TokenError) as e:
            logger.debug(f"Cookie token validation failed: {e}")
            return None
        try:
            user = self.get_user(validated_token)
        except Exception as e:
            logger.debug(f"User retrieval failed: {e}")
            return None
        return user, validated_token
