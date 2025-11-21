"""Utilities for creating error responses."""

from http import HTTPStatus


def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: int | HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, object]:
    """Create a standardized error response dictionary."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": str(
                code or (status_code.value if isinstance(status_code, HTTPStatus) else status_code)
            ),
        }
    }
