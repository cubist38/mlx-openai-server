"""Utilities for creating error responses."""

from http import HTTPStatus


def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: int | HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, object]:
    """
    Create a standardized error response dictionary.

    Parameters
    ----------
    message : str
        The error message to include in the response.
    err_type : str, optional
        The type of error, by default "internal_error".
    status_code : int or HTTPStatus, optional
        The HTTP status code, by default HTTPStatus.INTERNAL_SERVER_ERROR.
    param : str or None, optional
        The parameter that caused the error, by default None.
    code : str or None, optional
        The error code, by default None.

    Returns
    -------
    dict[str, object]
        A dictionary containing the error response structure.
    """
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
