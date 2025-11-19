"""Error response utilities for creating standardized API error messages."""

from http import HTTPStatus


def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: int | HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
    param: str | None = None,
    code: str | None = None,
):
    """Create a standardized error response dictionary.

    Parameters
    ----------
    message : str
        The error message.
    err_type : str, optional
        The error type, by default "internal_error".
    status_code : int | HTTPStatus, optional
        The HTTP status code, by default HTTPStatus.INTERNAL_SERVER_ERROR.
    param : str | None, optional
        The parameter that caused the error, by default None.
    code : str | None, optional
        The error code, by default None.

    Returns
    -------
    dict
        A dictionary containing the error response.
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
