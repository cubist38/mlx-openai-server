"""Network utility functions."""

from __future__ import annotations

from contextlib import suppress
import ipaddress
import socket

from ..const import DEFAULT_BIND_HOST


def is_port_available(host: str | None = None, port: int | None = None) -> bool:
    """Check if a port is available for binding on the specified host.

    Parameters
    ----------
    host : str, optional
        The host to check. If None, defaults to DEFAULT_BIND_HOST.
    port : int
        The port to check.

    Returns
    -------
    bool
        True if the port is available, False otherwise.

    Raises
    ------
    ValueError
        If port is None.
    """
    if port is None:
        raise ValueError("port must be specified")

    if host is None:
        host = DEFAULT_BIND_HOST

    family = socket.AF_INET6 if _is_ipv6_host(host) else socket.AF_INET
    bind_host = _normalize_host_for_binding(host, family)
    sock: socket.socket | None = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_address = (bind_host, port, 0, 0) if family == socket.AF_INET6 else (bind_host, port)
        sock.bind(bind_address)
    except OSError:
        return False
    finally:
        if sock is not None:
            with suppress(Exception):
                sock.close()
    return True


def _is_ipv6_host(host: str) -> bool:
    """Check if the host string represents an IPv6 address.

    Parameters
    ----------
    host : str
        Host string to inspect. May include surrounding brackets (``[::1]``).

    Returns
    -------
    bool
        True if the host parses as an IPv6 address, False otherwise.
    """
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        addr = ipaddress.ip_address(value)
        return isinstance(addr, ipaddress.IPv6Address)
    except ValueError:
        return False


def _normalize_host_for_binding(host: str, family: int) -> str:
    """Normalize a host string for socket binding.

    Parameters
    ----------
    host : str
        Host string provided by user or configuration.
    family : int
        Socket address family (``socket.AF_INET`` or ``socket.AF_INET6``).

    Returns
    -------
    str
        Host string adjusted for binding. For IPv6 this strips surrounding
        brackets when present; for empty values returns a suitable wildcard
        address for the requested family.
    """
    value = host.strip()
    if not value:
        return "::" if family == socket.AF_INET6 else DEFAULT_BIND_HOST
    if family == socket.AF_INET6 and value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    if family == socket.AF_INET6:
        return value
    if value == "::":
        return DEFAULT_BIND_HOST
    return value
