"""Test-environment shim for Windows loopback socketpair flakiness.

Windows binds for socketpair's AF_INET fallback can intermittently hit reserved
ephemeral ports (WinError 10013), which breaks anyio/TestClient portals. Use an
explicit loopback port outside the excluded ranges instead of port 0.
"""
import socket

_orig_socketpair = socket.socketpair
_base = None


def _pair_on_port(port):
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lsock.bind(("127.0.0.1", port))
        lsock.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        server, _ = lsock.accept()
        lsock.close()
        return server, client
    except Exception:
        lsock.close()
        raise


def _socketpair(*args, **kwargs):
    if args or kwargs:  # non-default family/type: defer to stdlib
        return _orig_socketpair(*args, **kwargs)
    for port in range(51200, 51400):
        try:
            return _pair_on_port(port)
        except OSError:
            continue
    return _orig_socketpair(*args, **kwargs)


socket.socketpair = _socketpair
