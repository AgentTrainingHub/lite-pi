"""Expose the live TLS socket while handshaking so cancellation can interrupt it."""

import http.client
import ssl


class CancellableHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, *, timeout: float, context: ssl.SSLContext) -> None:
        super().__init__(host, timeout=timeout, context=context)
        self.tls_context = context

    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        if self.sock is None:
            raise RuntimeError("HTTPS connection has no socket after connect")
        self.sock = self.tls_context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.sock.do_handshake()
