"""App logging for the apps in ~/claude: the journal as before, plus structured logs in Graylog.

    import applog
    log = applog.setup("kidsai")                          # once, at startup
    log.info("comic ready", extra={"kid": "Ava", "ms": 812})   # extra= fields become searchable Graylog fields
    logging.getLogger(__name__).warning(...)              # anywhere

    class Handler(applog.RequestLog, BaseHTTPRequestHandler): ...   # one structured access log per request

Journal lines look like `WARNING kidsai: ...` (same format as Agentora's logsetup). Every record also goes to Graylog
as GELF over UDP with app, logger, file, line, thread and any extra= fields; exceptions carry their traceback. Access
logs (logger "<app>.access": method, path, status, duration_ms, client) go to Graylog only, so the journal stays quiet.
UDP is fire-and-forget: Graylog being down or slow never blocks or breaks the app.

Settings (environment): LOG_LEVEL (default INFO), APPLOG_GELF_HOST / APPLOG_GELF_PORT (127.0.0.1:12201),
APPLOG_GELF=0 turns Graylog off, APPLOG_ACCESS=0 turns access logs off.
Uncaught exceptions, in the main thread or any other, are logged as ERROR with their traceback.

Standard library only. The canonical copy lives in ~/claude/monitoring/applog/; apps keep a copy next to their code.
"""
import itertools, json, logging, os, re, socket, sys, threading, time, traceback, urllib.parse
from pathlib import Path

FMT = "%(levelname)s %(name)s: %(message)s"
PEERS = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "agentora-peers"     # written by tools/tailnet-relay.py
_SYSLOG = {logging.CRITICAL: 2, logging.ERROR: 3, logging.WARNING: 4, logging.INFO: 6, logging.DEBUG: 7}
_STD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "console"}
_KEY = re.compile(r"[^\w.\-]")
_app = None


class GelfUdpHandler(logging.Handler):
    """Send each record to Graylog as one GELF 1.1 message over UDP (chunked when it's bigger than a datagram)."""
    CHUNK = 8000          # payload bytes per chunk; Graylog accepts up to 8192 per datagram, 128 chunks per message

    def __init__(self, app, host="127.0.0.1", port=12201):
        super().__init__()
        self.app, self.addr = app, (host, int(port))
        self.host = socket.gethostname()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ids = itertools.count(int.from_bytes(os.urandom(4), "big") << 32)

    def gelf(self, record):
        msg = record.getMessage()
        full = msg
        if record.exc_info:
            full += "\n" + "".join(traceback.format_exception(*record.exc_info))
        elif record.stack_info:
            full += "\n" + record.stack_info
        g = {"version": "1.1", "host": self.host, "short_message": msg.split("\n", 1)[0][:1000] or "(empty)",
             "timestamp": round(record.created, 3), "level": _SYSLOG.get(record.levelno, 6),
             "_app": self.app, "_logger": record.name, "_level_name": record.levelname, "_file": record.pathname,
             "_line": record.lineno, "_func": record.funcName, "_thread": record.threadName, "_pid": record.process}
        if full != g["short_message"]:
            g["full_message"] = full
        if record.exc_info and record.exc_info[0]:
            g["_exception"] = record.exc_info[0].__name__
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, path, _, status = record.args          # uvicorn's own access log (Mealie, FastAPI apps)
            path = urllib.parse.urlsplit(str(path)).path
            g.update(short_message=f"{method} {path} {status}", _http_method=method, _http_path=path,
                     _http_status=int(status), _client=str(client).rsplit(":", 1)[0])
            g.pop("full_message", None)
        for k, v in vars(record).items():
            if k in _STD or k.startswith("_"):
                continue
            k = "_" + _KEY.sub("_", k)
            if k == "_id":
                k = "_id_"
            g[k] = v if isinstance(v, (int, float, bool)) or v is None else str(v)
        return json.dumps(g, ensure_ascii=False, default=str).encode()

    def emit(self, record):
        try:
            data = self.gelf(record)
            if len(data) <= 8192:
                self.sock.sendto(data, self.addr)
                return
            parts = [data[i:i + self.CHUNK] for i in range(0, len(data), self.CHUNK)][:128]
            mid = next(self.ids).to_bytes(8, "big")
            for n, part in enumerate(parts):
                self.sock.sendto(b"\x1e\x0f" + mid + bytes([n, len(parts)]) + part, self.addr)
        except Exception:
            pass          # never let logging break the app; the journal still has the line


class _NotAccess(logging.Filter):
    def filter(self, record):
        return getattr(record, "console", True)


def setup(app, level=None):
    """Configure the root logger for this process and return logging.getLogger(app)."""
    global _app
    root = logging.getLogger()
    if _app == app:
        return logging.getLogger(app)
    for h in list(root.handlers):
        root.removeHandler(h)
    level = level or getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root.setLevel(level)
    out = logging.StreamHandler(sys.stdout)                 # -> the systemd journal (it adds the timestamp)
    out.setFormatter(logging.Formatter(FMT))
    out.addFilter(_NotAccess())
    root.addHandler(out)
    if os.environ.get("APPLOG_GELF", "1") != "0":
        root.addHandler(GelfUdpHandler(app, os.environ.get("APPLOG_GELF_HOST", "127.0.0.1"),
                                       os.environ.get("APPLOG_GELF_PORT", "12201")))
    for noisy in ("urllib3", "asyncio", "PIL", "matplotlib", "numba", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    def excepthook(t, v, tb):
        if issubclass(t, KeyboardInterrupt):
            return sys.__excepthook__(t, v, tb)
        logging.getLogger(app).error("uncaught %s", t.__name__, exc_info=(t, v, tb))

    def thread_hook(a):
        if issubclass(a.exc_type, (SystemExit, BrokenPipeError, ConnectionResetError)):
            return
        logging.getLogger(app).error("uncaught %s in thread %s", a.exc_type.__name__, a.thread.name if a.thread else "?",
                                     exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    _app = app
    return logging.getLogger(app)


def real_ip(ip, port=None):
    """The device's own address. Connections through the tailnet relay arrive from 127.0.0.1; the relay notes each one's
    real client under the local port it connects from."""
    if ip in ("127.0.0.1", "::1") and port:
        try:
            return (PEERS / str(port)).read_text().strip() or ip
        except OSError:
            pass
    return ip


def event(logger, message, level=logging.INFO, console=True, **fields):
    """Log message with fields as searchable Graylog fields. Any field name is fine (names logging itself uses, like
    name or args, get an f_ prefix). console=False sends it to Graylog only."""
    lg = logger if isinstance(logger, logging.Logger) else logging.getLogger(logger)
    lg.log(level, "%s", message, extra={"console": console,
                                        **{("f_" + k if k in _STD else k): v for k, v in fields.items()}})


def access(method, path, status, duration_ms, client=None, **fields):
    """One structured access-log record (Graylog only). path should be the path alone: query strings can hold tokens."""
    if os.environ.get("APPLOG_ACCESS", "1") == "0":
        return
    lvl = logging.ERROR if isinstance(status, int) and status >= 500 else \
        logging.WARNING if isinstance(status, int) and status >= 400 and status not in (401, 404) else logging.INFO
    logging.getLogger(f"{_app or 'app'}.access").log(
        lvl, "%s %s %s %.0fms", method, path, status, duration_ms,
        extra={"console": False, "http_method": method, "http_path": path, "http_status": status,
               "duration_ms": round(duration_ms, 1), "client": client, **fields})


class ServerErrors:
    """Mix into a socketserver/http.server server: a crash in a request handler is logged with its traceback (a client
    that went away isn't an error)."""

    def handle_error(self, request, client_address):
        t = sys.exc_info()[0]
        if t and issubclass(t, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        logging.getLogger(f"{_app or 'app'}.http").error("request from %s failed", real_ip(*client_address[:2]),
                                                          exc_info=True)


def server(addr, handler):
    """A ThreadingHTTPServer that logs handler crashes (ServerErrors)."""
    from http.server import ThreadingHTTPServer

    class Server(ServerErrors, ThreadingHTTPServer):
        pass
    return Server(addr, handler)


class RequestLog:
    """Mix into a BaseHTTPRequestHandler subclass (before it): logs every request with applog.access.
    duration_ms runs from reading the request line to sending the status line, so streamed bodies aren't counted."""

    def parse_request(self):
        self._applog_t0 = time.monotonic()
        return super().parse_request()

    def log_request(self, code="-", size="-"):
        t0 = getattr(self, "_applog_t0", None)
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = str(code)
        access(getattr(self, "command", None) or "-", urllib.parse.urlsplit(getattr(self, "path", "") or "").path or "-",
               status, (time.monotonic() - t0) * 1000 if t0 else 0, real_ip(*self.client_address[:2]),
               user_agent=(self.headers.get("User-Agent") if getattr(self, "headers", None) else None))
