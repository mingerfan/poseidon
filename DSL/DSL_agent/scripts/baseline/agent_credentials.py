"""Explicit project-root .env loading; no shell evaluation or dotenv dependency."""
import os
from pathlib import Path
import stat

NAME = "DEEPSEEK_API_KEY"
FILENAME = ".env"
LIMIT = 4096
PROVIDERS = {
    "deepseek": NAME,
}


class CredentialError(ValueError):
    """Safe messages only; never include file contents or key values."""


def validate_key(value, key_name=NAME):
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise CredentialError(key_name + " is empty or invalid; edit .env.")
    if any(not 33 <= ord(c) <= 126 or c in "\\\"'`$\\\\" for c in value):
        raise CredentialError(key_name + " must be a literal token, not a shell expression.")
    return value


def parse_env(text, key_name=NAME):
    """Read the project's single supported DeepSeek credential."""
    allowed = frozenset(PROVIDERS.values())
    if key_name not in allowed:
        raise CredentialError("Unsupported credential variable.")
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or name not in allowed or name in values:
            raise CredentialError("Invalid or duplicate provider assignment in .env.")
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise CredentialError("Invalid quoting in .env.")
            value = value[1:-1]
        values[name] = validate_key(value, name) if value else ""
    if not values.get(key_name):
        raise CredentialError("Set " + key_name + " in the project-root .env before using --live.")
    return values[key_name]


def load_api_key(root, environ=None, *, provider="deepseek"):
    if provider not in PROVIDERS:
        raise CredentialError("Unsupported credential provider.")
    key_name = PROVIDERS[provider]
    env = os.environ if environ is None else environ
    if env.get(key_name):
        return validate_key(env[key_name], key_name)
    path = Path(root) / FILENAME
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            raise CredentialError(".env must be a small regular file, not a symlink.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise CredentialError(".env must be a regular file.")
            raw = stream.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise CredentialError(".env exceeds the size limit.")
        return parse_env(raw.decode("utf-8-sig"), key_name)
    except FileNotFoundError:
        raise CredentialError("Missing .env; set " + key_name + " locally.") from None
    except (OSError, UnicodeError):
        raise CredentialError("Cannot read .env as UTF-8; check file permissions and encoding.") from None
