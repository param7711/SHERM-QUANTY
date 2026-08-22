"""
Wire protocol between the Python agents and the MQL5 Expert Advisor.

Newline-delimited JSON over TCP. Chosen over a binary format because it is
inspectable with netcat while debugging a live terminal, and over a file
drop because sockets give backpressure and immediate failure rather than
silent staleness.

Every request carries a client-generated request_id. This is the core
safety property of the whole bridge: the EA records which ids it has
already executed and replays the stored result for a duplicate instead of
acting twice. Without it, a reply lost in transit leaves the caller unable
to distinguish "order filled" from "order never arrived", and neither
retrying nor not-retrying is safe.
"""

import json
import uuid

PROTOCOL_VERSION = 1

# --- Commands ------------------------------------------------------------
CMD_PING           = 'ping'
CMD_ACCOUNT        = 'account'         # balance, equity, margin
CMD_POSITIONS      = 'positions'       # open positions — the source of truth
CMD_QUOTE          = 'quote'           # bid/ask/spread for one symbol
CMD_OPEN           = 'open'            # open with stop attached, atomically
CMD_CLOSE          = 'close'
CMD_MODIFY_STOP    = 'modify_stop'     # 2.5R breakeven move, trailing

ALL_COMMANDS = {
    CMD_PING, CMD_ACCOUNT, CMD_POSITIONS, CMD_QUOTE,
    CMD_OPEN, CMD_CLOSE, CMD_MODIFY_STOP,
}

# Commands that change broker state. Only these need idempotency
# protection; reads are naturally safe to repeat.
MUTATING_COMMANDS = {CMD_OPEN, CMD_CLOSE, CMD_MODIFY_STOP}

# --- Status codes --------------------------------------------------------
STATUS_OK          = 'ok'
STATUS_ERROR       = 'error'
STATUS_REJECTED    = 'rejected'        # broker refused (bad stop level, margin)
STATUS_DUPLICATE   = 'duplicate'       # replayed result for a known request_id


class BridgeError(Exception):
    """Transport or protocol failure. The command may or may not have run."""


class BridgeRejected(Exception):
    """Broker actively refused the command. It definitively did not run."""


def new_request_id() -> str:
    return str(uuid.uuid4())


def encode(message: dict) -> bytes:
    """Serialise one message. Newline terminates the frame."""
    return (json.dumps(message, separators=(',', ':')) + '\n').encode('utf-8')


def decode(line: bytes) -> dict:
    """Parse one frame. Raises BridgeError on malformed input."""
    try:
        return json.loads(line.decode('utf-8').strip())
    except (ValueError, UnicodeDecodeError) as e:
        raise BridgeError(f"malformed frame: {e}") from e


def build_request(command: str, request_id: str = None, **params) -> dict:
    if command not in ALL_COMMANDS:
        raise ValueError(f"unknown command {command!r}")
    return {
        'v':          PROTOCOL_VERSION,
        'command':    command,
        'request_id': request_id or new_request_id(),
        'params':     params,
    }


def build_response(request_id: str, status: str, **data) -> dict:
    return {
        'v':          PROTOCOL_VERSION,
        'request_id': request_id,
        'status':     status,
        'data':       data,
    }
