"""Thread-safe CSV and JSON telemetry persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from threading import RLock

from pydantic import ValidationError

from .models import ConversationLog, ConversationLogDocument, ToolCallEvent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOL_CALLS_PATH = PROJECT_ROOT / "artifacts" / "logs" / "tool_calls.csv"
DEFAULT_CONVERSATIONS_PATH = (
    PROJECT_ROOT / "artifacts" / "logs" / "conversation_logs.json"
)
MAX_CONVERSATION_LOGS = 5_000
TOOL_CALL_COLUMNS = (
    "timestamp",
    "customer",
    "tool",
    "confidence",
    "latency",
    "retry",
    "status",
    "loop_count",
    "reason",
)

_circuit_lock = RLock()
_circuit_states: dict[str, str] = {}


def record_circuit_state(name: str, state: str) -> None:
    """Update the process-wide circuit snapshot included in request logs."""

    with _circuit_lock:
        _circuit_states[name] = state


def circuit_state_snapshot() -> dict[str, str]:
    with _circuit_lock:
        return dict(_circuit_states)


class ObservabilityStore:
    """Append tool CSV events and atomically update bounded conversation JSON."""

    def __init__(
        self,
        tool_calls_path: str | Path = DEFAULT_TOOL_CALLS_PATH,
        conversation_logs_path: str | Path = DEFAULT_CONVERSATIONS_PATH,
    ) -> None:
        self.tool_calls_path = Path(tool_calls_path)
        self.conversation_logs_path = Path(conversation_logs_path)
        self._lock = RLock()

    def ensure_files(self) -> None:
        """Create empty telemetry files with stable schemas."""

        with self._lock:
            self.tool_calls_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.tool_calls_path.exists():
                with self.tool_calls_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as csv_file:
                    csv.DictWriter(
                        csv_file,
                        fieldnames=TOOL_CALL_COLUMNS,
                    ).writeheader()

            self.conversation_logs_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.conversation_logs_path.exists():
                self._write_conversations_unlocked(ConversationLogDocument())

    def log_tool_call(self, event: ToolCallEvent) -> None:
        """Append one tool event using the exact dashboard CSV columns."""

        with self._lock:
            self.ensure_files()
            row = event.model_dump(mode="json")
            with self.tool_calls_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                csv.DictWriter(
                    csv_file,
                    fieldnames=TOOL_CALL_COLUMNS,
                ).writerow(row)

    def _read_conversations_unlocked(self) -> ConversationLogDocument:
        if not self.conversation_logs_path.exists():
            return ConversationLogDocument()
        try:
            payload = json.loads(
                self.conversation_logs_path.read_text(encoding="utf-8")
            )
            return ConversationLogDocument.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RuntimeError(f"Conversation telemetry is invalid: {exc}") from exc

    def _write_conversations_unlocked(
        self,
        document: ConversationLogDocument,
    ) -> None:
        temporary_path = self.conversation_logs_path.with_suffix(
            self.conversation_logs_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            document.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.conversation_logs_path)

    def log_conversation(self, event: ConversationLog) -> None:
        """Persist every request while bounding long-running local log growth."""

        with self._lock:
            self.ensure_files()
            document = self._read_conversations_unlocked()
            document.requests = [
                *document.requests,
                event,
            ][-MAX_CONVERSATION_LOGS:]
            self._write_conversations_unlocked(document)
