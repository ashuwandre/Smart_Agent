"""Atomic JSON persistence for bounded multi-turn conversation memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import ValidationError

from .models import (
    ConversationMemory,
    Interaction,
    MemoryContext,
    MemoryDocument,
    OfferMemory,
    RefundMemory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_PATH = (
    PROJECT_ROOT / "artifacts" / "memory" / "conversation_memory.json"
)
MAX_INTERACTIONS = 10
MAX_TECHNICAL_STEPS = 10


class ConversationMemoryStore:
    """Persist conversation state in one bounded, atomically replaced JSON file."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = Path(path)
        self._lock = RLock()

    @staticmethod
    def _conversation_id(value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("conversation_id must not be empty")
        if len(clean) > 200:
            raise ValueError("conversation_id must not exceed 200 characters")
        return clean

    @staticmethod
    def _customer_id(value: str | None) -> str | None:
        return value.strip().upper() if value else None

    def _read_unlocked(self) -> MemoryDocument:
        if not self.path.exists():
            return MemoryDocument()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return MemoryDocument.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            # Corrupted memory must fail closed instead of silently replacing
            # customer history with an empty document.
            raise RuntimeError(f"Conversation memory is invalid: {exc}") from exc

    def _write_unlocked(self, document: MemoryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            document.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        # os.replace semantics make each write atomic for readers on the same
        # filesystem; they observe either the old or complete new document.
        temporary_path.replace(self.path)

    def _memory(
        self,
        document: MemoryDocument,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> ConversationMemory:
        memory = document.conversations.get(conversation_id)
        if memory is None:
            memory = ConversationMemory(
                conversation_id=conversation_id,
                customer_id=customer_id,
            )
            document.conversations[conversation_id] = memory
        elif customer_id and memory.customer_id and customer_id != memory.customer_id:
            # A conversation cannot switch customers because that could leak one
            # customer's history into another customer's model context.
            raise ValueError(
                f"Conversation {conversation_id!r} already belongs to "
                f"{memory.customer_id}."
            )
        elif customer_id and memory.customer_id is None:
            memory.customer_id = customer_id
        return memory

    def get_memory(self, conversation_id: str) -> ConversationMemory:
        """Return persisted memory or an empty, non-persisted conversation."""

        clean_id = self._conversation_id(conversation_id)
        with self._lock:
            document = self._read_unlocked()
            memory = document.conversations.get(clean_id)
            return (
                memory.model_copy(deep=True)
                if memory
                else ConversationMemory(conversation_id=clean_id)
            )

    def get_context(self, conversation_id: str) -> MemoryContext:
        """Return only interactions approved for reuse by the planner."""

        memory = self.get_memory(conversation_id)
        return MemoryContext(
            conversation_id=memory.conversation_id,
            customer_id=memory.customer_id,
            previous_messages=[
                interaction
                for interaction in memory.interactions
                if interaction.include_in_context
            ],
            offer_given=memory.offer_given,
            refund_requested=memory.refund_requested,
            technical_steps=list(memory.technical_steps),
        )

    def append_interaction(
        self,
        conversation_id: str,
        role: str,
        message: str,
        *,
        customer_id: str | None = None,
        include_in_context: bool = True,
    ) -> ConversationMemory:
        """Append one turn and retain only the ten most recent interactions."""

        clean_conversation_id = self._conversation_id(conversation_id)
        clean_customer_id = self._customer_id(customer_id)
        interaction = Interaction(
            interaction_id=uuid4().hex,
            role=role,
            message=message.strip(),
            timestamp=datetime.now(UTC),
            include_in_context=include_in_context,
        )

        with self._lock:
            document = self._read_unlocked()
            memory = self._memory(
                document,
                clean_conversation_id,
                clean_customer_id,
            )
            memory.interactions = [
                *memory.interactions,
                interaction,
            ][-MAX_INTERACTIONS:]
            memory.updated_at = datetime.now(UTC)
            self._write_unlocked(document)
            return memory.model_copy(deep=True)

    def append_turn(
        self,
        conversation_id: str,
        customer_message: str,
        assistant_message: str,
        *,
        customer_id: str | None = None,
        customer_include_in_context: bool = True,
    ) -> ConversationMemory:
        """Persist both sides of a turn in one atomic JSON update."""

        clean_conversation_id = self._conversation_id(conversation_id)
        clean_customer_id = self._customer_id(customer_id)
        now = datetime.now(UTC)
        interactions = [
            Interaction(
                interaction_id=uuid4().hex,
                role="customer",
                message=customer_message.strip(),
                timestamp=now,
                include_in_context=customer_include_in_context,
            ),
            Interaction(
                interaction_id=uuid4().hex,
                role="assistant",
                message=assistant_message.strip(),
                timestamp=now,
            ),
        ]

        with self._lock:
            document = self._read_unlocked()
            memory = self._memory(
                document,
                clean_conversation_id,
                clean_customer_id,
            )
            memory.interactions = [
                *memory.interactions,
                *interactions,
            ][-MAX_INTERACTIONS:]
            memory.updated_at = now
            self._write_unlocked(document)
            return memory.model_copy(deep=True)

    def record_offer(
        self,
        conversation_id: str,
        percentage: float,
        status: str,
        *,
        customer_id: str | None = None,
    ) -> ConversationMemory:
        """Remember the latest retention offer outcome for later turns."""

        clean_conversation_id = self._conversation_id(conversation_id)
        with self._lock:
            document = self._read_unlocked()
            memory = self._memory(
                document,
                clean_conversation_id,
                self._customer_id(customer_id),
            )
            memory.offer_given = OfferMemory(
                percentage=percentage,
                status=status.strip(),
                recorded_at=datetime.now(UTC),
            )
            memory.updated_at = datetime.now(UTC)
            self._write_unlocked(document)
            return memory.model_copy(deep=True)

    def record_refund(
        self,
        conversation_id: str,
        amount: float,
        status: str,
        *,
        customer_id: str | None = None,
    ) -> ConversationMemory:
        """Remember the latest refund request and approval state."""

        clean_conversation_id = self._conversation_id(conversation_id)
        with self._lock:
            document = self._read_unlocked()
            memory = self._memory(
                document,
                clean_conversation_id,
                self._customer_id(customer_id),
            )
            memory.refund_requested = RefundMemory(
                amount=amount,
                status=status.strip(),
                recorded_at=datetime.now(UTC),
            )
            memory.updated_at = datetime.now(UTC)
            self._write_unlocked(document)
            return memory.model_copy(deep=True)

    def record_technical_step(
        self,
        conversation_id: str,
        step: str,
        *,
        customer_id: str | None = None,
    ) -> ConversationMemory:
        """Remember a completed troubleshooting step without unbounded growth."""

        clean_conversation_id = self._conversation_id(conversation_id)
        clean_step = step.strip()
        if not clean_step:
            raise ValueError("technical step must not be empty")

        with self._lock:
            document = self._read_unlocked()
            memory = self._memory(
                document,
                clean_conversation_id,
                self._customer_id(customer_id),
            )
            if not memory.technical_steps or memory.technical_steps[-1] != clean_step:
                memory.technical_steps = [
                    *memory.technical_steps,
                    clean_step,
                ][-MAX_TECHNICAL_STEPS:]
            memory.updated_at = datetime.now(UTC)
            self._write_unlocked(document)
            return memory.model_copy(deep=True)
