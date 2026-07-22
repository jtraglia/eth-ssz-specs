"""Abstract bases for the SSZ type system."""

import io
from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Any, Final, Self

from pydantic import ConfigDict

from ssz.base import StrictBaseModel
from ssz.exceptions import SSZDefinitionError, SSZSerializationError

BYTES_PER_LENGTH_OFFSET: Final = 4
"""Width of an SSZ offset prefixing each variable-size element.

Encoded as a uint32 in little-endian byte order."""


class SSZType(ABC):
    """Abstract base for every SSZ-encodable type."""

    @classmethod
    @abstractmethod
    def is_fixed_size(cls) -> bool:
        """
        Whether every instance encodes to the same number of bytes.

        Returns:
            True for fixed-size types, False for variable-size.
        """
        ...

    @classmethod
    @abstractmethod
    def get_byte_length(cls) -> int:
        """
        Fixed encoded byte length of this type.

        Returns:
            The constant byte width every instance encodes to.

        Raises:
            SSZTypeError: If the type is variable-size.
        """
        ...

    @abstractmethod
    def serialize(self, stream: IO[bytes]) -> int:
        """
        Write the SSZ encoding to a binary stream.

        Args:
            stream: Output binary stream.

        Returns:
            Number of bytes written.
        """
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, stream: IO[bytes], scope: int) -> Self:
        """
        Read one value from a binary stream within a bounded byte budget.

        Args:
            stream: Source binary stream.
            scope: Number of bytes belonging to this value.

        Returns:
            A new instance reconstructed from the stream.
        """
        ...

    def encode_bytes(self) -> bytes:
        """
        Encode this value to its SSZ byte representation.

        Returns:
            Serialized bytes.
        """
        stream = io.BytesIO()
        self.serialize(stream)
        return stream.getvalue()

    def hash_tree_root(self) -> bytes:
        """
        Compute this value's 32-byte Merkle tree root.

        Method form of the ``hash_tree_root`` function, for fluent use on any
        SSZ value.
        """
        # Deferred import: the merkleization module imports the SSZ types.
        from ssz.merkleization import hash_tree_root

        return hash_tree_root(self)

    @classmethod
    def decode_bytes(cls, data: bytes) -> Self:
        """
        Decode SSZ bytes into a new instance.

        Rejects trailing bytes left over after the stream-based decoder finishes.
        A spec decoder must accept exactly one canonical encoding per value.

        Args:
            data: SSZ-encoded bytes containing exactly one value.

        Returns:
            A new instance reconstructed from the input.

        Raises:
            SSZSerializationError: If the input carries bytes past the decoded value.
        """
        stream = io.BytesIO(data)
        instance = cls.deserialize(stream, len(data))

        # Spec contract: each canonical encoding maps to exactly one value.
        #
        # Any unread bytes mean the input either over-allocated or carries noise.
        leftover = len(data) - stream.tell()
        if leftover:
            raise SSZSerializationError(f"{cls.__name__}: {leftover} trailing byte(s) after decode")
        return instance


class SSZModel(StrictBaseModel, SSZType):
    """
    Pydantic-backed SSZ base used by containers, lists, vectors, and bitfields.

    Two shapes share this base:

    - Collections wrap an inner sequence in one Pydantic field called data.
    - Containers expose multiple named Pydantic fields that map to a struct on the wire.

    The default length and string forms switch on which shape the subclass uses.
    """

    def copy(self) -> Self:  # ty: ignore[invalid-method-override]
        """
        Return a deep, independent copy of this value.

        Replaces Pydantic's deprecated shallow ``copy`` with the deep copy that
        value types need: mutating the copy never affects the original.
        """
        return self.model_copy(deep=True)

    def __eq__(self, other: object) -> bool:
        """
        Value equality by Merkle tree root.

        Two SSZ values are the same value exactly when their trees agree,
        regardless of which equivalent class holds them — a sync committee
        converted across spec versions still equals its source. Non-SSZ
        operands defer to the other operand's equality.
        """
        if isinstance(other, SSZModel):
            return self.hash_tree_root() == other.hash_tree_root()
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by Merkle tree root — equal values hash equally."""
        return hash(self.hash_tree_root())

    def __len__(self) -> int:
        """Element count for collections, field count for containers."""
        data_field = getattr(self, "data", None)
        if data_field is not None:
            return len(data_field)
        return len(type(self).model_fields)

    def __repr__(self) -> str:
        """Show collection contents as data=[...] or container fields as name=value pairs."""
        cls_name = type(self).__name__
        data_field = getattr(self, "data", None)
        if data_field is not None:
            return f"{cls_name}(data={list(data_field)!r})"
        field_strs = [f"{name}={getattr(self, name)!r}" for name in type(self).model_fields]
        return f"{cls_name}({' '.join(field_strs)})"


class SSZCollection(SSZModel):
    """
    Pydantic-backed SSZ base for collections that wrap their contents in one data field.

    Sequences, bitfields, and byte lists all share this base.
    Containers do not — their contents live in named fields, not a single data field.

    Construction passes the field by keyword, or the elements positionally
    through the `of` factory:

        Uint8List4(data=[1, 2, 3])
        Uint8List4.of(1, 2, 3)

    A subclass pins its size by declaring LENGTH (exact element count) or LIMIT
    (maximum element count). The declared bound must be a Uint64 — a plain int
    is rejected at class definition time, keeping size bounds as strictly typed
    as every other spec value.

        class Uint8List4(List[Uint8]):
            LIMIT = Uint64(4)

    Unlike containers, collections are mutable: element assignment, append, and
    pop revalidate the whole collection, so size bounds and element coercion
    behave exactly as they do at construction. Fixed-length shapes accept
    element assignment but reject any length change at revalidation.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    if TYPE_CHECKING:
        # Each concrete subclass declares the real data field with its own type.
        # This annotation only teaches type checkers the attribute exists here,
        # where the shared mutation methods assign it.
        data: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject subclasses whose declared size bound is not a Uint64."""
        super().__init_subclass__(**kwargs)
        # Deferred import: the uint module imports this one.
        from ssz.uint import Uint64

        for bound_name in ("LENGTH", "LIMIT"):
            if bound_name in cls.__dict__ and not isinstance(cls.__dict__[bound_name], Uint64):
                raise SSZDefinitionError(cls.__name__, f"{bound_name} as a Uint64")

    def __eq__(self, other: object) -> bool:
        """
        Value equality, relaxed against plain sequences.

        Against another SSZ value, equality stays Merkle-root based, exactly as
        for any SSZ value. Against a plain list or tuple, equality is
        element-wise: equal length, and every element equal by value — the same
        relaxation that already lets a uint compare equal to a plain int. This
        is what lets an SSZ sequence compare equal to the plain Python list a
        spec helper builds up, without materializing either side first.

        Element-level strictness is untouched: each element is compared by its
        own equality, so a sequence of typed bytes still raises when compared
        against plain bytes elements. Byte lists keep their own strict equality
        by overriding this method, so this relaxation never reaches them.
        """
        if isinstance(other, SSZModel):
            return self.hash_tree_root() == other.hash_tree_root()
        if isinstance(other, (list, tuple)):
            elements = list(self.data)
            if len(elements) != len(other):
                return False
            return all(mine == theirs for mine, theirs in zip(elements, other))
        return NotImplemented

    # Defining __eq__ would otherwise clear the inherited hash; keep values
    # hashable by their Merkle root, staying consistent with the equality above.
    __hash__ = SSZModel.__hash__

    def __setitem__(self, index: Any, value: Any) -> None:
        """Replace the element(s) at ``index``, revalidating the collection."""
        elements = list(self.data)
        elements[index] = value
        self.data = elements

    def append(self, value: Any) -> None:
        """Add one element at the end, revalidating the collection."""
        self.data = [*self.data, value]

    def pop(self) -> Any:
        """Remove and return the last element, revalidating the collection."""
        *rest, last = self.data
        self.data = rest
        return last

    @classmethod
    def of(cls, *elements: Any) -> Self:
        r"""
        Build an instance from the given elements.

        Pydantic models are keyword-only, so the data field is normally passed by
        keyword. This factory is the positional form: each argument is exactly one
        element, and no argument is ever spread. Classmethods are inherited without
        re-synthesis, so unlike a custom constructor this form stays fully visible
        to static type checkers on every subclass.

            Uint8List4.of(1, 2, 3)     ==  Uint8List4(data=[1, 2, 3])
            Uint8List4.of()            ==  Uint8List4(data=[])
            Uint8List4.of(*existing)   spreads an existing sequence
            ByteList10.of(0xDE, 0xAD)  ==  ByteList10(data=b"\xde\xad")

        Args:
            *elements: The elements of the new collection.

        Returns:
            A new instance holding exactly the given elements.
        """
        return cls(data=elements)
