"""
Generalized indices and Merkle proofs over SSZ Merkle trees.

A generalized index names a node in a Merkle tree by its position in a
breadth-first walk: the root is 1, and the children of node ``i`` are
``2 * i`` (left) and ``2 * i + 1`` (right). Every SSZ value has a
deterministic tree shape, so a path of container field names and
collection element indices resolves to one generalized index, and any
node in the tree can be proven against the root with a branch of
sibling roots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Union, cast

from ssz.bitfields import BaseBitlist, BaseBitvector
from ssz.boolean import Boolean
from ssz.byte_arrays import BaseByteList, BaseBytes
from ssz.collections import List, Vector
from ssz.container import Container
from ssz.exceptions import SSZTypeError, SSZValueError
from ssz.merkleization import (
    BITS_PER_CHUNK,
    BYTES_PER_CHUNK,
    _next_pow2,
    _pack_bits,
    _pack_bytes,
    hash_tree_root,
    merkleize,
    mix_in_length,
)
from ssz.ssz_base import SSZType
from ssz.uint import BaseUint

PathElement = Union[int, str]


def _chunk_count(cls: type) -> int:
    """
    Number of leaf chunks the type's data subtree is padded to.

    This is the capacity of the tree, not the number of occupied chunks:

    - Basic-element collections pack serialized elements into chunks.
    - Composite-element collections use one chunk per element root.
    - Bitfields pack 256 bits per chunk.
    - Byte arrays pack 32 bytes per chunk.
    """
    if issubclass(cls, (List, Vector)):
        bound = int(cls.LIMIT) if issubclass(cls, List) else int(cls.LENGTH)
        element_type = cls.ELEMENT_TYPE
        if issubclass(element_type, (BaseUint, Boolean)):
            return math.ceil(bound * element_type.get_byte_length() / BYTES_PER_CHUNK)
        return bound
    if issubclass(cls, BaseByteList):
        return math.ceil(int(cls.LIMIT) / BYTES_PER_CHUNK)
    if issubclass(cls, BaseBytes):
        return math.ceil(int(cls.LENGTH) / BYTES_PER_CHUNK)
    if issubclass(cls, BaseBitlist):
        return math.ceil(int(cls.LIMIT) / BITS_PER_CHUNK)
    if issubclass(cls, BaseBitvector):
        return math.ceil(int(cls.LENGTH) / BITS_PER_CHUNK)
    raise SSZTypeError(f"cannot compute chunk count for {cls.__name__}")


def _chunk_index(cls: type, element_index: int) -> int:
    """Leaf chunk that holds the element at the given index."""
    if issubclass(cls, (List, Vector)):
        element_type = cls.ELEMENT_TYPE
        if issubclass(element_type, (BaseUint, Boolean)):
            return element_index * element_type.get_byte_length() // BYTES_PER_CHUNK
        return element_index
    if issubclass(cls, (BaseByteList, BaseBytes)):
        return element_index // BYTES_PER_CHUNK
    # Bitfields: 256 bits per chunk.
    return element_index // BITS_PER_CHUNK


def _element_type(cls: type) -> type | None:
    """
    Type reached by indexing into a collection, when navigable.

    Only composite sequence elements have their own subtree; packed basic
    elements, bytes, and bits terminate at their chunk.
    """
    if issubclass(cls, (List, Vector)):
        element_type = cls.ELEMENT_TYPE
        if not issubclass(element_type, (BaseUint, Boolean)):
            return element_type
    return None


def get_generalized_index(cls: type[SSZType], *path: PathElement) -> int:
    """
    Resolve a path of field names and element indices to a generalized index.

    - A container consumes a field name.
    - A sequence consumes an element index, or the literal ``"__len__"``
      for the mixed-in length of a variable-size sequence.
    - Packed leaves (basic elements, bytes, bits) resolve to the chunk
      that holds them; the path cannot descend further.

    For example, with the beacon state's tree shape::

        get_generalized_index(BeaconState, "finalized_checkpoint", "root")

    walks into the ``finalized_checkpoint`` field's subtree and then to
    its ``root`` field, multiplying out one generalized index.
    """
    current: type | None = cls
    gindex = 1
    for element in path:
        if current is None:
            raise SSZTypeError("path descends into a packed leaf chunk")
        if issubclass(current, Container):
            if not isinstance(element, str) or element not in current.model_fields:
                raise SSZValueError(f"{current.__name__} has no field {element!r}")
            field_names = list(current.model_fields)
            gindex = gindex * _next_pow2(len(field_names)) + field_names.index(element)
            annotation = current.model_fields[element].annotation
            current = annotation
            continue

        is_variable_size = issubclass(current, (List, BaseByteList, BaseBitlist))
        if isinstance(element, str) and element == "__len__":
            if not is_variable_size:
                raise SSZValueError(f"{current.__name__} has no mixed-in length")
            gindex = gindex * 2 + 1
            current = None
            continue
        if not isinstance(element, int):
            raise SSZValueError(f"cannot index {current.__name__} with {element!r}")
        # Typed indices (e.g. Uint64 subclasses) restrict arithmetic; use the
        # plain integer value.
        element = int(element)
        if is_variable_size:
            # Step into the data subtree; the mixed-in length is the right child.
            gindex = gindex * 2
        gindex = gindex * _next_pow2(_chunk_count(current)) + _chunk_index(current, element)
        current = _element_type(current)
    return gindex


class _Node:
    """A virtual node in an SSZ Merkle tree, materialized lazily."""

    __slots__ = ()

    def root(self) -> bytes:
        raise NotImplementedError

    def children(self) -> tuple[_Node, _Node]:
        raise NotImplementedError


class _Leaf(_Node):
    """A 32-byte chunk with no children."""

    __slots__ = ("chunk",)

    def __init__(self, chunk: bytes) -> None:
        self.chunk = chunk

    def root(self) -> bytes:
        return self.chunk

    def children(self) -> tuple[_Node, _Node]:
        raise SSZValueError("generalized index descends below a leaf chunk")


class _Padded(_Node):
    """
    A perfect binary tree over a fixed number of leaf slots.

    Occupied slots hold nodes; the remaining slots are virtual zero
    chunks, so padding is never allocated.
    """

    __slots__ = ("slots", "width")

    def __init__(self, slots: Sequence[_Node], width: int) -> None:
        self.slots = slots
        self.width = width

    def root(self) -> bytes:
        return merkleize([slot.root() for slot in self.slots], limit=self.width)

    def children(self) -> tuple[_Node, _Node]:
        if self.width == 1:
            # A single slot is not a subtree of its own; the slot node takes over.
            node = self.slots[0] if self.slots else _Leaf(b"\x00" * BYTES_PER_CHUNK)
            return node.children()
        half = self.width // 2
        return (
            _Padded(self.slots[:half], half),
            _Padded(self.slots[half:], half),
        )


class _Mix(_Node):
    """A length mix-in: the data subtree on the left, the length chunk on the right."""

    __slots__ = ("data", "length")

    def __init__(self, data: _Node, length: int) -> None:
        self.data = data
        self.length = length

    def root(self) -> bytes:
        return mix_in_length(self.data.root(), self.length)

    def children(self) -> tuple[_Node, _Node]:
        return self.data, _Leaf(self.length.to_bytes(BYTES_PER_CHUNK, "little"))


class _Value(_Node):
    """An SSZ value, decomposed into its tree shape on first descent."""

    __slots__ = ("value",)

    def __init__(self, value: SSZType) -> None:
        self.value = value

    def root(self) -> bytes:
        return hash_tree_root(self.value)

    def children(self) -> tuple[_Node, _Node]:
        return _decompose(self.value).children()


def _decompose(value: SSZType) -> _Node:
    """Expand one SSZ value into its top-level tree structure."""
    if isinstance(value, Container):
        slots: list[_Node] = [_Value(getattr(value, name)) for name in type(value).model_fields]
        return _Padded(slots, _next_pow2(len(slots)))
    if isinstance(value, (BaseUint, Boolean)):
        raise SSZValueError("generalized index descends into a basic value")
    cls = type(value)
    if isinstance(value, (List, Vector)):
        elements = cast("list[Any]", value.data)
        width = _next_pow2(_chunk_count(cls))
        if _element_type(cls) is not None:
            slots = [_Value(element) for element in elements]
        else:
            packed = b"".join(element.encode_bytes() for element in elements)
            slots = [_Leaf(chunk) for chunk in _pack_bytes(packed)]
        data: _Node = _Padded(slots, width)
        return _Mix(data, len(elements)) if isinstance(value, List) else data
    if isinstance(value, (BaseByteList, BaseBytes)):
        width = _next_pow2(_chunk_count(cls))
        payload = value.encode_bytes()
        data = _Padded([_Leaf(chunk) for chunk in _pack_bytes(payload)], width)
        return _Mix(data, len(payload)) if isinstance(value, BaseByteList) else data
    if isinstance(value, (BaseBitlist, BaseBitvector)):
        width = _next_pow2(_chunk_count(cls))
        data = _Padded([_Leaf(chunk) for chunk in _pack_bits(value.data)], width)
        return _Mix(data, len(value.data)) if isinstance(value, BaseBitlist) else data
    raise SSZTypeError(f"cannot decompose {type(value).__name__}")


def compute_merkle_proof(value: SSZType, index: int) -> list[bytes]:
    """
    Build the Merkle branch proving the node at a generalized index.

    The branch lists sibling roots from the target's own sibling up to
    the child of the root, the order expected by the usual
    ``is_valid_merkle_branch`` verification loop.

    Raises:
        SSZValueError: If the index is not reachable in the value's tree.
    """
    if index < 1:
        raise SSZValueError(f"invalid generalized index: {index}")
    node: _Node = _Value(value)
    branch: list[bytes] = []
    # Visit the path bits from just below the root down to the target,
    # keeping the opposite-hand sibling root at every step.
    for depth in range(index.bit_length() - 2, -1, -1):
        left, right = node.children()
        if index & (1 << depth):
            branch.append(left.root())
            node = right
        else:
            branch.append(right.root())
            node = left
    branch.reverse()
    return branch
