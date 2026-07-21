"""Tests for generalized indices and Merkle proof construction."""

from hashlib import sha256
from typing import Any, cast

import pytest

from ssz import (
    Boolean,
    Container,
    List,
    SSZTypeError,
    SSZValueError,
    Uint16,
    Uint64,
    Vector,
    compute_merkle_proof,
    get_generalized_index,
)
from ssz.bitfields import BaseBitlist, BaseBitvector
from ssz.byte_arrays import BaseByteList, BaseBytes
from ssz.merkle_proofs import _decompose, _Node
from ssz.merkleization import hash_tree_root


class Bytes32(BaseBytes):
    LENGTH = 32


class Bytes48(BaseBytes):
    LENGTH = 48


class ByteList64(BaseByteList):
    LIMIT = Uint64(64)


class Bitlist600(BaseBitlist):
    LIMIT = Uint64(600)


class Bitvector300(BaseBitvector):
    LENGTH = Uint64(300)


class Checkpoint(Container):
    epoch: Uint64
    root: Bytes32


class Uint64List16(List[Uint64]):
    LIMIT = Uint64(16)


class CheckpointList4(List[Checkpoint]):
    LIMIT = Uint64(4)


class Uint16Vector8(Vector[Uint16]):
    LENGTH = Uint64(8)


class CheckpointVector2(Vector[Checkpoint]):
    LENGTH = Uint64(2)


class Wrapper(Container):
    """Five fields pad to eight leaf slots, giving a depth-three top layer."""

    tag: Uint64
    checkpoint: Checkpoint
    scores: Uint64List16
    history: CheckpointList4
    flag: Boolean


def is_valid_merkle_branch(leaf: bytes, branch: list[bytes], gindex: int, root: bytes) -> bool:
    """Fold a branch from the leaf up and compare against the expected root."""
    node = leaf
    for depth, sibling in enumerate(branch):
        if (gindex >> depth) & 1:
            node = sha256(sibling + node).digest()
        else:
            node = sha256(node + sibling).digest()
    return node == root


def sample_wrapper() -> Wrapper:
    return Wrapper(
        tag=Uint64(7),
        checkpoint=Checkpoint(epoch=Uint64(3), root=Bytes32(b"\xaa" * 32)),
        scores=Uint64List16.of(*[Uint64(i) for i in range(6)]),
        history=CheckpointList4.of(
            Checkpoint(epoch=Uint64(1), root=Bytes32(b"\x01" * 32)),
            Checkpoint(epoch=Uint64(2), root=Bytes32(b"\x02" * 32)),
        ),
        flag=Boolean(True),
    )


class TestGetGeneralizedIndex:
    """Path resolution across every navigable SSZ shape."""

    def test_empty_path_is_root(self) -> None:
        assert (
            get_generalized_index(
                Wrapper,
            )
            == 1
        )

    def test_container_fields(self) -> None:
        """Five fields pad to eight slots: field i lives at 8 + i."""
        assert get_generalized_index(Wrapper, "tag") == 8
        assert get_generalized_index(Wrapper, "checkpoint") == 9
        assert get_generalized_index(Wrapper, "flag") == 12

    def test_nested_container_field(self) -> None:
        """Checkpoint has two fields, so 'root' doubles the parent index plus one."""
        assert get_generalized_index(Wrapper, "checkpoint", "root") == 9 * 2 + 1

    def test_beacon_state_shape(self) -> None:
        """The altair beacon state constants fall out of the same arithmetic."""

        class BeaconState25(Container):
            """Twenty-five fields pad to thirty-two slots."""

            f00: Uint64
            f01: Uint64
            f02: Uint64
            f03: Uint64
            f04: Uint64
            f05: Uint64
            f06: Uint64
            f07: Uint64
            f08: Uint64
            f09: Uint64
            f10: Uint64
            f11: Uint64
            f12: Uint64
            f13: Uint64
            f14: Uint64
            f15: Uint64
            f16: Uint64
            f17: Uint64
            f18: Uint64
            f19: Uint64
            finalized_checkpoint: Checkpoint
            f21: Uint64
            current_sync_committee: Checkpoint
            next_sync_committee: Checkpoint
            f24: Uint64

        assert get_generalized_index(BeaconState25, "finalized_checkpoint", "root") == 105
        assert get_generalized_index(BeaconState25, "current_sync_committee") == 54
        assert get_generalized_index(BeaconState25, "next_sync_committee") == 55

    def test_basic_list_element_chunk(self) -> None:
        """Four Uint64 pack per chunk, so element five lives in chunk one of four."""
        assert get_generalized_index(Uint64List16, 5) == 2 * 4 + 1

    def test_composite_list_element(self) -> None:
        assert get_generalized_index(CheckpointList4, 2) == 2 * 4 + 2

    def test_composite_list_element_field(self) -> None:
        assert get_generalized_index(CheckpointList4, 2, "epoch") == (2 * 4 + 2) * 2

    def test_list_length(self) -> None:
        assert get_generalized_index(Uint64List16, "__len__") == 3

    def test_typed_uint_index(self) -> None:
        """Indexing with a Uint64 subclass must not trip strict comparisons."""

        class BlobIndex(Uint64):
            pass

        assert get_generalized_index(CheckpointList4, BlobIndex(2)) == 2 * 4 + 2

    def test_basic_vector_element_chunk(self) -> None:
        """Sixteen Uint16 fit in one chunk, so all eight elements share it."""
        assert get_generalized_index(Uint16Vector8, 7) == 1

    def test_composite_vector_element(self) -> None:
        assert get_generalized_index(CheckpointVector2, 1) == 3

    def test_bytes_chunk(self) -> None:
        assert get_generalized_index(Bytes32, 31) == 1
        assert get_generalized_index(Bytes48, 40) == 3

    def test_bytelist_chunk(self) -> None:
        assert get_generalized_index(ByteList64, 40) == 2 * 2 + 1

    def test_bitlist_chunk(self) -> None:
        """Bit 300 of 600 lands in chunk one of the four-slot data tree."""
        assert get_generalized_index(Bitlist600, 300) == 2 * 4 + 1

    def test_bitvector_chunk(self) -> None:
        assert get_generalized_index(Bitvector300, 299) == 2 + 1

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(SSZValueError, match="no field"):
            get_generalized_index(Wrapper, "missing")

    def test_int_into_container_raises(self) -> None:
        with pytest.raises(SSZValueError, match="no field"):
            get_generalized_index(Wrapper, 0)

    def test_length_of_fixed_size_raises(self) -> None:
        with pytest.raises(SSZValueError, match="no mixed-in length"):
            get_generalized_index(Uint16Vector8, "__len__")

    def test_string_into_sequence_raises(self) -> None:
        with pytest.raises(SSZValueError, match="cannot index"):
            get_generalized_index(Uint64List16, "epoch")

    def test_descend_past_packed_leaf_raises(self) -> None:
        with pytest.raises(SSZTypeError, match="packed leaf"):
            get_generalized_index(Uint64List16, 0, 0)

    def test_basic_type_has_no_chunk_count(self) -> None:
        with pytest.raises(SSZTypeError, match="chunk count"):
            get_generalized_index(Wrapper, "tag", 0)


class TestComputeMerkleProof:
    """Branches verify against the value's hash tree root."""

    @pytest.mark.parametrize(
        "path",
        [
            ("tag",),
            ("checkpoint",),
            ("checkpoint", "root"),
            ("flag",),
            ("scores", "__len__"),
            ("history", 1),
            ("history", 1, "root"),
        ],
    )
    def test_wrapper_paths_verify(self, path: tuple) -> None:
        value = sample_wrapper()
        gindex = get_generalized_index(Wrapper, *path)
        branch = compute_merkle_proof(value, gindex)

        # Resolve the leaf being proven by following the path by hand.
        leaf_value = cast(Any, value)
        for element in path:
            if element == "__len__":
                leaf = len(leaf_value.data).to_bytes(32, "little")
                break
            leaf_value = (
                getattr(leaf_value, element) if isinstance(element, str) else leaf_value[element]
            )
        else:
            leaf = hash_tree_root(leaf_value)
        assert len(branch) == gindex.bit_length() - 1
        assert is_valid_merkle_branch(leaf, branch, gindex, hash_tree_root(value))

    def test_packed_element_chunk_proof(self) -> None:
        """A basic list element proves the packed chunk that holds it."""
        value = sample_wrapper()
        gindex = get_generalized_index(Wrapper, "scores", 5)
        branch = compute_merkle_proof(value, gindex)
        # Chunk one packs elements four and five, padded with zero bytes.
        chunk = b"".join(i.to_bytes(8, "little") for i in (4, 5)).ljust(32, b"\x00")
        assert is_valid_merkle_branch(chunk, branch, gindex, hash_tree_root(value))

    def test_root_index_has_empty_branch(self) -> None:
        assert compute_merkle_proof(sample_wrapper(), 1) == []

    def test_padding_slots_prove_zero_chunks(self) -> None:
        """A slot beyond the occupied fields verifies as an all-zero chunk."""
        value = sample_wrapper()
        branch = compute_merkle_proof(value, 15)
        assert is_valid_merkle_branch(b"\x00" * 32, branch, 15, hash_tree_root(value))

    def test_bytes_chunk_proof(self) -> None:
        value = Bytes48(bytes(range(48)))
        gindex = get_generalized_index(Bytes48, 40)
        branch = compute_merkle_proof(value, gindex)
        chunk = bytes(range(32, 48)).ljust(32, b"\x00")
        assert is_valid_merkle_branch(chunk, branch, gindex, hash_tree_root(value))

    def test_bytelist_length_proof(self) -> None:
        value = ByteList64(data=b"\x11" * 40)
        branch = compute_merkle_proof(value, 3)
        leaf = (40).to_bytes(32, "little")
        assert is_valid_merkle_branch(leaf, branch, 3, hash_tree_root(value))

    def test_bitlist_chunk_proof(self) -> None:
        value = Bitlist600.of(*([True] * 300))
        gindex = get_generalized_index(Bitlist600, 299)
        branch = compute_merkle_proof(value, gindex)
        # Bits 256 through 299 land in chunk one: forty-four set bits.
        chunk = sum(1 << i for i in range(300 - 256)).to_bytes(32, "little")
        assert is_valid_merkle_branch(chunk, branch, gindex, hash_tree_root(value))

    def test_bitvector_chunk_proof(self) -> None:
        value = Bitvector300.of(*([True] * 300))
        gindex = get_generalized_index(Bitvector300, 0)
        branch = compute_merkle_proof(value, gindex)
        chunk = b"\xff" * 32
        assert is_valid_merkle_branch(chunk, branch, gindex, hash_tree_root(value))

    def test_composite_vector_element_proof(self) -> None:
        value = CheckpointVector2(
            data=[
                Checkpoint(epoch=Uint64(1), root=Bytes32(b"\x01" * 32)),
                Checkpoint(epoch=Uint64(2), root=Bytes32(b"\x02" * 32)),
            ]
        )
        branch = compute_merkle_proof(value, 3)
        assert is_valid_merkle_branch(
            hash_tree_root(value.data[1]), branch, 3, hash_tree_root(value)
        )

    def test_invalid_index_raises(self) -> None:
        with pytest.raises(SSZValueError, match="invalid generalized index"):
            compute_merkle_proof(sample_wrapper(), 0)

    def test_descend_below_basic_field_raises(self) -> None:
        with pytest.raises(SSZValueError, match="basic value"):
            compute_merkle_proof(sample_wrapper(), 8 * 2)

    def test_descend_below_leaf_chunk_raises(self) -> None:
        with pytest.raises(SSZValueError, match="below a leaf"):
            compute_merkle_proof(Bytes32(b"\xaa" * 32), 2)

    def test_descend_into_basic_value_raises(self) -> None:
        with pytest.raises(SSZValueError, match="basic value"):
            compute_merkle_proof(Uint64(5), 2)

    def test_decompose_rejects_foreign_values(self) -> None:
        foreign = cast(Any, object())
        with pytest.raises(SSZTypeError, match="cannot decompose"):
            _decompose(foreign)

    def test_node_base_methods_are_abstract(self) -> None:
        node = _Node()
        with pytest.raises(NotImplementedError):
            node.root()
        with pytest.raises(NotImplementedError):
            node.children()

    def test_mix_root_matches_list_root(self) -> None:
        """A decomposed list node reports the same root as its hash tree root."""
        scores = Uint64List16.of(Uint64(1), Uint64(2))
        assert _decompose(scores).root() == hash_tree_root(scores)
