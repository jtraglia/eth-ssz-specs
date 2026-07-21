"""Tests for SSZModel and SSZType base class behavior."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from ssz import Uint8, Uint16, Uint64
from ssz.bitfields import BaseBitlist, BaseBitvector
from ssz.boolean import Boolean
from ssz.byte_arrays import BaseByteList, BaseBytes
from ssz.collections import List, Vector
from ssz.container import Container
from ssz.exceptions import SSZTypeError, SSZValueError


class Uint16List4(List[Uint16]):
    """A list with up to 4 Uint16 values."""

    LIMIT = Uint64(4)


class Uint16Vector2(Vector[Uint16]):
    """A vector of exactly 2 Uint16 values."""

    LENGTH = Uint64(2)


class SmallBitvector(BaseBitvector):
    """A bitvector with exactly 3 bits."""

    LENGTH = Uint64(3)


class SmallByteList(BaseByteList):
    """A byte list with up to 10 bytes."""

    LIMIT = Uint64(10)


class TwoFieldContainer(Container):
    """A container with two fixed-size fields."""

    x: Uint8
    y: Uint16


class ThreeFieldContainer(Container):
    """A container with three fields, one variable-size."""

    a: Uint8
    b: Uint64
    c: Uint16List4


class SmallBitlist(BaseBitlist):
    """A bitlist with a small limit, used to test SSZModel.__len__ data path."""

    LIMIT = Uint64(8)


class TestSSZModelLength:
    """
    Tests for SSZModel.__len__() on both collection and container models.

    Uses BaseBitlist (not List) for the data-path because List overrides
    __len__ with its own implementation. BaseBitlist inherits SSZModel's version.
    """

    def test_length_data_path_via_bitlist(self) -> None:
        """BaseBitlist delegates to SSZModel.__len__ which returns len(data)."""
        bl = SmallBitlist(data=(Boolean(True), Boolean(False), Boolean(True)))
        assert len(bl) == 3

    def test_length_empty_data_path_via_bitlist(self) -> None:
        bl = SmallBitlist(data=())
        assert len(bl) == 0

    def test_length_container_returns_field_count(self) -> None:
        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        assert len(container) == 2

    def test_length_three_field_container(self) -> None:
        container = ThreeFieldContainer(a=Uint8(5), b=Uint64(42), c=Uint16List4(data=[Uint16(1)]))
        assert len(container) == 3


class TestSSZModelRepr:
    """Tests for SSZModel.__repr__() on both collection and container models."""

    def test_repr_collection_shows_data(self) -> None:
        assert repr(Uint16List4(data=[Uint16(10), Uint16(20)])) == (
            "Uint16List4(data=[Uint16(10), Uint16(20)])"
        )

    def test_repr_empty_collection(self) -> None:
        assert repr(Uint16List4(data=[])) == "Uint16List4(data=[])"

    def test_repr_container_shows_fields(self) -> None:
        assert repr(TwoFieldContainer(x=Uint8(1), y=Uint16(2))) == (
            "TwoFieldContainer(x=Uint8(1) y=Uint16(2))"
        )

    def test_repr_three_field_container(self) -> None:
        container = ThreeFieldContainer(a=Uint8(5), b=Uint64(42), c=Uint16List4(data=[Uint16(1)]))
        assert repr(container) == (
            "ThreeFieldContainer(a=Uint8(5) b=Uint64(42) c=Uint16List4(data=[Uint16(1)]))"
        )


class TestSSZTypeEncodeDecode:
    """
    Tests for encode_bytes/decode_bytes on SSZType.

    These methods wrap the stream-based serialize/deserialize interface
    so callers can work with plain byte strings instead.
    """

    def test_encode_bytes_fixed_container(self) -> None:
        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        encoded = container.encode_bytes()
        assert encoded == b"\x01\x02\x00"

    def test_decode_bytes_fixed_container(self) -> None:
        assert TwoFieldContainer.decode_bytes(b"\x01\x02\x00") == TwoFieldContainer(
            x=Uint8(1), y=Uint16(2)
        )

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding must recover the original object."""
        original = TwoFieldContainer(x=Uint8(255), y=Uint16(1000))
        assert TwoFieldContainer.decode_bytes(original.encode_bytes()) == original


class TestSSZCollectionOf:
    """
    Tests for the `of` factory classmethod.

    `of` is the positional construction form: each argument is exactly one
    element, and no argument is ever spread.
    """

    def test_of_builds_from_elements(self) -> None:
        """Each argument becomes one element."""
        assert Uint16List4.of(1, 2, 3) == Uint16List4(data=[Uint16(1), Uint16(2), Uint16(3)])

    def test_of_with_no_elements_builds_empty(self) -> None:
        """No arguments build an empty collection."""
        assert Uint16List4.of() == Uint16List4(data=[])

    def test_of_single_element_is_never_spread(self) -> None:
        """One argument is one element, never a whole data value."""
        assert Uint16List4.of(7) == Uint16List4(data=[Uint16(7)])

    def test_of_vector(self) -> None:
        """Vectors build from exactly LENGTH element arguments."""
        assert Uint16Vector2.of(1, 2) == Uint16Vector2(data=[Uint16(1), Uint16(2)])

    def test_of_bitvector(self) -> None:
        """Bitfields build from one bool argument per bit."""
        expected = SmallBitvector(data=[Boolean(True), Boolean(False), Boolean(True)])
        assert SmallBitvector.of(True, False, True) == expected

    def test_of_bitlist_accepts_splatted_bits(self) -> None:
        """An existing bit sequence splats into element arguments."""
        bits = [True, False]
        assert SmallBitlist.of(*bits) == SmallBitlist(data=[Boolean(True), Boolean(False)])

    def test_of_byte_list_elements_are_ints(self) -> None:
        """A byte list's elements are individual byte values."""
        assert SmallByteList.of(0xDE, 0xAD) == SmallByteList(data=b"\xde\xad")

    def test_of_returns_the_subclass_type(self) -> None:
        """The factory binds to the concrete subclass, not the base."""
        assert type(Uint16List4.of(1)) is Uint16List4

    def test_constructors_stay_keyword_only(self) -> None:
        """Positional constructor arguments stay rejected — `of` is the positional form."""
        with pytest.raises(TypeError):
            cast(Any, Uint16List4)([1, 2])
        with pytest.raises(TypeError):
            cast(Any, TwoFieldContainer)(Uint8(1), Uint16(2))


class TestSSZCollectionDataDonation:
    """
    Tests that a collection given as the data value donates its own data.

    A Pydantic model iterates as field pairs, so without explicit handling a
    collection input would corrupt the iterable path. Donation enables
    copy-construction and conversion between classes sharing an element shape.
    """

    def test_list_accepts_an_equivalent_list(self) -> None:
        """A list built from another list class holds the same elements."""

        class OtherUint16List4(List[Uint16]):
            """A distinct class with the same element shape."""

            LIMIT = Uint64(4)

        source = OtherUint16List4(data=[Uint16(1), Uint16(2)])
        assert Uint16List4(data=cast(Any, source)) == Uint16List4(data=[Uint16(1), Uint16(2)])

    def test_vector_accepts_an_equivalent_vector(self) -> None:
        """A vector built from another vector class holds the same elements."""

        class OtherUint16Vector2(Vector[Uint16]):
            """A distinct class with the same element shape."""

            LENGTH = Uint64(2)

        source = OtherUint16Vector2(data=[Uint16(1), Uint16(2)])
        expected = Uint16Vector2(data=[Uint16(1), Uint16(2)])
        assert Uint16Vector2(data=cast(Any, source)) == expected

    def test_bitvector_accepts_an_equivalent_bitvector(self) -> None:
        """A bitvector built from another bitvector class holds the same bits."""

        class OtherSmallBitvector(BaseBitvector):
            """A distinct class with the same bit count."""

            LENGTH = Uint64(3)

        source = OtherSmallBitvector(data=[Boolean(True), Boolean(False), Boolean(True)])
        expected = SmallBitvector(data=[Boolean(True), Boolean(False), Boolean(True)])
        assert SmallBitvector(data=cast(Any, source)) == expected

    def test_bitlist_accepts_an_equivalent_bitlist(self) -> None:
        """A bitlist built from another bitlist class holds the same bits."""

        class OtherSmallBitlist(BaseBitlist):
            """A distinct class with the same limit."""

            LIMIT = Uint64(8)

        source = OtherSmallBitlist(data=[Boolean(True), Boolean(False)])
        expected = SmallBitlist(data=[Boolean(True), Boolean(False)])
        assert SmallBitlist(data=cast(Any, source)) == expected

    def test_byte_list_accepts_an_equivalent_byte_list(self) -> None:
        """A byte list built from another byte list class holds the same payload."""

        class OtherSmallByteList(BaseByteList):
            """A distinct class with the same limit."""

            LIMIT = Uint64(10)

        source = OtherSmallByteList(data=b"\xde\xad")
        assert SmallByteList(data=cast(Any, source)) == SmallByteList(data=b"\xde\xad")


class TestSSZCollectionBoundTypes:
    """Tests that declared size bounds must be Uint64 values."""

    def test_plain_int_length_rejected(self) -> None:
        """A LENGTH declared as a plain int is rejected at class definition."""
        with pytest.raises(SSZTypeError) as exception_info:

            class IntLengthBitvector(BaseBitvector):
                LENGTH = 3

        assert str(exception_info.value) == "IntLengthBitvector must define LENGTH as a Uint64"

    def test_plain_int_limit_rejected(self) -> None:
        """A LIMIT declared as a plain int is rejected at class definition."""
        with pytest.raises(SSZTypeError) as exception_info:

            class IntLimitList(List[Uint16]):
                LIMIT = 4

        assert str(exception_info.value) == "IntLimitList must define LIMIT as a Uint64"

    def test_uint64_subclass_bound_accepted(self) -> None:
        """A Uint64 subclass (a typed spec constant) is a valid bound."""

        class Slot(Uint64):
            """A Uint64 subtype, as spec constants are."""

        class SlotBoundList(List[Uint16]):
            LIMIT = Slot(4)

        assert SlotBoundList(data=[Uint16(1)]).encode_bytes() == b"\x01\x00"


class TestSSZCollectionMutation:
    """
    Tests for in-place collection mutation.

    Collections are mutable, unlike containers: element assignment, append, and
    pop revalidate the whole collection, so size bounds and element coercion
    behave exactly as they do at construction. The data tuple itself stays
    immutable, so validation cannot be bypassed through the field.
    """

    def test_setitem_replaces_and_coerces(self) -> None:
        """Integer index assignment coerces the value into the element type."""
        values = Uint16List4(data=[Uint16(1), Uint16(2)])
        values[1] = 9
        assert values == Uint16List4(data=[Uint16(1), Uint16(9)])

    def test_setitem_slice_revalidates(self) -> None:
        """Slice assignment replaces a range of elements."""
        bits = SmallBitvector(data=[Boolean(True), Boolean(True), Boolean(True)])
        bits[1:] = [Boolean(False), Boolean(False)]
        assert bits == SmallBitvector(data=[Boolean(True), Boolean(False), Boolean(False)])

    def test_append_grows_within_limit(self) -> None:
        """Append adds one element while under the limit."""
        values = Uint16List4(data=[Uint16(1)])
        values.append(Uint16(2))
        assert values == Uint16List4(data=[Uint16(1), Uint16(2)])

    def test_append_beyond_limit_rejected(self) -> None:
        """Append past the limit fails revalidation and raises."""
        values = Uint16List4(data=[Uint16(1)] * 4)
        with pytest.raises((SSZValueError, ValidationError)):
            values.append(Uint16(5))

    def test_append_on_fixed_length_rejected(self) -> None:
        """A fixed-length shape rejects any length change at revalidation."""
        vec = Uint16Vector2(data=[Uint16(1), Uint16(2)])
        with pytest.raises((SSZValueError, ValidationError)):
            vec.append(Uint16(3))

    def test_pop_returns_last_and_shrinks(self) -> None:
        """Pop removes and returns the final element."""
        values = Uint16List4(data=[Uint16(1), Uint16(2)])
        assert values.pop() == Uint16(2)
        assert values == Uint16List4(data=[Uint16(1)])

    def test_byte_list_setitem_replaces_byte(self) -> None:
        """Byte lists mutate by integer byte value."""
        payload = SmallByteList(data=b"\xde\xad")
        payload[0] = 0xBE
        assert payload == SmallByteList(data=b"\xbe\xad")

    def test_direct_data_assignment_revalidates(self) -> None:
        """Assigning the data field directly runs the same validation as construction."""
        values = Uint16List4(data=[Uint16(1)])
        values.data = cast(Any, [2, 3])
        assert values == Uint16List4(data=[Uint16(2), Uint16(3)])
        with pytest.raises((SSZValueError, ValidationError)):
            values.data = cast(Any, [1, 2, 3, 4, 5])

    def test_container_field_assignment_coerces(self) -> None:
        """Containers are mutable; assigned values coerce into the field type."""
        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        container.x = 3
        assert container == TwoFieldContainer(x=Uint8(3), y=Uint16(2))

    def test_container_collection_field_assignment_coerces(self) -> None:
        """A raw payload assigned to a collection field goes through its type."""
        container = ThreeFieldContainer()  # ty: ignore[missing-argument]
        container.c = [1, 2]
        assert container.c == Uint16List4(data=[Uint16(1), Uint16(2)])

    def test_container_assignment_of_typed_value_passes_through(self) -> None:
        """An already-typed value is assigned without re-coercion."""
        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        container.y = Uint16(9)
        assert container.y == Uint16(9)

    def test_container_unknown_attribute_assignment_raises(self) -> None:
        """Assigning an attribute that is not a field still raises."""
        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        with pytest.raises((AttributeError, ValueError)):
            container.unknown = 1

    def test_container_hashes_by_tree_root(self) -> None:
        """Containers hash by Merkle root, so they work as dict keys."""
        first = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        second = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        assert hash(first) == hash(second)
        lookup = {first: "found"}
        assert lookup[second] == "found"


class TestSSZDefaults:
    """
    Tests that zero-argument construction yields the SSZ default of every type.

    The SSZ spec defines a default (zero) value for each type; construction with
    no arguments produces it, recursively for composite types.
    """

    def test_boolean_defaults_to_false(self) -> None:
        """Boolean() is False."""
        assert Boolean() == Boolean(False)

    def test_uint_defaults_to_zero(self) -> None:
        """Uint64() is zero."""
        assert Uint64() == Uint64(0)

    def test_bytes_default_to_zero_fill(self) -> None:
        """A fixed byte array defaults to all-zero bytes."""

        class Bytes4(BaseBytes):
            LENGTH = 4

        assert Bytes4() == Bytes4(b"\x00\x00\x00\x00")

    def test_list_defaults_to_empty(self) -> None:
        """A list defaults to no elements."""
        assert Uint16List4() == Uint16List4(data=[])

    def test_vector_defaults_to_length_default_elements(self) -> None:
        """A vector defaults to LENGTH default-valued elements."""
        assert Uint16Vector2() == Uint16Vector2(data=[Uint16(0), Uint16(0)])

    def test_bitvector_defaults_to_length_false_bits(self) -> None:
        """A bitvector defaults to LENGTH false bits."""
        assert SmallBitvector() == SmallBitvector(data=[Boolean(False)] * 3)

    def test_bitlist_defaults_to_empty(self) -> None:
        """A bitlist defaults to no bits."""
        assert SmallBitlist() == SmallBitlist(data=[])

    def test_byte_list_defaults_to_empty(self) -> None:
        """A byte list defaults to no bytes."""
        assert SmallByteList() == SmallByteList(data=b"")

    def test_container_fills_missing_fields_with_defaults(self) -> None:
        """Unspecified container fields take their SSZ default values."""
        container = TwoFieldContainer(x=Uint8(7))  # ty: ignore[missing-argument]
        assert container == TwoFieldContainer(x=Uint8(7), y=Uint16(0))

    def test_empty_container_is_the_default_container(self) -> None:
        """A container with no arguments defaults every field recursively."""
        container = ThreeFieldContainer()  # ty: ignore[missing-argument]
        expected = ThreeFieldContainer(a=Uint8(0), b=Uint64(0), c=Uint16List4(data=[]))
        assert container == expected

    def test_container_coerces_raw_collection_payloads(self) -> None:
        """A raw list for a collection field is coerced through the field's type."""
        container = ThreeFieldContainer(c=cast(Any, [1, 2]))  # ty: ignore[missing-argument]
        assert container.c == Uint16List4(data=[Uint16(1), Uint16(2)])

    def test_non_dict_input_passes_through_to_pydantic(self) -> None:
        """Non-dict inputs bypass default filling and fail model validation itself."""
        with pytest.raises(ValidationError):
            TwoFieldContainer.model_validate([1, 2])


class TestSSZMethodForms:
    """Tests for the fluent method forms of hash_tree_root and copy."""

    def test_hash_tree_root_method_matches_function(self) -> None:
        """The method form returns exactly what the function computes."""
        from ssz.merkleization import hash_tree_root

        container = TwoFieldContainer(x=Uint8(1), y=Uint16(2))
        assert container.hash_tree_root() == hash_tree_root(container)
        assert Uint64(7).hash_tree_root() == hash_tree_root(Uint64(7))

    def test_copy_is_deep_and_independent(self) -> None:
        """Mutating a copy never affects the original."""
        original = Uint16List4(data=[Uint16(1), Uint16(2)])
        duplicate = original.copy()
        duplicate[0] = 9
        assert original == Uint16List4(data=[Uint16(1), Uint16(2)])
        assert duplicate == Uint16List4(data=[Uint16(9), Uint16(2)])


class TestValueEquality:
    """SSZ models compare by Merkle tree root."""

    def test_equivalent_classes_compare_equal(self) -> None:
        class PointA(Container):
            x: Uint64
            y: Uint64

        class PointB(Container):
            x: Uint64
            y: Uint64

        a = PointA(x=Uint64(1), y=Uint64(2))
        b = PointB(x=Uint64(1), y=Uint64(2))
        assert a == b
        assert hash(a) == hash(b)
        assert a != PointB(x=Uint64(1), y=Uint64(3))

    def test_collections_compare_by_root(self) -> None:
        class Nums8A(List[Uint64]):
            LIMIT = Uint64(8)

        class Nums8B(List[Uint64]):
            LIMIT = Uint64(8)

        assert Nums8A.of(Uint64(1)) == Nums8B.of(Uint64(1))
        assert hash(Nums8A.of(Uint64(1))) == hash(Nums8B.of(Uint64(1)))

    def test_non_ssz_operand_is_not_equal(self) -> None:
        class Point(Container):
            x: Uint64

        assert Point(x=Uint64(1)) != {"x": 1}
