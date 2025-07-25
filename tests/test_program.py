# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LGPL-2.1-or-later
import ctypes
import itertools
import os
import sys
import tempfile
import unittest.mock

from _drgn_util.elf import ET, PT
from drgn import (
    Architecture,
    FaultError,
    FindObjectFlags,
    Language,
    NoDefaultProgramError,
    Object,
    ObjectNotFoundError,
    Platform,
    PlatformFlags,
    Program,
    ProgramFlags,
    Qualifiers,
    TypeKind,
    TypeMember,
    get_default_prog,
    host_platform,
    set_default_prog,
)
from tests import (
    DEFAULT_LANGUAGE,
    MOCK_32BIT_PLATFORM,
    MOCK_PLATFORM,
    MockMemorySegment,
    MockObject,
    MockProgramTestCase,
    TestCase,
    mock_program,
)
from tests.elfwriter import ElfSection, create_elf_file


def zero_memory_read(address, count, offset, physical):
    return bytes(count)


class TestProgram(TestCase):
    def test_default_program(self):
        self.assertRaises(NoDefaultProgramError, get_default_prog)
        prog = Program()
        prog2 = Program()
        try:
            set_default_prog(prog)
            self.assertIs(get_default_prog(), prog)
            set_default_prog(prog2)
            self.assertIs(get_default_prog(), prog2)
        finally:
            set_default_prog(None)
        self.assertRaises(NoDefaultProgramError, get_default_prog)

    def test_default_program_reference_counting(self):
        try:
            set_default_prog(Program())
            self.assertGreater(sys.getrefcount(get_default_prog()), 1)
        finally:
            set_default_prog(None)

    def test_set_pid(self):
        # Debug the running Python interpreter itself.
        prog = Program()
        self.assertIsNone(prog.platform)
        self.assertFalse(prog.flags & ProgramFlags.IS_LIVE)
        prog.set_pid(os.getpid())
        self.assertIsNone(prog.core_dump_path)
        self.assertEqual(prog.platform, host_platform)
        self.assertTrue(prog.flags & ProgramFlags.IS_LIVE)
        self.assertRaisesRegex(
            ValueError,
            "program memory was already initialized",
            prog.set_pid,
            os.getpid(),
        )

    def test_pid_memory(self):
        data = b"hello, world!"
        buf = ctypes.create_string_buffer(data)
        address = ctypes.addressof(buf)

        # QEMU user-mode emulation doesn't seem to emulate /proc/$pid/mem
        # correctly on a 64-bit host with a 32-bit guest; see
        # https://gitlab.com/qemu-project/qemu/-/issues/698. Packit uses mock
        # to cross-compile and test packages, which in turn uses QEMU user-mode
        # emulation. Skip this test if /proc/$pid/mem doesn't work so that
        # those builds succeed.
        try:
            with open("/proc/self/mem", "rb") as f:
                f.seek(address)
                functional_proc_pid_mem = f.read(len(data)) == data
        except OSError:
            functional_proc_pid_mem = False
        if not functional_proc_pid_mem:
            self.skipTest("/proc/$pid/mem is not functional")

        prog = Program()
        prog.set_pid(os.getpid())

        self.assertEqual(prog.read(ctypes.addressof(buf), len(data)), data)

    def test_object_not_found_error(self):
        prog = mock_program()

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find constant 'foo'$"
        ) as cm:
            prog.constant("foo")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find constant 'foo' in 'foo.c'$"
        ) as cm:
            prog.constant("foo", "foo.c")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find function 'foo'$"
        ) as cm:
            prog.function("foo")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find function 'foo' in 'foo.c'$"
        ) as cm:
            prog.function("foo", "foo.c")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find variable 'foo'$"
        ) as cm:
            prog.variable("foo")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find variable 'foo' in 'foo.c'$"
        ) as cm:
            prog.variable("foo", "foo.c")
        self.assertEqual(cm.exception.name, "foo")

        with self.assertRaisesRegex(
            ObjectNotFoundError, "^could not find 'foo'$"
        ) as cm:
            prog["foo"]
        self.assertEqual(cm.exception.name, "foo")

        # If name isn't a string, prog.object(name) should raise TypeError, and
        # prog[name] should raise KeyError (not ObjectNotFoundError).
        self.assertRaises(TypeError, prog.object, 9)
        with self.assertRaises(KeyError) as cm:
            prog[9]
        self.assertIs(type(cm.exception), KeyError)

    def test_type_lookup_error(self):
        prog = mock_program()

        self.assertRaisesRegex(LookupError, "^could not find 'foo'$", prog.type, "foo")
        self.assertRaisesRegex(
            LookupError, "^could not find 'foo' in 'foo.c'$", prog.type, "foo", "foo.c"
        )

    def test_flags(self):
        self.assertIsInstance(mock_program().flags, ProgramFlags)

    def test_debug_info(self):
        Program().load_debug_info([])

    def test_language(self):
        prog = Program()
        self.assertEqual(prog.language, DEFAULT_LANGUAGE)
        prog.language = Language.CPP
        self.assertEqual(prog.language, Language.CPP)
        prog.language = Language.C
        self.assertEqual(prog.language, Language.C)
        self.assertRaisesRegex(
            TypeError, "language must be Language", setattr, prog, "language", "CPP"
        )

    def test_language_del(self):
        with self.assertRaises(AttributeError):
            del Program().language


class TestMemory(TestCase):
    def test_simple_read(self):
        data = b"hello, world"
        prog = mock_program(segments=[MockMemorySegment(data, 0xFFFF0000, 0xA0)])
        self.assertEqual(prog.read(0xFFFF0000, len(data)), data)
        self.assertEqual(prog.read(0xA0, len(data), True), data)

    def test_read_unsigned(self):
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        for word_size in [8, 4]:
            for byteorder in ["little", "big"]:
                flags = PlatformFlags(0)
                if word_size == 8:
                    flags |= PlatformFlags.IS_64_BIT
                if byteorder == "little":
                    flags |= PlatformFlags.IS_LITTLE_ENDIAN
                prog = mock_program(
                    Platform(Architecture.UNKNOWN, flags),
                    segments=[MockMemorySegment(data, 0xFFFF0000, 0xA0)],
                )
                for size in [1, 2, 4, 8]:
                    read_fn = getattr(prog, f"read_u{8 * size}")
                    value = int.from_bytes(data[:size], byteorder)
                    self.assertEqual(read_fn(0xFFFF0000), value)
                    self.assertEqual(read_fn(0xA0, True), value)
                    if size == word_size:
                        self.assertEqual(prog.read_word(0xFFFF0000), value)
                        self.assertEqual(prog.read_word(0xA0, True), value)

        prog = mock_program(
            MOCK_32BIT_PLATFORM, segments=[MockMemorySegment(data, 0xFFFF0000, 0xA0)]
        )

    def test_bad_address(self):
        data = b"hello, world!"
        prog = mock_program(segments=[MockMemorySegment(data, 0xFFFF0000)])
        self.assertRaisesRegex(
            FaultError, "could not find memory segment", prog.read, 0xDEADBEEF, 4
        )
        self.assertRaisesRegex(
            FaultError,
            "could not find physical memory segment",
            prog.read,
            0xFFFF0000,
            4,
            True,
        )

    def test_segment_overflow(self):
        data = b"hello, world!"
        prog = mock_program(segments=[MockMemorySegment(data, 0xFFFF0000)])
        self.assertRaisesRegex(
            FaultError,
            "could not find memory segment",
            prog.read,
            0xFFFF0000,
            len(data) + 1,
        )

    def test_adjacent_segments(self):
        data = b"hello, world!\0foobar"
        prog = mock_program(
            segments=[
                MockMemorySegment(data[:4], 0xFFFF0000),
                MockMemorySegment(data[4:14], 0xFFFF0004),
                MockMemorySegment(data[14:], 0xFFFFF000),
            ]
        )
        self.assertEqual(prog.read(0xFFFF0000, 14), data[:14])

    def test_address_overflow(self):
        for bits in (64, 32):
            with self.subTest(bits=bits):
                prog = mock_program(
                    segments=[
                        MockMemorySegment(b"cd", 0x0),
                        MockMemorySegment(b"abyz", 2**bits - 2),
                    ],
                    platform=MOCK_PLATFORM if bits == 64 else MOCK_32BIT_PLATFORM,
                )
                for start in range(3):
                    for size in range(4 - start):
                        self.assertEqual(
                            prog.read((2**bits - 2 + start) % 2**64, size),
                            b"abcd"[start : start + size],
                        )

    def test_overlap_same_address_smaller_size(self):
        # Existing segment: |_______|
        # New segment:      |___|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 128, segment1)
        prog.add_memory_segment(0xFFFF0000, 64, segment2)
        prog.read(0xFFFF0000, 128)
        segment1.assert_called_once_with(0xFFFF0040, 64, 64, False)
        segment2.assert_called_once_with(0xFFFF0000, 64, 0, False)

    def test_overlap_within_segment(self):
        # Existing segment: |_______|
        # New segment:        |___|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 128, segment1)
        prog.add_memory_segment(0xFFFF0020, 64, segment2)
        prog.read(0xFFFF0000, 128)
        segment1.assert_has_calls(
            [
                unittest.mock.call(0xFFFF0000, 32, 00, False),
                unittest.mock.call(0xFFFF0060, 32, 96, False),
            ]
        )
        segment2.assert_called_once_with(0xFFFF0020, 64, 0, False)

    def test_overlap_same_segment(self):
        # Existing segment: |_______|
        # New segment:      |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 128, segment1)
        prog.add_memory_segment(0xFFFF0000, 128, segment2)
        prog.read(0xFFFF0000, 128)
        segment1.assert_not_called()
        segment2.assert_called_once_with(0xFFFF0000, 128, 0, False)

    def test_overlap_same_address_larger_size(self):
        # Existing segment: |___|
        # New segment:      |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 64, segment1)
        prog.add_memory_segment(0xFFFF0000, 128, segment2)
        prog.read(0xFFFF0000, 128)
        segment1.assert_not_called()
        segment2.assert_called_once_with(0xFFFF0000, 128, 0, False)

    def test_overlap_segment_tail(self):
        # Existing segment: |_______|
        # New segment:          |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 128, segment1)
        prog.add_memory_segment(0xFFFF0040, 128, segment2)
        prog.read(0xFFFF0000, 192)
        segment1.assert_called_once_with(0xFFFF0000, 64, 0, False)
        segment2.assert_called_once_with(0xFFFF0040, 128, 0, False)

    def test_overlap_subsume_after(self):
        # Existing segments:   |_|_|_|_|
        # New segment:       |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment3 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0020, 32, segment1)
        prog.add_memory_segment(0xFFFF0040, 32, segment1)
        prog.add_memory_segment(0xFFFF0060, 32, segment1)
        prog.add_memory_segment(0xFFFF0080, 64, segment2)
        prog.add_memory_segment(0xFFFF0000, 128, segment3)
        prog.read(0xFFFF0000, 192)
        segment1.assert_not_called()
        segment2.assert_called_once_with(0xFFFF0080, 64, 0, False)
        segment3.assert_called_once_with(0xFFFF0000, 128, 0, False)

    def test_overlap_segment_head(self):
        # Existing segment:     |_______|
        # New segment:      |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0040, 128, segment1)
        prog.add_memory_segment(0xFFFF0000, 128, segment2)
        prog.read(0xFFFF0000, 192)
        segment1.assert_called_once_with(0xFFFF0080, 64, 64, False)
        segment2.assert_called_once_with(0xFFFF0000, 128, 0, False)

    def test_overlap_segment_head_and_tail(self):
        # Existing segment: |_______||_______|
        # New segment:          |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment3 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 128, segment1)
        prog.add_memory_segment(0xFFFF0080, 128, segment2)
        prog.add_memory_segment(0xFFFF0040, 128, segment3)
        prog.read(0xFFFF0000, 256)
        segment1.assert_called_once_with(0xFFFF0000, 64, 0, False)
        segment2.assert_called_once_with(0xFFFF00C0, 64, 64, False)
        segment3.assert_called_once_with(0xFFFF0040, 128, 0, False)

    def test_overlap_subsume_at_and_after(self):
        # Existing segments: |_|_|_|_|
        # New segment:       |_______|
        prog = Program(MOCK_PLATFORM)
        segment1 = unittest.mock.Mock(side_effect=zero_memory_read)
        segment2 = unittest.mock.Mock(side_effect=zero_memory_read)
        prog.add_memory_segment(0xFFFF0000, 32, segment1)
        prog.add_memory_segment(0xFFFF0020, 32, segment1)
        prog.add_memory_segment(0xFFFF0040, 32, segment1)
        prog.add_memory_segment(0xFFFF0060, 32, segment1)
        prog.add_memory_segment(0xFFFF0000, 128, segment2)
        prog.read(0xFFFF0000, 128)
        segment1.assert_not_called()
        segment2.assert_called_once_with(0xFFFF0000, 128, 0, False)

    def test_invalid_read_fn(self):
        prog = mock_program()

        self.assertRaises(TypeError, prog.add_memory_segment, 0xFFFF0000, 8, b"foo")

        prog.add_memory_segment(0xFFFF0000, 8, lambda: None)
        self.assertRaises(TypeError, prog.read, 0xFFFF0000, 8)

        prog.add_memory_segment(
            0xFFFF0000, 8, lambda address, count, offset, physical: None
        )
        self.assertRaises(TypeError, prog.read, 0xFFFF0000, 8)

        prog.add_memory_segment(
            0xFFFF0000, 8, lambda address, count, offset, physical: "asdf"
        )
        self.assertRaises(TypeError, prog.read, 0xFFFF0000, 8)

        prog.add_memory_segment(
            0xFFFF0000, 8, lambda address, count, offset, physical: b""
        )
        self.assertRaisesRegex(
            ValueError,
            r"memory read callback returned buffer of length 0 \(expected 8\)",
            prog.read,
            0xFFFF0000,
            8,
        )

    def test_python_fault_error(self):
        def fault_memory_reader(address, count, offset, physical):
            raise FaultError("fault from Python", address)

        prog = Program(MOCK_PLATFORM)
        prog.add_memory_segment(0xFFFF0000, 8, fault_memory_reader)

        with self.assertRaises(FaultError) as cm:
            Object(prog, "int", address=0xFFFF0004).read_()
        self.assertEqual(cm.exception.message, "fault from Python")
        self.assertEqual(cm.exception.address, 0xFFFF0004)

        # If the FaultError from Python is translated to a drgn_error
        # correctly, then this shouldn't raise an exception.
        str(Object(prog, "int *", 0xFFFF0004))

    def test_python_fault_error_invalid_message(self):
        def fault_memory_reader(address, count, offset, physical):
            raise FaultError(None, address)

        prog = Program(MOCK_PLATFORM)
        prog.add_memory_segment(0xFFFF0000, 8, fault_memory_reader)

        # Just test that it doesn't crash.
        self.assertRaises(Exception, Object(prog, "int", address=0xFFFF0004).read_)

    def test_python_fault_error_invalid_address(self):
        def fault_memory_reader(address, count, offset, physical):
            raise FaultError("fault from Python", None)

        prog = Program(MOCK_PLATFORM)
        prog.add_memory_segment(0xFFFF0000, 8, fault_memory_reader)

        # Just test that it doesn't crash.
        self.assertRaises(Exception, Object(prog, "int", address=0xFFFF0004).read_)


class TestTypeFinder(TestCase):
    def test_register(self):
        prog = Program(MOCK_PLATFORM)

        # We don't test every corner case because the symbol finder tests cover
        # the shared part.
        self.assertEqual(prog.registered_type_finders(), {"dwarf"})
        self.assertEqual(prog.enabled_type_finders(), ["dwarf"])

        prog.register_type_finder(
            "foo", lambda prog, kinds, name, filename: None, enable_index=-1
        )
        self.assertEqual(prog.registered_type_finders(), {"dwarf", "foo"})
        self.assertEqual(prog.enabled_type_finders(), ["dwarf", "foo"])

        prog.set_enabled_type_finders(["foo"])
        self.assertEqual(prog.registered_type_finders(), {"dwarf", "foo"})
        self.assertEqual(prog.enabled_type_finders(), ["foo"])

    def test_add_type_finder(self):
        prog = Program(MOCK_PLATFORM)

        def dummy(kind, name, filename):
            if kind == TypeKind.TYPEDEF and name == "foo":
                return prog.typedef_type("foo", prog.void_type())
            else:
                return None

        prog.add_type_finder(dummy)
        self.assertTrue(any("dummy" in name for name in prog.registered_type_finders()))
        self.assertIn("dummy", prog.enabled_type_finders()[0])
        self.assertIdentical(
            prog.type("foo"), prog.typedef_type("foo", prog.void_type())
        )

    def test_register_invalid(self):
        prog = Program(MOCK_PLATFORM)
        self.assertRaises(TypeError, prog.register_type_finder, "foo", "foo")
        prog.register_type_finder(
            "foo", lambda prog, kinds, name, filename: "foo", enable_index=0
        )
        self.assertRaises(TypeError, prog.type, "int")

    def test_add_invalid(self):
        prog = Program(MOCK_PLATFORM)
        self.assertRaises(TypeError, prog.add_type_finder, "foo")
        prog.add_type_finder(lambda kind, name, filename: "foo")
        self.assertRaises(TypeError, prog.type, "int")

    def test_register_wrong_program(self):
        def finder(prog, kinds, name, filename):
            if TypeKind.TYPEDEF in kinds and name == "foo":
                prog = Program()
                return prog.typedef_type("foo", prog.void_type())
            else:
                return None

        prog = Program(MOCK_PLATFORM)
        prog.register_type_finder("foo", finder, enable_index=0)
        self.assertRaisesRegex(
            ValueError,
            "type find callback returned type from wrong program",
            prog.type,
            "foo",
        )

    def test_add_wrong_program(self):
        def finder(kind, name, filename):
            if kind == TypeKind.TYPEDEF and name == "foo":
                prog = Program()
                return prog.typedef_type("foo", prog.void_type())
            else:
                return None

        prog = Program(MOCK_PLATFORM)
        prog.add_type_finder(finder)
        self.assertRaisesRegex(
            ValueError,
            "type find callback returned type from wrong program",
            prog.type,
            "foo",
        )

    def test_register_wrong_kind(self):
        prog = Program(MOCK_PLATFORM)
        prog.register_type_finder(
            "foo", lambda prog, kinds, name, filename: prog.void_type(), enable_index=0
        )
        self.assertRaises(TypeError, prog.type, "int")

    def test_add_wrong_kind(self):
        prog = Program(MOCK_PLATFORM)
        prog.add_type_finder(lambda kind, name, filename: prog.void_type())
        self.assertRaises(TypeError, prog.type, "int")

    def test_register_not_found(self):
        prog = Program(MOCK_PLATFORM)
        prog.register_type_finder(
            "foo", lambda prog, kinds, name, filename: None, enable_index=0
        )
        self.assertRaises(LookupError, prog.type, "struct foo")

    def test_add_not_found(self):
        prog = Program(MOCK_PLATFORM)
        prog.add_type_finder(lambda kind, name, filename: None)
        self.assertRaises(LookupError, prog.type, "struct foo")


class TestObjectFinder(TestCase):
    def test_register(self):
        prog = Program(MOCK_PLATFORM)

        # We don't test every corner case because the symbol finder tests cover
        # the shared part.
        self.assertEqual(prog.registered_object_finders(), {"dwarf"})
        self.assertEqual(prog.enabled_object_finders(), ["dwarf"])

        prog.register_object_finder(
            "foo", lambda prog, name, flags, filename: None, enable_index=-1
        )
        self.assertEqual(prog.registered_object_finders(), {"dwarf", "foo"})
        self.assertEqual(prog.enabled_object_finders(), ["dwarf", "foo"])

        prog.set_enabled_object_finders(["foo"])
        self.assertEqual(prog.registered_object_finders(), {"dwarf", "foo"})
        self.assertEqual(prog.enabled_object_finders(), ["foo"])

    def test_add_object_finder(self):
        prog = Program(MOCK_PLATFORM)

        def dummy(prog, name, flags, filename):
            return Object(prog, "int", 1)

        prog.add_object_finder(dummy)
        self.assertTrue(
            any("dummy" in name for name in prog.registered_object_finders())
        )
        self.assertIn("dummy", prog.enabled_object_finders()[0])
        self.assertIdentical(prog.object("foo"), Object(prog, "int", 1))

    def test_register_invalid(self):
        prog = Program(MOCK_PLATFORM)
        self.assertRaises(TypeError, prog.register_object_finder, "foo", "foo")
        prog.register_object_finder(
            "foo", lambda prog, name, flags, filename: "foo", enable_index=0
        )
        self.assertRaises(TypeError, prog.object, "foo")

    def test_add_invalid(self):
        prog = Program(MOCK_PLATFORM)
        self.assertRaises(TypeError, prog.add_object_finder, "foo")
        prog.add_object_finder(lambda prog, name, flags, filename: "foo")
        self.assertRaises(TypeError, prog.object, "foo")

    def test_wrong_program(self):
        prog = Program(MOCK_PLATFORM)
        prog.register_object_finder(
            "foo",
            lambda prog, name, flags, filename: Object(
                Program(MOCK_PLATFORM), "int", 1
            ),
            enable_index=0,
        )
        self.assertRaisesRegex(
            ValueError,
            "different program",
            prog.object,
            "foo",
        )

    def test_not_found(self):
        prog = Program(MOCK_PLATFORM)
        self.assertRaises(LookupError, prog.object, "foo")
        prog.register_object_finder(
            "foo", lambda prog, name, flags, filename: None, enable_index=0
        )
        self.assertRaises(LookupError, prog.object, "foo")
        self.assertFalse("foo" in prog)


class TestTypes(MockProgramTestCase):
    def test_already_type(self):
        self.assertIdentical(
            self.prog.type(self.prog.pointer_type(self.prog.void_type())),
            self.prog.pointer_type(self.prog.void_type()),
        )

    def test_invalid_argument_type(self):
        self.assertRaises(TypeError, self.prog.type, 1)

    def test_default_primitive_types(self):
        def spellings(tokens, num_optional=0):
            for i in range(len(tokens) - num_optional, len(tokens) + 1):
                for perm in itertools.permutations(tokens[:i]):
                    yield " ".join(perm)

        for word_size in [8, 4]:
            prog = mock_program(
                MOCK_PLATFORM if word_size == 8 else MOCK_32BIT_PLATFORM
            )
            self.assertIdentical(prog.type("_Bool"), prog.bool_type("_Bool", 1))
            self.assertIdentical(prog.type("char"), prog.int_type("char", 1, True))
            for spelling in spellings(["signed", "char"]):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("signed char", 1, True)
                )
            for spelling in spellings(["unsigned", "char"]):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("unsigned char", 1, False)
                )
            for spelling in spellings(["short", "signed", "int"], 2):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("short", 2, True)
                )
            for spelling in spellings(["short", "unsigned", "int"], 1):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("unsigned short", 2, False)
                )
            for spelling in spellings(["int", "signed"], 1):
                self.assertIdentical(prog.type(spelling), prog.int_type("int", 4, True))
            for spelling in spellings(["unsigned", "int"]):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("unsigned int", 4, False)
                )
            for spelling in spellings(["long", "signed", "int"], 2):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("long", word_size, True)
                )
            for spelling in spellings(["long", "unsigned", "int"], 1):
                self.assertIdentical(
                    prog.type(spelling),
                    prog.int_type("unsigned long", word_size, False),
                )
            for spelling in spellings(["long", "long", "signed", "int"], 2):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("long long", 8, True)
                )
            for spelling in spellings(["long", "long", "unsigned", "int"], 1):
                self.assertIdentical(
                    prog.type(spelling), prog.int_type("unsigned long long", 8, False)
                )
            self.assertIdentical(prog.type("float"), prog.float_type("float", 4))
            self.assertIdentical(prog.type("double"), prog.float_type("double", 8))
            for spelling in spellings(["long", "double"]):
                self.assertIdentical(
                    prog.type(spelling), prog.float_type("long double", 16)
                )
            self.assertIdentical(
                prog.type("size_t"),
                prog.typedef_type(
                    "size_t", prog.int_type("unsigned long", word_size, False)
                ),
            )
            self.assertIdentical(
                prog.type("ptrdiff_t"),
                prog.typedef_type("ptrdiff_t", prog.int_type("long", word_size, True)),
            )

    def test_primitive_type(self):
        self.types.append(self.prog.int_type("long", 4, True))
        self.assertIdentical(
            self.prog.type("long"), self.prog.int_type("long", 4, True)
        )

    def test_primitive_type_invalid(self):
        # unsigned long with signed=True isn't valid, so it should be ignored.
        self.types.append(self.prog.int_type("unsigned long", 4, True))
        self.assertIdentical(
            self.prog.type("unsigned long"),
            self.prog.int_type("unsigned long", 8, False),
        )

    def test_size_t_and_ptrdiff_t(self):
        # 64-bit architecture with 4-byte long/unsigned long.
        types = []
        prog = mock_program(types=types)
        types.append(prog.int_type("long", 4, True))
        types.append(prog.int_type("unsigned long", 4, False))
        self.assertIdentical(
            prog.type("size_t"),
            prog.typedef_type("size_t", prog.type("unsigned long long")),
        )
        self.assertIdentical(
            prog.type("ptrdiff_t"),
            prog.typedef_type("ptrdiff_t", prog.type("long long")),
        )

        # 32-bit architecture with 8-byte long/unsigned long.
        types = []
        prog = mock_program(MOCK_32BIT_PLATFORM, types=types)
        types.append(prog.int_type("long", 8, True))
        types.append(prog.int_type("unsigned long", 8, False))
        self.assertIdentical(
            prog.type("size_t"), prog.typedef_type("size_t", prog.type("unsigned int"))
        )
        self.assertIdentical(
            prog.type("ptrdiff_t"), prog.typedef_type("ptrdiff_t", prog.type("int"))
        )

        # Nonsense sizes.
        types = []
        prog = mock_program(types=types)
        types.append(prog.int_type("int", 1, True))
        types.append(prog.int_type("unsigned int", 1, False))
        types.append(prog.int_type("long", 1, True))
        types.append(prog.int_type("unsigned long", 1, False))
        types.append(prog.int_type("long long", 2, True))
        types.append(prog.int_type("unsigned long long", 2, False))
        self.assertRaisesRegex(
            ValueError, "no suitable integer type for size_t", prog.type, "size_t"
        )
        self.assertRaisesRegex(
            ValueError, "no suitable integer type for ptrdiff_t", prog.type, "ptrdiff_t"
        )

    def test_not_size_t_or_ptrdiff_t(self):
        self.types.append(
            self.prog.typedef_type(
                "size_tea", self.prog.int_type("unsigned char", 1, False)
            )
        )
        self.types.append(
            self.prog.typedef_type("ptrdiff_tee", self.prog.int_type("char", 1, True))
        )
        self.assertIdentical(
            self.prog.type("size_tea"),
            self.prog.typedef_type(
                "size_tea", self.prog.int_type("unsigned char", 1, False)
            ),
        )
        self.assertIdentical(
            self.prog.type("ptrdiff_tee"),
            self.prog.typedef_type("ptrdiff_tee", self.prog.int_type("char", 1, True)),
        )

    def test_tagged_type(self):
        self.types.append(self.point_type)
        self.types.append(self.option_type)
        self.types.append(self.color_type)
        self.assertIdentical(self.prog.type("struct point"), self.point_type)
        self.assertIdentical(self.prog.type("union option"), self.option_type)
        self.assertIdentical(self.prog.type("enum color"), self.color_type)

    def test_class_type(self):
        struct_class = self.prog.struct_type(
            "class",
            8,
            (TypeMember(self.prog.pointer_type(self.prog.void_type()), "ptr"),),
        )
        class_point = self.prog.class_type(
            "Point",
            8,
            (
                TypeMember(self.prog.int_type("int", 4, True), "x", 0),
                TypeMember(self.prog.int_type("int", 4, True), "y", 32),
            ),
        )
        self.types.append(struct_class)
        self.types.append(class_point)
        self.prog.language = Language.C
        self.assertIdentical(self.prog.type("struct class"), struct_class)
        self.prog.language = Language.CPP
        self.assertRaisesRegex(
            SyntaxError,
            "expected identifier after 'struct'",
            self.prog.type,
            "struct class",
        )
        self.assertIdentical(self.prog.type("class Point"), class_point)

    def test_typedef(self):
        self.types.append(self.pid_type)
        self.assertIdentical(self.prog.type("pid_t"), self.pid_type)

    def test_pointer(self):
        self.assertIdentical(
            self.prog.type("int *"),
            self.prog.pointer_type(self.prog.int_type("int", 4, True)),
        )

    def test_pointer_to_const(self):
        self.assertIdentical(
            self.prog.type("const int *"),
            self.prog.pointer_type(
                self.prog.int_type("int", 4, True, qualifiers=Qualifiers.CONST)
            ),
        )

    def test_const_pointer(self):
        self.assertIdentical(
            self.prog.type("int * const"),
            self.prog.pointer_type(
                self.prog.int_type("int", 4, True), qualifiers=Qualifiers.CONST
            ),
        )

    def test_pointer_to_pointer(self):
        self.assertIdentical(
            self.prog.type("int **"),
            self.prog.pointer_type(
                self.prog.pointer_type(self.prog.int_type("int", 4, True))
            ),
        )
        self.assertIdentical(self.prog.type("int *((*))"), self.prog.type("int **"))

    def test_pointer_to_const_pointer(self):
        self.assertIdentical(
            self.prog.type("int * const *"),
            self.prog.pointer_type(
                self.prog.pointer_type(
                    self.prog.int_type("int", 4, True), qualifiers=Qualifiers.CONST
                )
            ),
        )

    def test_array(self):
        self.assertIdentical(
            self.prog.type("int [20]"),
            self.prog.array_type(self.prog.int_type("int", 4, True), 20),
        )

    def test_array_hexadecimal(self):
        self.assertIdentical(
            self.prog.type("int [0x20]"),
            self.prog.array_type(self.prog.int_type("int", 4, True), 32),
        )

    def test_array_octal(self):
        self.assertIdentical(
            self.prog.type("int [020]"),
            self.prog.array_type(self.prog.int_type("int", 4, True), 16),
        )

    def test_incomplete_array(self):
        self.assertIdentical(
            self.prog.type("int []"),
            self.prog.array_type(self.prog.int_type("int", 4, True)),
        )

    def test_array_two_dimensional(self):
        self.assertIdentical(
            self.prog.type("int [2][3]"),
            self.prog.array_type(
                self.prog.array_type(self.prog.int_type("int", 4, True), 3), 2
            ),
        )

    def test_array_three_dimensional(self):
        self.assertIdentical(
            self.prog.type("int [2][3][4]"),
            self.prog.array_type(
                self.prog.array_type(
                    self.prog.array_type(self.prog.int_type("int", 4, True), 4), 3
                ),
                2,
            ),
        )

    def test_array_of_pointers(self):
        self.assertIdentical(
            self.prog.type("int *[2][3]"),
            self.prog.array_type(
                self.prog.array_type(
                    self.prog.pointer_type(self.prog.int_type("int", 4, True)), 3
                ),
                2,
            ),
        )

    def test_pointer_to_array(self):
        self.assertIdentical(
            self.prog.type("int (*)[2]"),
            self.prog.pointer_type(
                self.prog.array_type(self.prog.int_type("int", 4, True), 2)
            ),
        )

    def test_pointer_to_two_dimensional_array(self):
        self.assertIdentical(
            self.prog.type("int (*)[2][3]"),
            self.prog.pointer_type(
                self.prog.array_type(
                    self.prog.array_type(self.prog.int_type("int", 4, True), 3), 2
                )
            ),
        )

    def test_pointer_to_pointer_to_array(self):
        self.assertIdentical(
            self.prog.type("int (**)[2]"),
            self.prog.pointer_type(
                self.prog.pointer_type(
                    self.prog.array_type(self.prog.int_type("int", 4, True), 2)
                )
            ),
        )

    def test_pointer_to_array_of_pointers(self):
        self.assertIdentical(
            self.prog.type("int *(*)[2]"),
            self.prog.pointer_type(
                self.prog.array_type(
                    self.prog.pointer_type(self.prog.int_type("int", 4, True)), 2
                )
            ),
        )
        self.assertIdentical(
            self.prog.type("int *((*)[2])"), self.prog.type("int *(*)[2]")
        )

    def test_array_of_pointers_to_array(self):
        self.assertIdentical(
            self.prog.type("int (*[2])[3]"),
            self.prog.array_type(
                self.prog.pointer_type(
                    self.prog.array_type(self.prog.int_type("int", 4, True), 3)
                ),
                2,
            ),
        )


class TestObjects(MockProgramTestCase):
    def test_constant(self):
        self.objects.append(
            MockObject("PAGE_SIZE", self.prog.int_type("int", 4, True), value=4096)
        )
        self.assertIdentical(
            self.prog["PAGE_SIZE"],
            Object(self.prog, self.prog.int_type("int", 4, True), value=4096),
        )
        self.assertIdentical(
            self.prog.object("PAGE_SIZE", FindObjectFlags.CONSTANT),
            self.prog["PAGE_SIZE"],
        )
        self.assertTrue("PAGE_SIZE" in self.prog)

    def test_function(self):
        self.objects.append(
            MockObject(
                "func",
                self.prog.function_type(self.prog.void_type(), (), False),
                address=0xFFFF0000,
            )
        )
        self.assertIdentical(
            self.prog["func"],
            Object(
                self.prog,
                self.prog.function_type(self.prog.void_type(), (), False),
                address=0xFFFF0000,
            ),
        )
        self.assertIdentical(
            self.prog.object("func", FindObjectFlags.FUNCTION), self.prog["func"]
        )
        self.assertTrue("func" in self.prog)

    def test_variable(self):
        self.objects.append(
            MockObject(
                "counter", self.prog.int_type("int", 4, True), address=0xFFFF0000
            )
        )
        self.assertIdentical(
            self.prog["counter"],
            Object(self.prog, self.prog.int_type("int", 4, True), address=0xFFFF0000),
        )
        self.assertIdentical(
            self.prog.object("counter", FindObjectFlags.VARIABLE), self.prog["counter"]
        )
        self.assertTrue("counter" in self.prog)


class TestCoreDump(TestCase):
    def test_not_core_dump(self):
        prog = Program()
        self.assertRaisesRegex(
            ValueError, "not an ELF core file", prog.set_core_dump, "/dev/null"
        )
        with tempfile.NamedTemporaryFile() as f:
            f.write(create_elf_file(ET.EXEC, []))
            f.flush()
            self.assertRaisesRegex(
                ValueError, "not an ELF core file", prog.set_core_dump, f.name
            )

    def test_twice(self):
        prog = Program()
        with tempfile.NamedTemporaryFile() as f:
            f.write(create_elf_file(ET.CORE, []))
            f.flush()
            prog.set_core_dump(f.name)
            self.assertRaisesRegex(
                ValueError,
                "program memory was already initialized",
                prog.set_core_dump,
                f.name,
            )

    def test_simple(self):
        data = b"hello, world"
        prog = Program()
        self.assertIsNone(prog.core_dump_path)
        with tempfile.NamedTemporaryFile() as f:
            f.write(
                create_elf_file(
                    ET.CORE, [ElfSection(p_type=PT.LOAD, vaddr=0xFFFF0000, data=data)]
                )
            )
            f.flush()
            prog.set_core_dump(f.name)
        self.assertEqual(prog.core_dump_path, f.name)
        self.assertEqual(prog.read(0xFFFF0000, len(data)), data)
        self.assertRaises(FaultError, prog.read, 0x0, len(data), physical=True)

    def test_physical(self):
        data = b"hello, world"
        prog = Program()
        with tempfile.NamedTemporaryFile() as f:
            f.write(
                create_elf_file(
                    ET.CORE,
                    [
                        ElfSection(
                            p_type=PT.LOAD, vaddr=0xFFFF0000, paddr=0xA0, data=data
                        ),
                    ],
                )
            )
            f.flush()
            prog.set_core_dump(f.name)
        self.assertEqual(prog.read(0xFFFF0000, len(data)), data)
        self.assertEqual(prog.read(0xA0, len(data), physical=True), data)

    def test_unsaved(self):
        data = b"hello, world"
        prog = Program()
        with tempfile.NamedTemporaryFile() as f:
            f.write(
                create_elf_file(
                    ET.CORE,
                    [
                        ElfSection(
                            p_type=PT.LOAD,
                            vaddr=0xFFFF0000,
                            data=data,
                            memsz=len(data) + 4,
                        ),
                    ],
                )
            )
            f.flush()
            prog.set_core_dump(f.name)
        with self.assertRaisesRegex(FaultError, "memory not saved in core dump") as cm:
            prog.read(0xFFFF0000, len(data) + 4)
        self.assertEqual(cm.exception.address, 0xFFFF000C)


def dummy_symbol_finder(prog, name, address, one):
    return ()


class TestSymbolFinders(TestCase):
    def test_registered(self):
        prog = Program()
        self.assertEqual(prog.registered_symbol_finders(), {"elf"})
        prog.register_symbol_finder("foo", dummy_symbol_finder)
        self.assertEqual(prog.registered_symbol_finders(), {"elf", "foo"})

    def test_register_duplicate(self):
        self.assertRaisesRegex(
            ValueError,
            "duplicate symbol finder",
            Program().register_symbol_finder,
            "elf",
            dummy_symbol_finder,
        )

    def test_default_enabled(self):
        self.assertEqual(Program().enabled_symbol_finders(), ["elf"])

    def test_disable_all(self):
        prog = Program()
        prog.set_enabled_symbol_finders(())
        self.assertEqual(prog.enabled_symbol_finders(), [])

    def test_register_then_enable(self):
        prog = Program()
        prog.register_symbol_finder("foo", dummy_symbol_finder)
        self.assertEqual(prog.enabled_symbol_finders(), ["elf"])

        prog.set_enabled_symbol_finders(["foo", "elf"])
        self.assertEqual(prog.enabled_symbol_finders(), ["foo", "elf"])

    def test_register_enable_index(self):
        prog = Program()
        with self.subTest("None"):
            prog.register_symbol_finder("ghost", dummy_symbol_finder, enable_index=None)
            self.assertEqual(prog.enabled_symbol_finders(), ["elf"])

        with self.subTest("first"):
            prog.register_symbol_finder("foo", dummy_symbol_finder, enable_index=0)
            self.assertEqual(prog.enabled_symbol_finders(), ["foo", "elf"])

        with self.subTest("middle"):
            prog.register_symbol_finder("bar", dummy_symbol_finder, enable_index=1)
            self.assertEqual(prog.enabled_symbol_finders(), ["foo", "bar", "elf"])

        with self.subTest("end"):
            prog.register_symbol_finder("baz", dummy_symbol_finder, enable_index=3)
            self.assertEqual(
                prog.enabled_symbol_finders(), ["foo", "bar", "elf", "baz"]
            )

        with self.subTest("past end"):
            prog.register_symbol_finder("qux", dummy_symbol_finder, enable_index=10)
            self.assertEqual(
                prog.enabled_symbol_finders(), ["foo", "bar", "elf", "baz", "qux"]
            )

        with self.subTest("DRGN_HANDLER_REGISTER_DONT_ENABLE"):
            prog.register_symbol_finder(
                "quux",
                dummy_symbol_finder,
                enable_index=(2**64 if sys.maxsize > 2**32 else 2**32) - 2,
            )
            self.assertEqual(
                prog.enabled_symbol_finders(),
                ["foo", "bar", "elf", "baz", "qux", "quux"],
            )

        with self.subTest("last"):
            prog.register_symbol_finder("quuux", dummy_symbol_finder, enable_index=-1)
            self.assertEqual(
                prog.enabled_symbol_finders(),
                ["foo", "bar", "elf", "baz", "qux", "quux", "quuux"],
            )

    def test_register_enable_index_invalid(self):
        self.assertRaises(
            OverflowError,
            Program().register_symbol_finder,
            "foo",
            dummy_symbol_finder,
            enable_index=-2,
        )
