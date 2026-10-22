"""Plain-array equivalence and bounded emitter tests; no backend or packages."""
import math
import random
import unittest

from capacity_broadcast_runs import broadcast_run_plan, emit_broadcast_runs


class Cell:
    def __init__(self, emitter, data):
        self.em = emitter
        self.data = tuple(data)

    def __add__(self, other):
        self.em.additions += 1
        return Cell(self.em, [a + b for a, b in zip(self.data, other.data)])


class ArrayEmitter:
    def __init__(self, period):
        self.period = period
        self.rotations = 0
        self.additions = 0

    def rotate(self, value, shift):
        shift %= self.period
        self.rotations += shift.bit_count()
        return Cell(self, value.data[shift:] + value.data[:shift])

    def scalar(self, value):
        return Cell(self, [value] + [0] * (self.period - 1))


def coordinates(index, shape):
    result = []
    for axis, width in enumerate(shape):
        stride = math.prod(shape[axis + 1:])
        result.append(index // stride % width)
    return result


def source_index(coord, shape):
    index = 0
    for digit, width in zip(coord, shape):
        index = index * width + (0 if width == 1 else digit)
    return index


class BroadcastRunTests(unittest.TestCase):
    def check_offsets(self, offsets, period=256, value=7):
        emitter = ArrayEmitter(period)
        result = emit_broadcast_runs(emitter, emitter.scalar(value), offsets)
        expected = tuple(value if i in offsets else 0 for i in range(period))
        self.assertEqual(result.data, expected)
        plan = broadcast_run_plan(offsets, period)
        self.assertEqual(emitter.rotations, plan["primitive_rotations"])
        self.assertEqual(emitter.additions, plan["additions"])
        self.assertLessEqual(plan["work"], plan["individual_work"])
        return plan

    def check_broadcast(self, source_shape, target_shape, period):
        shape = (1,) * (len(target_shape) - len(source_shape)) + tuple(source_shape)
        self.assertTrue(all(a == 1 or a == b for a, b in zip(shape, target_shape)))
        source = [(-1 if i % 2 else 1) * (i + 1) for i in range(math.prod(source_shape))]
        groups = {}
        expected = []
        for destination in range(math.prod(target_shape)):
            index = source_index(coordinates(destination, target_shape), shape)
            chunk, offset = divmod(destination, period)
            groups.setdefault((index, chunk), []).append(offset)
            expected.append(source[index])
        actual = [[0] * period for _ in range(math.ceil(len(expected) / period))]
        for (index, chunk), offsets in groups.items():
            emitter = ArrayEmitter(period)
            value = emit_broadcast_runs(emitter, emitter.scalar(source[index]), offsets)
            actual[chunk] = [a + b for a, b in zip(actual[chunk], value.data)]
        flat = [v for chunk in actual for v in chunk]
        self.assertEqual(flat[:len(expected)], expected)
        self.assertEqual(flat[len(expected):], [0] * (len(flat) - len(expected)))

    def test_singleton(self):
        for position in (0, 1, 7, 31):
            with self.subTest(position=position):
                self.check_offsets([position], 32)

    def test_arithmetic_power_of_two_counts(self):
        for count in (2, 4, 8, 16, 32):
            self.check_offsets([1 + 3 * i for i in range(count)])

    def test_arithmetic_non_power_of_two_counts(self):
        for count in (3, 5, 7, 9, 17, 36):
            self.check_offsets([5 + 6 * i for i in range(count)])

    def test_non_arithmetic_fallback(self):
        plan = self.check_offsets([0, 1, 3, 6, 10, 15, 21], 32)
        self.assertTrue(any(row["mode"] == "individual" for row in plan["runs"]))

    def test_disjoint_runs_and_tail(self):
        self.check_offsets([0, 4, 8, 12, 19, 23, 27, 31, 63], 64)

    def test_all_lanes(self):
        for period in (4, 8, 32, 256, 4096):
            self.check_offsets(list(range(period)), period)

    def test_negative_and_zero_values(self):
        for value in (-13, 0, 9):
            self.check_offsets([1, 4, 7, 10, 13], 16, value)

    def test_deterministic_sparse_destinations(self):
        rng = random.Random(407)
        for period in (8, 16, 32, 64):
            for _ in range(8):
                offsets = sorted(rng.sample(range(period), rng.randint(1, period)))
                self.check_offsets(offsets, period)

    def test_singleton_broadcast(self):
        self.check_broadcast((1,), (2, 3, 5), 16)

    def test_inner_axis_broadcast(self):
        self.check_broadcast((2, 1, 3), (2, 5, 3), 32)

    def test_cross_chunk_broadcast(self):
        self.check_broadcast((2, 1), (2, 9), 8)

    def test_tail_broadcast(self):
        self.check_broadcast((3,), (5, 3), 8)

    def test_original_axis8_broadcast(self):
        self.check_broadcast((2, 3), (1, 1, 2, 3, 2, 3, 2, 3), 256)

    def test_original_axis8_cost(self):
        plans = [broadcast_run_plan(list(range(i, 216, 6)), 256) for i in range(6)]
        old_rotations = sum((-i % 256).bit_count() for i in range(216))
        self.assertEqual(old_rotations, 922)
        old_work = old_rotations + 210  # per-cell accumulation, before combining six cells
        self.assertEqual(sum(p["individual_work"] for p in plans), old_work)
        self.assertLess(sum(p["work"] for p in plans), old_work // 3)

    def test_invalid_destinations(self):
        for offsets in ([], [0, 0], [2, 1], [-1], [8], [True], [1.0], "12", {1, 2}):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                broadcast_run_plan(offsets, 8)

    def test_invalid_period(self):
        for period in (0, 2, 6, 8192, True, 8.0):
            with self.subTest(period=period), self.assertRaises(ValueError):
                broadcast_run_plan([0], period)

    def test_work_limit_and_no_partial_emission(self):
        offsets = [1, 3, 7, 12, 18, 25, 33]
        plan = broadcast_run_plan(offsets, 64)
        self.assertEqual(broadcast_run_plan(offsets, 64, max_work=plan["work"]), plan)
        emitter = ArrayEmitter(64)
        with self.assertRaisesRegex(ValueError, "primitive work budget"):
            emit_broadcast_runs(emitter, emitter.scalar(5), offsets, max_work=plan["work"] - 1)
        self.assertEqual((emitter.rotations, emitter.additions), (0, 0))

    def test_invalid_work_limits(self):
        for limit in (0, 1025, True, 1.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                broadcast_run_plan([0], 8, max_work=limit)

    def test_large_irregular_map_respects_budget(self):
        offsets = [i for i in range(4096) if i % 7 in (0, 2, 5)]
        with self.assertRaisesRegex(ValueError, "primitive work budget"):
            broadcast_run_plan(offsets, 4096)


if __name__ == "__main__":
    unittest.main()
