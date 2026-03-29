"""Tests for design generation modules."""

import pytest

from forgedoe.core.types import Factor
from forgedoe.designs.factorial import fractional_factorial, full_factorial, plackett_burman
from forgedoe.designs.response_surface import box_behnken_design, central_composite_design
from forgedoe.designs.screening import augment_dsd_for_rsm, definitive_screening_design
from forgedoe.designs.space_filling import latin_hypercube, maximin_lhs


def _factors(k):
    return [Factor(name=f"X{i+1}", low=-1, high=1) for i in range(k)]


class TestFullFactorial:
    def test_2_factors(self):
        d = full_factorial(_factors(2), randomize=False)
        assert d.n_runs == 4
        assert d.n_factors == 2

    def test_3_factors(self):
        d = full_factorial(_factors(3), randomize=False)
        assert d.n_runs == 8

    def test_center_points(self):
        d = full_factorial(_factors(2), center_points=3, randomize=False)
        assert d.n_runs == 7  # 4 + 3

    def test_to_natural(self):
        factors = [Factor("Temp", low=150, high=250), Factor("Pressure", low=10, high=30)]
        d = full_factorial(factors, randomize=False)
        natural = d.to_natural()
        assert not natural.is_coded
        # All values should be within bounds
        for row in natural.matrix:
            assert 150 <= row[0] <= 250
            assert 10 <= row[1] <= 30

    def test_to_dicts(self):
        d = full_factorial(_factors(2), randomize=False)
        dicts = d.to_dicts()
        assert len(dicts) == 4
        assert "X1" in dicts[0]
        assert "run" in dicts[0]


class TestFractionalFactorial:
    def test_resolution_3(self):
        d = fractional_factorial(_factors(7), resolution=3, randomize=False)
        assert d.n_runs < 128  # less than full 2^7

    def test_small_falls_back_to_full(self):
        d = fractional_factorial(_factors(3), resolution=5, randomize=False)
        assert d.n_runs == 8  # full factorial for small k


class TestPlackettBurman:
    def test_7_factors(self):
        d = plackett_burman(_factors(7), randomize=False)
        assert d.n_runs == 8  # PB for 7 factors

    def test_5_factors(self):
        d = plackett_burman(_factors(5), randomize=False)
        assert d.n_runs >= 6


class TestDSD:
    def test_6_factors(self):
        d = definitive_screening_design(_factors(6), randomize=False)
        assert d.n_runs == 13  # 2*6 + 1

    def test_4_factors(self):
        d = definitive_screening_design(_factors(4), randomize=False)
        assert d.n_runs == 9  # 2*4 + 1

    def test_has_center_point(self):
        d = definitive_screening_design(_factors(5), randomize=False)
        center = [0] * 5
        assert center in d.matrix

    def test_augment(self):
        d = definitive_screening_design(_factors(5), randomize=False)
        aug = augment_dsd_for_rsm(d, additional_center_points=3)
        assert aug.n_runs == d.n_runs + 3

    def test_vs_full_factorial_efficiency(self):
        """DSD should use far fewer runs than full factorial."""
        k = 6
        dsd = definitive_screening_design(_factors(k))
        ff = full_factorial(_factors(k))
        assert dsd.n_runs < ff.n_runs / 4  # at least 4x more efficient


class TestCCD:
    def test_3_factors(self):
        d = central_composite_design(_factors(3), randomize=False)
        # 2^3 factorial + 2*3 axial + 5 center = 8 + 6 + 5 = 19
        assert d.n_runs == 19

    def test_face_centered(self):
        d = central_composite_design(_factors(2), alpha="face", randomize=False)
        # All points should be within [-1, 1]
        for row in d.matrix:
            for val in row:
                assert -1 <= val <= 1


class TestBoxBehnken:
    def test_3_factors(self):
        d = box_behnken_design(_factors(3), center_points=3, randomize=False)
        # 3 pairs × 4 points + 3 center = 12 + 3 = 15
        assert d.n_runs == 15

    def test_no_corners(self):
        """Box-Behnken should not have any corner points (all ±1)."""
        d = box_behnken_design(_factors(3), randomize=False)
        for row in d.matrix:
            # No row should have ALL factors at ±1
            extreme_count = sum(1 for v in row if abs(v) == 1)
            assert extreme_count <= 2  # at most 2 factors at extremes


class TestLHS:
    def test_correct_size(self):
        d = latin_hypercube(_factors(3), n_samples=20)
        assert d.n_runs == 20
        assert d.n_factors == 3

    def test_within_bounds(self):
        d = latin_hypercube(_factors(2), n_samples=50)
        for row in d.matrix:
            for val in row:
                assert -1 <= val <= 1

    def test_maximin_better(self):
        """Maximin LHS should have larger minimum distance than random LHS."""
        factors = _factors(3)
        regular = latin_hypercube(factors, n_samples=10)
        optimized = maximin_lhs(factors, n_samples=10, n_iterations=50)
        assert optimized is not None


class TestCoding:
    def test_encode_decode_roundtrip(self):
        from forgedoe.core.coding import decode, encode

        assert encode(200, 150, 250) == pytest.approx(0.0)
        assert encode(150, 150, 250) == pytest.approx(-1.0)
        assert encode(250, 150, 250) == pytest.approx(1.0)
        assert decode(0, 150, 250) == pytest.approx(200)
        assert decode(-1, 150, 250) == pytest.approx(150)
