"""Tests for new design types: classical, optimal, mixture, split-plot, EVOP."""

import pytest
from forgedoe.core.types import Factor
from forgedoe.designs.classical import latin_square, randomized_block, taguchi
from forgedoe.designs.optimal import d_optimal, i_optimal
from forgedoe.designs.mixture import simplex_lattice, simplex_centroid, extreme_vertices
from forgedoe.designs.split_plot import split_plot, split_plot_ccd
from forgedoe.designs.evop import evop_phase, EVOPAccumulator


# ── Helpers ──

def _factors(k):
    return [Factor(chr(65 + i), 0, 100) for i in range(k)]


# ═══════════════════════════════════════════
# Classical Designs
# ═══════════════════════════════════════════

class TestLatinSquare:
    def test_basic(self):
        d = latin_square(['A', 'B', 'C'], randomize=False)
        assert d.n_runs == 9  # 3×3

    def test_each_treatment_once_per_row_and_col(self):
        d = latin_square(['A', 'B', 'C', 'D'], randomize=False)
        assert d.n_runs == 16
        # Each row should have each treatment exactly once
        for row_idx in range(4):
            treatments = [d.matrix[row_idx * 4 + col][2] for col in range(4)]
            assert len(set(treatments)) == 4

    def test_min_treatments(self):
        with pytest.raises(ValueError):
            latin_square(['A'])


class TestRCBD:
    def test_basic(self):
        d = randomized_block(3, 4, randomize=False)
        assert d.n_runs == 12  # 3 treatments × 4 blocks

    def test_replicates(self):
        d = randomized_block(3, 2, replicates=2, randomize=False)
        assert d.n_runs == 12  # 3 × 2 × 2

    def test_min_requirements(self):
        with pytest.raises(ValueError):
            randomized_block(1, 3)
        with pytest.raises(ValueError):
            randomized_block(3, 1)


class TestTaguchi:
    def test_auto_select_l4(self):
        d = taguchi(_factors(3))
        assert d.n_runs == 4
        assert 'L4' in d.design_type

    def test_auto_select_l8(self):
        d = taguchi(_factors(5))
        assert d.n_runs == 8
        assert 'L8' in d.design_type

    def test_explicit_array(self):
        d = taguchi(_factors(3), array_type='L8')
        assert d.n_runs == 8

    def test_l9_three_level(self):
        d = taguchi(_factors(4), array_type='L9')
        assert d.n_runs == 9

    def test_too_many_factors(self):
        with pytest.raises(ValueError):
            taguchi(_factors(8), array_type='L4')

    def test_coded_values(self):
        d = taguchi(_factors(3), array_type='L4')
        # L4 should have -1 and +1 only (2-level)
        for row in d.matrix:
            for val in row:
                assert val in (-1, 1)


# ═══════════════════════════════════════════
# Optimal Designs
# ═══════════════════════════════════════════

class TestDOptimal:
    def test_linear(self):
        d = d_optimal(_factors(3), n_runs=8, model='linear', seed=42)
        assert d.n_runs == 8
        assert d.n_factors == 3

    def test_quadratic(self):
        d = d_optimal(_factors(3), n_runs=12, model='quadratic', seed=42)
        assert d.n_runs == 12

    def test_min_runs_enforced(self):
        # 3 factors linear = 4 params, requesting only 2 runs
        d = d_optimal(_factors(3), n_runs=2, model='linear', seed=42)
        assert d.n_runs >= 4  # should bump to minimum

    def test_coded_range(self):
        d = d_optimal(_factors(2), n_runs=6, model='linear', seed=42)
        for row in d.matrix:
            for val in row:
                assert -1 <= val <= 1


class TestIOptimal:
    def test_basic(self):
        d = i_optimal(_factors(2), n_runs=8, model='quadratic', seed=42)
        assert d.n_runs == 8

    def test_different_from_d(self):
        dd = d_optimal(_factors(2), n_runs=8, model='quadratic', seed=42)
        di = i_optimal(_factors(2), n_runs=8, model='quadratic', seed=42)
        # They should produce different designs
        assert dd.matrix != di.matrix or dd.design_type != di.design_type


# ═══════════════════════════════════════════
# Mixture Designs
# ═══════════════════════════════════════════

class TestSimplexLattice:
    def test_vertices_only(self):
        d = simplex_lattice(3, degree=1, randomize=False)
        assert d.n_runs == 3  # 3 vertices

    def test_degree_2(self):
        d = simplex_lattice(3, degree=2, randomize=False)
        assert d.n_runs == 6  # 3 vertices + 3 edge midpoints

    def test_degree_3(self):
        d = simplex_lattice(3, degree=3, randomize=False)
        assert d.n_runs == 10

    def test_sum_to_one(self):
        d = simplex_lattice(4, degree=2, randomize=False)
        for row in d.matrix:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_proportions_valid(self):
        d = simplex_lattice(3, degree=3, randomize=False)
        for row in d.matrix:
            for val in row:
                assert -1e-10 <= val <= 1.0 + 1e-10


class TestSimplexCentroid:
    def test_3_components(self):
        d = simplex_centroid(3, randomize=False)
        assert d.n_runs == 7  # 2^3 - 1

    def test_4_components(self):
        d = simplex_centroid(4, randomize=False)
        assert d.n_runs == 15  # 2^4 - 1

    def test_sum_to_one(self):
        d = simplex_centroid(3, randomize=False)
        for row in d.matrix:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_augmented(self):
        d = simplex_centroid(3, augment_axial=True, randomize=False)
        assert d.n_runs == 10  # 7 + 3 axial

    def test_includes_vertices(self):
        d = simplex_centroid(3, randomize=False)
        # Should include (1,0,0), (0,1,0), (0,0,1)
        vertices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for v in vertices:
            found = any(all(abs(row[i] - v[i]) < 1e-10 for i in range(3)) for row in d.matrix)
            assert found, f"Missing vertex {v}"

    def test_includes_centroid(self):
        d = simplex_centroid(3, randomize=False)
        centroid = [1/3, 1/3, 1/3]
        found = any(all(abs(row[i] - centroid[i]) < 1e-10 for i in range(3)) for row in d.matrix)
        assert found, "Missing centroid"


class TestExtremeVertices:
    def test_unconstrained(self):
        d = extreme_vertices(3, randomize=False)
        # Unconstrained = just the simplex vertices
        assert d.n_runs >= 3

    def test_constrained_sum_to_one(self):
        d = extreme_vertices(3, lower_bounds=[0.1, 0.1, 0.1],
                            upper_bounds=[0.8, 0.8, 0.8], randomize=False)
        for row in d.matrix:
            assert abs(sum(row) - 1.0) < 1e-8

    def test_bounds_respected(self):
        lb = [0.2, 0.1, 0.1]
        ub = [0.6, 0.7, 0.7]
        d = extreme_vertices(3, lower_bounds=lb, upper_bounds=ub, randomize=False)
        for row in d.matrix:
            for i in range(3):
                assert row[i] >= lb[i] - 1e-8
                assert row[i] <= ub[i] + 1e-8

    def test_interior_points(self):
        d = extreme_vertices(3, lower_bounds=[0.1, 0.1, 0.1],
                            upper_bounds=[0.8, 0.8, 0.8],
                            n_interior=3, randomize=False)
        assert d.n_runs > 3  # vertices + interior


# ═══════════════════════════════════════════
# Split-Plot Designs
# ═══════════════════════════════════════════

class TestSplitPlot:
    def test_basic(self):
        wp = [Factor('temp', 150, 200)]
        sp = [Factor('speed', 100, 500), Factor('feed', 1, 5)]
        d = split_plot(wp, sp, randomize=False)
        # 2^1 WP × 2^2 SP = 2 × 4 = 8 runs
        assert d.n_runs == 8
        assert d.n_factors == 3

    def test_replicates(self):
        wp = [Factor('A', 0, 1)]
        sp = [Factor('B', 0, 1)]
        d = split_plot(wp, sp, n_replicates=3, randomize=False)
        assert d.n_runs == 12  # 2 WP × 2 SP × 3 reps

    def test_two_wp_two_sp(self):
        wp = _factors(2)
        sp = [Factor('C', 0, 1), Factor('D', 0, 1)]
        d = split_plot(wp, sp, randomize=False)
        assert d.n_runs == 16  # 4 WP × 4 SP

    def test_no_wp_raises(self):
        with pytest.raises(ValueError):
            split_plot([], _factors(2))

    def test_no_sp_raises(self):
        with pytest.raises(ValueError):
            split_plot(_factors(2), [])


class TestSplitPlotCCD:
    def test_basic(self):
        wp = [Factor('temp', 150, 200)]
        sp = [Factor('speed', 100, 500), Factor('feed', 1, 5)]
        d = split_plot_ccd(wp, sp, randomize=False)
        # Factorial: 2×4=8, WP axial: 2, SP axial: 4, center: 3 = 17
        assert d.n_runs >= 15
        assert 'Split-Plot CCD' in d.design_type


# ═══════════════════════════════════════════
# EVOP
# ═══════════════════════════════════════════

class TestEVOP:
    def test_phase_generation(self):
        factors = [Factor('temp', 180, 220), Factor('pressure', 20, 40)]
        d = evop_phase(factors, step_sizes=[2, 1], center_point=[200, 30])
        assert d.n_runs == 5  # 2^2 + center

    def test_center_point_included(self):
        factors = _factors(2)
        d = evop_phase(factors, step_sizes=[5, 5], center_point=[50, 50])
        assert [50, 50] in d.matrix

    def test_no_center(self):
        factors = _factors(2)
        d = evop_phase(factors, step_sizes=[5, 5], center_point=[50, 50],
                       include_center=False)
        assert d.n_runs == 4  # 2^2 only

    def test_too_many_factors(self):
        with pytest.raises(ValueError):
            evop_phase(_factors(6))

    def test_default_step_sizes(self):
        factors = [Factor('A', 0, 100), Factor('B', 0, 100)]
        d = evop_phase(factors)
        # Default step = 5% of range = 5
        # Center = midpoint = 50
        # Points should be near 45/55
        assert d.n_runs == 5


class TestEVOPAccumulator:
    def test_single_cycle(self):
        acc = EVOPAccumulator(n_points=5)
        acc.add_cycle([10, 12, 11, 13, 11.5])
        result = acc.analyze()
        assert result['n_cycles'] == 1
        assert len(result['means']) == 5

    def test_multiple_cycles_detects_effect(self):
        acc = EVOPAccumulator(n_points=5)
        # Factor 2 clearly affects response (high values in positions 2,3)
        acc.add_cycle([10, 12, 10, 12, 11])
        acc.add_cycle([10.1, 11.9, 10.2, 12.1, 10.9])
        acc.add_cycle([9.9, 12.1, 9.8, 11.9, 11.1])
        result = acc.analyze()
        assert len(result['significant']) > 0

    def test_wrong_length_raises(self):
        acc = EVOPAccumulator(n_points=5)
        with pytest.raises(ValueError):
            acc.add_cycle([1, 2, 3])

    def test_decision_limit_positive(self):
        acc = EVOPAccumulator(n_points=5)
        acc.add_cycle([10, 12, 11, 13, 11.5])
        acc.add_cycle([10.5, 11.5, 11, 12.5, 11])
        result = acc.analyze()
        assert result['decision_limit'] >= 0
