"""Feature-engineering tests. These guard the conventions the model depends on."""
import numpy as np
import pandas as pd
import pytest

from pitchquality import features


def _pitch(**over):
    base = dict(
        pitch_type="FF", description="swinging_strike", p_throws="R", stand="R",
        pitcher=1, batter=9, player_name="Test, Arm", game_date="2026-05-01",
        release_speed=95.0, release_spin_rate=2300.0, release_extension=6.5,
        release_pos_x=-1.8, release_pos_z=6.0, pfx_x=0.8, pfx_z=1.4,
        plate_x=0.1, plate_z=2.5, spin_axis=210.0, arm_angle=42.0,
        balls=0, strikes=1,
        vx0=5.0, vy0=-138.0, vz0=-4.0, ax=-10.0, ay=28.0, az=-14.0,
    )
    base.update(over)
    return base


def frame(rows):
    return pd.DataFrame([_pitch(**r) for r in rows])


class TestTargets:
    def test_whiff_is_a_subset_of_swing(self):
        df = features.add_targets(frame([
            {"description": "swinging_strike"},
            {"description": "swinging_strike_blocked"},
            {"description": "foul"},
            {"description": "hit_into_play"},
            {"description": "ball"},
            {"description": "called_strike"},
        ]))
        assert df["is_swing"].tolist() == [True, True, True, True, False, False]
        assert df["is_whiff"].tolist() == [True, True, False, False, False, False]
        # Every whiff must also be a swing, or the conditioning is broken.
        assert (~df["is_whiff"] | df["is_swing"]).all()

    def test_foul_tip_is_a_swing_but_not_a_whiff(self):
        # Foul tip is contact. Counting it as a whiff would inflate every
        # grade for pitchers who generate weak contact at the top of the zone.
        df = features.add_targets(frame([{"description": "foul_tip"}]))
        assert bool(df["is_swing"].iloc[0]) is True
        assert bool(df["is_whiff"].iloc[0]) is False

    def test_bunts_are_excluded_from_swings(self):
        df = features.add_targets(frame([
            {"description": "missed_bunt"}, {"description": "foul_bunt"},
        ]))
        assert not df["is_swing"].any()


class TestMirroring:
    def test_lefty_horizontal_values_flip_sign(self):
        df = features.mirror_handedness(frame([
            {"p_throws": "R", "pfx_x": 0.8, "plate_x": 0.5, "release_pos_x": -1.8},
            {"p_throws": "L", "pfx_x": 0.8, "plate_x": 0.5, "release_pos_x": -1.8},
        ]))
        assert df["pfx_x_mir"].tolist() == [0.8, -0.8]
        assert df["plate_x_mir"].tolist() == [0.5, -0.5]
        assert df["release_pos_x_mir"].tolist() == [-1.8, 1.8]

    def test_vertical_values_are_untouched(self):
        df = features.mirror_handedness(frame([
            {"p_throws": "L", "pfx_z": 1.4, "plate_z": 2.5},
        ]))
        assert df["pfx_z"].iloc[0] == 1.4
        assert df["plate_z"].iloc[0] == 2.5

    def test_spin_axis_reflects_about_vertical(self):
        df = features.mirror_handedness(frame([
            {"p_throws": "L", "spin_axis": 210.0},
            {"p_throws": "L", "spin_axis": 0.0},
            {"p_throws": "R", "spin_axis": 210.0},
        ]))
        assert df["spin_axis_mir"].tolist() == [150.0, 0.0, 210.0]

    def test_same_hand_encodes_platoon_not_handedness(self):
        df = features.mirror_handedness(frame([
            {"p_throws": "R", "stand": "R"}, {"p_throws": "L", "stand": "L"},
            {"p_throws": "R", "stand": "L"}, {"p_throws": "L", "stand": "R"},
        ]))
        assert df["same_hand"].tolist() == [1, 1, 0, 0]


class TestApproachAngles:
    def test_vaa_is_negative_and_physically_plausible(self):
        df = features.add_approach_angles(frame([{}]))
        vaa = df["vaa"].iloc[0]
        # A pitch arrives on a downward plane; real four-seams sit near -4 to -6.
        assert -12.0 < vaa < 0.0

    def test_steeper_downward_velocity_gives_steeper_vaa(self):
        df = features.add_approach_angles(frame([
            {"vz0": -2.0}, {"vz0": -8.0},
        ]))
        assert df["vaa"].iloc[1] < df["vaa"].iloc[0]

    def test_haa_is_mirrored_for_lefties(self):
        df = features.add_approach_angles(
            features.mirror_handedness(frame([
                {"p_throws": "R", "vx0": 5.0}, {"p_throws": "L", "vx0": 5.0},
            ]))
        )
        assert df["haa"].iloc[0] == pytest.approx(-df["haa"].iloc[1], rel=1e-6)


class TestFastballDifferentials:
    def test_offspeed_is_measured_against_the_pitchers_own_fastball(self):
        df = frame([
            {"pitcher": 1, "pitch_type": "FF", "release_speed": 96.0},
            {"pitcher": 1, "pitch_type": "CH", "release_speed": 86.0},
        ])
        df = features.add_fastball_differentials(
            features.add_approach_angles(features.mirror_handedness(df))
        )
        ch = df[df["pitch_type"] == "CH"].iloc[0]
        assert ch["velo_diff_fb"] == pytest.approx(-10.0)

    def test_wrong_pipeline_order_fails_loudly(self):
        # Silent NaN differentials would degrade every offspeed grade without
        # surfacing anything, so the contract is enforced explicitly.
        with pytest.raises(ValueError, match="mirror_handedness"):
            features.add_fastball_differentials(frame([{}]))

    def test_pitcher_with_no_fastball_gets_zero_not_nan(self):
        df = frame([{"pitcher": 2, "pitch_type": "SL", "release_speed": 85.0}])
        df = features.add_fastball_differentials(
            features.add_approach_angles(features.mirror_handedness(df))
        )
        assert df["velo_diff_fb"].iloc[0] == 0.0
        assert not df["velo_diff_fb"].isna().any()


class TestBuild:
    def test_build_keeps_only_swings_and_modeled_pitch_types(self):
        df = features.build(frame([
            {"description": "swinging_strike", "pitch_type": "FF"},
            {"description": "ball", "pitch_type": "FF"},          # not a swing
            {"description": "swinging_strike", "pitch_type": "KN"},  # not modeled
        ]))
        assert len(df) == 1
        assert df["pitch_type"].iloc[0] == "FF"

    def test_build_drops_rows_missing_required_physics(self):
        df = features.build(frame([
            {"description": "foul", "release_speed": np.nan},
        ]))
        assert df.empty
