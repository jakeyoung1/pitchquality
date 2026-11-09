CREATE TABLE IF NOT EXISTS pitches (
    game_pk INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    pitch_number INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    pitcher INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    pitch_type TEXT NOT NULL,
    stuff_score REAL NOT NULL,
    expected_whiff REAL NOT NULL CHECK(expected_whiff BETWEEN 0 AND 1),
    actual_whiff INTEGER NOT NULL CHECK(actual_whiff IN (0, 1)),
    stuff_whiff REAL NOT NULL CHECK(stuff_whiff BETWEEN 0 AND 1),
    season INTEGER NOT NULL,
    game_type TEXT NOT NULL,
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);
CREATE INDEX IF NOT EXISTS pitches_pitcher_season ON pitches(pitcher, season);
CREATE INDEX IF NOT EXISTS pitches_game_date ON pitches(game_date);
CREATE TABLE IF NOT EXISTS leaderboard (
    pitcher INTEGER NOT NULL,
    season INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    pitches INTEGER NOT NULL,
    avg_stuff REAL NOT NULL,
    expected_whiff REAL NOT NULL,
    actual_whiff REAL NOT NULL,
    whiff_over_expected REAL NOT NULL,
    PRIMARY KEY (pitcher, season)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);
