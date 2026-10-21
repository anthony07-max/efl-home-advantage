"""Combine the football-data.co.uk season files into one tidy match table.
Output: data/processed/matches.csv (one row per league match, no betting odds)."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
TIER = {"E0": (1, "Premier League"), "E1": (2, "Championship"), "E2": (3, "League One"), "E3": (4, "League Two")}
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "Referee",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
# Matches played with no (or almost no) crowds: the June 2020 restart to the end of 2020-21.
# A few games in Dec 2020 and May 2021 allowed small, capped crowds; see README limitations.
BCD_START, BCD_END = pd.Timestamp("2020-06-01"), pd.Timestamp("2021-05-31")

def main():
    frames = []
    for f in sorted(RAW.glob("E*_*.csv")):
        div, code = f.stem.split("_")
        d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        d = d[[c for c in KEEP if c in d.columns]].copy()
        d["season"] = f"20{code[:2]}-{code[2:]}"
        d["div"], d["tier"], d["league"] = div, TIER[div][0], TIER[div][1]
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["Date"] = pd.to_datetime(m["Date"], dayfirst=True, format="mixed")
    for c in ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["FTR"] = m.FTR.where(m.FTR.isin(list("HDA")),
                           (m.FTHG > m.FTAG).map({True: "H"}).fillna((m.FTHG == m.FTAG).map({True: "D", False: "A"})))
    m["home_pts"] = m.FTR.map({"H": 3, "D": 1, "A": 0})
    m["away_pts"] = m.FTR.map({"H": 0, "D": 1, "A": 3})
    m["goal_diff"] = m.FTHG - m.FTAG
    m["no_crowd"] = m.Date.between(BCD_START, BCD_END)
    m = m.rename(columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away", "Referee": "referee"})
    m = m.sort_values(["date", "tier"]).reset_index(drop=True)
    m.to_csv(PROC / "matches.csv", index=False)
    print(f"{len(m):,} matches, {m.season.nunique()} seasons, {m.no_crowd.sum():,} played without crowds")
    print(m.groupby("league").size())

if __name__ == "__main__":
    main()
