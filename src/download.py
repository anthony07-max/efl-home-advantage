"""Download historical English league results from football-data.co.uk into data/raw/.

Source: Football-Data.co.uk (https://www.football-data.co.uk/englandm.php), free historical
results and match statistics. Leagues: E0 Premier League, E1 Championship, E2 League One,
E3 League Two. Seasons 2005-06 to 2025-26 (completed seasons only).
Raw files are not committed to this repo; please download them from the source.
"""
import pathlib, time, requests

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
DIVS = ["E0", "E1", "E2", "E3"]
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2005, 2026)]  # 0506 ... 2526
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project; football-data.co.uk historical results)"}

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for code in SEASONS:
        for div in DIVS:
            out = RAW / f"{div}_{code}.csv"
            if out.exists() and out.stat().st_size > 10_000:
                continue
            url = BASE.format(code=code, div=div)
            for attempt in range(5):  # the site can be slow, so retry with a back-off
                try:
                    r = requests.get(url, headers=HEADERS, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 4:
                        raise
                    print("Retrying", url, "after", type(e).__name__)
                    time.sleep(5 * (attempt + 1))
            out.write_bytes(r.content)
            print("Downloaded", url, f"({len(r.content) // 1024} KB)")
            time.sleep(0.5)  # be polite to a free service
    print("Files in data/raw:", len(list(RAW.glob("*.csv"))))

if __name__ == "__main__":
    main()
