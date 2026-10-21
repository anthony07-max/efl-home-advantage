"""Home advantage analysis: summaries, statistical tests, Poisson and logistic models, charts.
Run after clean.py. Writes charts/*.png, data/processed/*.csv and data/processed/key_findings.json."""
import json, pathlib, sqlite3, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, CH = ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 10})
LEAGUES = ["Premier League", "Championship", "League One", "League Two"]
COL = dict(zip(LEAGUES, ["#3D195B", "#1F77B4", "#E4002B", "#7F7F7F"]))
ORIENT_RED = "#C8102E"

m = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
m["period"] = np.where(m.no_crowd, "No crowds (Jun 2020-May 2021)",
                       np.where(m.date < "2020-06-01", "With crowds, before", "With crowds, after"))
m["crowd"] = (~m.no_crowd).astype(int)
F = {"n_matches": len(m), "first_season": m.season.min(), "last_season": m.season.max(),
     "n_no_crowd": int(m.no_crowd.sum())}

# 1. Season x league summary (SQL) -------------------------------------------------
con = sqlite3.connect(":memory:")
m.assign(date=m.date.dt.strftime("%Y-%m-%d")).to_sql("matches", con)
season = pd.read_sql("""
    SELECT season, league, tier, COUNT(*) AS matches,
           ROUND(100.0 * AVG(FTR = 'H'), 2) AS home_win_pct,
           ROUND(100.0 * AVG(FTR = 'D'), 2) AS draw_pct,
           ROUND(100.0 * AVG(FTR = 'A'), 2) AS away_win_pct,
           ROUND(AVG(home_pts), 3) AS home_ppg, ROUND(AVG(away_pts), 3) AS away_ppg,
           ROUND(AVG(goal_diff), 3) AS mean_goal_diff,
           ROUND(AVG(HY), 3) AS home_yellows, ROUND(AVG(AY), 3) AS away_yellows,
           ROUND(AVG(HF), 3) AS home_fouls, ROUND(AVG(AF), 3) AS away_fouls,
           SUM(no_crowd) AS no_crowd_matches
    FROM matches GROUP BY season, league, tier ORDER BY season, tier""", con)
season["ppg_gap"] = (season.home_ppg - season.away_ppg).round(3)
season.to_csv(PROC / "season_league_summary.csv", index=False)

overall = m.groupby("period").agg(matches=("FTR", "size"), home_win_pct=("FTR", lambda s: 100 * (s == "H").mean()),
                                  away_win_pct=("FTR", lambda s: 100 * (s == "A").mean()),
                                  home_ppg=("home_pts", "mean"), away_ppg=("away_pts", "mean"),
                                  mean_goal_diff=("goal_diff", "mean"),
                                  home_yellows=("HY", "mean"), away_yellows=("AY", "mean"),
                                  home_fouls=("HF", "mean"), away_fouls=("AF", "mean")).round(3)
overall.to_csv(PROC / "period_summary.csv")
by_period_league = m.groupby(["league", "period"]).agg(
    matches=("FTR", "size"), home_win_pct=("FTR", lambda s: 100 * (s == "H").mean()),
    home_ppg=("home_pts", "mean"), away_ppg=("away_pts", "mean"), mean_goal_diff=("goal_diff", "mean"),
    home_yellows=("HY", "mean"), away_yellows=("AY", "mean")).round(3).reset_index()
by_period_league["ppg_gap"] = (by_period_league.home_ppg - by_period_league.away_ppg).round(3)
by_period_league.to_csv(PROC / "period_league_summary.csv", index=False)

crowd = m[~m.no_crowd]; bcd = m[m.no_crowd]
F.update(home_win_pct_crowd=round(100 * (crowd.FTR == "H").mean(), 1),
         home_win_pct_no_crowd=round(100 * (bcd.FTR == "H").mean(), 1),
         away_win_pct_crowd=round(100 * (crowd.FTR == "A").mean(), 1),
         away_win_pct_no_crowd=round(100 * (bcd.FTR == "A").mean(), 1),
         ppg_gap_crowd=round(crowd.home_pts.mean() - crowd.away_pts.mean(), 3),
         ppg_gap_no_crowd=round(bcd.home_pts.mean() - bcd.away_pts.mean(), 3),
         gd_crowd=round(crowd.goal_diff.mean(), 3), gd_no_crowd=round(bcd.goal_diff.mean(), 3))

# 2. Statistical tests ----------------------------------------------------------------
tests = []
for lg in ["All leagues"] + LEAGUES:
    d = m if lg == "All leagues" else m[m.league == lg]
    a, b = d[~d.no_crowd], d[d.no_crowd]
    table = [[(a.FTR == "H").sum(), (a.FTR != "H").sum()], [(b.FTR == "H").sum(), (b.FTR != "H").sum()]]
    chi2, p_chi, *_ = stats.chi2_contingency(table)
    t, p_t = stats.ttest_ind(a.goal_diff, b.goal_diff, equal_var=False)
    yc = d.dropna(subset=["HY", "AY"])
    w, p_w = stats.wilcoxon(yc.AY, yc.HY)  # paired: away vs home yellows in the same match
    tests.append(dict(league=lg, home_win_pct_crowd=round(100 * (a.FTR == "H").mean(), 2),
                      home_win_pct_no_crowd=round(100 * (b.FTR == "H").mean(), 2), chi2_p=p_chi,
                      gd_crowd=round(a.goal_diff.mean(), 3), gd_no_crowd=round(b.goal_diff.mean(), 3), welch_t_p=p_t,
                      away_minus_home_yellows=round((yc.AY - yc.HY).mean(), 3), wilcoxon_yellows_p=p_w))
tests = pd.DataFrame(tests)
tests.to_csv(PROC / "statistical_tests.csv", index=False)
F["chi2_p_all"] = float(f"{tests.loc[0, 'chi2_p']:.2g}")
F["welch_p_all"] = float(f"{tests.loc[0, 'welch_t_p']:.2g}")
# one-sample: is home GD > 0 in crowd and in no-crowd matches?
F["gd_no_crowd_ttest_p"] = float(f"{stats.ttest_1samp(bcd.goal_diff, 0).pvalue:.2g}")

# trend in home win % across seasons with crowds (2005-06 to 2018-19, all four leagues pooled)
pre = season[season.season <= "2018-19"].groupby("season").apply(
    lambda g: np.average(g.home_win_pct, weights=g.matches)).reset_index(name="hw")
pre["x"] = np.arange(len(pre))
lr = stats.linregress(pre.x, pre.hw)
F.update(pre_covid_trend_pts_per_season=round(lr.slope, 3), pre_covid_trend_p=round(lr.pvalue, 3),
         home_win_2005=round(pre.hw.iloc[0], 1), home_win_2018=round(pre.hw.iloc[-1], 1))

# 3. Models -----------------------------------------------------------------------------
# 3a. Poisson model of goals (one model per league), controlling for team strength:
#     log(goals) = home + home:no_crowd + attack[team] + defence[opponent] + season
#     (club attack/defence effects pooled across seasons keep the model fast and stable)
long = pd.concat([
    pd.DataFrame({"goals": m.FTHG, "team": m.home, "opp": m.away, "season": m.season,
                  "home": 1, "no_crowd": m.no_crowd.astype(int), "league": m.league}),
    pd.DataFrame({"goals": m.FTAG, "team": m.away, "opp": m.home, "season": m.season,
                  "home": 0, "no_crowd": m.no_crowd.astype(int), "league": m.league})], ignore_index=True)
pois = []
for lg in LEAGUES:
    d = long[long.league == lg]
    fit = smf.glm("goals ~ home + home:no_crowd + C(team) + C(opp) + C(season)", data=d, family=sm.families.Poisson()).fit()
    b, bi = fit.params["home"], fit.params["home:no_crowd"]
    ci = fit.conf_int()
    pois.append(dict(league=lg, home_goal_multiplier=round(np.exp(b), 3),
                     home_ci_low=round(np.exp(ci.loc["home", 0]), 3), home_ci_high=round(np.exp(ci.loc["home", 1]), 3),
                     no_crowd_home_multiplier=round(np.exp(b + bi), 3),
                     interaction_p=float(f"{fit.pvalues['home:no_crowd']:.2g}")))
pois = pd.DataFrame(pois)
pois.to_csv(PROC / "poisson_home_effect.csv", index=False)
F["poisson"] = pois.set_index("league").to_dict(orient="index")

# 3b. Logistic regression: P(home win) ~ crowd + tier + season trend
m["season_idx"] = m.season.str[:4].astype(int) - 2005
m["home_win"] = (m.FTR == "H").astype(int)
logit = smf.logit("home_win ~ no_crowd + C(league, Treatment('Premier League')) + season_idx", data=m.assign(no_crowd=m.no_crowd.astype(int))).fit(disp=0)
F.update(logit_no_crowd_odds_ratio=round(np.exp(logit.params["no_crowd"]), 3),
         logit_no_crowd_p=float(f"{logit.pvalues['no_crowd']:.2g}"),
         logit_season_odds_ratio=round(np.exp(logit.params["season_idx"]), 4),
         logit_season_p=float(f"{logit.pvalues['season_idx']:.2g}"))
pd.DataFrame({"odds_ratio": np.exp(logit.params), "p_value": logit.pvalues}).round(4).to_csv(PROC / "logit_home_win.csv")

# 4. League One focus -----------------------------------------------------------------
l1 = m[m.league == "League One"]
l1c, l1b = l1[~l1.no_crowd], l1[l1.no_crowd]
F.update(l1_home_win_crowd=round(100 * (l1c.FTR == "H").mean(), 1), l1_home_win_no_crowd=round(100 * (l1b.FTR == "H").mean(), 1),
         l1_ppg_gap_crowd=round(l1c.home_pts.mean() - l1c.away_pts.mean(), 3),
         l1_ppg_gap_no_crowd=round(l1b.home_pts.mean() - l1b.away_pts.mean(), 3),
         l1_away_yellow_gap_crowd=round((l1c.AY - l1c.HY).mean(), 3),
         l1_away_yellow_gap_no_crowd=round((l1b.AY - l1b.HY).mean(), 3))
# team-level home edge in League One, last 10 completed seasons with crowds (excluding 2020-21)
rec = l1[(l1.season >= "2015-16") & ~l1.no_crowd]
h = rec.groupby("home").agg(home_games=("home_pts", "size"), home_ppg=("home_pts", "mean"))
a = rec.groupby("away").agg(away_games=("away_pts", "size"), away_ppg=("away_pts", "mean"))
teams = h.join(a).dropna()
teams = teams[teams.home_games >= 69].assign(ppg_gap=lambda t: t.home_ppg - t.away_ppg).round(3)  # >= 3 seasons
teams = teams.sort_values("ppg_gap", ascending=False).rename_axis("team").reset_index()
teams.to_csv(PROC / "league_one_team_home_edge.csv", index=False)
F.update(l1_teams_ranked=len(teams), l1_top_home_edge_team=teams.team.iloc[0], l1_top_home_edge=teams.ppg_gap.iloc[0],
         l1_bottom_home_edge_team=teams.team.iloc[-1], l1_bottom_home_edge=teams.ppg_gap.iloc[-1])

# 5. Leyton Orient ----------------------------------------------------------------------
LO = "Leyton Orient"
oh = m[m.home == LO].assign(venue="Home", pts=lambda d: d.home_pts, gf=lambda d: d.FTHG, ga=lambda d: d.FTAG,
                            opp=lambda d: d.away, yellows=lambda d: d.HY, opp_yellows=lambda d: d.AY)
oa = m[m.away == LO].assign(venue="Away", pts=lambda d: d.away_pts, gf=lambda d: d.FTAG, ga=lambda d: d.FTHG,
                            opp=lambda d: d.home, yellows=lambda d: d.AY, opp_yellows=lambda d: d.HY)
orient = pd.concat([oh, oa])[["date", "season", "league", "venue", "opp", "gf", "ga", "pts", "yellows", "opp_yellows", "no_crowd", "referee"]]
orient["result"] = np.select([orient.gf > orient.ga, orient.gf == orient.ga], ["W", "D"], "L")
orient = orient.sort_values("date")
orient.to_csv(PROC / "orient_matches.csv", index=False)
os_ = orient.groupby(["season", "league", "venue"]).agg(games=("pts", "size"), ppg=("pts", "mean"),
      win_pct=("result", lambda s: 100 * (s == "W").mean()), gf=("gf", "mean"), ga=("ga", "mean")).round(3).reset_index()
os_.to_csv(PROC / "orient_by_season.csv", index=False)
oc = orient[~orient.no_crowd]
ov = oc.groupby("venue").agg(games=("pts", "size"), ppg=("pts", "mean"), win_pct=("result", lambda s: 100 * (s == "W").mean()),
                             gd=("gf", "mean"), ga=("ga", "mean"))
# league-average home/away PPG in exactly the league-seasons Orient played (with crowds)
ls = m[~m.no_crowd].merge(orient[["season", "league"]].drop_duplicates(), on=["season", "league"])
o_bcd = orient[orient.no_crowd]
mw = stats.mannwhitneyu(oc[oc.venue == "Home"].pts, oc[oc.venue == "Away"].pts, alternative="greater")
F.update(orient_seasons=int(orient.season.nunique()), orient_matches=len(orient),
         orient_home_ppg=round(ov.loc["Home", "ppg"], 3), orient_away_ppg=round(ov.loc["Away", "ppg"], 3),
         orient_ppg_gap=round(ov.loc["Home", "ppg"] - ov.loc["Away", "ppg"], 3),
         orient_home_win_pct=round(ov.loc["Home", "win_pct"], 1), orient_away_win_pct=round(ov.loc["Away", "win_pct"], 1),
         orient_league_avg_gap=round(ls.home_pts.mean() - ls.away_pts.mean(), 3),
         orient_mwu_p=float(f"{mw.pvalue:.2g}"),
         orient_bcd_home_ppg=round(o_bcd[o_bcd.venue == "Home"].pts.mean(), 3),
         orient_bcd_away_ppg=round(o_bcd[o_bcd.venue == "Away"].pts.mean(), 3),
         orient_bcd_games=len(o_bcd))
ol1 = orient[(orient.league == "League One") & ~orient.no_crowd].groupby("venue").pts.mean()
F.update(orient_l1_home_ppg=round(ol1["Home"], 3), orient_l1_away_ppg=round(ol1["Away"], 3))
o_rank = teams[teams.team == LO]
F["orient_l1_rank"] = f"{o_rank.index[0] + 1} of {len(teams)}" if len(o_rank) else "not ranked (fewer than 3 seasons)"

# 6. Charts -----------------------------------------------------------------------------
def shade(ax):
    ax.axvspan(13.5, 15.5, color="grey", alpha=.15)

seasons = sorted(m.season.unique()); xi = {s: i for i, s in enumerate(seasons)}
fig, ax = plt.subplots(figsize=(11, 4.8))
for lg in LEAGUES:
    d = season[season.league == lg]
    ax.plot(d.season.map(xi), d.home_win_pct, marker="o", ms=3.5, lw=1.8, color=COL[lg], label=lg)
ax.axvspan(xi["2019-20"] - .3, xi["2020-21"] + .4, color="grey", alpha=.15)
ax.text(xi["2020-21"] + .45, 49, "2020-21: no crowds\n(and 2019-20 restart)", fontsize=8)
ax.set_xticks(range(len(seasons))); ax.set_xticklabels(seasons, rotation=60, fontsize=8)
ax.yaxis.set_major_formatter(mt.PercentFormatter()); ax.legend(ncol=4, loc="lower left", fontsize=8)
ax.set(title="Home win % by season, English leagues 2005-06 to 2025-26", ylabel="Home win %", ylim=(30, 52))
fig.tight_layout(); fig.savefig(CH / "01_home_win_pct_by_season.png"); plt.close(fig)

order = ["With crowds, before", "No crowds (Jun 2020-May 2021)", "With crowds, after"]
fig, ax = plt.subplots(figsize=(10, 4.5)); w = .26
for i, p in enumerate(order):
    d = by_period_league[by_period_league.period == p].set_index("league").reindex(LEAGUES)
    bars = ax.bar(np.arange(4) + (i - 1) * w, d.ppg_gap, w, label=p, color=["#1F4E79", "#BBBBBB", "#5B9BD5"][i])
    ax.bar_label(bars, fmt="%.2f", fontsize=7)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(4)); ax.set_xticklabels(LEAGUES)
ax.set(title="Home points per game minus away points per game", ylabel="PPG gap (home - away)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(CH / "02_ppg_gap_crowds_vs_no_crowds.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, (hc, ac, lab) in zip(axes, [("home_yellows", "away_yellows", "Yellow cards per team per match"),
                                     ("home_fouls", "away_fouls", "Fouls per team per match")]):
    d = overall.reindex(order)
    x = np.arange(3)
    b1 = ax.bar(x - .18, d[hc], .36, label="Home team", color="#1F4E79"); b2 = ax.bar(x + .18, d[ac], .36, label="Away team", color="#E4002B")
    ax.bar_label(b1, fmt="%.2f", fontsize=7); ax.bar_label(b2, fmt="%.2f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["Crowds,\nbefore", "No crowds", "Crowds,\nafter"]); ax.set_title(lab)
axes[0].legend(fontsize=8)
fig.suptitle("Referee decisions: away teams are booked more, but the gap vanished without crowds", fontsize=11)
fig.tight_layout(); fig.savefig(CH / "03_cards_fouls_home_vs_away.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(4)
ax.errorbar(pois.home_goal_multiplier, y + .12, xerr=[pois.home_goal_multiplier - pois.home_ci_low, pois.home_ci_high - pois.home_goal_multiplier],
            fmt="o", color="#1F4E79", label="With crowds (95% CI)")
ax.scatter(pois.no_crowd_home_multiplier, y - .12, color="#BBBBBB", edgecolor="black", zorder=3, label="No crowds (2020-21)")
ax.axvline(1, color="black", lw=.8, ls="--")
ax.set_yticks(y); ax.set_yticklabels(pois.league)
ax.set(title="Poisson model: home team's expected goals multiplier\n(controlling for club attack/defence and season)", xlabel="Multiplier on goals scored at home")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(CH / "04_poisson_home_effect.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 7))
t = teams.sort_values("ppg_gap")
ax.barh(t.team, t.ppg_gap, color=[ORIENT_RED if x == LO else "#9DB4C0" for x in t.team])
ax.axvline(teams.ppg_gap.median(), color="black", ls=":", lw=1)
ax.set(title="League One home edge by club, 2015-16 to 2025-26 (crowd seasons, 3+ seasons)",
       xlabel="Home PPG minus away PPG"); ax.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(CH / "05_league_one_club_home_edge.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.5))
osp = os_.pivot(index="season", columns="venue", values="ppg").reindex(seasons)
x = np.arange(len(seasons))
ax.bar(x - .2, osp.Home, .4, color=ORIENT_RED, label="Home"); ax.bar(x + .2, osp.Away, .4, color="#444444", label="Away")
for s in seasons:
    lg = orient.loc[orient.season == s, "league"]
    if len(lg):
        ax.text(xi[s], 2.75, {"League One": "L1", "League Two": "L2"}.get(lg.iloc[0], lg.iloc[0]), ha="center", fontsize=7)
    else:
        ax.text(xi[s], 0.1, "National\nLeague", ha="center", fontsize=6, color="grey")
ax.axvspan(xi["2020-21"] - .5, xi["2020-21"] + .5, color="grey", alpha=.15)
ax.set_xticks(x); ax.set_xticklabels(seasons, rotation=60, fontsize=8)
ax.set(title="Leyton Orient points per game, home vs away, by season (L1 / L2 labels; grey = no crowds)", ylabel="PPG", ylim=(0, 2.9))
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(CH / "06_orient_home_vs_away.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.3))
l1s = season[season.league == "League One"]
ax.plot(l1s.season.map(xi), l1s.home_ppg, marker="o", color=COL["League One"], label="Home PPG")
ax.plot(l1s.season.map(xi), l1s.away_ppg, marker="o", color="#444444", label="Away PPG")
ax.fill_between(l1s.season.map(xi), l1s.away_ppg, l1s.home_ppg, color=COL["League One"], alpha=.1)
ax.axvspan(xi["2020-21"] - .4, xi["2020-21"] + .4, color="grey", alpha=.15)
ax.set_xticks(range(len(seasons))); ax.set_xticklabels(seasons, rotation=60, fontsize=8)
ax.set(title="League One: home vs away points per game by season", ylabel="PPG"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(CH / "07_league_one_ppg_by_season.png"); plt.close(fig)

(PROC / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=2, default=str))
print(tests.to_string()); print(pois.to_string()); print(overall.to_string())
print(teams.head(5).to_string()); print(teams.tail(3).to_string())
