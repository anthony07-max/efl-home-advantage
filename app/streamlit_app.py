"""Streamlit dashboard: home advantage in English football. Run: streamlit run app/streamlit_app.py"""
import json, pathlib
import pandas as pd, plotly.express as px, streamlit as st

P = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="EFL Home Advantage", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv(P / "matches.csv", parse_dates=["date"]), pd.read_csv(P / "season_league_summary.csv"),
            pd.read_csv(P / "poisson_home_effect.csv"), pd.read_csv(P / "league_one_team_home_edge.csv"),
            pd.read_csv(P / "orient_matches.csv", parse_dates=["date"]), pd.read_csv(P / "orient_by_season.csv"),
            pd.read_csv(P / "statistical_tests.csv"), json.loads((P / "key_findings.json").read_text()))
m, season, pois, teams, orient, osea, tests, F = load()

st.title("Home advantage in English football, 2005-06 to 2025-26")
st.caption(f"{F['n_matches']:,} Premier League and EFL matches from Football-Data.co.uk. "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
c = st.columns(4)
c[0].metric("Home win % (with crowds)", f"{F['home_win_pct_crowd']}%")
c[1].metric("Home win % (no crowds)", f"{F['home_win_pct_no_crowd']}%", f"{F['home_win_pct_no_crowd'] - F['home_win_pct_crowd']:+.1f} pts")
c[2].metric("Home-away PPG gap (crowds)", f"{F['ppg_gap_crowd']:.2f}")
c[3].metric("Home-away PPG gap (no crowds)", f"{F['ppg_gap_no_crowd']:.2f}", f"{F['ppg_gap_no_crowd'] - F['ppg_gap_crowd']:+.2f}")

t1, t2, t3, t4, t5 = st.tabs(["By season", "Referees & discipline", "Model", "League One clubs", "Leyton Orient"])
with t1:
    metric = st.selectbox("Metric", ["home_win_pct", "away_win_pct", "draw_pct", "ppg_gap", "mean_goal_diff", "home_ppg", "away_ppg"])
    lgs = st.multiselect("Leagues", season.league.unique().tolist(), default=season.league.unique().tolist())
    fig = px.line(season[season.league.isin(lgs)], x="season", y=metric, color="league", markers=True)
    fig.add_vrect(x0="2019-20", x1="2020-21", fillcolor="grey", opacity=.15, line_width=0, annotation_text="no crowds")
    st.plotly_chart(fig, width="stretch")
with t2:
    lg = st.selectbox("League", ["All"] + season.league.unique().tolist(), key="ref")
    d = m if lg == "All" else m[m.league == lg]
    d = d.assign(period=d.no_crowd.map({True: "No crowds", False: "With crowds"}))
    s = d.groupby("period")[["HY", "AY", "HF", "AF", "HR", "AR"]].mean().round(3).T
    s.index = ["Home yellows", "Away yellows", "Home fouls", "Away fouls", "Home reds", "Away reds"]
    st.dataframe(s, width="stretch")
    st.dataframe(tests, width="stretch", hide_index=True)
    if lg in ("All", "Premier League"):
        r = m[(m.league == "Premier League") & m.referee.notna()].assign(yellow_gap=lambda x: x.AY - x.HY)
        r = r.groupby("referee").agg(matches=("yellow_gap", "size"), away_minus_home_yellows=("yellow_gap", "mean")).round(3)
        st.subheader("Premier League referees (100+ matches)")
        st.dataframe(r[r.matches >= 100].sort_values("away_minus_home_yellows", ascending=False), width="stretch")
with t3:
    st.write("Poisson regression of goals per team per match, with club attack and defence effects and season effects. "
             "The multiplier is how many times more goals a team is expected to score at home than away.")
    st.dataframe(pois, width="stretch", hide_index=True)
    st.write(f"Logistic regression of a home win: no-crowd odds ratio {F['logit_no_crowd_odds_ratio']} (p = {F['logit_no_crowd_p']}); "
             f"season trend odds ratio {F['logit_season_odds_ratio']} per season (p = {F['logit_season_p']}).")
with t4:
    st.plotly_chart(px.bar(teams.sort_values("ppg_gap"), x="ppg_gap", y="team", orientation="h", height=850,
                           hover_data=["home_games", "home_ppg", "away_ppg"],
                           title="League One home PPG minus away PPG, 2015-16 to 2025-26 (crowd seasons, 3+ seasons)"), width="stretch")
with t5:
    c = st.columns(3)
    c[0].metric("Orient home PPG", f"{F['orient_home_ppg']:.2f}")
    c[1].metric("Orient away PPG", f"{F['orient_away_ppg']:.2f}")
    c[2].metric("Orient gap vs league-average gap", f"{F['orient_ppg_gap']:.2f}", f"{F['orient_ppg_gap'] - F['orient_league_avg_gap']:+.2f}", delta_color="off")
    st.plotly_chart(px.bar(osea, x="season", y="ppg", color="venue", barmode="group", hover_data=["league", "games", "win_pct"],
                           color_discrete_map={"Home": "#C8102E", "Away": "#444444"}), width="stretch")
    s = st.selectbox("Season", sorted(orient.season.unique(), reverse=True))
    st.dataframe(orient[orient.season == s].drop(columns=["season"]), width="stretch", hide_index=True)
