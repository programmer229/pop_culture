#!/usr/bin/env python3
"""
Diagnostic script to explore pop-culture references in New Yorker caption contest data.

Pipeline:
1. Load contest summaries through caption_contest_data (patched to use local cache).
2. Build a lightweight pop-culture detector based on keyword lists with first-seen years.
3. Annotate captions with pop-culture metadata and normalize scores for length and contest effects.
4. Assign coarse image themes via keyword spotting on high-vote captions.
5. Produce comparative statistics, recency effects, polarization metrics, and yearly trend plots.
Outputs land in the analysis/ directory.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import caption_contest_data as ccd
from caption_contest_data import _web


# ---------------------------------------------------------------------------
# Patch caption_contest_data to read from the local cache we populated.
# ---------------------------------------------------------------------------
def _local_summary_map() -> Dict[str, str]:
    """
    Return {filename: pseudo-url} for summary CSVs that already exist in the cache.
    The pseudo-url pattern keeps caption_contest_data from trying to redownload.
    """
    cache = Path(ccd.__file__).parent / ".ccd-cache" / "summaries"
    if not cache.exists():
        raise FileNotFoundError(f"Expected summary cache at {cache}")
    return {p.name: f"local/{p.name}" for p in cache.glob("*.csv")}


_web._get_summary_fnames_web = _local_summary_map  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pop-culture lexicon — keyword rules with first-popular year annotations.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PopCultureEntry:
    label: str
    aliases: Tuple[str, ...]
    category: str
    first_year: int


def _build_pop_culture_lexicon() -> List[PopCultureEntry]:
    raw_entries = [
        # Musicians / entertainers
        ("taylor_swift", ("taylor swift", "swifties"), "music", 2006),
        ("beyonce", ("beyonce", "queen bey"), "music", 1998),
        ("jay_z", ("jay-z", "jay z"), "music", 1996),
        ("rihanna", ("rihanna",), "music", 2007),
        ("drake", ("drake",), "music", 2010),
        ("ariana_grande", ("ariana grande",), "music", 2013),
        ("lady_gaga", ("lady gaga",), "music", 2009),
        ("justin_bieber", ("justin bieber", "bieber"), "music", 2010),
        ("bruno_mars", ("bruno mars",), "music", 2010),
        ("ed_sheeran", ("ed sheeran",), "music", 2012),
        ("bts", ("bts", "bangtan boys"), "music", 2015),
        ("bad_bunny", ("bad bunny",), "music", 2018),
        ("weeknd", ("the weeknd", "weeknd"), "music", 2015),
        ("billie_eilish", ("billie eilish",), "music", 2019),
        ("dualipa", ("dua lipa",), "music", 2018),
        ("taylor_swift_songs", ("eras tour", "1989 tour"), "music", 2015),
        ("elton_john", ("elton john",), "music", 1970),
        ("madonna", ("madonna",), "music", 1984),
        ("prince", ("prince",), "music", 1984),
        ("beatles", ("the beatles", "beatles"), "music", 1964),
        ("rolling_stones", ("rolling stones",), "music", 1965),
        ("nirvana", ("nirvana", "kurt cobain"), "music", 1991),
        # Celebrities / pop figures
        ("kim_kardashian", ("kim kardashian", "kardashians"), "celebrity", 2007),
        ("kanye_west", ("kanye west", "yeezy"), "celebrity", 2004),
        ("oprah", ("oprah", "oprah winfrey"), "celebrity", 1986),
        ("martha_stewart", ("martha stewart",), "celebrity", 1995),
        ("gordon_ramsay", ("gordon ramsay",), "celebrity", 2004),
        ("anthony_bourdain", ("anthony bourdain",), "celebrity", 2005),
        ("martha_stewart_snoop", ("snoop dogg", "snoop lion"), "music", 1994),
        ("the_rock", ("the rock", "dwayne johnson"), "celebrity", 2001),
        ("keanu", ("keanu reeves",), "celebrity", 1999),
        ("elon_musk", ("elon musk",), "tech", 2015),
        ("jeff_bezos", ("jeff bezos",), "tech", 2015),
        ("mark_zuckerberg", ("mark zuckerberg",), "tech", 2010),
        ("greta_thunberg", ("greta thunberg",), "activism", 2019),
        ("simone_biles", ("simone biles",), "sports", 2016),
        ("serena_williams", ("serena williams",), "sports", 2002),
        ("lebron_james", ("lebron james",), "sports", 2003),
        ("tom_brady", ("tom brady",), "sports", 2002),
        ("lionel_messi", ("lionel messi", "messi"), "sports", 2010),
        ("cristiano_ronaldo", ("cristiano ronaldo", "ronaldo"), "sports", 2006),
        # Politics
        ("barack_obama", ("barack obama", "obama"), "politics", 2008),
        ("michelle_obama", ("michelle obama",), "politics", 2009),
        ("joe_biden", ("joe biden", "biden"), "politics", 2008),
        ("donald_trump", ("donald trump", "trump"), "politics", 2015),
        ("hillary_clinton", ("hillary clinton", "clinton"), "politics", 1992),
        ("bernie_sanders", ("bernie sanders", "feel the bern"), "politics", 2016),
        ("alexandria_ocasio_cortez", ("alexandria ocasio-cortez", "aoc"), "politics", 2018),
        ("vladimir_putin", ("vladimir putin", "putin"), "politics", 2000),
        ("zelenskyy", ("volodymyr zelensky", "zelenskyy"), "politics", 2022),
        # Brands / tech companies
        ("netflix", ("netflix",), "brand", 2007),
        ("amazon", ("amazon", "prime delivery", "prime video"), "brand", 2006),
        ("google", ("google", "googling"), "tech", 2004),
        ("apple", ("iphone", "ipad", "apple watch"), "tech", 2007),
        ("microsoft", ("microsoft", "windows"), "tech", 1995),
        ("tesla", ("tesla",), "tech", 2012),
        ("spacex", ("spacex",), "tech", 2016),
        ("uber", ("uber",), "tech", 2013),
        ("lyft", ("lyft",), "tech", 2014),
        ("airbnb", ("airbnb",), "tech", 2014),
        ("door_dash", ("doordash", "door dash"), "tech", 2019),
        ("zoom", ("zoom call", "zoom meeting", "zooming"), "tech", 2020),
        ("slack", ("slack",), "tech", 2015),
        ("instagram", ("instagram", "insta", "influencer"), "tech", 2013),
        ("facebook", ("facebook", "meta"), "tech", 2007),
        ("twitter", ("twitter", "tweet", "x dot com"), "tech", 2009),
        ("tiktok", ("tiktok", "tik tok"), "internet", 2019),
        ("youtube", ("youtube", "youtuber"), "internet", 2006),
        ("snapchat", ("snapchat", "snap streak"), "internet", 2015),
        ("reddit", ("reddit",), "internet", 2012),
        ("spotify", ("spotify", "playlist"), "tech", 2012),
        ("hulu", ("hulu",), "brand", 2009),
        ("disney_plus", ("disney plus", "disney+"), "brand", 2020),
        ("marvel", ("marvel", "mcu"), "tv_film", 2008),
        # TV, Film, Franchises
        ("game_of_thrones", ("game of thrones", "got"), "tv_film", 2011),
        ("house_of_cards", ("house of cards",), "tv_film", 2013),
        ("breaking_bad", ("breaking bad", "heisenberg"), "tv_film", 2008),
        ("better_call_saul", ("better call saul",), "tv_film", 2015),
        ("mad_men", ("mad men",), "tv_film", 2007),
        ("succession", ("succession",), "tv_film", 2018),
        ("the_crown", ("the crown",), "tv_film", 2016),
        ("stranger_things", ("stranger things", "hawkins"), "tv_film", 2016),
        ("squid_game", ("squid game",), "tv_film", 2021),
        ("the_bachelor", ("the bachelor", "rose ceremony"), "tv_film", 2002),
        ("american_idol", ("american idol",), "tv_film", 2002),
        ("the_voice", ("the voice",), "tv_film", 2011),
        ("survivor", ("survivor", "tribal council"), "tv_film", 2000),
        ("shark_tank", ("shark tank",), "tv_film", 2010),
        ("star_wars", ("star wars", "skywalker", "darth vader", "the force"), "tv_film", 1977),
        ("star_trek", ("star trek", "enterprise", "klingon"), "tv_film", 1966),
        ("harry_potter", ("harry potter", "hogwarts", "quidditch"), "tv_film", 1997),
        ("lord_of_the_rings", ("lord of the rings", "middle earth", "gollum"), "tv_film", 2001),
        ("avengers", ("avengers", "iron man", "captain america", "thor"), "tv_film", 2012),
        ("black_panther", ("black panther", "wakanda"), "tv_film", 2018),
        ("batman", ("batman", "gotham"), "tv_film", 1966),
        ("superman", ("superman", "kryptonite"), "tv_film", 1978),
        ("spider_man", ("spider-man", "spiderman", "peter parker"), "tv_film", 2002),
        ("wonder_woman", ("wonder woman",), "tv_film", 2017),
        ("joker", ("the joker", "joker"), "tv_film", 2019),
        ("frozen", ("frozen", "elsa", "let it go"), "tv_film", 2013),
        ("barbie", ("barbie", "ken"), "tv_film", 2023),
        ("jurassic_park", ("jurassic park", "jurassic world"), "tv_film", 1993),
        ("matrix", ("matrix", "neo", "red pill"), "tv_film", 1999),
        ("mission_impossible", ("mission impossible", "impossible mission force"), "tv_film", 1996),
        ("indiana_jones", ("indiana jones", "raiders"), "tv_film", 1981),
        ("ghostbusters", ("ghostbusters", "who ya gonna call"), "tv_film", 1984),
        ("simpsons", ("the simpsons", "homer simpson", "bart simpson"), "tv_film", 1989),
        ("family_guy", ("family guy", "stewie"), "tv_film", 1999),
        ("south_park", ("south park", "cartman"), "tv_film", 1997),
        ("seinfeld", ("seinfeld", "festivus"), "tv_film", 1989),
        ("friends", ("friends", "central perk", "pivot"), "tv_film", 1994),
        ("the_office", ("the office", "dunder mifflin", "michael scott"), "tv_film", 2005),
        ("parks_rec", ("parks and rec", "pawnee"), "tv_film", 2009),
        ("saturday_night_live", ("snl", "saturday night live"), "tv_film", 1975),
        ("daily_show", ("the daily show", "jon stewart", "trevor noah"), "tv_film", 2005),
        ("last_week_tonight", ("last week tonight", "john oliver"), "tv_film", 2014),
        ("colbert", ("colbert", "stephen colbert"), "tv_film", 2005),
        ("jimmy_fallon", ("jimmy fallon", "fallon"), "tv_film", 2014),
        ("jimmy_kimmel", ("jimmy kimmel",), "tv_film", 2003),
        # Internet culture / memes
        ("yolo", ("yolo",), "internet", 2012),
        ("ok_boomer", ("ok boomer",), "internet", 2019),
        ("rickroll", ("rickroll", "rick rolled"), "internet", 2007),
        ("harlem_shake", ("harlem shake",), "internet", 2013),
        ("gangnam_style", ("gangnam style",), "internet", 2012),
        ("ice_bucket", ("ice bucket challenge",), "internet", 2014),
        ("fidget_spinner", ("fidget spinner",), "internet", 2017),
        ("planking", ("planking",), "internet", 2011),
        ("dab", ("dabbing", "the dab"), "internet", 2015),
        ("mannequin_challenge", ("mannequin challenge",), "internet", 2016),
        ("doge", ("doge", "much wow"), "internet", 2013),
        ("nyan_cat", ("nyan cat",), "internet", 2011),
        ("pepe", ("pepe the frog",), "internet", 2008),
        ("grumpy_cat", ("grumpy cat",), "internet", 2012),
        ("area_51", ("area 51 raid", "storm area 51"), "internet", 2019),
        ("covfefe", ("covfefe",), "internet", 2017),
        ("wordle", ("wordle",), "internet", 2022),
        ("nft", ("nft", "non fungible token"), "internet", 2021),
        ("metaverse", ("metaverse",), "tech", 2021),
        ("ai_chatbot", ("chatgpt", "gpt-4", "openai"), "tech", 2023),
        ("bitcoin", ("bitcoin", "crypto", "cryptocurrency"), "finance", 2017),
        ("blockchain", ("blockchain",), "tech", 2017),
        ("meme_stock", ("meme stock", "gamestop", "gme", "diamond hands"), "finance", 2021),
        ("doomscroll", ("doomscroll", "doom scrolling"), "internet", 2020),
        ("quiet_quitting", ("quiet quitting",), "workplace", 2022),
        ("girl_boss", ("girlboss", "girl boss"), "internet", 2017),
        ("influencer", ("influencer", "content creator"), "internet", 2018),
        # Sports & events
        ("super_bowl", ("super bowl", "halftime show"), "sports", 1967),
        ("world_series", ("world series",), "sports", 1903),
        ("nba_finals", ("nba finals",), "sports", 1970),
        ("march_madness", ("march madness", "final four"), "sports", 1985),
        ("olympics", ("olympics", "olympic gold"), "sports", 1984),
        ("world_cup", ("world cup", "fifa"), "sports", 1994),
        ("stanley_cup", ("stanley cup",), "sports", 1990),
        ("tour_de_france", ("tour de france",), "sports", 1990),
        ("kentucky_derby", ("kentucky derby",), "sports", 1980),
        # Misc pop references
        ("mcdonalds", ("mcdonald's", "mickey d's"), "brand", 1970),
        ("starbucks", ("starbucks", "pumpkin spice"), "brand", 2004),
        ("chipotle", ("chipotle",), "brand", 2010),
        ("whole_foods", ("whole foods", "trader joe's", "trader joes"), "brand", 2008),
        ("ikea", ("ikea",), "brand", 2000),
        ("costco", ("costco", "costco card"), "brand", 2000),
        ("peloton", ("peloton",), "brand", 2019),
        ("crossfit", ("crossfit",), "fitness", 2012),
        ("soulcycle", ("soulcycle",), "fitness", 2014),
        ("hot_girl_summer", ("hot girl summer",), "internet", 2019),
        ("pumpkin_spice", ("pumpkin spice latte",), "brand", 2003),
        ("elf_on_shelf", ("elf on the shelf",), "internet", 2010),
        ("escape_room", ("escape room",), "recreation", 2017),
        ("marie_kondo", ("marie kondo", "spark joy"), "lifestyle", 2014),
        ("ikea_furniture", ("billy bookcase", "fjallbo", "malm"), "brand", 2005),
        ("stem_cell", ("stem cell",), "science", 2005),
        ("dna_test", ("23andme", "ancestry.com"), "science", 2017),
        ("avocado_toast", ("avocado toast",), "food", 2015),
        ("kombucha", ("kombucha",), "food", 2015),
        ("gluten_free", ("gluten free",), "food", 2012),
        ("paleo", ("paleo diet",), "food", 2013),
        ("keto", ("keto", "keto diet"), "food", 2018),
        ("vegan", ("vegan", "plant-based"), "food", 2014),
        ("goop", ("goop", "gwyneth paltrow"), "lifestyle", 2010),
        ("man_bun", ("man bun",), "style", 2015),
        ("normcore", ("normcore",), "style", 2014),
        ("athleisure", ("athleisure",), "style", 2016),
        ("cancel_culture", ("cancel culture", "cancelled"), "internet", 2019),
        ("wfh", ("work from home", "wfh"), "workplace", 2020),
        ("gig_economy", ("gig economy", "side hustle"), "workplace", 2016),
        ("crypto_bro", ("crypto bro",), "internet", 2021),
        ("met_gala", ("met gala",), "celebrity", 2010),
        ("comic_con", ("comic-con", "comic con"), "fan_events", 2010),
        ("new_yorker_shouts", ("shouts and murmurs",), "publishing", 2000),
    ]
    return [PopCultureEntry(label=label, aliases=tuple(aliases), category=category, first_year=first_year)
            for label, aliases, category, first_year in raw_entries]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized


LEXICON = _build_pop_culture_lexicon()
LEXICON_NORMALIZED = [
    (
        entry,
        tuple(
            re.sub(r"\s+", " ", _normalize_text(alias.lower())).strip()
            for alias in entry.aliases
        ),
    )
    for entry in LEXICON
]


def detect_pop_culture(caption: str) -> Tuple[bool, List[PopCultureEntry]]:
    norm = _normalize_text(caption.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    padded = f" {norm} "
    matches: List[PopCultureEntry] = []
    for entry, aliases in LEXICON_NORMALIZED:
        for alias in aliases:
            if not alias:
                continue
            alias_token = f" {alias} "
            if alias_token in padded:
                matches.append(entry)
                break
    # Deduplicate by label
    unique = []
    seen = set()
    for entry in matches:
        if entry.label not in seen:
            seen.add(entry.label)
            unique.append(entry)
    return (len(unique) > 0, unique)


def load_metadata() -> pd.DataFrame:
    path = Path("caption-contest-data-gh-pages/io/info-510-thru-659.yaml")
    if not path.exists():
        return pd.DataFrame(columns=["contest_id", "start_date"])
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)
    records = []
    for cid, payload in raw.items():
        start = payload.get("start")
        if isinstance(start, str):
            start_ts = pd.to_datetime(start)
        elif start is None:
            start_ts = pd.NaT
        else:
            # yaml already parsed into datetime.date
            start_ts = pd.to_datetime(start)
        records.append({"contest_id": int(cid), "start_date": start_ts})
    return pd.DataFrame(records)


def infer_contest_dates(contest_ids: Sequence[int], metadata: pd.DataFrame) -> pd.Series:
    contest_ids = np.array(sorted(contest_ids))
    known = metadata.dropna()
    if known.empty:
        # fallback: assume contest 500 started Jan 2016 at weekly cadence
        base_date = pd.Timestamp("2016-01-01")
        estimated = [base_date + pd.Timedelta(days=7 * (cid - 500)) for cid in contest_ids]
        return pd.Series(estimated, index=contest_ids, name="start_date")
    x = known["contest_id"].astype(float).to_numpy()
    y = known["start_date"].astype("int64").astype(float)
    slope, intercept = np.polyfit(x, y, deg=1)
    estimates = intercept + slope * contest_ids
    dates = pd.to_datetime(estimates.astype("int64"))
    inferred = pd.Series(dates, index=contest_ids, name="start_date")
    # Overwrite with observed dates where available
    for _, row in known.iterrows():
        inferred.loc[int(row["contest_id"])] = row["start_date"]
    return inferred.sort_index()


def parse_summary_id(summary_id: str) -> Tuple[int, str]:
    base = summary_id.replace(".csv", "")
    if "_" in base:
        contest, method = base.split("_", 1)
    else:
        contest, method = base, "final"
    return int(contest), method


def load_caption_summaries() -> pd.DataFrame:
    summary_ids = ccd.summary_ids()
    cache: Dict[str, pd.DataFrame] = {}
    contest_to_ids: Dict[int, List[str]] = defaultdict(list)
    for summary_id in summary_ids:
        sid = str(summary_id)
        contest_id, _ = parse_summary_id(sid)
        contest_to_ids[contest_id].append(sid)

    frames: List[pd.DataFrame] = []
    for contest_id, sids in sorted(contest_to_ids.items()):
        best_sid = None
        best_votes = -1
        for sid in sids:
            if sid not in cache:
                df = ccd.summary(sid).copy()
                df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
                cache[sid] = df
            total_votes = cache[sid]["votes"].sum()
            if total_votes > best_votes:
                best_sid = sid
                best_votes = total_votes
        if best_sid is None:
            continue
        best_df = cache[best_sid].copy()
        _, method = parse_summary_id(best_sid)
        if "rank" not in best_df.columns:
            # Derive rank from mean with stable tie-breaking on votes.
            best_df = best_df.sort_values(["mean", "votes"], ascending=[False, False]).reset_index(drop=True)
            best_df.insert(0, "rank", range(len(best_df)))
        best_df["contest_id"] = contest_id
        best_df["method"] = method
        frames.append(best_df)
    all_captions = pd.concat(frames, ignore_index=True)
    return all_captions


THEME_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "office": (
        "office",
        "meeting",
        "conference",
        "boss",
        "coworker",
        "cubicle",
        "memo",
        "corporate",
        "spreadsheet",
        "deadline",
    ),
    "medical": (
        "doctor",
        "nurse",
        "hospital",
        "patient",
        "clinic",
        "surgery",
        "waiting room",
        "prescription",
        "diagnosis",
    ),
    "politics": (
        "president",
        "white house",
        "congress",
        "campaign",
        "senate",
        "mayor",
        "debate",
        "election",
        "polling",
    ),
    "legal": ("court", "judge", "lawyer", "jury", "trial", "lawsuit"),
    "technology": (
        "robot",
        "algorithm",
        "app",
        "smartphone",
        "smart phone",
        "software",
        "internet",
        "ai ",
        "artificial intelligence",
        "data center",
    ),
    "sports": (
        "coach",
        "team",
        "stadium",
        "locker room",
        "player",
        "referee",
        "football",
        "baseball",
        "basketball",
    ),
    "finance": ("bank", "money", "investment", "market", "economy", "stock", "shareholder"),
    "animals": ("dog", "cat", "pet", "animal", "bird", "horse", "cow", "pig"),
    "relationship": ("marriage", "married", "wife", "husband", "dating", "couple", "relationship"),
    "food": ("restaurant", "menu", "chef", "kitchen", "dinner", "table", "wine"),
    "travel": ("airport", "flight", "plane", "pilot", "luggage", "hotel", "vacation"),
    "arts": ("gallery", "museum", "artist", "painting", "sculpture", "exhibit"),
}


def assign_image_theme(captions: pd.DataFrame) -> pd.Series:
    grouped = captions.sort_values("votes", ascending=False).groupby("contest_id")
    theme_map = {}
    for contest_id, group in grouped:
        top_text = " ".join(group.head(200)["caption"].tolist()).lower()
        counts = {}
        for theme, keywords in THEME_KEYWORDS.items():
            total = 0
            for kw in keywords:
                total += top_text.count(kw)
            counts[theme] = total
        best_theme, best_score = max(counts.items(), key=lambda kv: kv[1])
        theme_map[contest_id] = best_theme if best_score >= 3 else "general"
    return pd.Series(theme_map, name="image_theme")


def compute_polarization_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vote_spread"] = (df["funny"] - df["not_funny"]) / df["votes"].clip(lower=1)
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df.loc[~np.isfinite(df["precision"]), "precision"] = np.nan
    grouped = df.groupby("pop_reference")
    summary = grouped.agg(
        captions=("caption", "count"),
        mean_score=("mean", "mean"),
        median_spread=("vote_spread", "median"),
        iqr_spread=("vote_spread", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        high_polarization_share=("vote_spread", lambda s: (s.abs() > 0.5).mean()),
        mean_precision=("precision", "mean"),
        precision_p90=("precision", lambda s: s.quantile(0.9)),
    )
    summary.index = summary.index.map({0: "non_reference", 1: "reference"})
    return summary


def categorize_recency(recency_years: float) -> str:
    if math.isnan(recency_years):
        return "unknown"
    if recency_years <= 3:
        return "fresh (<3y)"
    if recency_years <= 10:
        return "current (3-10y)"
    if recency_years <= 20:
        return "nostalgic (10-20y)"
    return "retro (>20y)"


def summarize_reference_types(df: pd.DataFrame) -> pd.DataFrame:
    ref_df = df[df["pop_reference"] == 1].copy()
    if ref_df.empty:
        return pd.DataFrame()
    summary = (
        ref_df.groupby(["primary_category", "image_theme"])
        .agg(
            captions=("caption", "count"),
            mean_score=("mean", "mean"),
            recent_share=("recency_bucket", lambda s: (s == "fresh (<3y)").mean()),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )
    return summary


def build_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["contest_date"].dt.year
    yearly = df.groupby("year").agg(
        total_captions=("caption", "count"),
        pop_captions=("pop_reference", "sum"),
    )
    pop_mask = df["pop_reference"] == 1
    pop_means = df.loc[pop_mask].groupby("year")["mean"].mean()
    non_means = df.loc[~pop_mask].groupby("year")["mean"].mean()
    yearly["pop_share"] = yearly["pop_captions"] / yearly["total_captions"]
    yearly["pop_mean"] = pop_means
    yearly["non_mean"] = non_means
    yearly["pop_minus_non"] = yearly["pop_mean"] - yearly["non_mean"]
    yearly = yearly.reset_index()
    return yearly


def build_recency_table(df: pd.DataFrame) -> pd.DataFrame:
    ref_df = df[df["pop_reference"] == 1]
    if ref_df.empty:
        return pd.DataFrame()
    summary = (
        ref_df.groupby("recency_bucket")
        .agg(
            captions=("caption", "count"),
            mean_age=("reference_age", "mean"),
            mean_score=("mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )
    return summary


def plot_yearly_trends(yearly: pd.DataFrame, out_path: Path) -> None:
    if yearly.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(yearly["year"], yearly["pop_share"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("Pop-culture share")
    axes[0].set_title("Share of captions with pop-culture references")
    axes[0].grid(alpha=0.3)

    axes[1].plot(yearly["year"], yearly["pop_mean"], label="Pop references", color="#d62728")
    axes[1].plot(yearly["year"], yearly["non_mean"], label="Non references", color="#2ca02c")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Average score")
    axes[1].set_title("Average funniness by year")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[1].set_xlabel("Contest year")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    captions = load_caption_summaries()
    captions["caption"] = captions["caption"].astype(str)
    captions["caption_length"] = captions["caption"].str.len()
    captions["num_words"] = captions["caption"].str.split().apply(len)

    # Detect pop culture references
    pop_flags = []
    categories = []
    first_years = []
    for text in captions["caption"]:
        flag, matches = detect_pop_culture(text)
        pop_flags.append(1 if flag else 0)
        categories.append("|".join(sorted({m.category for m in matches})) if matches else None)
        first_years.append(min(m.first_year for m in matches) if matches else np.nan)
    captions["pop_reference"] = pop_flags
    captions["reference_categories"] = categories
    captions["reference_first_year"] = first_years
    captions["primary_category"] = captions["reference_categories"].str.split("|").str[0]

    # Contest dates
    metadata = load_metadata()
    contest_dates = infer_contest_dates(sorted(captions["contest_id"].unique()), metadata)
    captions = captions.merge(
        contest_dates.rename("contest_date"),
        left_on="contest_id",
        right_index=True,
        how="left",
    )
    captions["contest_year"] = captions["contest_date"].dt.year

    # Recency calculations
    captions["reference_age"] = captions["contest_year"] - captions["reference_first_year"]
    captions.loc[captions["reference_age"] < 0, "reference_age"] = np.nan
    captions["recency_bucket"] = captions["reference_age"].apply(categorize_recency)

    # Assign image themes
    theme_map = assign_image_theme(captions[["contest_id", "caption", "votes"]])
    captions = captions.merge(
        theme_map.rename("image_theme"),
        left_on="contest_id",
        right_index=True,
        how="left",
    )
    captions["image_theme"] = captions["image_theme"].fillna("general")

    # Polarization metrics
    polarization = compute_polarization_metrics(captions)
    polarization_path = out_dir / "polarization_summary.csv"
    polarization.to_csv(polarization_path)

    # Recency analysis
    recency_table = build_recency_table(captions)
    recency_path = out_dir / "recency_summary.csv"
    recency_table.to_csv(recency_path, index=False)

    # Reference type vs image theme
    type_theme_table = summarize_reference_types(captions)
    type_theme_path = out_dir / "reference_type_theme.csv"
    type_theme_table.to_csv(type_theme_path, index=False)

    # Yearly trends + plot
    yearly = build_yearly_trends(captions)
    yearly_path = out_dir / "yearly_trends.csv"
    yearly.to_csv(yearly_path, index=False)
    plot_yearly_trends(yearly, figures_dir / "yearly_pop_culture_trends.png")

    # Additional summary stats
    pop_share = captions["pop_reference"].mean()
    avg_scores = captions.groupby("pop_reference")["mean"].mean()

    summary_payload = {
        "total_captions": int(len(captions)),
        "total_contests": int(captions["contest_id"].nunique()),
        "pop_reference_share": float(pop_share),
        "mean_score_reference": float(avg_scores.get(1, np.nan)),
        "mean_score_non_reference": float(avg_scores.get(0, np.nan)),
    }
    summary_path = out_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("=== Pop Culture Caption Analysis ===")
    print(json.dumps(summary_payload, indent=2))
    print("\nPolarization summary:")
    print(polarization)
    if not recency_table.empty:
        print("\nRecency buckets:")
        print(recency_table)
    if not type_theme_table.empty:
        print("\nTop reference types by image theme:")
        print(type_theme_table.head(15))
    if not yearly.empty:
        print("\nYearly pop-culture share and average performance written to", yearly_path)
        print(f"Plot saved to {figures_dir / 'yearly_pop_culture_trends.png'}")


if __name__ == "__main__":
    main()
