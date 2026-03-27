import { useState, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Feature label mapping ────────────────────────────────────
const FEATURE_LABELS = {
  goal_below_cat_median: "Goal Below Category Average",
  sub_cat_code_te: "Category Success Rate",
  name_has_colon: "Subtitle in Campaign Name",
  goal_large: "High Funding Goal",
  goal_log: "Funding Goal Size",
  goal_x_duration: "Goal × Duration",
  usd_goal_real: "Amount Requested",
  main_cat_code_te: "Main Category Win Rate",
  "sub_Tabletop Games": "Tabletop Games Niche",
  duration_long: "Long Campaign Duration",
  duration_log: "Campaign Duration",
  duration_days: "Duration in Days",
  goal_log_sq: "Goal Non-linearity",
  name_word_count: "Title Word Count",
  goal_pct_in_cat: "Goal Rank in Category",
  name_has_number: "Number in Title",
  launch_year: "Launch Year",
  goal_vs_cat_median: "Goal vs Category Norm",
  name_has_film: "Film Keyword",
  launch_hour: "Launch Hour",
  name_has_game: "Game Keyword",
  duration_ideal: "Ideal Duration Range",
  launch_month: "Launch Month",
  launch_weekday: "Launch Day of Week",
  is_us: "US-Based Campaign",
  goal_medium: "Medium Funding Goal",
  goal_per_day: "Daily Funding Target",
  name_length: "Campaign Name Length",
  name_has_exclaim: "Exclamation in Title",
  launch_good_month: "Peak Launch Month",
  launch_weekend: "Weekend Launch",
  sub_cat_code_te2: "Sub-Category Win Rate",
  main_cat_code_te2: "Category Win Rate",
  country_code_te: "Country Success Rate",
  country_code_te2: "Country Success Rate",
};

function readableFeature(raw) {
  return FEATURE_LABELS[raw] || raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function featureGroup(raw) {
  if (raw.startsWith("goal") || raw === "usd_goal_real") return { label: "Funding Goal", color: "#6366f1" };
  if (raw.startsWith("sub_") || raw.startsWith("main_cat") || raw.startsWith("cat_")) return { label: "Category", color: "#a855f7" };
  if (raw.startsWith("name")) return { label: "Campaign Name", color: "#f97316" };
  if (raw.startsWith("duration")) return { label: "Duration", color: "#22c55e" };
  if (raw.startsWith("launch")) return { label: "Launch Timing", color: "#06b6d4" };
  if (raw.startsWith("country") || raw === "is_us") return { label: "Location", color: "#ec4899" };
  return { label: "Other", color: "#6c63ff" };
}

// ── Similar campaigns data ───────────────────────────────────
const SIMILAR_CAMPAIGNS = {
  Games: [
    { name: "Exploding Kittens", goal: 10000, raised: 8782571, backers: 219382, result: "Success", year: 2015 },
    { name: "Gloomhaven", goal: 100000, raised: 386104, backers: 4904, result: "Success", year: 2015 },
    { name: "Wingspan", goal: 25000, raised: 1366796, backers: 12743, result: "Success", year: 2019 },
    { name: "Dark Souls Board Game", goal: 50000, raised: 3771474, backers: 31198, result: "Success", year: 2016 },
  ],
  Technology: [
    { name: "Pebble Time Smartwatch", goal: 500000, raised: 20338986, backers: 78471, result: "Success", year: 2015 },
    { name: "Oculus Rift", goal: 250000, raised: 2437429, backers: 9522, result: "Success", year: 2012 },
    { name: "Flow Hive", goal: 70000, raised: 12173055, backers: 37533, result: "Success", year: 2015 },
    { name: "Healbe GoBe", goal: 100000, raised: 1056630, backers: 6328, result: "Success", year: 2014 },
  ],
  "Film & Video": [
    { name: "Veronica Mars Movie", goal: 2000000, raised: 5702153, backers: 91585, result: "Success", year: 2013 },
    { name: "Kung Fury", goal: 200000, raised: 630019, backers: 17713, result: "Success", year: 2013 },
    { name: "Blue Mountain State Movie", goal: 1500000, raised: 1895915, backers: 26473, result: "Success", year: 2015 },
    { name: "Lazer Team", goal: 650000, raised: 2480926, backers: 37427, result: "Success", year: 2014 },
  ],
  Music: [
    { name: "Amanda Palmer: Theatre", goal: 100000, raised: 1192793, backers: 24883, result: "Success", year: 2012 },
    { name: "Jill Sobule's Album", goal: 75000, raised: 88241, backers: 1611, result: "Success", year: 2009 },
    { name: "De La Soul Music", goal: 110000, raised: 601322, backers: 10795, result: "Success", year: 2015 },
    { name: "Super Duper Kyle EP", goal: 15000, raised: 16450, backers: 324, result: "Success", year: 2014 },
  ],
  Art: [
    { name: "Giant Panda Mural", goal: 5000, raised: 6200, backers: 118, result: "Success", year: 2016 },
    { name: "The Oatmeal Tesla Museum", goal: 850000, raised: 1370461, backers: 33492, result: "Success", year: 2012 },
    { name: "Reading Rainbow Revival", goal: 1000000, raised: 5408916, backers: 105857, result: "Success", year: 2014 },
    { name: "Banksy Art Book", goal: 15000, raised: 19320, backers: 441, result: "Success", year: 2017 },
  ],
  Design: [
    { name: "Coolest Cooler", goal: 50000, raised: 13285226, backers: 62642, result: "Success", year: 2014 },
    { name: "Micro: World's Smallest 3D Printer", goal: 50000, raised: 3401361, backers: 11381, result: "Success", year: 2013 },
    { name: "Suntory BOSS Rainbow Mountain", goal: 30000, raised: 32150, backers: 670, result: "Success", year: 2018 },
    { name: "Nimuno Loops Tape", goal: 6000, raised: 2437390, backers: 60875, result: "Success", year: 2017 },
  ],
  Fashion: [
    { name: "Ministry of Supply Dress Shirt", goal: 30000, raised: 430264, backers: 3472, result: "Success", year: 2012 },
    { name: "STACT Wine Rack", goal: 30000, raised: 239054, backers: 1788, result: "Success", year: 2013 },
    { name: "Wooly Bully Sock Suspenders", goal: 5000, raised: 5312, backers: 134, result: "Success", year: 2016 },
    { name: "Tommy John Underwear Campaign", goal: 50000, raised: 25000, backers: 312, result: "Failed", year: 2017 },
  ],
  Theater: [
    { name: "Sleep No More Expansion", goal: 20000, raised: 28400, backers: 542, result: "Success", year: 2016 },
    { name: "Hamilton Documentary", goal: 500000, raised: 1466837, backers: 6534, result: "Success", year: 2017 },
    { name: "Natasha Pierre & The Great Comet", goal: 75000, raised: 374331, backers: 2219, result: "Success", year: 2015 },
    { name: "Hadestown Off-Broadway", goal: 40000, raised: 41800, backers: 623, result: "Success", year: 2016 },
  ],
};

function getSimilar(category) {
  return SIMILAR_CAMPAIGNS[category] || SIMILAR_CAMPAIGNS["Games"];
}

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const CATEGORIES = ["Games","Technology","Music","Film & Video","Art","Design","Publishing","Fashion","Food","Theater","Dance","Comics","Photography","Crafts","Journalism"];
const COUNTRIES = ["US","GB","CA","AU","DE","FR","NL","SE","IT","ES","DK","NO","BE","AT","NZ","CH","MX","SG","HK","IE","IN"];
const CAMPAIGN_TYPES = ["Product Launch","Creative Project","Community Initiative","Social Cause","Educational Program"];
const TEAM_SIZES = ["Solo Creator","2–5 Members","6–10 Members","10+ Members"];
const PRIOR_CAMPAIGNS = ["First Campaign","1 Previous","2–3 Previous","4+ Previous"];
const SOCIAL_FOLLOWERS = ["Under 1K","1K–10K","10K–50K","50K–100K","100K+"];
const PROTOTYPE_STATUS = ["Concept Only","Early Prototype","Working Prototype","Production Ready"];

const css = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Playfair+Display:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #080b14;
  --bg2: #0d1221;
  --surface: #111827;
  --surface2: #1a2235;
  --surface3: #1f2a3f;
  --border: #1e2d45;
  --border2: #263548;
  --accent: #3b82f6;
  --accent2: #6366f1;
  --success: #10b981;
  --fail: #ef4444;
  --warn: #f59e0b;
  --gold: #f59e0b;
  --text: #e2e8f0;
  --text2: #94a3b8;
  --text3: #64748b;
  --font: 'Space Grotesk', sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --display: 'Playfair Display', serif;
  --radius: 12px;
  --shadow: 0 4px 24px rgba(0,0,0,0.4);
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Subtle grid background */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

.app {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── SIDEBAR ─────────────────────────────────── */
.layout { display: flex; min-height: 100vh; }

.sidebar {
  width: 240px;
  min-height: 100vh;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 0;
  position: fixed;
  left: 0; top: 0; bottom: 0;
  z-index: 100;
}

.sidebar-logo {
  padding: 28px 24px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-logo-title {
  font-family: var(--display);
  font-size: 18px;
  font-weight: 800;
  color: white;
  line-height: 1.2;
}
.sidebar-logo-sub {
  font-size: 10px;
  color: var(--text3);
  font-family: var(--mono);
  margin-top: 4px;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.sidebar-nav {
  padding: 20px 12px;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-section {
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text3);
  font-family: var(--mono);
  padding: 8px 12px 4px;
  margin-top: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: var(--text2);
  border: none;
  background: none;
  font-family: var(--font);
  transition: all 0.15s;
  width: 100%;
  text-align: left;
}
.nav-item:hover { background: var(--surface2); color: var(--text); }
.nav-item.active {
  background: rgba(59,130,246,0.15);
  color: var(--accent);
  border: 1px solid rgba(59,130,246,0.2);
}
.nav-icon { font-size: 15px; width: 20px; text-align: center; }

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid var(--border);
}
.accuracy-pill {
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.2);
  border-radius: 8px;
  padding: 10px 14px;
}
.accuracy-pill-label { font-size: 9px; color: var(--text3); font-family: var(--mono); text-transform: uppercase; letter-spacing: 1px; }
.accuracy-pill-value { font-size: 22px; font-weight: 700; color: var(--success); margin-top: 2px; }
.accuracy-pill-sub { font-size: 10px; color: var(--text3); font-family: var(--mono); margin-top: 2px; }

/* ── MAIN CONTENT ────────────────────────────── */
.main {
  margin-left: 240px;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.topbar {
  height: 60px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  position: sticky;
  top: 0;
  z-index: 50;
}

.topbar-title { font-size: 15px; font-weight: 600; color: var(--text); }
.topbar-badges { display: flex; gap: 8px; align-items: center; }
.tb-badge {
  font-size: 10px;
  font-family: var(--mono);
  padding: 4px 10px;
  border-radius: 20px;
  border: 1px solid var(--border);
  color: var(--text3);
  background: var(--surface2);
}
.tb-badge.green { color: var(--success); border-color: rgba(16,185,129,0.25); background: rgba(16,185,129,0.08); }
.tb-badge.blue { color: var(--accent); border-color: rgba(59,130,246,0.25); background: rgba(59,130,246,0.08); }

.content { padding: 32px; flex: 1; }

/* ── CARDS ───────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
}
.card-sm { padding: 18px; }
.card-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 16px;
  font-family: var(--mono);
}
.card-heading {
  font-size: 16px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}
.card-sub { font-size: 12px; color: var(--text3); font-family: var(--mono); margin-bottom: 20px; }

/* ── GRIDS ───────────────────────────────────── */
.g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.g3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.g4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.full { grid-column: 1 / -1; }

/* ── FORM ────────────────────────────────────── */
.form-section {
  margin-bottom: 28px;
}
.form-section-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--accent);
  font-family: var(--mono);
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.fg { display: flex; flex-direction: column; gap: 6px; }
.fl { font-size: 11px; font-weight: 500; color: var(--text2); letter-spacing: 0.5px; font-family: var(--mono); }
.fi {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 13px;
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  outline: none;
  width: 100%;
  transition: border-color 0.2s, background 0.2s;
}
.fi:focus { border-color: var(--accent); background: var(--surface3); }
.fi::placeholder { color: var(--text3); }
.fi option { background: var(--surface2); }

.form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.form-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }

/* ── BUTTON ──────────────────────────────────── */
.btn-predict {
  width: 100%;
  padding: 14px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  font-family: var(--font);
  transition: all 0.2s;
  margin-top: 8px;
  letter-spacing: 0.3px;
  box-shadow: 0 4px 16px rgba(59,130,246,0.3);
}
.btn-predict:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(59,130,246,0.4); }
.btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

/* ── STAT CARDS ──────────────────────────────── */
.stat {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  position: relative;
  overflow: hidden;
}
.stat::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--c, var(--accent));
}
.stat-v { font-size: 30px; font-weight: 800; line-height: 1; color: var(--c, var(--accent)); }
.stat-l { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-top: 6px; }
.stat-s { font-size: 10px; color: var(--text3); margin-top: 4px; }

/* ── GAUGE ───────────────────────────────────── */
.gauge-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0 12px;
}

/* ── RESULT PANEL ────────────────────────────── */
.result-panel {
  border-radius: var(--radius);
  border: 1px solid;
  padding: 28px;
  text-align: center;
}
.result-panel.s { background: rgba(16,185,129,0.06); border-color: rgba(16,185,129,0.3); }
.result-panel.f { background: rgba(239,68,68,0.06); border-color: rgba(239,68,68,0.3); }
.result-pct { font-size: 52px; font-weight: 800; line-height: 1; font-family: var(--display); }
.result-pct.s { color: var(--success); }
.result-pct.f { color: var(--fail); }
.result-pred { font-size: 18px; font-weight: 700; margin-top: 10px; }
.result-sub { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-top: 6px; }
.risk-badge {
  display: inline-block;
  margin-top: 12px;
  padding: 5px 16px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--mono);
  letter-spacing: 0.5px;
}

/* ── SHAP ────────────────────────────────────── */
.shap-row { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.shap-label { font-size: 12px; color: var(--text); width: 200px; flex-shrink: 0; font-weight: 500; }
.shap-group { font-size: 9px; color: var(--text3); font-family: var(--mono); }
.shap-track { flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
.shap-fill { height: 100%; border-radius: 4px; transition: width 0.7s ease; }
.shap-val { font-size: 11px; font-family: var(--mono); width: 52px; text-align: right; flex-shrink: 0; font-weight: 500; }

/* ── SIMILAR CAMPAIGNS ───────────────────────── */
.camp-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  transition: border-color 0.2s, transform 0.2s;
}
.camp-card:hover { border-color: var(--border2); transform: translateY(-2px); }
.camp-name { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 8px; line-height: 1.3; }
.camp-meta { display: flex; flex-direction: column; gap: 4px; }
.camp-row { display: flex; justify-content: space-between; align-items: center; }
.camp-key { font-size: 10px; color: var(--text3); font-family: var(--mono); }
.camp-val { font-size: 11px; font-weight: 600; color: var(--text2); font-family: var(--mono); }
.camp-badge {
  font-size: 9px;
  font-family: var(--mono);
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
  margin-top: 8px;
  display: inline-block;
}
.camp-badge.s { background: rgba(16,185,129,0.12); color: var(--success); border: 1px solid rgba(16,185,129,0.2); }
.camp-badge.f { background: rgba(239,68,68,0.12); color: var(--fail); border: 1px solid rgba(239,68,68,0.2); }

/* ── AI ROADMAP ──────────────────────────────── */
.roadmap-wrap { position: relative; }
.roadmap-text {
  font-size: 13px;
  line-height: 1.85;
  color: var(--text2);
  white-space: pre-wrap;
  font-family: var(--font);
}
.roadmap-loading {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 0;
  color: var(--text3);
  font-size: 13px;
  font-family: var(--mono);
}
.pulse-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 0.3; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.2); } }

.roadmap-section {
  margin-bottom: 16px;
  padding: 14px 16px;
  background: var(--surface2);
  border-radius: 8px;
  border-left: 3px solid var(--accent);
}
.roadmap-section-title {
  font-size: 12px;
  font-weight: 700;
  color: var(--accent);
  font-family: var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
}
.roadmap-section-body {
  font-size: 13px;
  line-height: 1.7;
  color: var(--text2);
}

/* ── FI BARS ─────────────────────────────────── */
.fi-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.fi-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.fi-name { font-size: 12px; color: var(--text); width: 220px; flex-shrink: 0; font-weight: 500; }
.fi-track { flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
.fi-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.fi-pct { font-size: 11px; font-family: var(--mono); color: var(--text3); width: 46px; text-align: right; }

.legend-strip { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: var(--text3); font-family: var(--mono); }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── TIP/INSIGHT ─────────────────────────────── */
.tip {
  font-size: 11px;
  color: var(--text3);
  font-family: var(--mono);
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--surface2);
  border-radius: 8px;
  border-left: 3px solid var(--accent);
  line-height: 1.6;
}
.insight {
  font-size: 12px;
  color: #6ee7b7;
  font-family: var(--mono);
  margin-top: 14px;
  padding: 12px 16px;
  background: rgba(16,185,129,0.07);
  border-radius: 8px;
  border-left: 3px solid var(--success);
  line-height: 1.6;
}

/* ── SCORE METER ─────────────────────────────── */
.score-meter { display: flex; gap: 3px; margin-top: 8px; }
.score-seg {
  height: 5px;
  border-radius: 3px;
  flex: 1;
  background: var(--surface3);
  transition: background 0.5s;
}

/* ── TABLE ───────────────────────────────────── */
.tbl { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
.tbl th { padding: 9px 12px; text-align: left; font-size: 10px; font-weight: 600; color: var(--text3); letter-spacing: 0.8px; border-bottom: 1px solid var(--border); text-transform: uppercase; }
.tbl td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text2); }
.tbl tr:last-child td { border-bottom: none; }
.tbl tr:hover td { background: var(--surface2); }

/* ── SPINNER ─────────────────────────────────── */
.spin { width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.2); border-top-color: white; border-radius: 50%; animation: spinning 0.6s linear infinite; display: inline-block; }
@keyframes spinning { to { transform: rotate(360deg); } }

/* ── EMPTY STATE ─────────────────────────────── */
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 48px 24px;
  color: var(--text3);
}
.empty-icon { font-size: 40px; }
.empty-text { font-size: 13px; text-align: center; line-height: 1.6; }

.divider { height: 1px; background: var(--border); margin: 24px 0; }
.mt { margin-top: 20px; }
.mb { margin-bottom: 20px; }

/* ── CONFIDENCE BAR ──────────────────────────── */
.conf-bar-wrap { height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; margin-top: 6px; }
.conf-bar { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 1s ease; }

/* ── SECTION HEADER ──────────────────────────── */
.sec-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.sec-title { font-size: 18px; font-weight: 700; color: var(--text); }
.sec-sub { font-size: 12px; color: var(--text3); font-family: var(--mono); margin-top: 2px; }

/* ── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
`;

// ── Gauge ────────────────────────────────────────────────────
function ProbGauge({ prob }) {
  const color = prob >= 70 ? "#10b981" : prob >= 50 ? "#3b82f6" : prob >= 35 ? "#f59e0b" : "#ef4444";
  const circ  = 2 * Math.PI * 60;
  return (
    <div className="gauge-wrap">
      <div style={{ position: "relative", width: 160, height: 160 }}>
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle cx="80" cy="80" r="60" fill="none" stroke="#1a2235" strokeWidth="12" />
          <circle cx="80" cy="80" r="60" fill="none" stroke={color} strokeWidth="12"
            strokeDasharray={`${(prob / 100) * circ} ${circ}`}
            strokeLinecap="round" transform="rotate(-90 80 80)"
            style={{ transition: "stroke-dasharray 1.2s ease" }} />
        </svg>
        <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
          <div style={{ fontSize: 30, fontWeight: 800, color, fontFamily: "var(--display)" }}>{prob}%</div>
          <div style={{ fontSize: 9, color: "var(--text3)", fontFamily: "var(--mono)", marginTop: 2, letterSpacing: 1 }}>SUCCESS PROB</div>
        </div>
      </div>
    </div>
  );
}

// ── SHAP chart ───────────────────────────────────────────────
function ShapChart({ data }) {
  if (!data?.length) return null;
  const maxAbs = Math.max(...data.map(d => Math.abs(d.shap_value)));
  return (
    <div>
      {data.map((d, i) => {
        const label = readableFeature(d.feature);
        const grp   = featureGroup(d.feature);
        const isPos = d.direction === "positive";
        const color = isPos ? "#10b981" : "#ef4444";
        return (
          <div key={i} className="shap-row">
            <div style={{ width: 200, flexShrink: 0 }}>
              <div className="shap-label">{label}</div>
              <div className="shap-group" style={{ color: grp.color }}>▪ {grp.label}</div>
            </div>
            <div className="shap-track">
              <div className="shap-fill" style={{ width: `${(Math.abs(d.shap_value)/maxAbs)*100}%`, background: color }} />
            </div>
            <div className="shap-val" style={{ color }}>
              {d.shap_value > 0 ? "+" : ""}{d.shap_value.toFixed(3)}
            </div>
          </div>
        );
      })}
      <div className="tip">Green = boosts success probability · Red = reduces it · Length = strength of influence</div>
    </div>
  );
}

// ── Feature Importance ───────────────────────────────────────
function FIChart({ data }) {
  if (!data?.length) return <div style={{ color:"var(--text3)", fontFamily:"var(--mono)", fontSize:12 }}>Loading...</div>;
  const KEEP = ["goal_below_cat_median","sub_cat_code_te","name_has_colon","goal_large","goal_log","goal_x_duration","main_cat_code_te","sub_Tabletop Games","duration_long","duration_days","name_word_count","goal_pct_in_cat","launch_year","name_has_number"];
  let filtered = data.filter(d => KEEP.includes(d.name));
  if (filtered.length < 5) filtered = data.slice(0, 14);
  const max = filtered[0]?.importance || 1;
  return (
    <div>
      <div className="legend-strip">
        {[["#6366f1","Funding Goal"],["#a855f7","Category"],["#f97316","Name Quality"],["#22c55e","Duration"],["#06b6d4","Launch Timing"]].map(([c,l]) => (
          <div key={l} className="legend-item"><div className="legend-dot" style={{ background:c }} />{l}</div>
        ))}
      </div>
      {filtered.map((d, i) => {
        const grp = featureGroup(d.name);
        return (
          <div key={i} className="fi-row">
            <div style={{ fontSize:10, color:"var(--text3)", fontFamily:"var(--mono)", width:20, textAlign:"right", flexShrink:0 }}>#{i+1}</div>
            <div className="fi-dot" style={{ background: grp.color }} />
            <div className="fi-name">{readableFeature(d.name)}</div>
            <div className="fi-track">
              <div className="fi-fill" style={{ width:`${(d.importance/max)*100}%`, background: grp.color, opacity:0.85 }} />
            </div>
            <div className="fi-pct">{(d.importance*100).toFixed(2)}%</div>
          </div>
        );
      })}
      <div className="insight">Key Insight: Setting a goal below your category average is 6× more predictive than any other single factor.</div>
    </div>
  );
}

// ── Similar Campaigns ────────────────────────────────────────
function SimilarCampaigns({ category, goal }) {
  const campaigns = getSimilar(category);
  return (
    <div>
      <div className="card-sub">Real Kickstarter campaigns in {category} — see how yours compares</div>
      <div className="g2" style={{ gap:12 }}>
        {campaigns.map((c, i) => (
          <div key={i} className="camp-card">
            <div className="camp-name">{c.name}</div>
            <div className="camp-meta">
              <div className="camp-row">
                <span className="camp-key">Goal</span>
                <span className="camp-val">${c.goal.toLocaleString()}</span>
              </div>
              <div className="camp-row">
                <span className="camp-key">Raised</span>
                <span className="camp-val" style={{ color: c.result === "Success" ? "var(--success)" : "var(--fail)" }}>
                  ${c.raised.toLocaleString()}
                </span>
              </div>
              <div className="camp-row">
                <span className="camp-key">Backers</span>
                <span className="camp-val">{c.backers.toLocaleString()}</span>
              </div>
              <div className="camp-row">
                <span className="camp-key">Multiplier</span>
                <span className="camp-val" style={{ color:"var(--gold)" }}>
                  {(c.raised / c.goal).toFixed(1)}×
                </span>
              </div>
            </div>
            <div className={`camp-badge ${c.result === "Success" ? "s" : "f"}`}>{c.result} · {c.year}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── AI Roadmap ───────────────────────────────────────────────
function AIRoadmap({ form, result, loading, roadmap }) {
  if (loading) return (
    <div className="roadmap-loading">
      <div className="pulse-dot" />
      <span>Generating personalised AI roadmap for your campaign...</span>
    </div>
  );
  if (!roadmap) return (
    <div className="empty">
      <div className="empty-icon">✦</div>
      <div className="empty-text">Run a prediction first to unlock your<br />AI-powered campaign strategy roadmap</div>
    </div>
  );

  // Parse sections from roadmap text
  const sections = roadmap.split(/\n(?=##\s)/).filter(Boolean);
  if (sections.length <= 1) {
    // Fallback: render as plain text
    return <div className="roadmap-text">{roadmap}</div>;
  }
  return (
    <div>
      {sections.map((s, i) => {
        const lines  = s.trim().split('\n');
        const title  = lines[0].replace(/^##\s*/, '').trim();
        const body   = lines.slice(1).join('\n').trim();
        const colors = ["#3b82f6","#10b981","#f59e0b","#a855f7","#ef4444","#06b6d4"];
        return (
          <div key={i} className="roadmap-section" style={{ borderLeftColor: colors[i % colors.length] }}>
            <div className="roadmap-section-title" style={{ color: colors[i % colors.length] }}>{title}</div>
            <div className="roadmap-section-body">{body}</div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]         = useState("predict");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [roadmap, setRoadmap] = useState(null);
  const [stats, setStats]     = useState(null);
  const [fiData, setFiData]   = useState([]);
  const [cats, setCats]       = useState({ categories: [], countries: [] });
  const [error, setError]     = useState("");

  const [form, setForm] = useState({
    // Core
    name: "", goal_usd: 10000, duration_days: 30,
    main_category: "Games", country: "US",
    launch_month: 4, launch_weekday: 1,
    // Extended
    campaign_type: "Product Launch",
    team_size: "Solo Creator",
    prior_campaigns: "First Campaign",
    social_followers: "1K–10K",
    prototype_status: "Working Prototype",
    has_video: "Yes",
    reward_tiers: "5",
    description: "",
  });

  const upd = (k, v) => setForm(f => ({ ...f, [k]: v }));

  useEffect(() => {
    fetch(`${API}/stats`).then(r => r.json()).then(setStats).catch(() => {});
    fetch(`${API}/feature-importance`).then(r => r.json()).then(d => setFiData(d.features || [])).catch(() => {});
    fetch(`${API}/categories`).then(r => r.json()).then(setCats).catch(() => {});
  }, []);

  const handlePredict = async () => {
    setLoading(true); setError(""); setResult(null); setRoadmap(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: form.name,
          goal_usd: Number(form.goal_usd),
          duration_days: Number(form.duration_days),
          main_category: form.main_category,
          country: form.country,
          launch_month: form.launch_month,
          launch_weekday: form.launch_weekday,
        })
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      const data = await res.json();
      setResult(data);
      setTab("result");
      // Auto-generate AI roadmap
      generateRoadmap(data);
    } catch (e) {
      setError(e.message || "Backend error — is the server running?");
    } finally { setLoading(false); }
  };

  const generateRoadmap = async (predResult) => {
    setAiLoading(true);
    try {
      const topFactors = (predResult.shap_explanation || [])
        .map(s => readableFeature(s.feature) + (s.direction === "positive" ? " (positive)" : " (negative)"))
        .join(", ");

      const payload = {
        campaign_name:    form.name || "Untitled Campaign",
        category:         form.main_category,
        goal_usd:         Number(form.goal_usd),
        duration_days:    Number(form.duration_days),
        country:          form.country,
        launch_month:     MONTHS[form.launch_month - 1],
        campaign_type:    form.campaign_type,
        team_size:        form.team_size,
        prior_campaigns:  form.prior_campaigns,
        social_followers: form.social_followers,
        prototype_status: form.prototype_status,
        has_video:        form.has_video,
        reward_tiers:     String(form.reward_tiers),
        description:      form.description || "",
        success_prob:     predResult.success_probability,
        risk_level:       predResult.risk_level,
        top_factors:      topFactors,
      };

      const res = await fetch(`${API}/roadmap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`Server error ${res.status}: ${text}`);
      }
      const data = JSON.parse(text);
      setRoadmap(data.roadmap || "No roadmap content returned.");
    } catch (e) {
      setRoadmap("Error: " + (e.message || "Unknown error. Check browser console (F12)."));
    } finally { setAiLoading(false); }
  };

  const isSuccess = result && result.success_probability >= 50;
  const riskColor = { "Low Risk":"#10b981","Moderate Risk":"#3b82f6","High Risk":"#f59e0b","Very High Risk":"#ef4444" };

  const NAV = [
    { id:"predict",    icon:"◈", label:"Predict" },
    { id:"result",     icon:"◉", label:"Results" },
    { id:"roadmap",    icon:"◎", label:"AI Roadmap" },
    { id:"similar",    icon:"⊞", label:"Similar Campaigns" },
    { id:"dashboard",  icon:"⊟", label:"Model Dashboard" },
    { id:"importance", icon:"⊠", label:"Feature Importance" },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="layout">

        {/* ── SIDEBAR ── */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="sidebar-logo-title">Kickstarter<br />Intelligence</div>
            <div className="sidebar-logo-sub">ML Prediction Suite</div>
          </div>

          <nav className="sidebar-nav">
            <div className="nav-section">Analysis</div>
            {NAV.map(n => (
              <button key={n.id} className={`nav-item ${tab === n.id ? "active" : ""}`} onClick={() => setTab(n.id)}>
                <span className="nav-icon">{n.icon}</span>
                {n.label}
                {n.id === "result" && result && (
                  <span style={{ marginLeft:"auto", fontSize:10, fontFamily:"var(--mono)", color: isSuccess ? "var(--success)" : "var(--fail)" }}>
                    {result.success_probability}%
                  </span>
                )}
              </button>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="accuracy-pill">
              <div className="accuracy-pill-label">Model Accuracy</div>
              <div className="accuracy-pill-value">70.5%</div>
              <div className="accuracy-pill-sub">AUC 0.774 · 331K trained</div>
            </div>
          </div>
        </aside>

        {/* ── MAIN ── */}
        <main className="main">
          <div className="topbar">
            <div className="topbar-title">
              {NAV.find(n => n.id === tab)?.label || "Kickstarter Intelligence"}
            </div>
            <div className="topbar-badges">
              <div className="tb-badge">XGB + LGB + CatBoost + RF</div>
              <div className="tb-badge blue">Stacking Ensemble</div>
              <div className="tb-badge green">AUC 0.7737</div>
              {result && <div className="tb-badge" style={{ color: riskColor[result.risk_level], borderColor: riskColor[result.risk_level]+"44", background: riskColor[result.risk_level]+"11" }}>{result.risk_level}</div>}
            </div>
          </div>

          <div className="content">

            {/* ══════════════ PREDICT TAB ══════════════ */}
            {tab === "predict" && (
              <div className="g2" style={{ alignItems:"start" }}>
                <div className="card">
                  <div className="card-heading">Campaign Details</div>
                  <div className="card-sub">Fill in your campaign information for an ML-powered success prediction</div>

                  <div className="form-section">
                    <div className="form-section-title">Core Campaign Info</div>
                    <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
                      <div className="fg">
                        <label className="fl">Campaign Name</label>
                        <input className="fi" placeholder='e.g. "Epic Quest: A Strategy Board Game for Families"'
                          value={form.name} onChange={e => upd("name", e.target.value)} />
                      </div>
                      <div className="form-grid-3">
                        <div className="fg">
                          <label className="fl">Funding Goal (USD)</label>
                          <input className="fi" type="number" min="1" value={form.goal_usd} onChange={e => upd("goal_usd", e.target.value)} />
                        </div>
                        <div className="fg">
                          <label className="fl">Duration (days)</label>
                          <input className="fi" type="number" min="1" max="90" value={form.duration_days} onChange={e => upd("duration_days", e.target.value)} />
                        </div>
                        <div className="fg">
                          <label className="fl">Category</label>
                          <select className="fi" value={form.main_category} onChange={e => upd("main_category", e.target.value)}>
                            {(cats.categories.length ? cats.categories : CATEGORIES).map(c => <option key={c}>{c}</option>)}
                          </select>
                        </div>
                      </div>
                      <div className="form-grid-3">
                        <div className="fg">
                          <label className="fl">Country</label>
                          <select className="fi" value={form.country} onChange={e => upd("country", e.target.value)}>
                            {([...new Set([...(cats.countries.length ? cats.countries : COUNTRIES), "IN"])]).map(c => <option key={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="fg">
                          <label className="fl">Launch Month</label>
                          <select className="fi" value={form.launch_month} onChange={e => upd("launch_month", Number(e.target.value))}>
                            {MONTHS.map((m, i) => <option key={i+1} value={i+1}>{m}</option>)}
                          </select>
                        </div>
                        <div className="fg">
                          <label className="fl">Launch Day</label>
                          <select className="fi" value={form.launch_weekday} onChange={e => upd("launch_weekday", Number(e.target.value))}>
                            {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="form-section">
                    <div className="form-section-title">Campaign Profile</div>
                    <div className="form-grid-2">
                      <div className="fg">
                        <label className="fl">Campaign Type</label>
                        <select className="fi" value={form.campaign_type} onChange={e => upd("campaign_type", e.target.value)}>
                          {CAMPAIGN_TYPES.map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                      <div className="fg">
                        <label className="fl">Team Size</label>
                        <select className="fi" value={form.team_size} onChange={e => upd("team_size", e.target.value)}>
                          {TEAM_SIZES.map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                      <div className="fg">
                        <label className="fl">Prior Campaigns</label>
                        <select className="fi" value={form.prior_campaigns} onChange={e => upd("prior_campaigns", e.target.value)}>
                          {PRIOR_CAMPAIGNS.map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                      <div className="fg">
                        <label className="fl">Prototype Status</label>
                        <select className="fi" value={form.prototype_status} onChange={e => upd("prototype_status", e.target.value)}>
                          {PROTOTYPE_STATUS.map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                    </div>
                  </div>

                  <div className="form-section">
                    <div className="form-section-title">Marketing & Reach</div>
                    <div className="form-grid-3">
                      <div className="fg">
                        <label className="fl">Social Followers</label>
                        <select className="fi" value={form.social_followers} onChange={e => upd("social_followers", e.target.value)}>
                          {SOCIAL_FOLLOWERS.map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                      <div className="fg">
                        <label className="fl">Has Promo Video</label>
                        <select className="fi" value={form.has_video} onChange={e => upd("has_video", e.target.value)}>
                          {["Yes","No","In Production"].map(c => <option key={c}>{c}</option>)}
                        </select>
                      </div>
                      <div className="fg">
                        <label className="fl">Reward Tiers</label>
                        <input className="fi" type="number" min="1" max="20" value={form.reward_tiers}
                          onChange={e => upd("reward_tiers", e.target.value)} placeholder="e.g. 5" />
                      </div>
                    </div>
                    <div className="fg" style={{ marginTop:14 }}>
                      <label className="fl">Campaign Description (optional — improves AI roadmap)</label>
                      <textarea className="fi" rows={3}
                        placeholder="Briefly describe your campaign idea, target audience, and what makes it unique..."
                        value={form.description} onChange={e => upd("description", e.target.value)}
                        style={{ resize:"vertical", minHeight:72 }} />
                    </div>
                  </div>

                  <button className="btn-predict" onClick={handlePredict} disabled={loading}>
                    {loading ? <span className="spin" /> : "Run ML Prediction + Generate AI Roadmap"}
                  </button>
                  {error && <div style={{ color:"var(--fail)", fontSize:12, marginTop:10, fontFamily:"var(--mono)", padding:"10px 14px", background:"rgba(239,68,68,0.08)", borderRadius:8 }}>{error}</div>}
                </div>

                {/* Right panel — quick stats or result preview */}
                <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
                  <div className="card">
                    <div className="card-title">Platform Statistics</div>
                    <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
                      {[
                        { v:"331,651", l:"Campaigns Trained On", c:"#3b82f6" },
                        { v:"40.4%",   l:"Average Success Rate", c:"#f59e0b" },
                        { v:"70.5%",   l:"Model Accuracy", c:"#10b981" },
                        { v:"0.774",   l:"ROC-AUC Score", c:"#a855f7" },
                        { v:"1.81×",   l:"Lift on Top 20%", c:"#f97316" },
                        { v:"123",     l:"Features Engineered", c:"#06b6d4" },
                      ].map((s, i) => (
                        <div key={i} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 14px", background:"var(--surface2)", borderRadius:8 }}>
                          <span style={{ fontSize:11, color:"var(--text3)", fontFamily:"var(--mono)" }}>{s.l}</span>
                          <span style={{ fontSize:16, fontWeight:800, color:s.c }}>{s.v}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-title">Success Rate by Category</div>
                    {[
                      { cat:"Theater",   pct:60 }, { cat:"Comics",    pct:55 },
                      { cat:"Dance",     pct:52 }, { cat:"Art",       pct:48 },
                      { cat:"Games",     pct:45 }, { cat:"Music",     pct:44 },
                      { cat:"Film",      pct:38 }, { cat:"Technology",pct:35 },
                      { cat:"Fashion",   pct:25 },
                    ].map(({ cat, pct }) => {
                      const active = form.main_category === cat || (cat === "Film" && form.main_category === "Film & Video");
                      return (
                        <div key={cat} style={{ display:"flex", alignItems:"center", gap:10, marginBottom:7 }}>
                          <div style={{ width:72, fontSize:10, color: active ? "white" : "var(--text3)", fontFamily:"var(--mono)", fontWeight: active ? 700 : 400 }}>{cat}</div>
                          <div style={{ flex:1, height:6, background:"var(--surface2)", borderRadius:3, overflow:"hidden" }}>
                            <div style={{ width:`${pct}%`, height:"100%", background: active ? "var(--accent)" : "var(--border2)", borderRadius:3, transition:"width 0.6s" }} />
                          </div>
                          <div style={{ fontSize:10, color: active ? "var(--accent)" : "var(--text3)", fontFamily:"var(--mono)", width:30, textAlign:"right", fontWeight: active ? 700 : 400 }}>{pct}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* ══════════════ RESULT TAB ══════════════ */}
            {tab === "result" && (
              !result ? (
                <div className="empty"><div className="empty-icon">◈</div><div className="empty-text">Run a prediction first to see results</div></div>
              ) : (
                <div className="g2" style={{ alignItems:"start" }}>
                  <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
                    <div className={`result-panel ${isSuccess ? "s" : "f"}`}>
                      <ProbGauge prob={result.success_probability} />
                      <div className={`result-pct ${isSuccess ? "s" : "f"}`}>{result.prediction}</div>
                      <div className="result-sub">vs {result.base_rate}% platform average · {result.vs_baseline > 0 ? "+" : ""}{result.vs_baseline}% difference</div>
                      <div className="risk-badge" style={{ background: riskColor[result.risk_level]+"18", color: riskColor[result.risk_level], border:`1px solid ${riskColor[result.risk_level]}33` }}>
                        {result.risk_level}
                      </div>
                    </div>

                    <div className="g2" style={{ gap:12 }}>
                      {[
                        { l:"Success Prob", v:`${result.success_probability}%`, c:"#10b981" },
                        { l:"vs Baseline",  v:`${result.vs_baseline > 0 ? "+" : ""}${result.vs_baseline}%`, c: result.vs_baseline > 0 ? "#10b981" : "#ef4444" },
                        { l:"Base Rate",    v:`${result.base_rate}%`, c:"#f59e0b" },
                        { l:"Risk Level",   v:result.risk_level.split(" ")[0], c: riskColor[result.risk_level] },
                      ].map((s, i) => (
                        <div key={i} className="stat" style={{ "--c": s.c }}>
                          <div className="stat-v">{s.v}</div>
                          <div className="stat-l">{s.l}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-title">Why did the model predict this?</div>
                    <div className="card-sub" style={{ marginBottom:16 }}>Top factors influencing this specific prediction</div>
                    <ShapChart data={result.shap_explanation} />
                    <div className="divider" />
                    <div style={{ fontSize:12, color:"var(--text3)", fontFamily:"var(--mono)" }}>
                      <span style={{ color:"var(--text2)", fontWeight:600 }}>Campaign:</span> {form.name || "Untitled"} &nbsp;·&nbsp;
                      <span style={{ color:"var(--text2)", fontWeight:600 }}>Goal:</span> ${Number(form.goal_usd).toLocaleString()} &nbsp;·&nbsp;
                      <span style={{ color:"var(--text2)", fontWeight:600 }}>Category:</span> {form.main_category}
                    </div>
                  </div>
                </div>
              )
            )}

            {/* ══════════════ AI ROADMAP TAB ══════════════ */}
            {tab === "roadmap" && (
              <div className="g2" style={{ alignItems:"start" }}>
                <div className="card" style={{ gridColumn:"1/-1" }}>
                  <div className="sec-head">
                    <div>
                      <div className="sec-title">AI Campaign Strategy Roadmap</div>
                      <div className="sec-sub">Personalised recommendations powered by Groq AI (Llama 3.1) based on your campaign inputs and ML prediction</div>
                    </div>
                    {result && (
                      <div style={{ textAlign:"right" }}>
                        <div style={{ fontSize:24, fontWeight:800, color: isSuccess ? "var(--success)" : "var(--fail)" }}>{result.success_probability}%</div>
                        <div style={{ fontSize:10, color:"var(--text3)", fontFamily:"var(--mono)" }}>current prediction</div>
                      </div>
                    )}
                  </div>
                  <AIRoadmap form={form} result={result} loading={aiLoading} roadmap={roadmap} />
                  {roadmap && !aiLoading && (
                    <div className="tip" style={{ marginTop:16 }}>
                      Roadmap generated for: {form.name || "Untitled"} · {form.main_category} · ${Number(form.goal_usd).toLocaleString()} goal · {form.duration_days} days
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ══════════════ SIMILAR CAMPAIGNS TAB ══════════════ */}
            {tab === "similar" && (
              <div>
                <div className="sec-head">
                  <div>
                    <div className="sec-title">Similar Campaigns — {form.main_category}</div>
                    <div className="sec-sub">Real Kickstarter campaigns from your category — use these as benchmarks</div>
                  </div>
                  {result && (
                    <div style={{ padding:"10px 18px", background:"var(--surface2)", borderRadius:10, border:"1px solid var(--border)", textAlign:"center" }}>
                      <div style={{ fontSize:11, color:"var(--text3)", fontFamily:"var(--mono)" }}>Your Prediction</div>
                      <div style={{ fontSize:22, fontWeight:800, color: isSuccess ? "var(--success)" : "var(--fail)" }}>{result.success_probability}%</div>
                    </div>
                  )}
                </div>
                <SimilarCampaigns category={form.main_category} goal={form.goal_usd} />

                <div className="divider" />
                <div className="card">
                  <div className="card-title">Category Benchmark</div>
                  <table className="tbl">
                    <thead>
                      <tr><th>Metric</th><th>Your Campaign</th><th>Category Avg</th><th>Top Campaign</th></tr>
                    </thead>
                    <tbody>
                      {(() => {
                        const similar = getSimilar(form.main_category);
                        const avgGoal = Math.round(similar.reduce((s,c) => s + c.goal, 0) / similar.length);
                        const avgRaised = Math.round(similar.reduce((s,c) => s + c.raised, 0) / similar.length);
                        const topCamp = similar.reduce((a,b) => a.raised > b.raised ? a : b);
                        return [
                          ["Funding Goal", `$${Number(form.goal_usd).toLocaleString()}`, `$${avgGoal.toLocaleString()}`, `$${topCamp.goal.toLocaleString()}`],
                          ["Amount Raised", "TBD", `$${avgRaised.toLocaleString()}`, `$${topCamp.raised.toLocaleString()}`],
                          ["Duration", `${form.duration_days} days`, "30 days", "30 days"],
                          ["Success Rate", result ? `${result.success_probability}%` : "—", "40.4%", "100%"],
                        ].map(([m, y, a, t]) => (
                          <tr key={m}>
                            <td style={{ fontWeight:600, color:"var(--text)" }}>{m}</td>
                            <td style={{ color:"var(--accent)", fontWeight:600 }}>{y}</td>
                            <td>{a}</td>
                            <td style={{ color:"var(--gold)" }}>{t}</td>
                          </tr>
                        ));
                      })()}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* ══════════════ DASHBOARD TAB ══════════════ */}
            {tab === "dashboard" && (
              <>
                <div className="sec-head">
                  <div>
                    <div className="sec-title">Model Performance Dashboard</div>
                    <div className="sec-sub">Stacking ensemble trained on 331,651 Kickstarter campaigns</div>
                  </div>
                </div>
                <div className="g4" style={{ marginBottom:20 }}>
                  {[
                    { v: stats?.accuracy + "%" || "70.5%", l:"Test Accuracy",    s:"66,331 held-out samples",   c:"#3b82f6" },
                    { v: stats?.roc_auc  || "0.7737",      l:"ROC-AUC Score",    s:"area under ROC curve",      c:"#10b981" },
                    { v: stats?.lift + "×" || "1.81×",     l:"Top-20% Lift",     s:"vs 40.4% baseline",         c:"#f59e0b" },
                    { v:"0.7748",                           l:"CV ROC-AUC",       s:"±0.0011 std across 5 folds",c:"#a855f7" },
                  ].map((s, i) => (
                    <div key={i} className="stat" style={{ "--c":s.c }}>
                      <div className="stat-v">{s.v}</div>
                      <div className="stat-l">{s.l}</div>
                      <div className="stat-s">{s.s}</div>
                    </div>
                  ))}
                </div>

                <div className="g2">
                  <div className="card">
                    <div className="card-title">Classification Report</div>
                    <table className="tbl">
                      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr></thead>
                      <tbody>
                        {[
                          { cls:"Failed",  p:0.74, r:0.78, f:0.76, s:"39,539", c:"#ef4444" },
                          { cls:"Success", p:0.65, r:0.59, f:0.62, s:"26,792", c:"#10b981" },
                        ].map(row => (
                          <tr key={row.cls}>
                            <td style={{ color:row.c, fontWeight:700 }}>{row.cls}</td>
                            <td>{row.p}</td><td>{row.r}</td>
                            <td style={{ fontWeight:700, color:row.c }}>{row.f}</td>
                            <td style={{ color:"var(--text3)" }}>{row.s}</td>
                          </tr>
                        ))}
                        <tr style={{ borderTop:"1px solid var(--border2)" }}>
                          <td style={{ color:"var(--text2)", fontWeight:600 }}>Weighted Avg</td>
                          <td style={{ fontWeight:600 }}>0.70</td>
                          <td style={{ fontWeight:600 }}>0.71</td>
                          <td style={{ fontWeight:600 }}>0.70</td>
                          <td style={{ color:"var(--text3)" }}>66,331</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="card">
                    <div className="card-title">Ensemble Architecture</div>
                    {[
                      { name:"XGBoost",             role:"Base · gradient boosting · depth=5 · n=500",   color:"#3b82f6" },
                      { name:"LightGBM",            role:"Base · leaf-wise boosting · leaves=31 · n=500", color:"#10b981" },
                      { name:"CatBoost",            role:"Base · ordered boosting · depth=5 · n=500",    color:"#f59e0b" },
                      { name:"Random Forest",       role:"Base · bagging · n=300 · depth=10",            color:"#f97316" },
                      { name:"Logistic Regression", role:"Meta-learner · stacking · C=1.0",              color:"#e2e8f0" },
                    ].map((m, i) => (
                      <div key={i} style={{ display:"flex", alignItems:"center", gap:12, padding:"10px 14px", background:"var(--surface2)", borderRadius:8, border:"1px solid var(--border)", marginBottom:8 }}>
                        <div style={{ width:8, height:8, borderRadius:"50%", background:m.color, flexShrink:0 }} />
                        <div>
                          <div style={{ fontSize:13, fontWeight:600 }}>{m.name}</div>
                          <div style={{ fontSize:10, color:"var(--text3)", fontFamily:"var(--mono)", marginTop:2 }}>{m.role}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="card">
                    <div className="card-title">Confusion Matrix</div>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginTop:8 }}>
                      {[
                        { label:"True Negative", sub:"Correctly predicted failure", v:"30,902", pct:"78%", c:"#ef4444" },
                        { label:"False Positive", sub:"Failed predicted as success",  v:"8,637",  pct:"22%", c:"#f59e0b" },
                        { label:"False Negative", sub:"Success predicted as failure", v:"11,034", pct:"41%", c:"#f59e0b" },
                        { label:"True Positive", sub:"Correctly predicted success",  v:"15,758", pct:"59%", c:"#10b981" },
                      ].map((cell, i) => (
                        <div key={i} style={{ padding:"14px", background:"var(--surface2)", borderRadius:8, border:`1px solid ${cell.c}22` }}>
                          <div style={{ fontSize:22, fontWeight:800, color:cell.c }}>{cell.v}</div>
                          <div style={{ fontSize:11, fontWeight:600, color:"var(--text2)", marginTop:4 }}>{cell.label}</div>
                          <div style={{ fontSize:10, color:"var(--text3)", fontFamily:"var(--mono)", marginTop:2 }}>{cell.sub}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-title">Data Sources</div>
                    {[
                      { src:"Kickstarter Projects CSV", rec:"378,661", role:"Core campaign features + labels", c:"#3b82f6" },
                      { src:"Reddit Posts",             rec:"35,084",  role:"Community sentiment signals",     c:"#f97316" },
                      { src:"YouTube Channels",         rec:"500+",    role:"Platform engagement metrics",     c:"#ef4444" },
                      { src:"World Bank GDP",           rec:"200 countries", role:"Macroeconomic context",    c:"#10b981" },
                    ].map((d, i) => (
                      <div key={i} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 0", borderBottom: i < 3 ? "1px solid var(--border)" : "none" }}>
                        <div>
                          <div style={{ fontSize:13, fontWeight:600, color:"var(--text)" }}>{d.src}</div>
                          <div style={{ fontSize:10, color:"var(--text3)", fontFamily:"var(--mono)", marginTop:3 }}>{d.role}</div>
                        </div>
                        <div style={{ fontSize:13, fontWeight:700, color:d.c, fontFamily:"var(--mono)" }}>{d.rec}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* ══════════════ FEATURE IMPORTANCE TAB ══════════════ */}
            {tab === "importance" && (
              <div>
                <div className="sec-head">
                  <div>
                    <div className="sec-title">What Drives Campaign Success?</div>
                    <div className="sec-sub">Top factors ranked by XGBoost feature importance — colour-coded by group</div>
                  </div>
                </div>
                <div className="g2" style={{ alignItems:"start" }}>
                  <div className="card">
                    <div className="card-title">Feature Importance Ranking</div>
                    <FIChart data={fiData} />
                  </div>
                  <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
                    <div className="card">
                      <div className="card-title">Key Insights</div>
                      {[
                        { icon:"◈", title:"Goal vs Category",    body:"Setting a goal below your category average is 6× more predictive than any other single factor. Research category medians before setting your ask.", c:"#6366f1" },
                        { icon:"◉", title:"Category Signal",     body:"Sub-category historical success rate is the #2 factor. Tabletop Games, Theater, and Comics have the highest baseline win rates.", c:"#a855f7" },
                        { icon:"◎", title:"Name Structure",      body:"Campaigns named 'X: Y' (with a colon subtitle) consistently outperform single-word titles. It signals professionalism and creativity.", c:"#f97316" },
                        { icon:"⊞", title:"Duration Sweet Spot", body:"15–35 day campaigns outperform longer ones. Urgency drives backer behavior — shorter windows convert better.", c:"#22c55e" },
                      ].map((item, i) => (
                        <div key={i} style={{ padding:"14px", background:"var(--surface2)", borderRadius:8, borderLeft:`3px solid ${item.c}`, marginBottom:10 }}>
                          <div style={{ fontSize:12, fontWeight:700, color:item.c, marginBottom:6, fontFamily:"var(--mono)" }}>{item.icon} {item.title}</div>
                          <div style={{ fontSize:12, color:"var(--text2)", lineHeight:1.65 }}>{item.body}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>
    </>
  );
}