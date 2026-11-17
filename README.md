# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dorkcheng/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dorkcheng/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A program that forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour. It reads one case file (the target's hour-1 print plus the comparables' past launch-day curves, all from the case itself) and writes five quantiles for each of hours 2–24. The model is a kernel-weighted log-ratio regression: each past launch is expressed as ratios to its own hour-1 count, and for every hour a line is fitted through log ratio against log hour-1, weighted by how close each comparable's hour-1 is to the target's. Launches that peaked in hours 1–4 (a PC simultaneous unlock that decayed from the starting gun) are left out, because GTA VI is console-only with a rolling midnight unlock and its shape is a rise, not a decay. The spread comes from the weighted residual standard deviation around the line, so hours where past launches disagreed get wide bands. If the fit ever has too few comparables it falls back to the house persistence baseline, so the program always writes a valid forecast. Deterministic, offline, no network.

Validation: leave-one-out over the 37 comparables with full release-day curves in the kit's practice cases, scored with the contest's exact pinball loss on the log scale — house persistence baseline 0.930, this model 0.382 (90% coverage 0.90, i.e. the spread is calibrated). On the big targets only (hour 1 ≥ 20,000 viewers, the GTA VI scale): baseline 0.321, this model 0.265. On the reconstructed Wolverine practice answer the model scores 0.205 against the baseline's 0.744.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=z7ynzx9d&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
