# MCMC 調律検証 — 集計

非収束率 nonconv_rate [Wilson 95% CI] (div~ = median divergences, reftemp のみ)

| 条件 | nt3_strong | nt2_strong |
|---|---|---|
| **A** centered base | 35% [18%–57%] | 85% [64%–95%] |
| **B** reftemp | 10% [3%–30%], div~0 | 65% [43%–82%], div~0 |

## 非収束内訳 (rhat / ess / both / error, 非収束分の構成比)

| 条件×セル | rhat | ess | both | error |
|---|---|---|---|---|
| A×nt3_strong | 86% | 0% | 14% | 0% |
| A×nt2_strong | 53% | 0% | 47% | 0% |
| B×nt3_strong | 50% | 0% | 50% | 0% |
| B×nt2_strong | 23% | 0% | 77% | 0% |
