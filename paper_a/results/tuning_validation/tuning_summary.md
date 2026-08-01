# MCMC 調律検証 — 集計

非収束率 nonconv_rate [Wilson 95% CI] (div~ = median divergences, reftemp のみ)

| 条件 | nt3_strong | nt2_strong | nt2_accurate |
|---|---|---|---|
| **A** centered base | 30% [23%–39%] | 71% [62%–78%] | 65% [56%–73%] |
| **B** reftemp | 12% [8%–20%], div~0 | 42% [34%–51%], div~0 | — |
| **C** centered t=.99 | 22% [15%–30%] | 68% [60%–76%] | — |
| **D** centered tune2k | 17% [11%–24%] | 55% [46%–64%] | — |
| **E** reftemp dense .99/2k | 2% [1%–7%], div~0 | 19% [13%–27%], div~0 | 13% [8%–21%], div~0 |

## 非収束内訳 (rhat / ess / both / error, 非収束分の構成比)

| 条件×セル | rhat | ess | both | error |
|---|---|---|---|---|
| A×nt3_strong | 69% | 0% | 31% | 0% |
| A×nt2_strong | 47% | 0% | 53% | 0% |
| A×nt2_accurate | 44% | 3% | 54% | 0% |
| B×nt3_strong | 40% | 0% | 60% | 0% |
| B×nt2_strong | 35% | 0% | 65% | 0% |
| C×nt3_strong | 85% | 0% | 15% | 0% |
| C×nt2_strong | 48% | 0% | 52% | 0% |
| D×nt3_strong | 80% | 0% | 20% | 0% |
| D×nt2_strong | 53% | 0% | 47% | 0% |
| E×nt3_strong | 33% | 0% | 67% | 0% |
| E×nt2_strong | 17% | 0% | 83% | 0% |
| E×nt2_accurate | 19% | 0% | 81% | 0% |
