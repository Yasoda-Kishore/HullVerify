# Step 7 report - tests, sensitivity, limits

## 1. AI rules vs naive baseline (held-out TEST split only, n=24)
Thresholds were tuned on the tune split only. Synthetic images - demonstrates the
pipeline, NOT real-world accuracy.

|                        | accuracy               | wrongly-'sufficient'   | meaning                |
| ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| our rules              | 96%                    | 0/12                   | 1 error, over-cautious direction |
| naive baseline         | 58%                    | 0/12                   | passes all murky images as 'clean' - the dangerous failure |
| logistic regression    | 100%                   | 0/12                   | perfect on THIS small synthetic set - do not oversell |

**Wrongly-'sufficient' rate (dirty area passed as clean): 0/12 on test.**
Report it even when it is bad; here it is 0 on a small synthetic set, which
may NOT hold on real imagery - say that out loud.

## 2. Sensitivity: how reliable must cleaning be before C stops paying?
Costs below use the default slider inputs {'clean_min': 3.0, 'check_min': 0.5, 'rephoto_min': 1.0, 'over_min': 1.0, 'rate_eur_h': 150.0, 'expert_eur': 40.0, 'penalty_eur': 60.0} -
all assumptions. Counts from 300 simulated hulls per setting.

| clean-prob x | cost A       | cost B       | cost C       | dirty A      | dirty B      | dirty C      | cheapest     |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| x0.6         | EUR 4,929    | EUR 4,968    | EUR 3,964    | 32.1         | 57.8         | 20.4         | C            |
| x0.8         | EUR 3,998    | EUR 3,940    | EUR 2,980    | 16.6         | 40.7         | 8.4          | C            |
| x1.0         | EUR 3,444    | EUR 2,996    | EUR 2,347    | 7.4          | 24.9         | 1.7          | C            |
| x1.1         | EUR 3,268    | EUR 2,533    | EUR 2,158    | 4.5          | 17.2         | 0.5          | C            |
| x1.2         | EUR 3,183    | EUR 2,163    | EUR 2,033    | 3.0          | 11.1         | 0.1          | C            |
| x1.3         | EUR 3,112    | EUR 2,011    | EUR 1,988    | 1.9          | 8.5          | 0.1          | C            |

## 3. Sensitivity: the penalty for a dirty area left behind
At penalty = 0, leaving fouling behind is 'free', so the cheapest strategy is
doing less. C's case rests on dirty areas having a real cost (fuel, re-dock, audit).

| penalty EUR  | cost A       | cost B       | cost C       | cheapest     |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| 0            | EUR 3,000    | EUR 1,500    | EUR 2,245    | B            |
| 20           | EUR 3,152    | EUR 2,000    | EUR 2,278    | B            |
| 40           | EUR 3,303    | EUR 2,500    | EUR 2,311    | C            |
| 60           | EUR 3,455    | EUR 2,999    | EUR 2,344    | C            |
| 120          | EUR 3,910    | EUR 4,499    | EUR 2,443    | C            |
| 200          | EUR 4,517    | EUR 6,498    | EUR 2,575    | C            |

## 4. Sensitivity: how slow may verification be?
| verify min/area | cost C       | cheapest     |
| -------------- | -------------- | -------------- |
| 0.5          | EUR 2,344    | C            |
| 1.0          | EUR 2,594    | C            |
| 2.0          | EUR 3,094    | B            |
| 3.0          | EUR 3,594    | B            |
| 4.0          | EUR 4,094    | B            |

## 5. Where C stops winning (say this in the pitch)
C is NOT the cheapest at: penalty EUR 0; penalty EUR 20; verify 2.0 min/area; verify 3.0 min/area; verify 4.0 min/area.
The pattern to expect: C loses when cleaning is already very reliable (little
to find), when leaving dirt behind costs nothing, or when verification is slow.
Even where C is not cheapest, it is the only strategy that KNOWS which areas
are still dirty - the others leave them silently.

## 6. Limits slide (paste as-is)
- Images are computer-generated stand-ins, not real hull photos.
- The robot and the hull are simulated; zone fouling/cleaning probabilities are assumptions.
- Test set is small (24 pairs); thresholds fitted to this synthetic distribution.
- Cost figures come from editable assumed inputs, not from operations data.
- AI error rates feed the simulation from a measured but synthetic test; real-water
  performance is unknown until tried on real imagery.
- IMO GreenVoyage2050 context ($5k-$50k per full clean) is background, not our result.