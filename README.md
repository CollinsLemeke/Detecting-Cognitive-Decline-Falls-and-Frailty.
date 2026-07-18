# Psychointelligence: Dementia and Falls in Older Adults

**Detecting signs of cognitive decline from how a person walks, using interpretable machine learning on the GSTRIDE dataset.**

## The short version

Dementia is usually caught late, once memory symptoms are obvious enough that someone books a clinical appointment. We wanted to know whether something cheaper and more passive could flag it earlier.

Walking is a reasonable candidate. It feels automatic, but staying steady requires attention, planning and balance control, all of which are brain work. So we asked: if you record how an older adult walks and nothing else, can you tell whether they are showing signs of cognitive decline?

Using six simple measurements from 163 older adults, a plain logistic regression flags roughly **72% of people with signs of cognitive decline** (ROC-AUC 0.752). More importantly, we show this is **not just detecting frail people** — the cognitive signal is measurably separate from the physical one.

**→ [Read the full analysis with all charts](notebooks/gstride_psychointelligence.ipynb)** (renders directly on GitHub, no setup needed)

## Reproducing this

The dataset is included, so this should work in one go.

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
python src/reproduce.py
```

That prints every headline number below in about two minutes. Everything is seeded with `RANDOM_STATE = 42`, so your output should match ours exactly. If it doesn't, that's a bug and we'd like to hear about it.

## What's in here

```
notebooks/gstride_psychointelligence.ipynb   Full analysis, 14 steps, charts included
src/reproduce.py                             Headline numbers without Jupyter
data/Database_register.xlsx                  GSTRIDE register, 163 participants
```

The notebook is committed with its outputs intact, so you can read the whole thing on GitHub without running anything. It's written for someone who has never trained a model, and every step explains why it's there, not just what it does.

## The data

[GSTRIDE](https://doi.org/10.5281/zenodo.8003441) is a public dataset of 163 community-dwelling older adults in Madrid, aged 70 to 98 (mean 82.6), 72% women. Each participant wore an inertial sensor on the foot during an extended walking test and completed a battery of clinical assessments.

What makes it unusual is the cognitive variable. Most public gait datasets either use simulated falls performed by young volunteers or contain no cognitive information at all. GSTRIDE has real older adults and a documented dementia stage.

| Outcome | Positive | Rate |
|---|---|---|
| Signs of cognitive decline (GDS ≥ 3) | 43 / 163 | 26% |
| Fell in the past year | 86 / 163 | 53% |
| Frail (Fried ≥ 2) | 49 / 163 | 30% |

Two things worth knowing before you use this data:

**The GDS here is the Global Deterioration Scale**, a dementia-staging instrument, not the Geriatric Depression Scale. These get confused constantly, and it changes what the variable means entirely.

**Fall history is retrospective.** It records falls in the year *before* testing, not a prospective follow-up. So any model built on it performs risk stratification, not forecasting.

### Traps in this dataset

**A trailing space.** Eight rows record the fall outcome as `"NO "`. Python treats that as a different value from `"NO"`, so without stripping whitespace those eight people fail to map to either class and quietly disappear. You'd analyse 155 people while believing you had 163, and nothing would warn you.

**Three header rows.** The file has a category row, a variable-name row and a description row stacked on top, so a naive `read_excel` returns nonsense. The real names are on row 1, data starts at row 3.

**Informative missingness.** Eighteen participants have `"Incapable"` instead of a fear-of-falling score. Mean GDS among those who completed it is 1.70; among those who couldn't it's 4.83. The non-responders are the severely impaired. Don't drop those rows and don't mean-impute them.

## How we built it

### Choosing features, and the mistake we made first

Our first version used ten walking measurements. It scored fine. The explanations were wrong.

The problem is that several gait metrics are the same information wearing different hats. Walking speed is approximately stride length times cadence, so feeding a model all three is like giving it someone's height in centimetres, inches and feet and asking which matters most. It can't tell them apart, so it splits the credit arbitrarily, and it will happily assign a negative weight to one variable to cancel out over-crediting another.

Variance inflation factor made the scale of the problem obvious:

| Feature | VIF before |
|---|---|
| `step_speed` | 88.2 |
| `stride_len` | 61.0 |
| `clearance_std` | 22.7 |
| `clearance` | 22.7 |
| `cadence` | 17.9 |

Anything above 10 is usually called severe. At 88, a coefficient is not a finding, it's noise with a decimal point.

Three of the "risk factors" in that first model had signs pointing the *opposite* way to their actual relationship with cognitive decline when measured on their own. The accuracy was unaffected. The story was false.

So we kept one representative per concept: **age, BMI, TUG time, walking speed, foot clearance, and stride-time variability**. All six now sit under VIF 3. Cross-validated AUC moved from 0.756 to 0.752, which is nothing, and every coefficient became readable.

**The lesson, stated plainly: collinearity does not damage prediction, it damages explanation.** If you are building an interpretable model, accuracy will not tell you when your interpretation has broken.

### Testing honestly

With 163 people, a single train/test split is a coin toss. We use 5-fold cross-validation repeated 10 times, so every participant is held out and tested on, and the reported figure is an average of 50 separate fits. Imputation and scaling sit inside the pipeline, fitted on training folds only.

We also run all three outcomes through the same pipeline with the same six features. That's what makes the separability test possible.

## Results

### Which algorithm wins depends on the task

| Task | Best model | ROC-AUC | Recall |
|---|---|---|---|
| **Dementia** | Logistic Regression | **0.752 ± 0.074** | 0.657 |
| Falls | Random Forest | 0.814 ± 0.079 | 0.749 |
| Frailty | Logistic Regression | 0.749 ± 0.087 | 0.706 |

The pattern isn't random. Logistic regression wins on the two rare outcomes; random forest wins on the balanced one. Tree ensembles partition data into progressively smaller groups, and with only 43 positive cases they run out of examples and turn conservative. You can see it in recall: on dementia, logistic regression catches 66% of cases while gradient boosting catches 32%.

The differences in AUC sit inside the error bars, so "best" here means a tendency, not a decisive win. The useful conclusion is that **the interpretable model is competitive-to-best on the task we care about**, so transparency costs nothing.

### Screening performance

Out of fold, at a 0.5 threshold: **31 of 43 cases caught, 12 missed, 35 false alarms.**

For a screening tool that trade is defensible. A false alarm means someone gets a cognitive assessment they didn't need. A missed case means nobody looks at all.

### What the model learned

Two variables reach statistical significance:

| Feature | Odds ratio (per 1 SD) | p |
|---|---|---|
| **TUG time** (slower) | 2.34 | 0.004 |
| **BMI** (lower) | 0.52 | 0.003 |

Slower walking, shorter strides, lower foot clearance and older age all point the same direction but overlap with TUG, which already captures walking capacity, so they support the picture rather than adding independent evidence.

Taken together this is a hesitant, shuffling, unsteady walk. It's a pattern a geriatrician would recognise, which is a reassuring sign that the model latched onto something real rather than an artefact.

One caveat we'd rather state than have someone find: **lower BMI may be a consequence of cognitive decline rather than a warning sign of it.** Weight loss is common in advancing dementia. It's still a useful flag, but the causal direction is not ours to claim.

### Is it actually the brain, or just frailty?

This is the question that decides whether any of the above means anything. Frail people fall, and frail people often have cognitive decline, so a model could score well on dementia while knowing nothing about cognition.

We tested it by comparing the per-person risk scores from all three tasks:

| Comparison | Correlation |
|---|---|
| Falls vs Frailty | **r = 0.93** |
| Dementia vs Frailty | r = 0.66 |
| Dementia vs Falls | r = 0.53 |

Falls and frailty are effectively the same signal measured twice. If cognitive decline were simply frailty in disguise, it would track at 0.93 as well. It tracks at 0.53 to 0.66.

**There are two distinguishable signals in a person's walk: a physical one and a cognitive one.** They overlap, but they are not the same thing.

## Where this connects to the wider framework

This work sits inside a broader idea we call psychointelligence: that health risk in older adults is cognitive and psychological as well as physical, and that tools which measure only the body are missing part of the picture.

Worth being honest about how the project got here. We started out assuming that adding cognitive information to a fall-risk model would improve it. We tested that and it failed — once the model knew a person's gait and frailty, dementia stage contributed nothing to fall prediction (OR 1.02, p = 0.87).

That null result is what pointed to the better question. Cognition added nothing *on top of* walking because the walking already contained it. Cognitive decline doesn't sit alongside gait as a separate risk factor; it expresses itself through gait. So we inverted the problem and used gait to detect the decline directly.

Which layers of the framework this study actually reached:

| Layer | Status |
|---|---|
| Physical mobility | Measured |
| Cognitive | Measured |
| Psycho-affective | Set aside (see below) |
| Behavioural routine | Not in this dataset |
| Environmental | Not in this dataset |
| Adaptive decision support | Partial — we output a risk score and a readable reason |

On the psycho-affective layer: GSTRIDE includes a fear-of-falling questionnaire (FES-I), and we intended to use it. Eighteen participants are marked "Incapable" rather than given a score, and that missingness is not random — mean GDS is 1.70 among those who completed it and 4.83 among those who couldn't.

That's worth stating as a finding in its own right: **self-reported psychological measures degrade exactly where cognitive impairment is most severe**, which is a real design constraint for anyone building assessment tools for this population.

Two of six layers, one honestly set aside, two out of scope. A first step, not a finished framework.

## Limitations

We'd rather list these ourselves than have them found.

**This is a screening signal, not a diagnosis.** The model flags who might benefit from a proper cognitive assessment. Only a clinician can diagnose dementia.

**The sample is small.** 163 people, 43 of them positive. Results are promising but need replication on a larger and more varied cohort.

**It's cross-sectional.** Gait and cognition were measured at the same visit, so we can say gait is associated with current cognitive status. We cannot say gait today predicts decline in five years, which is the clinically interesting question.

**Association, not causation.** Unsteady walking goes with cognitive decline. Neither we nor this data can say which drives which.

**The cohort skews.** 72% women, most aged 80+, all recruited in one region of Spain. Generalise carefully.

**Model selection and evaluation share a cross-validation loop.** Choosing the best algorithm using the same procedure that reports its score can inflate that score slightly. Nested cross-validation would close this, and it's the first thing we'd add.

## Citation

If this repository is useful to you, please cite the original GSTRIDE authors — they did the hard work of collecting the data.

> García-de-Villa, S., García-Villamil Neira, G., Neira Álvarez, M., Huertas-Hoyas, E., Ruiz Ruiz, L., del-Ama, A. J., Rodríguez Sánchez, M. C., & Jiménez Ruiz, A. (2023). A database with frailty, functional and inertial gait metrics for the research of fall causes in older adults. *Scientific Data*, 10, 566. https://doi.org/10.1038/s41597-023-02428-0

## Licence

Code is MIT. The GSTRIDE data redistributed in `data/` is **CC BY 4.0**, which permits reuse and redistribution provided the original authors are credited. That attribution is not optional.

## Contributing

Corrections are genuinely welcome, particularly on the statistics. If you spot a methodological problem, open an issue. We already found one significant mistake in our own work (the collinearity), and we'd rather find the rest.
