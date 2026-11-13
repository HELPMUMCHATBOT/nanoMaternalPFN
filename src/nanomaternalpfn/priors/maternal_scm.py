"""Maternal synthetic task generator. Assumptions are not clinically validated."""
import numpy as np

FAMILIES = ('hypertension', 'metabolic', 'history', 'messy')

def sample_task(n_rows=128, n_features=10, n_classes=3, prior='mixture', seed=None, rng=None, maternal_probability=0.7, augment_class_permutation=False, augment_feature_sign_flip=False, robust=False):
    if n_rows < 8 or not 2 <= n_features <= 32 or n_classes not in (2, 3):
        raise ValueError('unsupported task shape')
    rng = rng if rng is not None else np.random.default_rng(seed)
    if prior not in ('maternal', 'generic', 'mixture'):
        raise ValueError('prior must be maternal, generic, or mixture')
    maternal = prior == 'maternal' or (prior == 'mixture' and rng.random() < maternal_probability)
    if not maternal:
        latent = rng.normal(size=(n_rows, 6))
        x = np.tanh(latent @ rng.normal(size=(6, n_features))) + rng.normal(0, .25, size=(n_rows, n_features))
        risk = x @ rng.normal(size=n_features) + rng.normal(0, .5, size=n_rows)
        x[rng.random(x.shape) < rng.uniform(.01, .1)] = np.nan
        family = 'generic'
    else:
        family = str(rng.choice(FAMILIES))
        age = np.clip(14 + rng.gamma(3.8, 3.5, n_rows), 14, 49)
        parity = np.zeros(n_rows) if rng.random() < .20 else rng.poisson(np.maximum(.1, (age - 16) * .22))
        gest = rng.uniform(8, 40, n_rows)
        bmi = np.clip(15 + rng.lognormal(2.2, .4, n_rows), 15, 52)
        sbp = np.clip(110 + .35 * (age - 25) + .35 * (bmi - 23) + rng.normal(0, 10, n_rows) + (rng.random(n_rows) < .2) * rng.exponential(22, n_rows), 80, 210)
        dbp = np.clip(.62 * sbp + rng.normal(3, 6, n_rows), 48, 130)
        glucose = np.clip(3.9 + np.exp(.035 * (bmi - 22) + rng.normal(0, .25, n_rows)) + (rng.random(n_rows) < .15) * rng.exponential(3, n_rows), 3, 22)
        hb = np.clip(11.4 - .16 * parity + rng.normal(0, 1.4, n_rows), 4.5, 16.5)
        hr = np.clip(78 + .12 * (sbp - 115) - 1.8 * (hb - 11) + rng.normal(0, 8, n_rows), 55, 145)
        temp = np.clip(36.8 + rng.normal(0, .45, n_rows), 35.5, 41)
        base = np.column_stack([age, parity, sbp, dbp, glucose, bmi, hb, hr, temp, gest])
        if n_features <= base.shape[1]:
            x = base[:, :n_features].copy()
        else:
            extra = np.tanh((base / np.array([30,3,120,80,5,25,11,80,37,25])) @ rng.normal(size=(10, n_features-10)))
            x = np.column_stack([base, extra])
        age_curve = .05 * np.maximum(0, 19-age)**1.5 + .07 * np.maximum(0, age-34)**1.4
        bp_curve = .07 * np.maximum(0, sbp-135)**1.35 + .08 * np.maximum(0, dbp-88)**1.3
        history = .5 * (parity == 0) + .23 * np.maximum(0, parity-4)**1.3 + .30 * np.maximum(0, 10-hb)**1.4
        metab = .3 * np.maximum(0, glucose-5.5) * np.maximum(0, bmi-25)/5
        coeffs = {'hypertension':(1.8,.8,.5,1), 'metabolic':(.7,2,.5,.7), 'history':(.5,.5,2,1), 'messy':(1,1,1,1)}[family]
        risk = coeffs[0]*bp_curve + coeffs[1]*metab + coeffs[2]*history + coeffs[3]*age_curve + rng.normal(0,.8,n_rows)
        if robust:
            # Hypothetical hospital shift and nonlinear interactions.
            site = rng.integers(0, 3, n_rows)
            risk += rng.normal(0, .5, 3)[site]
            risk += rng.uniform(-.5, .5) * np.tanh((sbp-135)/20) * np.tanh((glucose-5.5)/3)
            if n_features >= 4:
                x[:, 2] += rng.normal(0, 3, 3)[site]
                x[:, 3] += rng.normal(0, 2, 3)[site]
        if n_features >= 4 and rng.random() < .65:
            heaped = rng.random(n_rows) < .6
            x[heaped, 2:4] = np.round(x[heaped, 2:4]/5)*5
        x[rng.random(x.shape) < rng.uniform(.01,.08)] = np.nan
        if n_features >= 6:
            mar = (age < 30) & (bmi < 25) & (rng.random(n_rows) < .35)
            x[mar, 4] = np.nan
            threshold = np.quantile(risk, .80 if rng.random() < .5 else .20)
            mnar = ((risk > threshold) if rng.random() < .5 else (risk < threshold)) & (rng.random(n_rows) < .3)
            x[mnar, 5] = np.nan
        if robust:
            # MAR-style missingness and heavy-tailed simulated measurement errors.
            p_missing = np.clip(.02 + .35 / (1 + np.exp(-np.clip((sbp-135)/15, -10, 10))), .02, .48)
            missing_col = int(rng.integers(n_features))
            x[rng.random(n_rows) < p_missing, missing_col] = np.nan
            corrupted = rng.random(x.shape) < rng.uniform(.005, .03)
            x[corrupted] += rng.standard_t(3, x.shape)[corrupted] * 2
        x = x[:, rng.permutation(n_features)]
    if n_classes == 2:
        prevalence = rng.uniform(.015, .48) if robust else rng.uniform(.1, .48)
        y = (risk >= np.quantile(risk, 1-prevalence)).astype(np.int64)
    else:
        y = np.digitize(risk, np.quantile(risk, [.48, .82])).astype(np.int64)
    if robust:
        noise_mask = rng.random(n_rows) < rng.uniform(0, .12)
        count = int(noise_mask.sum())
        if count:
            y[noise_mask] = (y[noise_mask] + rng.integers(1, n_classes, size=count)) % n_classes
    # v2 class augmentation changes the numerical labels, not the risk function.
    # Both generic and maternal tasks use the same policy. Single-pass
    # class-equivariance is encouraged, not mathematically guaranteed.
    flipped = False
    if augment_feature_sign_flip and rng.random() < 0.5:
        x = x * rng.choice([-1., 1.], size=n_features)[None, :]
        flipped = True
    label_mapping = rng.permutation(n_classes) if augment_class_permutation else np.arange(n_classes)
    y = label_mapping[y]
    meta = {'source': 'synthetic', 'family': family, 'prior': prior, 'robust': bool(robust)}
    if augment_class_permutation or augment_feature_sign_flip:
        meta.update({'class_mapping_original_to_augmented': label_mapping.tolist(),
                     'sign_flip_task_selected': flipped})
    return x.astype(np.float32), y, meta
