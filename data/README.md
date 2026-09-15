# Data

**No recordings are distributed with this repository.** Both corpora belong to
their original contributors and are hosted by PhysioNet; download them from
there. This directory is otherwise empty and is excluded from version control.

## What you need

| Corpus | PhysioNet slug | Target directory | Used for |
|:---|:---|:---|:---|
| MIT-BIH Arrhythmia Database | `mitdb` | `data/mit-bih-arrhythmia/` | utility, within-recording re-identification |
| ECG-ID Database | `ecgiddb` | `data/ecg-id-database/` | across-session re-identification, attribute inference |

## Getting them

The loaders fetch what they need on first use, so in most cases you do not have
to do anything: `src/data/mitbih.py` falls back to PhysioNet through `wfdb`
when a record is missing locally, and `src/data/ecgid.py` calls
`wfdb.dl_database` for the whole ECG-ID set.

To fetch them ahead of time:

```python
import wfdb
wfdb.dl_database("mitdb",         "data/mit-bih-arrhythmia")
wfdb.dl_database("ecgiddb/1.0.0", "data/ecg-id-database", keep_subdirs=True)
```

Or with the PhysioNet command-line tools:

```bash
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
wget -r -N -c -np https://physionet.org/files/ecgiddb/1.0.0/
```

Expect roughly 100 MB for MIT-BIH and 15 MB for ECG-ID.

## Which records are used

MIT-BIH follows the standard inter-patient split of de Chazal et al. (2004),
which the analysis code defines in `configs/config.py`:

- **DS1** (training, 22 records): 101 106 108 109 112 114 115 116 118 119 122
  124 201 203 205 207 208 209 215 220 223 230
- **DS2** (test, 22 records): 100 103 105 111 113 117 121 123 200 202 210 212
  213 214 219 221 222 228 231 232 233 234

From ECG-ID the code uses every subject with at least two sessions, which is 89
of the 90, split chronologically into an earlier half for the attacker to train
on and a later half to be evaluated against.

## Citing the data

Both corpora require citation of PhysioNet itself alongside the corpus:

- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
  Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- Lugovaya, T. S. (2005). *Biometric human identification based on
  electrocardiogram* [Master's thesis]. Electrotechnical University "LETI",
  Saint Petersburg.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation*, 101(23), e215–e220.

Their licence terms are those stated on PhysioNet and are not affected by the
licence of this repository, which covers only the code.
