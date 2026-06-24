"""
benchmark.py
Full benchmark 4 thuật toán + đối chiếu với solieu_*.md.

Methodology khớp với evaluate_*.py:
  LR  — train_time = model.fit() only (sau TF-IDF)
  NB  — train_time = TF-IDF + model.fit()
  KNN — train_time = tune_hyperparams() + model.fit() (sau TF-IDF); dùng tune thực sự
  NC  — train_time = TF-IDF + model.fit()
  Inference time (tất cả) = transform() + predict()

Legend so sánh:  ✓ khớp   ✗ lệch   ~ thời gian (luôn biến động)   N/A không có reference
"""

import sys, os, re, time, pickle
import numpy as np
from pathlib import Path
from io import StringIO
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ──────────────────────────────────────────────────────────
# 1. ENCODING + SYS.PATH
# ──────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR     = Path(__file__).resolve().parent
UTILS_DIR    = ROOT_DIR / "src" / "utils"
LOGISTIC_DIR = ROOT_DIR / "src" / "logistic-regression"
NAIVE_DIR    = ROOT_DIR / "src" / "naive-bayes"
KNN_DIR      = ROOT_DIR / "src" / "k-nearest-neighbor"
CENTROID_DIR = ROOT_DIR / "src" / "nearest-centroid"
MODEL_DIR    = ROOT_DIR / "src" / "model"

for _d in (UTILS_DIR, LOGISTIC_DIR, NAIVE_DIR, KNN_DIR, CENTROID_DIR):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

from tfidfcal    import TfIdfVectorizer                    # type: ignore  # noqa: F401
from labelEncode import LabelEncoder                       # type: ignore  # noqa: F401
from dataio      import load_split, INPUT_DIR              # type: ignore
from mrl         import MultinomialLogisticRegression      # type: ignore  # noqa: F401
from mnb         import MultinomialNaiveBayes              # type: ignore  # noqa: F401
from mknn        import KNearestNeighbors, tune_hyperparams  # type: ignore  # noqa: F401
from mnc         import NearestCentroid                    # type: ignore  # noqa: F401

import mknn as _mk, mnc as _mn  # type: ignore
sys.modules.setdefault('mknn', _mk)
sys.modules.setdefault('mnc',  _mn)


# ──────────────────────────────────────────────────────────
# 2. PARSE REFERENCE FILES (solieu_*.md)
# ──────────────────────────────────────────────────────────
def parse_solieu(path: Path):
    """Đọc solieu_*.md → dict. Trả None nếu file không tồn tại."""
    if not path.exists():
        return None
    txt = path.read_text(encoding='utf-8')

    def _f(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    top5 = re.findall(r'\d+\.\s*(\S+)\s*→\s*(\S+)\s*\(\d+ lần\)', txt)

    return {
        'train_acc':  _f(r'Train Accuracy\s*:\s*([\d.]+)%'),
        'val_acc':    _f(r'Val Accuracy\s*:\s*([\d.]+)%'),
        'test_acc':   _f(r'Test Accuracy\s*:\s*([\d.]+)%'),
        'overfit':    _f(r'Overfit gap\s*:\s*([\d.]+)%'),
        'f1':         _f(r'Macro F1\s*:\s*([\d.]+)'),
        'prec':       _f(r'Macro Precision\s*:\s*([\d.]+)'),
        'rec':        _f(r'Macro Recall\s*:\s*([\d.]+)'),
        'train_time': _f(r'Training Time\s*:\s*([\d.]+)'),
        'infer_ms':   _f(r'Inference Time\s*:\s*([\d.]+)'),
        'size':       _f(r'Model Size\s*:\s*([\d.]+)'),
        'top5':       [(a, b) for a, b in top5],
    }

REF = {
    'lr':  parse_solieu(ROOT_DIR / "src" / "logistic-regression" / "solieu_LR.md"),
    'nb':  parse_solieu(ROOT_DIR / "src" / "naive-bayes"         / "solieu_NB.md"),
    'knn': parse_solieu(ROOT_DIR / "src" / "k-nearest-neighbor"  / "solieu_KNN.md"),
    'nc':  parse_solieu(ROOT_DIR / "src" / "nearest-centroid"    / "solieu_NC.md"),
}


# ──────────────────────────────────────────────────────────
# 3. TIỆN ÍCH
# ──────────────────────────────────────────────────────────
class _Quiet:
    def __enter__(self): self._o = sys.stdout; sys.stdout = StringIO()
    def __exit__(self, *_): sys.stdout = self._o

def _filter(corpus, labels, enc):
    fc, yt = [], []
    for t, l in zip(corpus, labels):
        if l in enc.label_to_index:
            fc.append(t); yt.append(enc.label_to_index[l])
    return fc, np.array(yt)

def _top5(y_true, y_pred, enc):
    pairs = [(enc.index_to_label[t], enc.index_to_label[p])
             for t, p in zip(y_true, y_pred) if t != p]
    return Counter(pairs).most_common(5)

def _fmt_time(s):
    return f"{s:.4f} s" if s < 60 else f"{s/60:.2f} m  ({s:.1f} s)"

def _fmt_size(path):
    return f"{os.path.getsize(path)/1_048_576:.4f} MB"

def _cmp(ref_val, cur_val, tol=0.005):
    """So sánh hai số. Trả ✓/✗."""
    if ref_val is None or cur_val is None:
        return "?"
    return "✓" if abs(ref_val - cur_val) <= tol else f"✗"

def _cmp_top5(ref_pairs, cur_pairs):
    """So sánh top-5 pairs (chỉ label, không đếm số lần).
    ref_pairs: [(a,b), ...]  cur_pairs: [(a,b), ...]"""
    if not ref_pairs:
        return "?"
    return "✓" if ref_pairs == cur_pairs else "✗"


# ──────────────────────────────────────────────────────────
# 4. ĐỌC DỮ LIỆU
# ──────────────────────────────────────────────────────────
print("=" * 64)
print("  FULL BENCHMARK — 4 THUẬT TOÁN  +  ĐỐI CHIẾU SOURCE")
print("=" * 64)
print("\n[1/3] Đọc dữ liệu...")
train_c, train_l = load_split(INPUT_DIR/"train_corpus.txt", INPUT_DIR/"train_labels.txt")
val_c,   val_l   = load_split(INPUT_DIR/"val_corpus.txt",   INPUT_DIR/"val_labels.txt")
test_c,  test_l  = load_split(INPUT_DIR/"test_corpus.txt",  INPUT_DIR/"test_labels.txt")
print(f"  Train {len(train_c):,}  |  Val {len(val_c):,}  |  Test {len(test_c):,}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
results = {}   # key → dict of metrics

print("\n[2/3] Huấn luyện + đánh giá...\n")


# ══════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════
print("  ▶ Logistic Regression")
print("    TF-IDF...", end="", flush=True)
with _Quiet():
    vec = TfIdfVectorizer(); X_tr = vec.fit_transform(train_c)
    enc = LabelEncoder();    y_tr = enc.fit_transform(train_l)
print(" ok")
print("    Training...", end="", flush=True)
t0 = time.time()
with _Quiet():
    m = MultinomialLogisticRegression(learning_rate=0.5, epochs=500)
    m.fit(X_tr, y_tr)
tt = time.time() - t0
print(f" {_fmt_time(tt)}")

pkl = MODEL_DIR / "LogisticRegression_model.pkl"
with open(pkl,"wb") as f: pickle.dump({"vectorizer":vec,"label_encoder":enc,"model":m},f)

y_tr_p = m.predict(X_tr)
tr_acc = accuracy_score(y_tr, y_tr_p)*100

val_fc, y_val = _filter(val_c, val_l, enc)
with _Quiet(): X_val = vec.transform(val_fc)
va_acc = accuracy_score(y_val, m.predict(X_val))*100

test_fc, y_test = _filter(test_c, test_l, enc)
t_inf = time.time()
with _Quiet(): X_test = vec.transform(test_fc)
y_test_p = m.predict(X_test)
inf_ms = (time.time()-t_inf)/len(test_fc)*1000
te_acc = accuracy_score(y_test, y_test_p)*100

results['lr'] = {
    'name':'Logistic Regression',
    'train_acc':tr_acc, 'val_acc':va_acc, 'test_acc':te_acc,
    'overfit':tr_acc-va_acc,
    'f1':   f1_score(y_test, y_test_p, average='macro', zero_division=0),
    'prec': precision_score(y_test, y_test_p, average='macro', zero_division=0),
    'rec':  recall_score(y_test, y_test_p, average='macro', zero_division=0),
    'train_time':tt, 'infer_ms':inf_ms,
    'size_str': _fmt_size(pkl), 'size_f': os.path.getsize(pkl)/1_048_576,
    'top5': _top5(y_test, y_test_p, enc), 'pkl': pkl,
}
print(f"    Test Acc {te_acc:.2f}%  |  Inf {inf_ms:.4f} ms/câu\n")


# ══════════════════════════════════════════════════════════
# NAIVE BAYES
# ══════════════════════════════════════════════════════════
print("  ▶ Naive Bayes")
print("    TF-IDF + Training...", end="", flush=True)
t0 = time.time()
with _Quiet():
    vec = TfIdfVectorizer(); X_tr = vec.fit_transform(train_c)
    enc = LabelEncoder();    y_tr = enc.fit_transform(train_l)
    m   = MultinomialNaiveBayes(alpha=1.0); m.fit(X_tr, y_tr)
tt = time.time()-t0
print(f" {_fmt_time(tt)}")

pkl = MODEL_DIR / "NaiveBayes_model.pkl"
with open(pkl,"wb") as f: pickle.dump({"vectorizer":vec,"label_encoder":enc,"model":m},f)

y_tr_p = m.predict(X_tr)
tr_acc = accuracy_score(y_tr, y_tr_p)*100

val_fc, y_val = _filter(val_c, val_l, enc)
with _Quiet(): X_val = vec.transform(val_fc)
va_acc = accuracy_score(y_val, m.predict(X_val))*100

test_fc, y_test = _filter(test_c, test_l, enc)
t_inf = time.time()
with _Quiet(): X_test = vec.transform(test_fc)
y_test_p = m.predict(X_test)
inf_ms = (time.time()-t_inf)/len(test_fc)*1000
te_acc = accuracy_score(y_test, y_test_p)*100

results['nb'] = {
    'name':'Naive Bayes',
    'train_acc':tr_acc, 'val_acc':va_acc, 'test_acc':te_acc,
    'overfit':tr_acc-va_acc,
    'f1':   f1_score(y_test, y_test_p, average='macro', zero_division=0),
    'prec': precision_score(y_test, y_test_p, average='macro', zero_division=0),
    'rec':  recall_score(y_test, y_test_p, average='macro', zero_division=0),
    'train_time':tt, 'infer_ms':inf_ms,
    'size_str': _fmt_size(pkl), 'size_f': os.path.getsize(pkl)/1_048_576,
    'top5': _top5(y_test, y_test_p, enc), 'pkl': pkl,
}
print(f"    Test Acc {te_acc:.2f}%  |  Inf {inf_ms:.4f} ms/câu\n")


# ══════════════════════════════════════════════════════════
# K-NEAREST NEIGHBORS
# ══════════════════════════════════════════════════════════
print("  ▶ K-Nearest Neighbors")
print("    TF-IDF...", end="", flush=True)
with _Quiet():
    vec = TfIdfVectorizer(); X_tr = vec.fit_transform(train_c)
    enc = LabelEncoder();    y_tr = enc.fit_transform(train_l)
    X_val_raw = vec.transform(val_c)
print(" ok")
print("    Tuning + Training...", end="", flush=True)
t0 = time.time()
with _Quiet():
    tuner = KNearestNeighbors(k=15, weighted=True, normalize=True)
    tuner.fit(X_tr, y_tr)
    best_k, best_w = tune_hyperparams(tuner, X_val_raw, val_l, enc,
                                      candidate_ks=[1,3,5,7,9,11,15,21,31])
    m = KNearestNeighbors(k=best_k, weighted=best_w, normalize=True)
    m.fit(X_tr, y_tr)
tt = time.time()-t0
print(f" {_fmt_time(tt)}  (k={best_k}, weighted={best_w})")

pkl = MODEL_DIR / "KNN_model.pkl"
with open(pkl,"wb") as f: pickle.dump({"vectorizer":vec,"label_encoder":enc,"model":m},f)

print("    Eval train (slow)...", end="", flush=True)
y_tr_p = m.predict(X_tr)
tr_acc = accuracy_score(y_tr, y_tr_p)*100
print(f" {tr_acc:.2f}%")

val_fc, y_val = _filter(val_c, val_l, enc)
with _Quiet(): X_val = vec.transform(val_fc)
print("    Eval val (slow)...", end="", flush=True)
va_acc = accuracy_score(y_val, m.predict(X_val))*100
print(f" {va_acc:.2f}%")

test_fc, y_test = _filter(test_c, test_l, enc)
print("    Eval test (slow)...", end="", flush=True)
t_inf = time.time()
with _Quiet(): X_test = vec.transform(test_fc)
y_test_p = m.predict(X_test)
inf_ms = (time.time()-t_inf)/len(test_fc)*1000
te_acc = accuracy_score(y_test, y_test_p)*100
print(f" {te_acc:.2f}%  |  Inf {inf_ms:.4f} ms/câu\n")

results['knn'] = {
    'name':f'KNN (k={best_k})',
    'train_acc':tr_acc, 'val_acc':va_acc, 'test_acc':te_acc,
    'overfit':tr_acc-va_acc,
    'f1':   f1_score(y_test, y_test_p, average='macro', zero_division=0),
    'prec': precision_score(y_test, y_test_p, average='macro', zero_division=0),
    'rec':  recall_score(y_test, y_test_p, average='macro', zero_division=0),
    'train_time':tt, 'infer_ms':inf_ms,
    'size_str': _fmt_size(pkl), 'size_f': os.path.getsize(pkl)/1_048_576,
    'top5': _top5(y_test, y_test_p, enc), 'pkl': pkl,
}


# ══════════════════════════════════════════════════════════
# NEAREST CENTROID
# ══════════════════════════════════════════════════════════
print("  ▶ Nearest Centroid")
print("    TF-IDF + Training...", end="", flush=True)
t0 = time.time()
with _Quiet():
    vec = TfIdfVectorizer(); X_tr = vec.fit_transform(train_c)
    enc = LabelEncoder();    y_tr = enc.fit_transform(train_l)
    m   = NearestCentroid(normalize=True); m.fit(X_tr, y_tr)
tt = time.time()-t0
print(f" {_fmt_time(tt)}")

pkl = MODEL_DIR / "NearestCentroid_model.pkl"
with open(pkl,"wb") as f: pickle.dump({"vectorizer":vec,"label_encoder":enc,"model":m},f)

y_tr_p = m.predict(X_tr)
tr_acc = accuracy_score(y_tr, y_tr_p)*100

val_fc, y_val = _filter(val_c, val_l, enc)
with _Quiet(): X_val = vec.transform(val_fc)
va_acc = accuracy_score(y_val, m.predict(X_val))*100

test_fc, y_test = _filter(test_c, test_l, enc)
t_inf = time.time()
with _Quiet(): X_test = vec.transform(test_fc)
y_test_p = m.predict(X_test)
inf_ms = (time.time()-t_inf)/len(test_fc)*1000
te_acc = accuracy_score(y_test, y_test_p)*100

results['nc'] = {
    'name':'Nearest Centroid',
    'train_acc':tr_acc, 'val_acc':va_acc, 'test_acc':te_acc,
    'overfit':tr_acc-va_acc,
    'f1':   f1_score(y_test, y_test_p, average='macro', zero_division=0),
    'prec': precision_score(y_test, y_test_p, average='macro', zero_division=0),
    'rec':  recall_score(y_test, y_test_p, average='macro', zero_division=0),
    'train_time':tt, 'infer_ms':inf_ms,
    'size_str': _fmt_size(pkl), 'size_f': os.path.getsize(pkl)/1_048_576,
    'top5': _top5(y_test, y_test_p, enc), 'pkl': pkl,
}
print(f"    Test Acc {te_acc:.2f}%  |  Inf {inf_ms:.4f} ms/câu\n")


# ══════════════════════════════════════════════════════════
# [3/3] BẢNG TỔNG HỢP
# ══════════════════════════════════════════════════════════
ORDER = ['lr','nb','knn','nc']
KEYS  = ORDER

print("\n" + "="*64)
print("  [3/3] BẢNG TỔNG HỢP KẾT QUẢ")
print("="*64 + "\n")

C0, CW = 24, 20
SEP = "  " + "─"*(C0 + CW*4)

def _prow(lbl, cells):
    print(f"  {lbl:<{C0}}" + "".join(f"{str(c):<{CW}}" for c in cells))

_prow("", [results[k]['name'] for k in KEYS])
print(SEP)

STAT_ROWS = [
    ("Train Accuracy",       lambda r: f"{r['train_acc']:.2f}%"),
    ("Val Accuracy",         lambda r: f"{r['val_acc']:.2f}%"),
    ("Test Accuracy",        lambda r: f"{r['test_acc']:.2f}%"),
    ("Overfit Gap (tr-val)", lambda r: f"{r['overfit']:+.2f}%"),
    ("Macro F1 (test)",      lambda r: f"{r['f1']:.4f}"),
    ("Macro Precision",      lambda r: f"{r['prec']:.4f}"),
    ("Macro Recall",         lambda r: f"{r['rec']:.4f}"),
    ("Training Time",        lambda r: _fmt_time(r['train_time'])),
    ("Inference Time",       lambda r: f"{r['infer_ms']:.4f} ms/câu"),
    ("Model Size (.pkl)",    lambda r: r['size_str']),
]

for name, fmt in STAT_ROWS:
    _prow(name, [fmt(results[k]) for k in KEYS])

print(SEP)

# Top 5 per model
print()
for k in KEYS:
    r = results[k]
    print(f"  Top 5 confused [{r['name']}]:")
    top5 = r['top5']
    while len(top5) < 5: top5.append((("N/A","N/A"),0))
    for i,((tl,pl),cnt) in enumerate(top5,1):
        print(f"    {i}. {tl} → {pl}  ({cnt} lần)")
    print()


# ══════════════════════════════════════════════════════════
# ĐỐI CHIẾU VỚI SOURCE (solieu_*.md)
# ══════════════════════════════════════════════════════════
print("="*64)
print("  ĐỐI CHIẾU VỚI SOURCE  (solieu_*.md)")
print("  Legend:  ✓ khớp   ✗ lệch   ~ thời gian (luôn biến động)   N/A không có reference")
print("="*64)

SOURCE_FILES = {
    'lr':  "src/logistic-regression/solieu_LR.md",
    'nb':  "src/naive-bayes/solieu_NB.md",
    'knn': "src/k-nearest-neighbor/solieu_KNN.md",
    'nc':  "src/nearest-centroid/solieu_NC.md",
}

CMP_ROWS = [
    # (label, cur_key, ref_key, tol, is_time)
    ("Train Accuracy",   'train_acc', 'train_acc', 0.005, False),
    ("Val Accuracy",     'val_acc',   'val_acc',   0.005, False),
    ("Test Accuracy",    'test_acc',  'test_acc',  0.005, False),
    ("Overfit Gap",      'overfit',   'overfit',   0.005, False),
    ("Macro F1",         'f1',        'f1',        0.00005, False),
    ("Macro Precision",  'prec',      'prec',      0.00005, False),
    ("Macro Recall",     'rec',       'rec',       0.00005, False),
    ("Training Time",    'train_time',None,        None,  True),
    ("Inference Time",   'infer_ms',  'infer_ms',  None,  True),
    ("Model Size",       'size_f',    'size',      0.0001, False),
    ("Top 5 pairs",      'top5',      'top5',      None,  False),
]

for k in KEYS:
    r   = results[k]
    ref = REF[k]
    src = SOURCE_FILES[k]
    print(f"\n  ┌─ {r['name']}  ←→  {src}")

    if ref is None:
        print(f"  │  (Không tìm thấy file reference — chưa có solieu tương ứng)")
        print(f"  └{'─'*58}")
        continue

    CW2 = [22, 20, 20, 6]
    hdr = f"  │  {'Metric':<{CW2[0]}}{'Reference':<{CW2[1]}}{'Current':<{CW2[2]}}{'Status'}"
    print(hdr)
    print(f"  │  {'─'*66}")

    for label, cur_k, ref_k, tol, is_time in CMP_ROWS:
        if is_time:
            status = "~"
            ref_str = f"{ref.get('train_time' if cur_k=='train_time' else 'infer_ms', '?'):.4f} s" \
                      if not isinstance(ref.get('train_time'), type(None)) else "?"
            if cur_k == 'infer_ms':
                ref_val = ref.get('infer_ms')
                ref_str = f"{ref_val:.4f} ms/câu" if ref_val else "?"
                cur_str = f"{r['infer_ms']:.4f} ms/câu"
            else:
                ref_val = ref.get('train_time')
                ref_str = f"{ref_val:.4f} s" if ref_val else "?"
                cur_str = _fmt_time(r['train_time'])
        elif label == "Top 5 pairs":
            ref_pairs = ref.get('top5', [])
            cur_pairs = [(a,b) for (a,b),_ in r['top5']]
            status = _cmp_top5(ref_pairs, cur_pairs)
            ref_str = str(ref_pairs[:2]) + "…"
            cur_str = str(cur_pairs[:2]) + "…"
        else:
            ref_val = ref.get(ref_k) if ref_k else None
            cur_val = r.get(cur_k)
            status = _cmp(ref_val, cur_val, tol)
            if cur_k in ('train_acc','val_acc','test_acc','overfit'):
                ref_str = f"{ref_val:.2f}%" if ref_val is not None else "?"
                cur_str = f"{cur_val:.2f}%"
            elif cur_k in ('f1','prec','rec'):
                ref_str = f"{ref_val:.4f}" if ref_val is not None else "?"
                cur_str = f"{cur_val:.4f}"
            else:
                ref_str = f"{ref_val:.4f} MB" if ref_val is not None else "?"
                cur_str = f"{cur_val:.4f} MB"

        flag = f"  ◀ lệch" if status == "✗" else ""
        print(f"  │  {label:<{CW2[0]}}{ref_str:<{CW2[1]}}{cur_str:<{CW2[2]}}{status}{flag}")

    print(f"  └{'─'*58}")

print()
