"""dfd_lib.py - deepfake detection pipeline: file-based 2-GPU job queue, data discovery,
face extraction, CLIP-LN / EfficientNet training, evaluation, ensembling, inference."""
import os, json, time, re, hashlib, random, traceback, math, glob
from pathlib import Path
import numpy as np

WORK = Path(os.environ.get("DFD_WORK", "/kaggle/working/dfd"))
Q = WORK / "queue"
STATES = ["pending", "running", "done", "failed", "skipped"]
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
FACE = 256  # stored crop size

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

class Deferred(Exception):
    """raised by a job that ran out of SESSION time after saving a checkpoint: goes back to pending, not failed"""

def load_cfg():
    return json.load(open(WORK / "cfg.json"))

def _atomic_json(path, obj):
    path = Path(path); tmp = path.parent / (".tmp_" + path.name + "_" + str(os.getpid()))
    json.dump(obj, open(tmp, "w"), default=str)
    os.replace(tmp, path)

# ----------------------------------------------------------------------------- job queue
def qinit():
    for s in STATES:
        (Q / s).mkdir(parents=True, exist_ok=True)

def _jid(name):
    return name.split(".json")[0].split("_", 1)[1]

def enqueue(job):
    """job: id, type, prio, args, deps (hard), soft_deps, min_left_min"""
    qinit()
    job.setdefault("deps", []); job.setdefault("soft_deps", []); job.setdefault("prio", 5)
    if job_state(job["id"]) is not None:
        return False
    _atomic_json(Q / "pending" / f"{int(job['prio']):02d}_{job['id']}.json", job)
    return True

def snapshot():
    st = {}
    for s in STATES:
        for p in (Q / s).glob("*.json*"):
            if not p.name.startswith("."):
                st[_jid(p.name)] = s
    return st

def job_state(jid):
    return snapshot().get(jid)

def recover_running():
    """after a crash/restart: put jobs that were mid-flight back into pending"""
    for p in (Q / "running").glob("*.json.*"):
        base = p.name.split(".json")[0] + ".json"
        os.replace(p, Q / "pending" / base)

def drop_failed():
    """delete failed/skipped job files so the build cell can enqueue them again; done jobs are untouched"""
    n = []
    for st in ("failed", "skipped"):
        for p in (Q / st).glob("*.json*"):
            n.append(_jid(p.name)); os.remove(p)
    return sorted(n)

def drop_jobs(ids):
    """delete the listed jobs (any state) and their training checkpoints so the build cell enqueues them again"""
    gone = []
    for st in STATES:
        for p in (Q / st).glob("*.json*"):
            if _jid(p.name) in ids: os.remove(p); gone.append(_jid(p.name))
    for i in ids:
        ck = WORK / "ckpt" / f"{i}_last.pt"
        if ck.exists(): ck.unlink()
    return sorted(set(gone))

def _finish(running_path, job, state, **extra):
    job.update(extra); job["finished"] = time.time()
    base = Path(running_path).name.split(".json")[0] + ".json"
    _atomic_json(Q / state / base, job)
    try: os.unlink(running_path)
    except FileNotFoundError: pass

def claim(tag):
    snap = snapshot()
    for p in sorted((Q / "pending").glob("*.json")):
        try: job = json.load(open(p))
        except Exception: continue
        hard, soft = job.get("deps", []), job.get("soft_deps", [])
        if any(snap.get(d) in ("failed", "skipped") for d in hard):
            dst = Q / "running" / (p.name + ".dep")
            try: os.rename(p, dst)
            except OSError: continue
            _finish(dst, job, "failed", error="dependency failed"); continue
        if any(snap.get(d) != "done" for d in hard): continue
        if any(snap.get(d) not in ("done", "failed", "skipped") for d in soft): continue
        dst = Q / "running" / (p.name + "." + tag)
        try: os.rename(p, dst)          # atomic: only one worker wins
        except OSError: continue
        return job, dst
    return None

def queue_table():
    rows = []
    for s in STATES:
        for p in sorted((Q / s).glob("*.json*")):
            if p.name.startswith("."): continue
            try: j = json.load(open(p))
            except Exception: continue
            rows.append((s, j["id"], j["type"], p.name.split(".json.")[-1] if s == "running" else ""))
    return rows

def worker_loop(tag, deadline):
    qinit()
    log(f"worker {tag} up (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    ctx = {"tag": tag, "deadline": deadline, "cfg": load_cfg()}
    while True:
        got = claim(tag)
        if got is None:
            snap = snapshot()
            pend = [k for k, v in snap.items() if v == "pending"]
            runn = [k for k, v in snap.items() if v == "running"]
            if not pend and not runn: break
            if time.time() > deadline: log("deadline reached, worker exits"); break
            if pend and not runn:      # nothing can ever become claimable
                time.sleep(10)
                if claim_probe_stuck(): break
            time.sleep(5); continue
        job, rpath = got
        left = (deadline - time.time()) / 60
        if job.get("min_left_min") and left < job["min_left_min"]:
            log(f"skip {job['id']} (only {left:.0f} min left)")
            _finish(rpath, job, "skipped", error="not enough time left"); continue
        log(f"START {job['id']} [{job['type']}]")
        t0 = time.time()
        try:
            res = RUNNERS[job["type"]](job.get("args", {}), ctx, job)
            _finish(rpath, job, "done", result=res, secs=time.time() - t0)
            log(f"DONE  {job['id']} in {(time.time()-t0)/60:.1f} min")
        except Deferred as e:
            log(f"DEFER {job['id']}: {e}")
            os.replace(rpath, Q / "pending" / (Path(rpath).name.split(".json")[0] + ".json"))
            _free_gpu(); break
        except BaseException as e:
            tb = traceback.format_exc(); log(f"FAILED {job['id']}\n{tb}")
            _finish(rpath, job, "failed", error=repr(e), traceback=tb, secs=time.time() - t0)
            if isinstance(e, KeyboardInterrupt): raise
        _free_gpu()

def claim_probe_stuck():
    snap = snapshot()
    return not [k for k, v in snap.items() if v == "running"] and \
           not any(True for p in (Q / "pending").glob("*.json") if _claimable(p, snap))

def _claimable(p, snap):
    try: job = json.load(open(p))
    except Exception: return False
    return all(snap.get(d) == "done" for d in job.get("deps", [])) and \
           all(snap.get(d) in ("done", "failed", "skipped") for d in job.get("soft_deps", []))

def _free_gpu():
    try:
        import gc, torch; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass

# ----------------------------------------------------------------------------- discovery
FAKE_KEYS = ["fake", "synthesis", "manipulat", "face2face", "faceswap", "neuraltextures", "faceshifter", "forged", "generated"]
REAL_KEYS = ["real", "original", "pristine", "youtube", "actors", "authentic", "genuine"]
FF_METHODS = ["deepfakedetection", "deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter"]

def label_from_dirs(dirs):
    for d in reversed(dirs):               # deepest directory that says something wins
        d = d.lower(); f = any(k in d for k in FAKE_KEYS); r = any(k in d for k in REAL_KEYS)
        if f and not r: return 1
        if r and not f: return 0
    return None

def dataset_name(rel_parts):
    s = "/".join(rel_parts).lower()
    if "celeb" in s: return "celebdf"
    if "dfdc" in s or "deepfake-detection-challenge" in s: return "dfdc"
    if "faceforensics" in s or "ff++" in s or "ffpp" in s or re.search(r"(^|/)ff[-_]?c\d\d", s) or any(k in s for k in ("original_sequences", "manipulated_sequences", "neuraltextures", "face2face", "faceswap", "faceshifter")): return "ffpp"
    rp = rel_parts[2:] if rel_parts[:1] == ["datasets"] and len(rel_parts) >= 3 else rel_parts
    return "other_" + re.sub(r"[^a-z0-9]+", "", rp[0].lower())[:12] if rp else "other"

def group_tokens(ds, stem):
    if ds == "celebdf":
        t = re.findall(r"id\d+", stem)
        if t: return t
    if re.fullmatch(r"\d+(_\d+)?", stem): return stem.split("_")
    return [stem]

def discover(root="/kaggle/input"):
    """Walk /kaggle/input, return list of items {path, kind, dataset, label, method, tokens}"""
    items, unlabeled = [], 0
    root = str(root).rstrip("/")
    for dp, dn, fn in os.walk(root, followlinks=True):
        if os.path.basename(dp).startswith("dfd_") or os.path.basename(dp) == "deepfake_bundle": dn[:] = []; continue
        rel = os.path.relpath(dp, root).split(os.sep)
        rel = [] if rel == ["."] else rel
        fullrel = list(rel)
        if rel[:1] == ["datasets"] and len(rel) >= 3: rel = rel[3:]  # /kaggle/input/datasets/<owner>/<slug>/...
        top = (os.path.relpath(dp, root).split(os.sep) + [""])[0]
        meta = None
        if "metadata.json" in fn:
            try: meta = json.load(open(os.path.join(dp, "metadata.json")))
            except Exception: meta = None
        vids = [f for f in fn if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        imgs = [f for f in fn if os.path.splitext(f)[1].lower() in IMG_EXT]
        ds = dataset_name(fullrel if fullrel else [top])
        for f in vids:
            stem = os.path.splitext(f)[0]; lab = None; tokens = None
            if meta and f in meta and isinstance(meta[f], dict) and "label" in meta[f]:
                lab = 1 if str(meta[f]["label"]).upper() == "FAKE" else 0
                orig = meta[f].get("original"); tokens = [os.path.splitext(orig)[0]] if orig else [stem]
                ds = "dfdc"
            else:
                lab = label_from_dirs(rel[1:] if len(rel) > 1 else rel)
            if lab is None: unlabeled += 1; continue
            low = "/".join(rel).lower()
            method = next((m for m in FF_METHODS if m in low), "real" if lab == 0 else "fake")
            items.append(dict(path=os.path.join(dp, f), kind="video", dataset=ds, label=lab, method=method,
                              tokens=tokens or group_tokens(ds, stem)))
        if not vids and 8 <= len(imgs) <= 400:       # a folder of extracted frames = one video
            lab = label_from_dirs(rel[1:] if len(rel) > 1 else rel)
            if lab is None: unlabeled += 1; continue
            low = "/".join(rel).lower()
            method = next((m for m in FF_METHODS if m in low), "real" if lab == 0 else "fake")
            stem = os.path.basename(dp)
            items.append(dict(path=dp, kind="frames", dataset=ds, label=lab, method=method, tokens=group_tokens(ds, stem)))
    return items, unlabeled

def cap_items(items, caps, seed=0):
    """caps: {dataset: [max_real, max_fake]} with 'default'. Fakes are round-robined over methods."""
    rng = random.Random(seed); out = []
    for ds in sorted({i["dataset"] for i in items}):
        cr, cf = caps.get(ds, caps.get("default", [10**9, 10**9]))
        reals = [i for i in items if i["dataset"] == ds and i["label"] == 0]
        fakes = [i for i in items if i["dataset"] == ds and i["label"] == 1]
        rng.shuffle(reals); out += reals[:cr]
        by_m = {}
        for i in fakes: by_m.setdefault(i["method"], []).append(i)
        for v in by_m.values(): rng.shuffle(v)
        picked = []
        while len(picked) < cf and any(by_m.values()):
            for m in list(by_m):
                if by_m[m] and len(picked) < cf: picked.append(by_m[m].pop())
        out += picked
    return out

def assign_groups_and_splits(items, split_pct=(80, 10, 10)):
    """union-find on identity tokens so a real video and the fakes made from it never straddle splits."""
    for ds in sorted({i["dataset"] for i in items}):
        sub = [i for i in items if i["dataset"] == ds]
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for it in sub:
            t = it["tokens"]
            for a in t[1:]: parent[find(a)] = find(t[0])
            find(t[0])
        comp = {}
        for it in sub: comp.setdefault(find(it["tokens"][0]), []).append(it)
        biggest = max(len(v) for v in comp.values())
        giant = len(sub) >= 10 and biggest > 0.35 * len(sub)
        for it in sub:
            rep = it["tokens"][0] if giant else find(it["tokens"][0])
            it["group"] = f"{ds}:{rep}"
            h = int(hashlib.sha1(it["group"].encode()).hexdigest(), 16) % 100
            it["split"] = "train" if h < split_pct[0] else ("val" if h < split_pct[0] + split_pct[1] else "test")
        for sp in ("val", "test"):      # tiny datasets: never leave a split empty
            if not any(i["split"] == sp for i in sub) and len({i["group"] for i in sub}) >= 3:
                gs = sorted({i["group"] for i in sub if i["split"] == "train"}, key=lambda g: hashlib.sha1((g + sp).encode()).hexdigest())
                for i in sub:
                    if i["group"] == gs[0]: i["split"] = sp
        if giant: log(f"[{ds}] identity graph is one giant component -> splitting by first identity token instead")
    return items

# ----------------------------------------------------------------------------- face extraction
class FaceDet:
    def __init__(self, device):
        self.kind = None
        if os.environ.get("DFD_TEST_CENTER"): self.kind = "center"; return      # self-test hook
        try:
            from facenet_pytorch import MTCNN
            self.m = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7], min_face_size=48, post_process=False)
            self.kind = "mtcnn"
        except Exception as e:
            log("WARNING: MTCNN unavailable (%r) -> OpenCV Haar fallback (lower quality)" % (e,))
            try:
                import cv2
                self.h = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self.kind = "haar"
            except Exception as e2:
                log("WARNING: Haar unavailable too (%r) -> centre crop (poor)" % (e2,)); self.kind = "center"

    def boxes(self, frames_bgr):
        """one best box (x0,y0,x1,y1) or None per frame"""
        import cv2
        out = [None] * len(frames_bgr)
        if self.kind == "center":
            return [[f.shape[1] * .3, f.shape[0] * .25, f.shape[1] * .7, f.shape[0] * .75] for f in frames_bgr]
        scale = [1.0] * len(frames_bgr)
        small = []
        for i, f in enumerate(frames_bgr):
            h, w = f.shape[:2]; s = min(1.0, 640.0 / max(h, w)); scale[i] = s
            small.append(cv2.resize(f, (int(w * s), int(h * s))) if s < 1 else f)
        if self.kind == "mtcnn":
            from PIL import Image
            for i0 in range(0, len(small), 8):                    # same-size frames of one video
                chunk = [Image.fromarray(f[:, :, ::-1]) for f in small[i0:i0 + 8]]
                try: bx, pr = self.m.detect(chunk)
                except Exception: bx, pr = [None] * len(chunk), [None] * len(chunk)
                for j, (b, p) in enumerate(zip(bx, pr)):
                    if b is None or len(b) == 0: continue
                    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); score = np.where(p > 0.9, area, 0) + p
                    k = int(np.argmax(score)); out[i0 + j] = (b[k] / scale[i0 + j]).tolist()
        else:
            for i, f in enumerate(small):
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                r = self.h.detectMultiScale(g, 1.2, 5, minSize=(48, 48))
                if len(r):
                    x, y, w, h = max(r, key=lambda t: t[2] * t[3]); out[i] = (np.array([x, y, x + w, y + h]) / scale[i]).tolist()
        return out

def crop_face(img, box, margin=1.3, size=FACE):
    import cv2
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * margin
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2)); s = int(round(side))
    pad = max(0, -a, -b, a + s - img.shape[1], b + s - img.shape[0])
    if pad: img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0); a += pad; b += pad
    c = img[b:b + s, a:a + s]
    if c.size == 0: return None
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)

def read_frames(item, n):
    import cv2
    if item["kind"] == "frames":
        fs = sorted(p for p in glob.glob(os.path.join(item["path"], "*")) if os.path.splitext(p)[1].lower() in IMG_EXT)
        idx = np.linspace(0, len(fs) - 1, min(n, len(fs))).round().astype(int)
        return [f for f in (cv2.imread(fs[i]) for i in idx) if f is not None]
    cap = cv2.VideoCapture(item["path"]); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    if total <= 0:
        allf = []
        while len(allf) < 600:
            ok, f = cap.read()
            if not ok: break
            allf.append(f)
        cap.release()
        return [allf[i] for i in np.linspace(0, len(allf) - 1, min(n, len(allf))).round().astype(int)] if allf else []
    want = sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist())); wi = 0; i = 0
    while wi < len(want):
        if not cap.grab(): break
        if i == want[wi]:
            ok, f = cap.retrieve()
            if ok: frames.append(f)
            wi += 1
        i += 1
    cap.release()
    return frames

def extract_video(item, det, n):
    frames = read_frames(item, n)
    if not frames: return []
    boxes = det.boxes(frames); crops = []
    for f, b in zip(frames, boxes):
        if b is None: continue
        c = crop_face(f, b)
        if c is not None: crops.append(c)
    return crops

def run_extract(args, ctx, job):
    import cv2, pandas as pd
    cv2.setNumThreads(2)
    dev = "cuda" if _has_cuda() else "cpu"
    det = FaceDet(dev); items = json.load(open(args["items_file"])); rows = []; skipped = 0
    faces = WORK / "faces"
    for k, it in enumerate(items):
        vid = hashlib.sha1(it["path"].encode()).hexdigest()[:12]; d = faces / it["dataset"] / vid; d.mkdir(parents=True, exist_ok=True)
        have = sorted(d.glob("*.jpg"))
        if not have:
            crops = extract_video(it, det, it["n_frames"])
            if len(crops) < 2: skipped += 1; continue
            for j, c in enumerate(crops): cv2.imwrite(str(d / f"{j:02d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
            have = sorted(d.glob("*.jpg"))
        for p in have:
            rows.append((str(p), it["dataset"], vid, it["label"], it["split"], it["method"], it["group"]))
        if k % 50 == 0: log(f"  extract {job['id']}: {k}/{len(items)} videos, skipped {skipped}")
    out = WORK / "manifest_parts"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["path", "ds", "vid", "label", "split", "method", "group"]).to_csv(out / f"{job['id']}.csv", index=False)
    return {"videos": len(items), "skipped_no_face": skipped, "frames": len(rows)}

def run_merge(args, ctx, job):
    import pandas as pd
    parts = sorted((WORK / "manifest_parts").glob("*.csv"))
    man = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    man.to_csv(WORK / "manifest.csv", index=False)
    v = man.drop_duplicates("vid")
    tab = v.groupby(["ds", "split", "label"]).size().unstack(fill_value=0)
    log("manifest: %d frames, %d videos\n%s" % (len(man), len(v), tab.to_string()))
    return {"frames": int(len(man)), "videos": int(len(v)), "table": tab.to_string()}

def _has_cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

# ----------------------------------------------------------------------------- data / augmentation
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def select_rows(man, sel):
    import pandas as pd
    m = pd.Series(False, index=man.index)
    for ds, splits in sel:
        mm = man.ds == ds
        if "all" not in splits: mm &= man.split.isin(splits)
        m |= mm
    return man[m]

def per_video_cap(df, k):
    """evenly spaced <=k frames per video"""
    if k is None: return df
    keep = []
    for _, g in df.sort_values("path").groupby("vid", sort=False):
        idx = np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int); keep.append(g.iloc[np.unique(idx)])
    import pandas as pd
    return pd.concat(keep) if keep else df

def augment(img, rng):
    """img: BGR uint8 FACE x FACE. Mild geometric + photometric + compression/blur degradations."""
    import cv2
    if rng.random() < 0.5: img = img[:, ::-1]
    h, w = img.shape[:2]; s = rng.uniform(0.80, 1.0); ar = rng.uniform(0.92, 1.08)
    ch, cw = int(h * s), int(min(w, w * s * ar)); y0 = rng.integers(0, h - ch + 1); x0 = rng.integers(0, w - cw + 1)
    img = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    if rng.random() < 0.5:
        a = rng.uniform(0.75, 1.25); b = rng.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] *= rng.uniform(0.6, 1.4); hsv[..., 0] += rng.uniform(-6, 6)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.05: img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5])); img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:     # down/up-scale
        f = rng.uniform(0.4, 0.85); hh, ww = img.shape[:2]
        img = cv2.resize(cv2.resize(img, (max(16, int(ww * f)), max(16, int(hh * f))), interpolation=cv2.INTER_AREA), (ww, hh), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:     # JPEG re-compression
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 96))])
        if ok: img = cv2.imdecode(enc, 1)
    return img

class FrameDS:
    def __init__(self, df, size, train):
        self.p = df.path.tolist(); self.y = df.label.astype("float32").tolist(); self.size = size; self.train = train
        import cv2; cv2.setNumThreads(0)
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        import cv2, torch
        img = cv2.imread(self.p[i])
        if img is None: img = np.zeros((FACE, FACE, 3), np.uint8)
        if self.train:
            rng = np.random.default_rng((int(time.time() * 1e6) + i * 7919) % (2**32)); img = augment(img, rng)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))   # RGB CHW uint8
        return t, torch.tensor(self.y[i])

# ----------------------------------------------------------------------------- models
BACKBONES = {
    "clip_l14_ln": dict(size=224, mean=CLIP_MEAN, std=CLIP_STD, bs=16, accum=2, lr=1e-4, head_lr=1e-3),
    "clip_l14_svd": dict(size=224, mean=CLIP_MEAN, std=CLIP_STD, bs=16, accum=2, lr=2e-4, head_lr=1e-3),
    "effnet_b4":   dict(size=256, mean=IN_MEAN, std=IN_STD, bs=32, accum=1, lr=2e-4, head_lr=2e-4),
    "tiny":        dict(size=64, mean=IN_MEAN, std=IN_STD, bs=16, accum=1, lr=1e-3, head_lr=1e-3),   # CPU self-test only
}

_CLIP_OVERRIDE = {}      # set by Detector: CLIP base weights come from the exported file (no download); "infer" skips the SVD at load time

def _fit_prefix(sd, ref):
    """saved key names differ between transformers versions ("vit.vision_model." or "vit."): rename so every key exists in the model"""
    ref = set(ref)
    if all(k in ref for k in sd): return sd
    for f in (lambda k: k.replace("vit.", "vit.vision_model.", 1), lambda k: k.replace("vit.vision_model.", "vit.", 1)):
        m = {f(k): v for k, v in sd.items()}
        if all(k in ref for k in m): return m
    return sd

def _fit_base(sd, ref):
    ref = set(ref)
    if all(k in ref for k in sd): return sd
    for f in (lambda k: "vision_model." + k, lambda k: k.replace("vision_model.", "", 1)):
        m = {f(k): v for k, v in sd.items()}
        if all(k in ref for k in m): return m
    return sd

def build_model(name):
    import torch, torch.nn as nn, torch.nn.functional as F

    class CosHead(nn.Module):
        def __init__(self, d):
            super().__init__(); self.w = nn.Parameter(torch.randn(d) * 0.02); self.b = nn.Parameter(torch.zeros(1)); self.s = nn.Parameter(torch.tensor(10.0))
        def forward(self, z): return self.s * (z @ F.normalize(self.w, dim=0)) + self.b

    class ClipLN(nn.Module):
        """CLIP ViT-L/14 vision tower, only LayerNorm affine params trained (+ cosine head on L2-normalised embedding)."""
        def __init__(self):
            super().__init__()
            from transformers import CLIPVisionModel
            base = _CLIP_OVERRIDE.get("base")
            if base is not None:
                from transformers import CLIPVisionConfig
                self.vit = CLIPVisionModel(CLIPVisionConfig.from_dict(base["cfg"])); self.vit.load_state_dict(_fit_base(base["sd"], self.vit.state_dict())); self.vit = self.vit.half()
            else: self.vit = CLIPVisionModel.from_pretrained(os.environ.get("DFD_CLIP_PATH", "openai/clip-vit-large-patch14")).half()
            for p in self.vit.parameters(): p.requires_grad_(False)
            for m in self.vit.modules():
                if isinstance(m, nn.LayerNorm):
                    m.float()
                    for p in m.parameters(): p.requires_grad_(True)
            self.head = CosHead(self.vit.config.hidden_size); self.ckpt = False
        def enable_ckpt(self):
            self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); self.ckpt = True
        def forward(self, x):
            z = F.normalize(self.vit(pixel_values=x).pooler_output.float(), dim=-1)
            return self.head(z), z
        def ln_params(self): return [p for n, p in self.vit.named_parameters() if p.requires_grad]
        def head_params(self): return list(self.head.parameters())
        def trainable_state(self): return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("head.") or ("layer_norm" in k or "layrnorm" in k or "layernorm" in k)}
        def load_trainable(self, sd):
            r = self.load_state_dict(_fit_prefix(sd, self.state_dict()), strict=False)
            if r.unexpected_keys: raise RuntimeError(f"{len(r.unexpected_keys)} saved weights do not match this transformers version, e.g. {r.unexpected_keys[:2]}")

    class SVDLinear(nn.Module):
        """frozen weight with its smallest singular direction split off as a trainable rank-1 residual (Effort-style): y = x W_main^T + b + s (x.v) u"""
        def __init__(self, lin, infer=False):
            super().__init__(); out_f, in_f = lin.weight.shape; dt = lin.weight.dtype
            if infer:                       # exported model: W is replaced by load_trainable, no SVD needed
                self.register_buffer("W", lin.weight.data.clone()); u0, v0, s0 = torch.zeros(out_f), torch.zeros(in_f), torch.zeros(())
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                U, S, Vh = torch.linalg.svd(lin.weight.data.float().to(dev), full_matrices=False)
                u0, s0, v0 = U[:, -1].cpu().clone(), S[-1].cpu().clone(), Vh[-1].cpu().clone()
                self.register_buffer("W", (lin.weight.data.float() - s0 * torch.outer(u0, v0)).to(dt))
            self.register_buffer("bias", None if lin.bias is None else lin.bias.data.clone())
            self.u, self.v, self.s = nn.Parameter(u0.clone()), nn.Parameter(v0.clone()), nn.Parameter(s0.clone())
            self.register_buffer("u0", u0); self.register_buffer("v0", v0); self.register_buffer("s0", s0)
        def forward(self, x):
            y = F.linear(x, self.W, self.bias)
            with torch.autocast(device_type=x.device.type, enabled=False):
                r = (x.float() @ self.v) * self.s
                return y + (r.unsqueeze(-1) * self.u).to(y.dtype)

    class ClipSVD(ClipLN):
        """CLIP ViT-L/14, every attention projection keeps its principal part frozen and trains only the smallest singular direction (Effort-style); LayerNorms frozen."""
        def __init__(self):
            super().__init__()
            for m in self.vit.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters(): p.requires_grad_(False)
            infer = bool(_CLIP_OVERRIDE.get("infer"))
            for mod in list(self.vit.modules()):       # attention blocks found by structure (module paths differ between transformers versions)
                if all(isinstance(getattr(mod, nm, None), nn.Linear) for nm in ("q_proj", "k_proj", "v_proj", "out_proj")):
                    for nm in ("q_proj", "k_proj", "v_proj", "out_proj"): setattr(mod, nm, SVDLinear(getattr(mod, nm), infer))
        def trainable_state(self): return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("head.") or k.rsplit(".", 1)[-1] in ("u", "v", "s", "u0", "v0", "s0")}
        def load_trainable(self, sd):
            sd = _fit_prefix(sd, self.state_dict())
            for name, m in self.named_modules():       # rebuild W_main from the saved original residual, so the SVD of this run never has to match the training run
                if isinstance(m, SVDLinear) and f"{name}.u0" in sd:
                    dev = m.W.device; full = m.W.float() + m.s0.float() * torch.outer(m.u0.float(), m.v0.float())
                    m.W = (full - sd[f"{name}.s0"].to(dev).float() * torch.outer(sd[f"{name}.u0"].to(dev).float(), sd[f"{name}.v0"].to(dev).float())).to(m.W.dtype)
            r = self.load_state_dict(sd, strict=False)
            if r.unexpected_keys: raise RuntimeError(f"{len(r.unexpected_keys)} saved weights do not match this transformers version, e.g. {r.unexpected_keys[:2]}")

    class Eff(nn.Module):
        def __init__(self):
            super().__init__(); import timm
            last = None
            for nm in ["tf_efficientnet_b4.ns_jft_in1k", "efficientnet_b4.ra2_in1k"]:
                try: self.net = timm.create_model(nm, pretrained=not os.environ.get("DFD_NO_PRETRAINED"), num_classes=1, drop_rate=0.3, drop_path_rate=0.2); break
                except Exception as e: last = e
            else: raise last
            self.ckpt = False
        def enable_ckpt(self):
            try: self.net.set_grad_checkpointing(True); self.ckpt = True
            except Exception: pass
        def forward(self, x): return self.net(x.contiguous(memory_format=torch.channels_last)).squeeze(-1), None
        def ln_params(self): return list(self.net.parameters())
        def head_params(self): return []
        def trainable_state(self): return {k: v.detach().cpu() for k, v in self.state_dict().items()}
        def load_trainable(self, sd): self.load_state_dict(sd)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(), nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Linear(16, 1); self.ckpt = False
        def enable_ckpt(self): pass
        def forward(self, x): return self.head(self.f(x)).squeeze(-1), None
        def ln_params(self): return list(self.f.parameters())
        def head_params(self): return list(self.head.parameters())
        def trainable_state(self): return {k: v.detach().cpu() for k, v in self.state_dict().items()}
        def load_trainable(self, sd): self.load_state_dict(sd)

    return {"clip_l14_ln": ClipLN, "clip_l14_svd": ClipSVD, "effnet_b4": Eff, "tiny": Tiny}[name]()

def normalise(x_uint8, mean, std, dev):
    import torch
    x = x_uint8.to(dev, non_blocking=True).float() / 255.0
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1); s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    return (x - m) / s

# ----------------------------------------------------------------------------- metrics
def eer_from(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2)

def video_table(df, col="logit"):
    g = df.groupby("vid").agg(label=("label", "first"), ds=("ds", "first"), logit=(col, "mean"), n=(col, "size")).reset_index()
    return g

def best_threshold(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y.tolist())) < 2: return 0.0
    c = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))); best, bt = -1, 0.0
    for t in c:
        p = s >= t; ba = 0.5 * ((p & (y == 1)).sum() / max(1, (y == 1).sum()) + (~p & (y == 0)).sum() / max(1, (y == 0).sum()))
        if ba > best: best, bt = ba, float(t)
    return bt

def metrics_at(vdf, thr, fdf=None):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
    y, s = vdf.label.values.astype(int), vdf.logit.values; out = {"n_videos": int(len(vdf)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    two = len(set(y.tolist())) == 2
    out["video_auc"] = float(roc_auc_score(y, s)) if two else float("nan")
    out["video_ap"] = float(average_precision_score(y, s)) if two else float("nan")
    out["video_eer"] = eer_from(y, s) if two else float("nan")
    p = (s >= thr).astype(int)
    out.update(acc=float(accuracy_score(y, p)), precision=float(precision_score(y, p, zero_division=0)), recall=float(recall_score(y, p, zero_division=0)),
               f1=float(f1_score(y, p, zero_division=0)), mcc=float(matthews_corrcoef(y, p)) if len(set(p.tolist())) > 1 else 0.0, thr=float(thr))
    if fdf is not None and len(set(fdf.label.tolist())) == 2: out["frame_auc"] = float(roc_auc_score(fdf.label.values, fdf.logit.values))
    return out

# ----------------------------------------------------------------------------- predict / train
def predict_df(model, df, bb, dev, bs=64, tta=False, workers=2):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(FrameDS(df, bb["size"], False), batch_size=bs, shuffle=False, num_workers=workers, pin_memory=(dev == "cuda"))
    model.eval(); outs = []
    with torch.no_grad(), torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.float16 if dev == "cuda" else torch.bfloat16, enabled=(dev == "cuda")):
        for x, _ in dl:
            x = normalise(x, bb["mean"], bb["std"], dev); l = model(x)[0].float()
            if tta: l = 0.5 * (l + model(torch.flip(x, dims=[3]))[0].float())
            outs.append(l.cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0)

def run_train(args, ctx, job):
    import torch, torch.nn.functional as F, pandas as pd
    from torch.utils.data import DataLoader, WeightedRandomSampler
    cfg = ctx["cfg"]; name = args["name"]; bbn = args["backbone"]; bb = BACKBONES[bbn]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.get("seed", 0); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    man = pd.read_csv(WORK / "manifest.csv")
    tr = select_rows(man, args["train_sel"]); va = per_video_cap(select_rows(man, args["val_sel"]), args.get("val_frames", 6))
    if len(tr) == 0 or len(va) == 0: raise RuntimeError("empty train/val selection")
    # class-balanced, video-uniform sampling weights
    nv = tr.groupby("label").vid.nunique(); nf = tr.groupby("vid").path.transform("size")
    w = (0.5 / tr.label.map(nv).values) / nf.values
    spe = args["samples_per_epoch"]; epochs = args["epochs"]; total_samples = spe * epochs
    log(f"[{name}] train frames {len(tr)} ({tr.vid.nunique()} videos), val frames {len(va)}, backbone {bbn}, device {dev}")
    _chk = man.path.sample(min(300, len(man)), random_state=0).tolist(); _miss = sum(not os.path.exists(p) for p in _chk)
    if _miss > 0.02 * len(_chk): raise RuntimeError(f"{_miss} of {len(_chk)} sampled face crops are missing on disk (e.g. {next(p for p in _chk if not os.path.exists(p))}); re-run the extraction instead of training on blank images")
    model = build_model(bbn).to(dev)
    if bbn == "effnet_b4": model = model.to(memory_format=torch.channels_last)
    if args.get("grad_ckpt"): model.enable_ckpt()
    groups = [{"params": model.ln_params(), "lr": bb["lr"], "weight_decay": 0.0 if bbn.startswith("clip") else 1e-2}]
    if model.head_params(): groups.append({"params": model.head_params(), "lr": bb["head_lr"], "weight_decay": 1e-4})
    for g in groups: g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    trainable = [p for g in groups for p in g["params"]]
    log(f"[{name}] trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")
    bs, accum = args.get("bs", bb["bs"]), args.get("accum", bb["accum"])
    ckdir = WORK / "ckpt"; ckdir.mkdir(exist_ok=True); last = ckdir / f"{name}_last.pt"
    sig = hashlib.sha1(json.dumps([name, bbn, spe, epochs, seed, len(tr), args["train_sel"]], default=str).encode()).hexdigest()[:10]
    step = it = seen = 0; best = (-1.0, None); next_eval = spe; hist = []; run_loss = 0.0; elapsed0 = 0.0
    if last.exists():
        try:
            S = torch.load(last, map_location="cpu", weights_only=False)
            if S.get("sig") != sig: raise RuntimeError("checkpoint is from different settings/data")
            model.load_trainable(S["model"]); opt.load_state_dict(S["opt"]); scaler.load_state_dict(S["scaler"])
            step, it, seen, next_eval, hist, run_loss, elapsed0 = S["step"], S["it"], S["seen"], S["next_eval"], S["hist"], S["run_loss"], S["elapsed"]
            best = (S["best_auc"], S["best_state"])
            log(f"[{name}] RESUMED from checkpoint: {seen}/{total_samples} samples, step {step}, {elapsed0/60:.1f} train-min so far, best val AUC {best[0]:.4f}")
        except Exception as e:
            log(f"[{name}] checkpoint ignored ({e!r}) -> starting fresh")
            step = it = seen = 0; best = (-1.0, None); next_eval = spe; hist = []; run_loss = 0.0; elapsed0 = 0.0
    remaining = total_samples - seen
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=max(1, remaining), replacement=True)
    dl = DataLoader(FrameDS(tr, bb["size"], True), batch_size=bs, sampler=sampler, num_workers=args.get("workers", 2), pin_memory=(dev == "cuda"), drop_last=True, persistent_workers=False)
    steps_total = max(1, total_samples // bs // accum); max_s = max(60.0, args["max_minutes"] * 60)   # job budget = TRAINING minutes summed over sessions
    session_end = ctx["deadline"] - args.get("eval_reserve_min", 12) * 60
    if session_end - time.time() < 300 and remaining >= bs: raise Deferred("less than 5 min of session left before training could start")
    t0 = time.time() - elapsed0; stopped = "complete"; last_save = time.time(); ckpt_every = float(os.environ.get("DFD_CKPT_MIN", "15")) * 60
    ck = WORK / "models"; ck.mkdir(exist_ok=True)

    def evaluate_val():
        vv = va.copy(); vv["logit"] = predict_df(model, vv, bb, dev, workers=2); model.train()
        vt = video_table(vv)
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(vt.label, vt.logit)) if vt.label.nunique() == 2 else 0.5

    def save_last():
        try:
            S = dict(sig=sig, model=model.trainable_state(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, it=it, seen=seen,
                     next_eval=next_eval, hist=hist, run_loss=run_loss, elapsed=time.time() - t0, best_auc=best[0], best_state=best[1])
            tmp = ckdir / f"tmp_{name}.pt"; torch.save(S, tmp); os.replace(tmp, last)   # (a leading "." in the name makes torch.save fail)
        except Exception as e:
            log(f"[{name}] WARNING checkpoint save failed, training continues without it: {e!r}")

    def lr_scale():
        prog = max(step / steps_total, (time.time() - t0) / max_s)
        warm = min(1.0, (step + 1) / 200)
        return warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    model.train(); opt.zero_grad(set_to_none=True)
    for x, y in dl:
        try:
            x = normalise(x, bb["mean"], bb["std"], dev); y = y.to(dev)
            ys = y * 0.95 + 0.025
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                logit, z = model(x)
                loss = F.binary_cross_entropy_with_logits(logit.float(), ys)
                ua = args.get("ua")
                if ua and z is not None and z.size(0) > 2:      # alignment + uniformity of L2-normalised features (weights: [alignment, uniformity])
                    zf = F.normalize(z.float(), dim=-1); iu = torch.triu_indices(zf.size(0), zf.size(0), 1, device=zf.device)
                    d2 = (zf[iu[0]] - zf[iu[1]]).pow(2).sum(-1); same_pair = (y[iu[0]] == y[iu[1]])
                    loss = loss + ua[1] * torch.log(torch.exp(-2 * d2).mean() + 1e-8)
                    if same_pair.any(): loss = loss + ua[0] * d2[same_pair].mean()
                if z is not None and hasattr(model, "head"):      # latent slerp augmentation (same-class interpolation)
                    idx = torch.randperm(z.size(0), device=dev); same = (y == y[idx])
                    if same.sum() > 1:
                        z1, z2 = z[same], z[idx][same]; t = torch.rand(z1.size(0), 1, device=dev)
                        om = torch.acos((z1 * z2).sum(-1, keepdim=True).clamp(-0.999, 0.999))
                        zt = (torch.sin((1 - t) * om) * z1 + torch.sin(t * om) * z2) / torch.sin(om)
                        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(model.head(zt.float()), ys[same])
            scaler.scale(loss / accum).backward()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            if not model.ckpt: log(f"[{name}] OOM -> enabling gradient checkpointing"); model.enable_ckpt(); continue
            raise
        it += 1; seen += x.size(0); run_loss = 0.98 * run_loss + 0.02 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % accum == 0:
            s = lr_scale()
            for g in opt.param_groups: g["lr"] = g["base_lr"] * s
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                el = time.time() - t0; log(f"[{name}] step {step}/{steps_total} loss {run_loss:.4f} lr x{s:.3f} {seen/el:.1f} img/s elapsed {el/60:.1f}m")
        if seen >= next_eval:
            auc = evaluate_val(); hist.append((seen, auc)); log(f"[{name}] VAL video AUC {auc:.4f} after {seen} samples")
            if auc > best[0]: best = (auc, model.trainable_state()); torch.save(best[1], ck / f"{name}.pt")
            next_eval += spe; save_last(); last_save = time.time()
        if time.time() - t0 > max_s: stopped = "time"; log(f"[{name}] time budget hit, stopping early"); break
        if time.time() - last_save > ckpt_every: save_last(); last_save = time.time()
        if time.time() > session_end:
            save_last(); raise Deferred(f"session time over at {seen}/{total_samples} samples; checkpoint saved - run the launch cell again in a new session")
    if not hist or hist[-1][0] != seen:
        auc = evaluate_val(); hist.append((seen, auc)); log(f"[{name}] VAL video AUC {auc:.4f} (final)")
        if auc > best[0]: best = (auc, model.trainable_state()); torch.save(best[1], ck / f"{name}.pt")
    model.load_trainable(best[1]); model.eval()
    return finish_eval(name, args, model, bb, dev, man, va, cfg, hist, stopped, t0)

def finish_eval(name, args, model, bb, dev, man, va, cfg, hist, stopped, t0):
    pr = WORK / "preds"; pr.mkdir(exist_ok=True)
    va = va.copy(); va["logit"] = predict_df(model, va, bb, dev); va.to_csv(pr / f"{name}__VAL.csv", index=False)
    vt = video_table(va); thr = best_threshold(vt.label, vt.logit); res = {"val_thr": thr, "history": hist, "stopped": stopped, "sets": {}}
    for sname, sel in args["test_sets"].items():
        df = per_video_cap(select_rows(man, sel), args.get("eval_frames", 8))
        if len(df) == 0: continue
        df = df.copy(); df["logit"] = predict_df(model, df, bb, dev, tta=args.get("tta", False)); df.to_csv(pr / f"{name}__{sname}.csv", index=False)
        res["sets"][sname] = metrics_at(video_table(df), thr, df)
        log(f"[{name}] {sname}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in res["sets"][sname].items() if k in ("video_auc", "frame_auc", "acc", "n_videos")))
    res["train_minutes"] = (time.time() - t0) / 60
    return res

# ----------------------------------------------------------------------------- ensemble / report
def _fit_ensemble_weights(X, y, ds):
    """non-negative weights summing to 1: logistic fit on validation videos, every (dataset, class) cell weighted equally, mild pull toward equal weights"""
    from scipy.optimize import minimize
    y = np.asarray(y).astype(int); ds = np.asarray(ds); M = X.shape[1]
    key = [(d, c) for d, c in zip(ds, y)]; cnt = {}
    for k in key: cnt[k] = cnt.get(k, 0) + 1
    sw = np.array([1.0 / cnt[k] for k in key]); sw /= sw.sum()
    def loss(p):
        th, a, b = p[:M], p[M], p[M + 1]; w = np.exp(th - th.max()); w /= w.sum()
        return float((sw * np.logaddexp(0, -(2 * y - 1) * (a * (X @ w) + b))).sum() + 1e-3 * np.sum(th ** 2))
    r = minimize(loss, np.r_[np.zeros(M), 1.0, 0.0], method="L-BFGS-B"); th = r.x[:M]; w = np.exp(th - th.max())
    return w / w.sum()

def _macro_threshold(z, y, ds):
    """threshold maximising the mean over datasets of balanced accuracy (so one easy dataset cannot drag the threshold)"""
    y = np.asarray(y).astype(int); z = np.asarray(z); ds = np.asarray(ds)
    groups = [np.where(ds == d)[0] for d in np.unique(ds)]; groups = [g for g in groups if len(set(y[g].tolist())) == 2]
    if not groups: return best_threshold(y, z)
    best, bt = -1.0, 0.0
    for t in np.unique(np.quantile(z, np.linspace(0.01, 0.99, 197))):
        p = z >= t
        s = np.mean([0.5 * ((p[g] & (y[g] == 1)).sum() / (y[g] == 1).sum() + (~p[g] & (y[g] == 0)).sum() / (y[g] == 0).sum()) for g in groups])
        if s > best: best, bt = s, float(t)
    return bt

def run_ensemble(args, ctx, job):
    import pandas as pd
    pr = WORK / "preds"; names = [n for n in args["models"] if (pr / f"{n}__VAL.csv").exists()]
    if not names: raise RuntimeError("no model predictions found")
    scales = {}; vals = {}
    for n in names:
        v = pd.read_csv(pr / f"{n}__VAL.csv"); scales[n] = float(max(v.logit.std(), 1e-3)); vals[n] = video_table(v).set_index("vid")
    common = sorted(set.intersection(*[set(v.index) for v in vals.values()]))
    Xv = np.stack([vals[n].loc[common, "logit"].values / scales[n] for n in names], axis=1)
    yv = vals[names[0]].loc[common, "label"].values.astype(int); dsv = vals[names[0]].loc[common, "ds"].values
    eq = np.ones(len(names)) / len(names); w = eq.copy()
    if args.get("weighting", "val") == "val" and len(names) > 1:
        try: w = _fit_ensemble_weights(Xv, yv, dsv)
        except Exception as e: log("weight fitting failed, using equal weights:", repr(e))
    cnn = np.array([str(args.get("specs", {}).get(n, {}).get("backbone", "")).startswith("effnet") for n in names]); cap = float(args.get("max_cnn_share", 1.0))
    if cnn.any() and (~cnn).any() and w[cnn].sum() > cap: w = np.where(cnn, w / w[cnn].sum() * cap, w / w[~cnn].sum() * (1 - cap)); log(f"EfficientNet weight share capped at {cap}")
    wd = dict(zip(names, [float(x) for x in w])); zv = Xv @ w
    thr = _macro_threshold(zv, yv, dsv) if args.get("weighting", "val") == "val" else best_threshold(yv, zv); thr_old = best_threshold(yv, Xv @ eq)
    log("ensemble weights:", {k: round(v, 3) for k, v in wd.items()}, "| threshold %.3f (old equal-weight pooled threshold %.3f)" % (thr, thr_old))
    out = {"models": names, "scales": scales, "weights": wd, "weighting": args.get("weighting", "val"), "threshold": thr, "ensemble": {}, "ensemble_equal": {}, "single": {}}
    for sname in args["sets"]:
        per = {}
        for n in names:
            f = pr / f"{n}__{sname}.csv"
            if f.exists(): per[n] = pd.read_csv(f)
        if not per: continue
        tabs = {n: video_table(d).set_index("vid") for n, d in per.items()}
        for n, t in tabs.items():
            tv = best_threshold(vals[n].label, vals[n].logit); out["single"].setdefault(n, {})[sname] = metrics_at(t.reset_index(), tv, per[n])
        com = sorted(set.intersection(*[set(t.index) for t in tabs.values()])); ks = list(tabs)
        wk = np.array([wd[n] for n in ks]); wk = wk / wk.sum()
        Lk = np.stack([tabs[n].loc[com, "logit"].values / scales[n] for n in ks], axis=1); lab = tabs[ks[0]].loc[com, "label"].values
        e = pd.DataFrame({"vid": com, "label": lab, "logit": Lk @ wk}); ee = pd.DataFrame({"vid": com, "label": lab, "logit": Lk.mean(1)})
        out["ensemble"][sname] = metrics_at(e, thr); out["ensemble_equal"][sname] = metrics_at(ee, thr_old)
        e.to_csv(pr / f"ENSEMBLE__{sname}.csv", index=False)
    aucs = [v["video_auc"] for v in out["ensemble"].values() if v["video_auc"] == v["video_auc"]]
    out["ensemble_macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    calib_a = 1.0
    try:      # steepness of a sigmoid centred on the threshold, fitted on class-balanced validation videos
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(fit_intercept=False, class_weight="balanced", C=1e4).fit((zv - thr).reshape(-1, 1), yv); calib_a = float(max(lr.coef_[0, 0], 1e-3))
    except Exception as e: log("calibration skipped:", repr(e))
    json.dump({"models": names, "scales": scales, "weights": wd, "threshold": thr, "calib_a": calib_a, "specs": args.get("specs", {})}, open(WORK / "ensemble_config.json", "w"), indent=1)
    json.dump(out, open(WORK / "ensemble_results.json", "w"), indent=1)
    log("ensemble macro video AUC: %.4f" % out["ensemble_macro_auc"])
    return out

# ----------------------------------------------------------------------------- inference on a new video
def predict_video(path, work=None, n_frames=16, device=None):
    """Load saved ensemble and score one video. Returns dict(prob_fake, label, per_model_logits, frames_used)."""
    import torch
    global WORK
    if work: WORK = Path(work)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ec = json.load(open(WORK / "ensemble_config.json")); det = FaceDet(dev)
    crops = extract_video(dict(path=str(path), kind="video"), det, n_frames)
    if len(crops) < 2: return {"error": "no face found"}
    per = {}
    for n in ec["models"]:
        spec = ec["specs"][n]; bb = BACKBONES[spec["backbone"]]; m = build_model(spec["backbone"]).to(dev)
        m.load_trainable(torch.load(WORK / "models" / f"{n}.pt", map_location="cpu")); m.eval()
        import cv2
        x = torch.from_numpy(np.stack([cv2.resize(c, (bb["size"], bb["size"]), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1) for c in crops]).copy())
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            per[n] = float(m(normalise(x, bb["mean"], bb["std"], dev))[0].float().mean()) / ec["scales"][n]
        del m; _free_gpu()
    z = float(np.mean(list(per.values()))) - ec["threshold"]
    return {"score": z, "label": "FAKE" if z > 0 else "REAL", "pseudo_prob_fake": float(1 / (1 + math.exp(-z))), "per_model": per, "frames_used": len(crops)}

def run_official_eval(args, ctx, job):
    """standard-protocol check: FF++-only models scored on the Celeb-DF-v2 official test videos (frame-level and video-level AUC are computed from these predictions)"""
    import torch, cv2, pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"; items = json.load(open(args["items_file"])); det = FaceDet(dev); crops = {}
    for it in items:
        cs = extract_video(dict(path=it["path"], kind="video"), det, args.get("frames", 12))
        if len(cs) >= 2: crops[it["path"]] = cs
    log(f"official test videos with a detected face: {len(crops)}/{len(items)}"); done = []
    for name in args["models"]:
        f = WORK / "models" / f"{name}.pt"
        if not f.exists(): log(f"{name}: no trained weights, skipped"); continue
        bbn = args["specs"][name]["backbone"]; bb = BACKBONES[bbn]; m = build_model(bbn); m.load_trainable(torch.load(f, map_location="cpu")); m = m.to(dev).eval(); rows = []
        with torch.no_grad():
            for it in items:
                cs = crops.get(it["path"])
                if not cs: continue
                x = torch.from_numpy(np.stack([cv2.resize(c, (bb["size"], bb["size"]), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1) for c in cs]).copy()); outs = []
                for i in range(0, len(x), 8):
                    xb = normalise(x[i:i + 8], bb["mean"], bb["std"], dev)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                        l = 0.5 * (m(xb)[0].float() + m(torch.flip(xb, dims=[3]))[0].float())
                    outs.append(l.cpu().numpy())
                vid = "/".join(it["path"].replace("\\", "/").split("/")[-2:])
                rows += [dict(path=it["path"], ds="celebdf", vid=vid, label=it["label"], logit=float(v)) for v in np.concatenate(outs)]
        pd.DataFrame(rows).to_csv(WORK / "preds" / f"{name}__celebdf_official.csv", index=False); done.append(name); del m
        if dev == "cuda": torch.cuda.empty_cache()
    return {"models": done, "videos": len(crops)}

def export_single(out_path, ec, work=None):
    """ONE file with every model's trained weights, the ensemble config and the CLIP base weights, so nothing has to be downloaded to run it"""
    import torch
    work = Path(work or WORK); obj = {"format": "dfd-1", "config": ec, "models": {n: torch.load(work / "models" / f"{n}.pt", map_location="cpu") for n in ec["models"]}, "clip_base": None}
    if any(ec["specs"][n]["backbone"].startswith("clip") for n in ec["models"]):
        base = _CLIP_OVERRIDE.get("base")
        if base is None:
            from transformers import CLIPVisionModel
            v = CLIPVisionModel.from_pretrained(os.environ.get("DFD_CLIP_PATH", "openai/clip-vit-large-patch14")).half()
            base = {"cfg": v.config.to_dict(), "sd": {k: t.detach().cpu() for k, t in v.state_dict().items()}}
        obj["clip_base"] = base
    torch.save(obj, out_path); return str(out_path)

class Detector:
    """Inference on a video or face image with the exported ensemble:  Detector("deepfake_model/deepfake_detector.pkl").predict("clip.mp4")"""
    def __init__(self, bundle, device=None):
        """bundle: path to deepfake_detector.pkl (the single file written by export_single)"""
        import torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(bundle, map_location="cpu", weights_only=False)
        self.cfg = obj["config"]; _CLIP_OVERRIDE["base"] = obj.get("clip_base"); _CLIP_OVERRIDE["infer"] = True
        os.environ["DFD_NO_PRETRAINED"] = "1"
        try:
            self.det = FaceDet(self.dev); self.models = {}
            for n in self.cfg["models"]:
                bbn = self.cfg["specs"][n]["backbone"]; m = build_model(bbn)
                m.load_trainable(obj["models"][n]); m = m.to(self.dev).eval()
                if self.dev == "cpu": m = m.float()
                self.models[n] = (m, BACKBONES[bbn])
        finally: _CLIP_OVERRIDE.clear()

    def predict(self, path, n_frames=16, tta=None):
        import torch, cv2
        path = Path(path); tta = (self.dev == "cuda") if tta is None else tta
        if path.suffix.lower() in IMG_EXT:
            img = cv2.imread(str(path))
            if img is None: return {"error": "could not read the image"}
            crops = [c for c in (crop_face(img, b) for b in self.det.boxes([img]) if b is not None) if c is not None]; need = 1
        else:
            crops = extract_video(dict(path=str(path), kind="video"), self.det, n_frames); need = 2
        if len(crops) < need: return {"error": "no face found in the input"}
        per = {}
        with torch.no_grad():
            for n, (m, bb) in self.models.items():
                x = torch.from_numpy(np.stack([cv2.resize(c, (bb["size"], bb["size"]), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1) for c in crops]).copy())
                outs = []
                for i in range(0, len(x), 8):
                    xb = normalise(x[i:i + 8], bb["mean"], bb["std"], self.dev)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.dev == "cuda")):
                        l = m(xb)[0].float()
                        if tta: l = 0.5 * (l + m(torch.flip(xb, dims=[3]))[0].float())
                    outs.append(l.cpu().numpy())
                per[n] = np.concatenate(outs) / self.cfg["scales"][n]
        w = self.cfg.get("weights"); fs = (sum(w.get(n, 0.0) * per[n] for n in per) / max(sum(w.get(n, 0.0) for n in per), 1e-9) if w else np.mean(list(per.values()), axis=0)) - self.cfg["threshold"]; z = float(fs.mean()); a = float(self.cfg.get("calib_a", 1.0))
        return dict(label="FAKE" if z > 0 else "REAL", score=z, prob_fake=float(1 / (1 + math.exp(-a * z))), frame_scores=fs.tolist(),
                    per_model={n: float(v.mean()) for n, v in per.items()}, crops=crops)

RUNNERS = {"extract": run_extract, "merge": run_merge, "train": run_train, "ensemble": run_ensemble, "official_eval": run_official_eval}
