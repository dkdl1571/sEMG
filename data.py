"""data.py - 원신호(CSV) -> 필터 -> 윈도우 -> 시간영역 특징 -> dataset.npz

실행:  python data.py
(원본 데이터 semg-auth/data 가 있을 때만 필요. 이미 만들어진 dataset.npz 가 있으면 main.py 만 실행하면 된다.)
"""
import glob
import os

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 설정 ──
DATA_DIR = "semg-auth/data"     # 폴더 이름(A~E)이 곧 클래스
FS = 1000                       # 샘플링 주파수(가정값)
START, LEN = 500, 2000          # 3초(3000행) 중 가운데 2초
WIN, HOP = 200, 100             # 윈도우 200샘플(200ms), 간격 100샘플(오버랩 50%)
SEED = 42


def load(path):
    """CSV 1개 읽기. 첫 줄은 열 이름(헤더)이라 건너뛴다. 반환: (3000, 2)"""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def preprocess(x, fs=FS):
    """60Hz 노치 + 20~450Hz 4차 Butterworth 대역통과 (양방향 filtfilt)"""
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def cut(x, start=START, n=LEN):
    """분석 구간 자르기. 반환: (2000, 2)"""
    return x[start:start + n]


def make_windows(x, win=WIN, hop=HOP):
    """슬라이딩 윈도우. 반환: (19, 200, 2)"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop:i * hop + win] for i in range(n)])


def td_features(w, th=0.01):
    """윈도우 (200, 2) -> 10차원. 순서: RMS2, MAV2, WL2, ZC2, SSC2"""
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1] * w[1:]) < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1] * d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])


def build_dataset(out="dataset.npz"):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
    classes = sorted({os.path.basename(os.path.dirname(p)) for p in files})
    cls2id = {c: i for i, c in enumerate(classes)}
    label_of = lambda p: cls2id[os.path.basename(os.path.dirname(p))]
    labels = [label_of(p) for p in files]

    # 파일(시행) 단위로 먼저 분할 -> 겹치는 윈도우가 학습/테스트에 섞이는 데이터 누수 방지
    tr_f, te_f = train_test_split(files, test_size=0.2, stratify=labels, random_state=SEED)

    def build(flist):
        X, y = [], []
        for p in flist:
            seg = cut(preprocess(load(p)))
            for wnd in make_windows(seg):
                X.append(td_features(wnd))
                y.append(label_of(p))
        return np.stack(X), np.array(y)

    Xtr, ytr = build(tr_f)
    Xte, yte = build(te_f)

    sc = StandardScaler()            # 학습셋에서만 fit
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    np.savez(out, Xtr=Xtr_s, ytr=ytr, Xte=Xte_s, yte=yte, classes=np.array(classes))
    print(f"저장 완료: {out}  학습 {Xtr_s.shape}  테스트 {Xte_s.shape}  클래스 {classes}")


if __name__ == "__main__":
    build_dataset()
