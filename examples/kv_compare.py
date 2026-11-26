# kv_quality の出力 2 つ (基準と比べる方) と、その位置の正解のトークンから、KL ダイバージェンス、最上位の一致率、正解の負の対数尤度を出す
# 使い方: python kv_compare.py <基準の出力> <比べる出力> <@トークン ID のファイル> <prefill したトークン数>
import sys
import numpy as np

V = 200058  # forward と prefill が返す logits の長さ (語彙)
a = np.fromfile(sys.argv[1], np.float32).reshape(-1, V)
b = np.fromfile(sys.argv[2], np.float32).reshape(-1, V)
ids = np.array(open(sys.argv[3].lstrip('@')).read().strip().split(','), np.int64)
y = ids[int(sys.argv[4]):][:len(a)]


def logp(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


la, lb = logp(a.astype(np.float64)), logp(b.astype(np.float64))
kl = (np.exp(la) * (la - lb)).sum(1)
r = np.arange(len(y))
print(f'位置 {len(a)}: KL 平均 {kl.mean():.5f} 最大 {kl.max():.4f}、最上位の一致 {(a.argmax(1) == b.argmax(1)).mean() * 100:.2f}%、'
      f'正解の NLL 基準 {-la[r, y].mean():.4f} 比べる方 {-lb[r, y].mean():.4f}')
