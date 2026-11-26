//! 待機中もスピンし続けるスレッドプール。1トークンで約1000回並列処理を呼ぶので、
//! rayon のようにスレッドを眠らせて起こすと、その往復が計算時間より長くなる。
//! ponytail: 待機中も全スレッドが CPU を使い続ける。常駐サーバーにするなら一定時間後に眠らせる。

use std::sync::LazyLock;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering::SeqCst};

struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}

struct Shared {
    job: AtomicPtr<Job>,
    epoch: AtomicUsize,
    active: AtomicUsize,
}

static SHARED: Shared = Shared { job: AtomicPtr::new(std::ptr::null_mut()), epoch: AtomicUsize::new(0), active: AtomicUsize::new(0) };

static WORKERS: LazyLock<usize> = LazyLock::new(|| {
    let n = std::env::var("OCTO_THREADS").ok().and_then(|s| s.parse().ok());
    // 帯域律速なので SMT の2本目は効かない。論理コア数の半分を物理コア数とみなす
    let n = n.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1)));
    for _ in 1..n {
        std::thread::spawn(worker);
    }
    n
});

fn drain(job: &Job) {
    loop {
        let i = job.next.fetch_add(1, SeqCst);
        if i >= job.n {
            return;
        }
        // SAFETY: f は run() が返るまで生きている。run() は全タスクの完了と active == 0 を待つ
        unsafe { (*job.f)(i) };
        job.done.fetch_add(1, SeqCst);
    }
}

fn worker() {
    let mut seen = 0;
    loop {
        let e = SHARED.epoch.load(SeqCst);
        if e == seen {
            std::hint::spin_loop();
            continue;
        }
        seen = e;
        SHARED.active.fetch_add(1, SeqCst);
        let p = SHARED.job.load(SeqCst);
        if !p.is_null() {
            // SAFETY: active を増やしてから読んだので、run() は戻らず Job は生きている
            drain(unsafe { &*p });
        }
        SHARED.active.fetch_sub(1, SeqCst);
    }
}

/// f(0), f(1), ..., f(n-1) を全スレッドで分けて実行する。呼び出し元のスレッドも働く。
/// 入れ子にして呼んではいけない。
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if *WORKERS == 1 || n == 1 {
        (0..n).for_each(f);
        return;
    }
    // SAFETY: 寿命を消すだけ。f はこの関数が返るまで使われ終わる
    let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
    let job = Job { f, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0) };
    SHARED.job.store(&job as *const Job as *mut Job, SeqCst);
    SHARED.epoch.fetch_add(1, SeqCst);
    drain(&job);
    while job.done.load(SeqCst) < n {
        std::hint::spin_loop();
    }
    SHARED.job.store(std::ptr::null_mut(), SeqCst);
    while SHARED.active.load(SeqCst) > 0 {
        std::hint::spin_loop();
    }
}

/// out を chunk 要素ずつに分け、f(チャンク番号, チャンク) を並列に呼ぶ。
pub fn chunks_mut<T: Send>(out: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let len = out.len();
    let base = out.as_mut_ptr() as usize;
    run(len.div_ceil(chunk), &|c| {
        let start = c * chunk;
        let n = chunk.min(len - start);
        // SAFETY: チャンクは互いに重ならず、out の借用は run() が返るまで続く
        f(c, unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(start), n) });
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn covers_every_index_once() {
        let mut v = vec![0u32; 10_007];
        for _ in 0..100 {
            super::chunks_mut(&mut v, 64, |_, c| c.iter_mut().for_each(|x| *x += 1));
        }
        assert!(v.iter().all(|&x| x == 100));
    }
}
