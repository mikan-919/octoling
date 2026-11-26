//! safetensors を mmap して、名前でテンソルのバイト列を引く。

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct Tensor<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

pub struct SafeTensors {
    mmap: Mmap,
    base: usize,
    index: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: 重みファイルは読み取り専用で、実行中に書き換えられない前提
        let mmap = unsafe { Mmap::map(&file)? };
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&mmap[8..8 + n]).map_err(std::io::Error::other)?;
        let mut index = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap().to_owned();
            let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let off = v["data_offsets"].as_array().unwrap();
            let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            if 8 + n + e > mmap.len() {
                return Err(std::io::Error::other(format!("{name}: data_offsets がファイル外を指している")));
            }
            index.insert(name, (dtype, shape, s, e));
        }
        Ok(Self { mmap, base: 8 + n, index })
    }

    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let (dtype, shape, s, e) = self.index.get(name)?;
        Some(Tensor { dtype: dtype.clone(), shape: shape.clone(), data: &self.mmap[self.base + s..self.base + e] })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}
