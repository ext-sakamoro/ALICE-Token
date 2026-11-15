//! Fixtures of the binary vocabulary format (`save_binary` / `load_binary`), captured from
//! bincode 1.3.3
//!
//! `tests/fixtures/bincode1/` holds the bytes `save_binary` wrote for two fixed vocabularies
//! (byte-level plus a few merges, and one trained on a fixed corpus) while this crate used
//! bincode 1.3.3; `SHA256SUMS` names each file. The tests pin that the bytes written now are
//! those bytes, that the files decode to the same vocabulary, and that writing the same
//! vocabulary again, in this process or another, gives the same bytes.

use std::path::{Path, PathBuf};

use alice_token::{byte_level_builder, load_binary, save_binary, Trainer, TrainerConfig, Vocab};

fn small() -> Vocab {
    let mut b = byte_level_builder();
    b.add_merge(b"a", b"b");
    b.add_merge(b"c", b"d");
    b.add_merge(b"ab", b"cd");
    b.add_merge("é".as_bytes(), b"t");
    b.build()
}

const CORPUS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps; the quick fox runs",
    "トークン化は決定論的であるべき",
    "rust rust rust trust",
];

fn trained() -> Vocab {
    let corpus: Vec<&[u8]> = CORPUS.iter().map(|s| s.as_bytes()).collect();
    Trainer::new(TrainerConfig {
        vocab_size: 320,
        ..TrainerConfig::default()
    })
    .train(&corpus)
}

fn fixtures() -> [(&'static str, Vocab); 2] {
    [
        ("vocab_small.bin", small()),
        ("vocab_trained.bin", trained()),
    ]
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bincode1")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(b).iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// Same vocabulary: same length, merges and token bytes for every id.
fn assert_same(a: &Vocab, b: &Vocab, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len");
    assert_eq!(a.merge_count(), b.merge_count(), "{what}: merges");
    #[allow(clippy::cast_possible_truncation)]
    for id in 0..a.len() as u32 {
        assert_eq!(a.get_token(id), b.get_token(id), "{what}: token {id}");
    }
}

/// Writes the fixtures (run once with bincode 1.3.3:
/// `ALICE_TOKEN_WRITE_BINCODE1_FIXTURES=1 cargo test --test bincode1_fixtures write -- --ignored`).
#[test]
#[ignore = "writes tests/fixtures/bincode1; run only to capture"]
fn write_bincode1_fixtures() {
    if std::env::var_os("ALICE_TOKEN_WRITE_BINCODE1_FIXTURES").is_none() {
        return;
    }
    std::fs::create_dir_all(dir()).unwrap();
    let mut sums = String::new();
    for (name, v) in fixtures() {
        let b = save_binary(&v).unwrap();
        std::fs::write(dir().join(name), &b).unwrap();
        sums.push_str(&format!("{}  {name}\n", sha256_hex(&b)));
    }
    std::fs::write(dir().join("SHA256SUMS"), sums).unwrap();
}

#[test]
fn every_fixture_matches_its_recorded_sha256() {
    let sums = std::fs::read_to_string(dir().join("SHA256SUMS")).unwrap();
    let mut checked = 0;
    for line in sums.lines() {
        let (hash, name) = line.split_once("  ").expect("`<sha256>  <name>`");
        assert_eq!(sha256_hex(&read(name)), hash, "{name}");
        checked += 1;
    }
    assert_eq!(checked, fixtures().len(), "SHA256SUMS lists every fixture");
}

#[test]
fn the_bytes_written_now_are_the_bytes_bincode_wrote() {
    for (name, v) in fixtures() {
        assert!(save_binary(&v).unwrap() == read(name), "{name}");
    }
}

#[test]
fn files_bincode_wrote_decode_to_the_same_vocabulary() {
    for (name, v) in fixtures() {
        let back = load_binary(&read(name)).unwrap();
        assert_same(&back, &v, name);
        assert!(
            save_binary(&back).unwrap() == read(name),
            "{name} re-encodes to itself"
        );
    }
}

#[test]
fn the_trained_fixture_is_not_trivial() {
    let v = trained();
    assert!(
        v.merge_count() >= 20,
        "trained vocabulary has {} merges",
        v.merge_count()
    );
}

#[test]
fn writing_the_same_vocabulary_again_gives_the_same_bytes() {
    for (name, _) in fixtures() {
        let first =
            save_binary(&fixtures().into_iter().find(|(n, _)| *n == name).unwrap().1).unwrap();
        for round in 0..4 {
            let again =
                save_binary(&fixtures().into_iter().find(|(n, _)| *n == name).unwrap().1).unwrap();
            assert!(
                again == first,
                "{name}: round {round} wrote different bytes"
            );
        }
    }
}

const CHILD_OUT: &str = "ALICE_TOKEN_DETERMINISM_OUT";

#[test]
fn another_process_writes_the_same_bytes() {
    let mine: Vec<Vec<u8>> = fixtures()
        .iter()
        .map(|(_, v)| save_binary(v).unwrap())
        .collect();
    if let Some(out) = std::env::var_os(CHILD_OUT) {
        let mut all = Vec::new();
        for b in &mine {
            all.extend_from_slice(&u32::try_from(b.len()).unwrap().to_le_bytes());
            all.extend_from_slice(b);
        }
        std::fs::write(out, all).unwrap();
        return;
    }
    let out = std::env::temp_dir().join(format!(
        "alice-token-determinism-{}.bin",
        std::process::id()
    ));
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "another_process_writes_the_same_bytes",
            "--exact",
            "--test-threads=1",
        ])
        .env(CHILD_OUT, &out)
        .status()
        .unwrap();
    assert!(status.success(), "child process failed");
    let theirs = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    let mut at = 0;
    for b in &mine {
        let len = u32::from_le_bytes(theirs[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        assert!(
            theirs[at..at + len] == b[..],
            "the other process wrote different bytes"
        );
        at += len;
    }
    assert_eq!(at, theirs.len());
}

#[test]
fn a_vocabulary_with_bytes_after_it_is_refused() {
    // the whole input is one vocabulary (bincode 1 ignored trailing bytes; this does not)
    let mut b = read("vocab_small.bin");
    assert!(load_binary(&b).is_ok());
    b.push(0);
    assert!(load_binary(&b).is_err());
}
