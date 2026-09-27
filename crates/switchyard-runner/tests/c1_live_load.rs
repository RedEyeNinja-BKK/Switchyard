//! C1 gate probe: does the EXACT live deployment file load end-to-end through
//! the real parser and runtime builder? This is the return gate for the
//! config-compatibility lane, not an F1 unit test.
//!
//! `SWITCHYARD_C1_CONFIG` overrides the path at runtime so a probe can point at
//! a temporary copy. The default is the live file, read-only.

/// GREEN as of F7 (2026-09-26): the exact live file loads end-to-end through the
/// real parser and runtime builder. The test stays `#[ignore]`d so a routine
/// `cargo test` does not require production credentials; run it explicitly:
/// `cargo test -p switchyard-runner --test c1_live_load -- --ignored`.
/// It needs the six credential VARIABLES to be present and non-empty (values
/// are never read by the load path). With that env empty the gate reports RED —
/// the discriminating negative control for this pass.
#[test]
#[ignore = "C1 gate: requires the six credential variables; opt in with -- --ignored"]
fn c1_exact_live_file_loads_through_the_real_parser_and_builder() {
    let path = std::env::var("SWITCHYARD_C1_CONFIG").unwrap_or_else(|_| {
        "/home/vincent/.local/lib/localclaw-switchyard/routes.toml".to_string()
    });
    match switchyard_runner::Runner::load(&path) {
        Ok(_) => {}
        Err(error) => panic!("C1 RED: live config did not load: {error}"),
    }
}
