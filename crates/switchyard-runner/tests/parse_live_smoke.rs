use switchyard_runner::Runner;

static SAFETY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn live_routes_toml_parses_unchanged() {
    let source =
        std::fs::read_to_string("/home/vincent/.local/lib/localclaw-switchyard/routes.toml")
            .expect("live routes.toml readable");
    // This smoke test validates SCHEMA parsing only: satisfy the fail-loud
    // credential guard with placeholder values for every env-var NAME the file
    // references (names are configuration, not secrets; no real credential
    // values are read or printed here).
    for name in source.lines().filter_map(|line| {
        let (key, value) = line.split_once('=')?;
        let key = key.trim();
        let wants = key == "api_key_env" || key == "auth_token_env";
        wants.then(|| value.trim().trim_matches('"').to_string())
    }) {
        if std::env::var(&name).is_err() {
            // Single-threaded smoke test; no concurrent env reads to race.
            let _guard = SAFETY_LOCK.lock().unwrap();
            // SAFETY: see above - single-threaded, placeholder-only.
            unsafe {
                std::env::set_var(&name, "smoke-test-placeholder");
            }
        }
    }
    match Runner::from_toml(&source) {
        Ok(runner) => {
            let count = runner.models().count();
            println!("PARSED OK: {count} routes");
            assert!(count > 50, "expected ~58 routes, got {count}");
            assert!(runner.fleet_state().is_some(), "fleet state present");
            assert!(
                runner.fleet_readiness().is_some(),
                "readiness config present"
            );
            assert!(
                !runner.capabilities().is_empty(),
                "capability routes present"
            );
            assert!(
                runner.capability_clients().len() >= 2,
                "capability clients present"
            );
        }
        Err(e) => panic!("LIVE CONFIG FAILED TO PARSE: {e}"),
    }
}
