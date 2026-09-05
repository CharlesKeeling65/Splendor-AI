"""
Browser adaptation layer: wraps the game.hullqin.cn/ccbs web page as a
Splendor environment sharing the SplendorEnvBase protocol.

The layer's single responsibility is translating between the DOM world
(HTML snapshots, clicks) and the engine world (SplendorState, engine rule
checks) - every rule or feature computation is delegated to the engine
itself, never re-implemented here.

Module map (plan phase-2):
    driver          BrowserDriver protocol + offline MockBrowserDriver
    dom_extractor   EXTRACT_SNAPSHOT_JS, Snapshot schema, schema validation
    card_registry   web card faces -> engine Card objects (P0 deliverable)
    state_builder   Snapshot -> engine-isomorphic pseudo state (the pivot)
    action_executor ALL_ACTIONS index -> measured click sequences
    browser_env     BrowserSplendorEnv: one web seat as a gym.Env
    session         room lifecycle, double-identity cookie recipes
    monitor         engine-mask vs DOM-affordance parity attribution
    fixtures/       offline HTML fixtures (registry-verified card faces)
"""
