"""
Browser adaptation layer: wraps the game.hullqin.cn/ccbs web page as a
Splendor environment sharing the SplendorEnvBase protocol.

The layer's single responsibility is translating between the DOM world
(HTML snapshots, clicks) and the engine world (SplendorState, engine rule
checks) - every rule or feature computation is delegated to the engine
itself, never re-implemented here.
"""
