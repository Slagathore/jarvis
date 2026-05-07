"""Per-room audio drivers: mic sources and speaker sinks. See base.py for
the protocol. Concrete drivers live in sibling files; the factory wires
them up in modules/voice/mic_manager.py + speaker_manager.py based on the
`mic.type` / `speaker.type` toggle in each room's config.yaml block.
"""
