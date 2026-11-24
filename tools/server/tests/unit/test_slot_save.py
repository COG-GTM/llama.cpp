import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


def test_slot_actions_isolated_between_api_keys(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_SLOTS_DEBUG", "1")
    server.api_key = "key-alice"
    server.extra_api_keys = ["key-mallory"]
    server.server_slots = True
    server.start()
    alice = {"Authorization": "Bearer key-alice"}
    mallory = {"Authorization": "Bearer key-mallory"}

    # Alice fills slot 1
    res = server.make_request("POST", "/completion", headers=alice, data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    alice_prompt_n = res.body["timings"]["prompt_n"]

    # Mallory cannot save, restore into or erase Alice's slot
    res = server.make_request("POST", "/slots/1?action=save", headers=mallory, data={"filename": "stolen.bin"})
    assert res.status_code == 403
    res = server.make_request("POST", "/slots/1?action=erase", headers=mallory)
    assert res.status_code == 403
    res = server.make_request("POST", "/slots/1?action=restore", headers=mallory, data={"filename": "stolen.bin"})
    assert res.status_code == 403

    # Alice can save her own slot
    res = server.make_request("POST", "/slots/1?action=save", headers=alice, data={"filename": "alice.bin"})
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Mallory cannot restore Alice's save file, even into a free slot
    res = server.make_request("POST", "/slots/0?action=restore", headers=mallory, data={"filename": "alice.bin"})
    assert res.status_code == 400

    # Even in debug mode, /slots only shows prompt and generated text to the slot's owner
    res = server.make_request("GET", "/slots", headers=mallory)
    assert res.status_code == 200
    slot1 = next(s for s in res.body if s["id"] == 1)
    assert "prompt" not in slot1
    assert "generated" not in slot1
    res = server.make_request("GET", "/slots", headers=alice)
    assert res.status_code == 200
    slot1 = next(s for s in res.body if s["id"] == 1)
    assert "France" in slot1["prompt"]

    # Alice can still restore and erase her own slot
    res = server.make_request("POST", "/slots/0?action=restore", headers=alice, data={"filename": "alice.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    # restoring gives Alice ownership of slot 0, so Mallory cannot touch it
    res = server.make_request("POST", "/slots/0?action=erase", headers=mallory)
    assert res.status_code == 403
    res = server.make_request("POST", "/slots/1?action=erase", headers=alice)
    assert res.status_code == 200

    # Once Mallory claims a slot with her own request, she can act on it
    res = server.make_request("POST", "/completion", headers=mallory, data={
        "prompt": "Hello",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/1?action=save", headers=mallory, data={"filename": "mallory.bin"})
    assert res.status_code == 200
    # and now Alice is the one locked out
    res = server.make_request("POST", "/slots/1?action=erase", headers=alice)
    assert res.status_code == 403

    # Claiming a slot drops the previous key's cached prompt instead of reusing it
    res = server.make_request("POST", "/completion", headers=alice, data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", headers=mallory, data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == alice_prompt_n
