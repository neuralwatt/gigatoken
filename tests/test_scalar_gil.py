"""Scalar native work releases Python without sharing mutable encode state."""

import sys
import threading
import time

import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers

from gigatoken import Tokenizer
from gigatoken._hf_compat import _gpt2_unicode_to_byte


def make_encoder(kind):
    if kind == "bpe":
        hf = HFTokenizer(models.BPE(_gpt2_unicode_to_byte(), []))
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    else:
        vocab = {"<unk>": 0, **{f"<0x{b:02X}>": b + 1 for b in range(256)}}
        hf = HFTokenizer(models.BPE(vocab, [], unk_token="<unk>", byte_fallback=True))
        hf.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="never")
    tokenizer = Tokenizer(hf)
    encode = tokenizer.encode_no_normalize if kind == "sentencepiece_normalized" else tokenizer.encode
    return tokenizer, encode, hf, kind


@pytest.fixture(params=["bpe", "sentencepiece", "sentencepiece_normalized"])
def scalar_encoder(request):
    return make_encoder(request.param)


def test_scalar_encode_releases_gil_and_retains_cache_borrow(scalar_encoder):
    tokenizer, encode, _, _ = scalar_encoder
    # Large enough that an awakened observer can run on an ordinary CI host.
    text = " abcdefghij 12345" * 1_000_000
    expected = encode(text)
    ready = threading.Event()
    start = threading.Event()
    observations = []

    def observe():
        ready.set()
        if not start.wait(10):
            return
        try:
            tokenizer.cache_entries()
        except RuntimeError as exc:
            observations.append((time.perf_counter(), str(exc)))

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    assert ready.wait(10)
    previous_interval = sys.getswitchinterval()
    try:
        # Do not mistake a Python timeslice between start.set() and encode()
        # for a thread running during native encoding.
        sys.setswitchinterval(10)
        started = time.perf_counter()
        start.set()
        actual = encode(text)
        finished = time.perf_counter()
    finally:
        sys.setswitchinterval(previous_interval)
        start.set()
        observer.join(10)
    assert not observer.is_alive()
    assert observations, "another Python thread never ran during native encode"
    observed_at, error = observations[0]
    assert started < observed_at < finished
    assert "borrow" in error.lower()
    assert (actual == expected).all()
    # The exclusive borrow is released on return and the cache stays usable.
    assert tokenizer.cache_entries() >= 0
    assert encode("test").size > 0


@pytest.mark.parametrize("as_bytes", [False, True])
def test_scalar_ids_stay_exact(scalar_encoder, as_bytes):
    _, encode, hf, kind = scalar_encoder
    text = " café 日本語 12345\n"
    # encode_no_normalize consumes text after the Metaspace transformation.
    if kind == "sentencepiece_normalized":
        text = text.replace(" ", "▁")
    if as_bytes and kind == "sentencepiece_normalized":
        with pytest.raises(TypeError):
            encode(text.encode())
        return
    value = text.encode() if as_bytes else text
    expected = hf.encode(text, add_special_tokens=False).ids
    assert encode(value).tolist() == expected


@pytest.mark.parametrize("kind", ["bpe", "sentencepiece"])
def test_mutable_input_is_snapshotted_before_detach(kind):
    _, encode, _, _ = make_encoder(kind)
    value = bytearray(b" abcdefghij 12345" * 1_000_000)
    expected = encode(bytes(value))
    ready = threading.Event()
    start = threading.Event()
    mutated = []

    def mutate():
        ready.set()
        if start.wait(10):
            value[:] = b"x" * len(value)
            mutated.append(time.perf_counter())

    worker = threading.Thread(target=mutate, daemon=True)
    worker.start()
    assert ready.wait(10)
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(10)
        started = time.perf_counter()
        start.set()
        actual = encode(value)
        finished = time.perf_counter()
    finally:
        sys.setswitchinterval(previous_interval)
        start.set()
        worker.join(10)
    assert not worker.is_alive()
    assert mutated and started < mutated[0] < finished
    assert (actual == expected).all()
