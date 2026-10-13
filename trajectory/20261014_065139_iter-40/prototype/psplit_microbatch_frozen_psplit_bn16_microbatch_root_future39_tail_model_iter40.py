"""Independent explicit tail-epoch model; does not import the generator model."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import sys

ROOT = Path("/share/home/liuyao/workspace/MegaFMM")
PREFIX = ROOT / ".scratch/psplit_bn16_microbatch_root_future39_"
DOMAINS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 16, 17, 18, 33, 34]
OWNED = [(0, 1, 2), (3, 4, 5, 6)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        mode, flags = args[1] or "", args[2] if len(args) > 2 else 0
        writes = isinstance(mode, str) and any(c in mode for c in "wax+")
        writes |= isinstance(flags, int) and bool(flags & (1 | 2 | 64 | 512 | 1024))
        if writes and not str(Path(args[0]).absolute()).startswith(str(PREFIX)):
            raise RuntimeError("tail model write outside unique prefix")


sys.addaudithook(guard)
main = json.loads(Path(str(PREFIX)+"review.json").read_text())
assert main["candidate_sha256"] == sha(str(PREFIX)+"candidate.cu")
protected = main["protected_before_after_sha256"]
assert {p: sha(p) for p in protected} == protected


class EpochViolation(Exception):
    pass


def need(ok, tag):
    if not ok:
        raise EpochViolation(tag)


class TailEpoch:
    """Explicit token arrivals abstract each128-thread role; all U owners literal."""
    def __init__(self, stages):
        self.stages = stages
        self.epoch = 0
        self.last_generations = [max((s for s in range(stages) if s % 2 == b), default=None) for b in range(2)]
        self.retired = [set() if g is not None else {0, 1} for g in self.last_generations]
        self.producer_tail_banks = set()
        self.drains = set()
        self.bar6 = set()
        self.bar0 = set()
        self.stores = set()
        self.store_counts = [0, 0]
        self.loads = Counter()
        self.owners = {}
        self.outputs = set()
        self.trace = []

    def retire(self, bank, wg):
        need(self.epoch == 0, "RETIRE_AFTER_TYPED_ALIAS")
        need(wg not in self.retired[bank], "DUPLICATE_FINAL_BANK_WG_RETIRE")
        self.retired[bank].add(wg)

    def producer_tail_acquire(self, bank):
        need(self.epoch == 0, "PRODUCER_TAIL_AFTER_ALIAS")
        need(self.retired[bank] == {0, 1}, "PRODUCER_TAIL_BANK_BEFORE_256_RELEASES")
        self.producer_tail_banks.add(bank)

    def drain(self, wg):
        need(all(wg in r for r in self.retired), "CONSUMER_DRAIN_WITH_LIVE_BANK")
        self.drains.add(wg)

    def arrive6(self, role):
        if role == 2:
            need(self.producer_tail_banks == {0, 1}, "PRODUCER_BAR6_WITHOUT_TWO_BANK_TAIL")
        else:
            need(role in self.drains, "CONSUMER_BAR6_WITHOUT_DRAIN")
        self.bar6.add(role)

    def activate_typed_alias(self):
        need(self.epoch == 0, "ALIAS_ACTIVATED_TWICE")
        need(all(r == {0, 1} for r in self.retired), "ALIAS_BEFORE_BOTH_BANKS_DRAINED")
        need(self.producer_tail_banks == {0, 1} and self.drains == {0, 1}, "ALIAS_BEFORE_PRODUCER_TAIL_AND_TWO_DRAINS")
        need(self.bar6 == {0, 1, 2}, "ALIAS_BEFORE_ALL384_NONALIGNED_BAR6")
        self.epoch = 1

    def store(self, wg, product, row, col):
        need(self.epoch == 1, "TAIL_STORE_OUTSIDE_TYPED_WRITE_EPOCH")
        need(product in OWNED[wg], "TAIL_P_STORE_WRONG_WG")
        mask = ((row & 2) >> 1) | ((row & 4) << 1)
        word = product*1024+row*16+(col ^ mask)
        need(word not in self.stores, "DUPLICATE_TAIL_WORD")
        self.stores.add(word)
        self.store_counts[wg] += 1

    def arrive0(self, role):
        need(self.epoch == 1, "POSTTAIL_BAR0_OUTSIDE_WRITE_EPOCH")
        if role < 2:
            need(self.store_counts[role] == 1024*len(OWNED[role]), "CONSUMER_BAR0_BEFORE_OWN_P_STORES")
        need(role not in self.bar0, "DUPLICATE_BAR0_ROLE")
        self.bar0.add(role)
        if self.bar0 == {0, 1, 2}:
            need(self.stores == set(range(7168)), "BAR0_COMPLETED_WITH_UNSTORED_P_WORDS")
            self.epoch = 2

    def u(self, tid, linear, beta_nonzero):
        need(self.epoch == 2 and self.bar0 == {0, 1, 2}, "U_BEFORE_ALL384_BAR0")
        need(linear in range(tid, 1024, 384), "U_CHANGED_FIXED_TID_OWNER")
        need(linear not in self.owners, "DUPLICATE_UNIFIED_U_OWNER")
        self.owners[linear] = tid
        row, col = divmod(linear, 16)
        mask = ((row & 2) >> 1) | ((row & 4) << 1)
        for p in range(7):
            word = 1024*p+row*16+(col ^ mask)
            need(word in self.stores, "U_UNSTORED_P_LOAD")
            self.loads[word] += 1
        order = ["P1..7", "u1_RNadd", "u2_RNadd", "u3_RNadd", "u4_RNadd",
                 "u5_RNadd", "u6_RNsub", "u7_RNadd", "alphaRN11", "alphaRN12", "alphaRN21", "alphaRN22"]
        if beta_nonzero:
            order += ["C011", "C012", "C021", "C022", "betaFMA11", "betaFMA12", "betaFMA21", "betaFMA22"]
        order += ["D11", "D12", "D21", "D22"]
        if beta_nonzero:
            need(max(order.index(x) for x in ("C011", "C012", "C021", "C022")) < order.index("D11"), "D_BEFORE_ALL_FOUR_C0_LOADS")
        for quadrant in range(4):
            need((quadrant, linear) not in self.outputs, "D_OUTPUT_DUPLICATED")
            self.outputs.add((quadrant, linear))
        self.trace.append((tid, linear, order))


def drained(stages, bar6=True):
    t = TailEpoch(stages)
    for bank in range(2):
        if t.last_generations[bank] is not None:
            for wg in range(2):
                t.retire(bank, wg)
        t.producer_tail_acquire(bank)
    for wg in range(2):
        t.drain(wg)
    if bar6:
        for role in range(3):
            t.arrive6(role)
    return t


store_events = []
for p in range(7):
    wg = 0 if p < 3 else 1
    for tid in range(128):
        for slot in range(8):
            t0, t1, t2 = tid % 4, (tid // 4) % 8, tid // 32
            v0, v1, v2 = slot % 2, (slot // 2) % 2, slot // 4
            store_events.append((wg, p, t1+16*t2+8*v1, 2*t0+v0+8*v2))

valid = []
for stages in DOMAINS:
    for beta_nonzero in (False, True):
        t = drained(stages)
        t.activate_typed_alias()
        # Producer's first128 bar0 arrivals can happen before either WG's stores.
        t.arrive0(2)
        shuffled = list(store_events)
        random.Random(stages).shuffle(shuffled)
        for e in shuffled:
            t.store(*e)
        t.arrive0(0)
        t.arrive0(1)
        for tid in range(384):
            for linear in range(tid, 1024, 384):
                t.u(tid, linear, beta_nonzero)
        need(t.epoch == 2 and set(t.owners) == set(range(1024)), "TAIL_OR_U_INCOMPLETE")
        need(t.loads == Counter({w: 1 for w in range(7168)}), "U_NOT_ONCE_PER_P_WORD")
        need(len(t.outputs) == 4096, "D_FOUR_QUADRANTS_INCOMPLETE")
        valid.append({"stages": stages, "beta_nonzero": beta_nonzero, "tail_epochs": [0, 1, 2],
                      "tail_store_words": len(t.stores), "U_owners": len(t.owners), "D_words": len(t.outputs),
                      "fixedU_order_trace_sha256": hashlib.sha256(json.dumps(t.trace).encode()).hexdigest()})

negative = []


def reject(name, fn, expected):
    try:
        fn()
    except EpochViolation as exc:
        need(str(exc) == expected, "WRONG_TAIL_NEGATIVE_REJECTION")
        negative.append({"control": name, "rejection": str(exc)})
        return
    raise EpochViolation("TAIL_NEGATIVE_ACCEPTED")


reject("early_producer_tail_bank", lambda: TailEpoch(3).producer_tail_acquire(0), "PRODUCER_TAIL_BANK_BEFORE_256_RELEASES")
reject("alias_before_both_banks_drain", lambda: TailEpoch(3).activate_typed_alias(), "ALIAS_BEFORE_BOTH_BANKS_DRAINED")
reject("alias_before_bar6", lambda: drained(3, False).activate_typed_alias(), "ALIAS_BEFORE_ALL384_NONALIGNED_BAR6")
reject("tail_store_before_alias", lambda: drained(3).store(0, 0, 0, 0), "TAIL_STORE_OUTSIDE_TYPED_WRITE_EPOCH")
t = drained(3)
t.activate_typed_alias()
reject("U_before_bar0", lambda: t.u(0, 0, True), "U_BEFORE_ALL384_BAR0")
reject("consumer_bar0_before_all_own_P_stores", lambda: t.arrive0(0), "CONSUMER_BAR0_BEFORE_OWN_P_STORES")
for e in store_events:
    t.store(*e)
for role in range(3):
    t.arrive0(role)
reject("rewrite_tail_during_U_epoch", lambda: t.store(0, 0, 0, 0), "TAIL_STORE_OUTSIDE_TYPED_WRITE_EPOCH")
reject("changed_fixedU_owner", lambda: t.u(1, 0, True), "U_CHANGED_FIXED_TID_OWNER")
assert {p: sha(p) for p in protected} == protected
report = {
    "status": "PASS_CPU_EXPLICIT_TAIL_THREE_EPOCHS_FIXEDU_ONLY",
    "model_sha256": sha(__file__), "candidate_sha256": main["candidate_sha256"],
    "stage_domains": DOMAINS, "valid_beta0_and_beta_nonzero_cases": valid,
    "negative_controls": negative, "negative_controls_rejected": len(negative),
    "literal_operand_payload_bytes": 2*(4*64*32+7*16*32)*2,
    "literal_typed_tail_payload_bytes": 7*64*16*4,
    "actual_shared_ABI_bytes": None, "actual_two_CTA_residency": None,
    "all_protected_old_bytes_unchanged": True,
    "scope": "independent logical lifetime/arrival/coordinate model; no C++ instantiation, compiler ABI, device barrier timing, or GPU precision/performance proof",
}
out = Path(str(PREFIX)+"tail_review.json")
assert not out.exists(), "refuse overwrite of frozen tail review"
out.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"status": report["status"], "valid_cases": len(valid),
                  "negative_controls_rejected": len(negative), "tail_review_sha256": sha(out),
                  "candidate_sha256": main["candidate_sha256"], "all_protected_old_bytes_unchanged": True}, indent=2))
