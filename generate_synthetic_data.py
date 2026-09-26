
"""
generate_synthetic_data.py
===========================
FOR DEVELOPMENT / PIPELINE TESTING ONLY.
This produces MOCK data — it is NOT the official Amazon ML Challenge dataset
and must never be submitted or presented as such. It exists solely to test
that step1-step6 of the existing pipeline run correctly end-to-end and that
precision/recall/F0.5 behave sensibly before real data is used.

Run:
    python generate_synthetic_data.py
"""

import os
import csv
import random
import argparse
import math

random.seed(42)

# ───────────────────────── CONFIG ─────────────────────────
TRAIN_S1_COUNT   = 1000
TEST_S1_COUNT    = 350          # scaled down from train (see notes at bottom)
TARGET_S2_TRAIN  = 2200
TARGET_S3_TRAIN  = 2200
TARGET_S2_TEST   = 800
TARGET_S3_TEST   = 800
FRANCE_FRACTION_TEST = 0.15      # France appears ONLY in test, per problem statement
GENERIC_NAME_FRACTION = 0.05
SHARED_ADDRESS_FRACTION = 0.05
DECOY_SAME_NAME_FRACTION = 0.15
DECOY_SAME_ADDR_FRACTION = 0.15
NOISE_PROFILE = "standard"

OUT_TRAIN = "dataset/train"
OUT_TEST  = "dataset/test"

# ───────────────────────── VOCAB POOLS ─────────────────────────
INDIA_NAMES = ["Sharma","Gupta","Reddy","Iyer","Khan","Verma","Patel","Nair","Singh","Mehta",
               "Chatterjee","Rao","Kapoor","Malhotra","Joshi","Desai","Bose","Kulkarni","Menon","Agarwal"]
US_NAMES    = ["Smith","Johnson","Miller","Brown","Davis","Wilson","Anderson","Thompson","Garcia",
               "Martinez","Clark","Lewis","Walker","Hall","Young","King","Wright","Scott","Green","Baker"]
FRANCE_NAMES = ["Dubois","Moreau","Laurent","Simon","Michel","Lefevre","Roux","David","Bertrand","Morel"]

BIZ_WORDS = ["Electronics","Traders","Enterprises","Textiles","Foods","Motors","Pharmacy",
             "Consultants","Industries","Solutions","Bakery","Hardware","Autoparts","Logistics",
             "Apparel","Furniture","Constructions","Technologies","Retail","Jewellers","Opticals",
             "Stationery","Diagnostics","Travels","Caterers","Sweets","Garments","Cafe","Stores"]

GENERIC_NAME_TEMPLATES = ["Krishna Traders","City Electronics","Metro Stores","Sunrise Enterprises",
                           "National Traders","Om Sai Enterprises"]

SUFFIX_MAP = {
    "India":  {"Private Limited": ["Pvt Ltd","Pvt. Ltd.","Private Ltd","P Ltd"],
               "Limited": ["Ltd","Ltd."], "and Sons": ["& Sons"],
               "and Company": ["& Co","& Company","and Co"]},
    "US":     {"LLC": ["L.L.C.","L L C"], "Incorporated": ["Inc","Inc."],
               "Corporation": ["Corp","Corp."], "and Company": ["& Co","& Company"]},
    "France": {"SARL": ["S.A.R.L."], "and Fils": ["& Fils"], "SA": ["S.A."]},
}

INDIA_CITIES = {"Bengaluru": ["Bangalore","Bengaluru"], "Mumbai": ["Mumbai","Bombay"],
                 "Delhi": ["Delhi","New Delhi"], "Chennai": ["Chennai","Madras"],
                 "Kolkata": ["Kolkata","Calcutta"], "Hyderabad": ["Hyderabad"],
                 "Pune": ["Pune","Poona"], "Ahmedabad": ["Ahmedabad"]}
US_CITIES    = {"New York": ["New York","NYC","New York City"], "Chicago": ["Chicago"],
                 "Austin": ["Austin"], "Houston": ["Houston"], "Denver": ["Denver"],
                 "Seattle": ["Seattle"], "Boston": ["Boston"], "Atlanta": ["Atlanta"]}
FRANCE_CITIES = {"Paris": ["Paris"], "Lyon": ["Lyon"], "Marseille": ["Marseille"], "Toulouse": ["Toulouse"]}

INDIA_ROADS = ["MG Road","Church Street","Anna Salai","Brigade Road","Park Street",
               "Ring Road","Station Road","Gandhi Nagar","Nehru Street","Linking Road"]
US_STREETS  = ["Main","Oak","Maple","Cedar","Pine","Elm","Washington","Lincoln","Park","Highland"]
US_TYPES    = {"Street":"St","Road":"Rd","Avenue":"Ave","Boulevard":"Blvd","Drive":"Dr","Lane":"Ln"}
FRANCE_ROADS = ["Rue de la Paix","Avenue des Champs","Boulevard Voltaire","Rue Victor Hugo","Rue de Rivoli"]
LANDMARKS   = ["Near SBI ATM","Near City Mall","Opp Bus Stand","Near Central Park","Behind Railway Station"]

# ───────────────────────── NAME / ADDRESS BUILDERS ─────────────────────────

def build_name(country):
    if country == "India":
        core = f"{random.choice(INDIA_NAMES)} {random.choice(BIZ_WORDS)}"
    elif country == "US":
        core = f"{random.choice(US_NAMES)} {random.choice(BIZ_WORDS)}"
    else:
        core = f"{random.choice(FRANCE_NAMES)} {random.choice(BIZ_WORDS)}"
    suffix = random.choice(list(SUFFIX_MAP[country].keys()))
    return f"{core} {suffix}"


def build_address(country):
    if country == "India":
        city = random.choice(list(INDIA_CITIES.keys()))
        door = random.randint(1, 999)
        road = random.choice(INDIA_ROADS)
        pin = random.randint(400001, 700099)
        return f"{door} {road}, {city} {pin}", city
    elif country == "US":
        num = random.randint(100, 9999)
        street = random.choice(US_STREETS)
        stype = random.choice(list(US_TYPES.keys()))
        city = random.choice(list(US_CITIES.keys()))
        zip_ = random.randint(10000, 99999)
        return f"{num} {street} {stype}, {city} {zip_}", city
    else:
        num = random.randint(1, 200)
        road = random.choice(FRANCE_ROADS)
        city = random.choice(list(FRANCE_CITIES.keys()))
        pcode = random.randint(75000, 75020)
        return f"{num} {road}, {city} {pcode}", city


def make_entity(country):
    name = build_name(country)
    addr, city = build_address(country)
    return {"business_name": name, "business_address": addr, "country": country, "_city": city}


# ───────────────────────── NOISE FUNCTIONS ─────────────────────────

def typo(s):
    if len(s) < 4:
        return s
    i = random.randint(0, len(s) - 2)
    op = random.choice(["swap", "delete", "dup", "sub"])
    if op == "swap":
        return s[:i] + s[i+1] + s[i] + s[i+2:]
    if op == "delete":
        return s[:i] + s[i+1:]
    if op == "dup":
        return s[:i] + s[i] + s[i:]
    return s[:i] + random.choice("aeiou") + s[i+1:]


def random_case(s):
    return {"upper": s.upper(), "lower": s.lower(), "title": s.title(), "asis": s}[
        random.choice(["upper", "lower", "title", "asis"])]


def punctuation_noise(s):
    if random.random() < 0.5:
        s = s.replace(",", "")
    if random.random() < 0.2:
        s = s.replace(" ", "  ", 1)
    return s


def transliterate(s):
    for a, b in [("oo", "u"), ("ph", "f"), ("Kapoor", "Kapur"), ("Bazaar", "Bazar")]:
        if a in s and random.random() < 0.5:
            s = s.replace(a, b, 1)
    return s


def char_sub(s):
    subs = {"O": "0", "l": "1", "S": "5"}
    return "".join(subs.get(c, c) if random.random() < 0.08 else c for c in s)


def abbreviate_suffix(name, country):
    for canon, variants in SUFFIX_MAP[country].items():
        if canon in name and random.random() < 0.75:
            return name.replace(canon, random.choice(variants))
    return name


def abbreviate_street(addr):
    for full, ab in US_TYPES.items():
        if full in addr and random.random() < 0.6:
            addr = addr.replace(full, ab)
    return addr


def drop_component(addr, p_city=0.15, p_code=0.2):
    parts = addr.split(",")
    if len(parts) > 1 and random.random() < p_city:
        parts = [parts[0]]
    addr = ",".join(parts)
    if random.random() < p_code:
        addr = " ".join(w for w in addr.split() if not w.isdigit() or len(w) < 4)
    return addr


def use_landmark(addr, country):
    if country == "India" and random.random() < 0.15:
        return f"{random.choice(LANDMARKS)}, {addr.split(',')[-1].strip()}"
    return addr


def reorder_name_tokens(name):
    tokens = name.split()
    if len(tokens) > 2:
        random.shuffle(tokens)
    return " ".join(tokens)


def inject_irrelevant_name_token(name):
    tokens = name.split()
    tokens.insert(random.randint(0, len(tokens)),
                  random.choice(["branch", "online", "official", "group", "holdings", "shop"]))
    return " ".join(tokens)


def inject_irrelevant_address_detail(addr, country):
    extras = ["Unit 4", "Floor 2", "Building B", "Near Market", "Block C"]
    if country == "India" and random.random() < 0.5:
        extras.append(random.choice(LANDMARKS))
    return f"{addr}, {random.choice(extras)}"


def remove_or_duplicate_spaces(value):
    if random.random() < 0.5:
        return value.replace(" ", "")
    return value.replace(" ", "  ")


def noisy_name(name, country, level, profile="standard"):
    ops = [lambda s: abbreviate_suffix(s, country), random_case, punctuation_noise,
           typo, transliterate, char_sub, reorder_name_tokens,
           inject_irrelevant_name_token, remove_or_duplicate_spaces]
    n_ops = {"easy": 1, "medium": 2, "hard": 4}[level]
    if profile == "aggressive":
        n_ops = {"easy": 2, "medium": 4, "hard": 6}[level]
    for op in random.sample(ops, min(n_ops, len(ops))):
        name = op(name)
    return name


def noisy_address(addr, country, level, profile="standard"):
    ops = [abbreviate_street, punctuation_noise, typo, char_sub,
           lambda a: drop_component(a), lambda a: use_landmark(a, country),
           lambda a: inject_irrelevant_address_detail(a, country),
           remove_or_duplicate_spaces]
    n_ops = {"easy": 1, "medium": 2, "hard": 4}[level]
    if profile == "aggressive":
        n_ops = {"easy": 2, "medium": 4, "hard": 6}[level]
    for op in random.sample(ops, min(n_ops, len(ops))):
        addr = op(addr)
    return addr


def pick_noise_level():
    if NOISE_PROFILE == "aggressive":
        return random.choices(["easy", "medium", "hard"], weights=[0.2, 0.4, 0.4])[0]
    return random.choices(["easy", "medium", "hard"], weights=[0.4, 0.4, 0.2])[0]


# ───────────────────────── S1 GENERATION WITH FORCED COLLISIONS ─────────────────────────

def generate_s1_pool(n, country_weights, used_signatures):
    pool = []
    for _ in range(n):
        country = random.choices(list(country_weights.keys()), weights=list(country_weights.values()))[0]
        e = make_entity(country)
        sig = (e["business_name"], e["business_address"])
        while sig in used_signatures:
            e = make_entity(country)
            sig = (e["business_name"], e["business_address"])
        used_signatures.add(sig)
        pool.append(e)

    # Force generic-name collisions
    n_generic = int(len(pool) * GENERIC_NAME_FRACTION)
    for idx in random.sample(range(len(pool)), n_generic):
        pool[idx]["business_name"] = f"{random.choice(GENERIC_NAME_TEMPLATES)} " \
                                      f"{random.choice(list(SUFFIX_MAP[pool[idx]['country']].keys()))}"

    # Force shared-address / different-business collisions
    n_shared = int(len(pool) * SHARED_ADDRESS_FRACTION)
    for idx in random.sample(range(len(pool)), n_shared):
        donor = random.choice(pool)
        if donor is not pool[idx]:
            pool[idx]["business_address"] = donor["business_address"]

    return pool


# ───────────────────────── MATCH / DECOY / DISTRACTOR GENERATION ─────────────────────────

def match_plan():
    return random.choices([0, 1, 2, 3, 4], weights=[25, 45, 20, 7, 3])[0]


def generate_matches_and_decoys(s1_pool):
    s2_records, s3_records = [], []
    gt = {}  # index in s1_pool -> list of ("S2"/"S3", record)

    for i, biz in enumerate(s1_pool):
        n_matches = match_plan()
        gt[i] = []
        for _ in range(n_matches):
            level = pick_noise_level()
            variant = {
                "business_name": noisy_name(biz["business_name"], biz["country"], level, NOISE_PROFILE),
                "business_address": noisy_address(biz["business_address"], biz["country"], level, NOISE_PROFILE),
                "country": biz["country"],
            }
            target = random.choice(["S2", "S3"])
            (s2_records if target == "S2" else s3_records).append(variant)
            gt[i].append((target, len(s2_records) - 1 if target == "S2" else len(s3_records) - 1))

        # Hard negative: same name, materially different address (NOT linked in gt)
        if random.random() < DECOY_SAME_NAME_FRACTION:
            decoy_addr, _ = build_address(biz["country"])
            decoy = {"business_name": biz["business_name"], "business_address": decoy_addr, "country": biz["country"]}
            (s2_records if random.random() < 0.5 else s3_records).append(decoy)

        # Hard negative: same/near address, different business (NOT linked in gt)
        if random.random() < DECOY_SAME_ADDR_FRACTION:
            decoy = {"business_name": build_name(biz["country"]),
                     "business_address": biz["business_address"], "country": biz["country"]}
            (s2_records if random.random() < 0.5 else s3_records).append(decoy)

    return s2_records, s3_records, gt


def pad_distractors(records, target_total, country_weights):
    needed = max(0, target_total - len(records))
    for _ in range(needed):
        country = random.choices(list(country_weights.keys()), weights=list(country_weights.values()))[0]
        e = make_entity(country)
        rec = {"business_name": e["business_name"], "business_address": e["business_address"], "country": country}
        if random.random() < 0.05:  # realistic missing values
            rec["business_address"] = ""
        records.append(rec)
    return records


# ───────────────────────── ID ASSIGNMENT + WRITE ─────────────────────────

def assign_ids(records, prefix):
    for i, r in enumerate(records, start=1):
        r["entity_id"] = f"{prefix}-{i:06d}"
    return records


def write_tsv(path, records, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(columns)
        for r in records:
            w.writerow([r.get(c, "") for c in columns])


def write_ground_truth(path, s1_records, gt, s2_records, s3_records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for i, biz in enumerate(s1_records):
            ids = []
            for target, idx in gt[i]:
                pool = s2_records if target == "S2" else s3_records
                ids.append(pool[idx]["entity_id"])
            w.writerow([biz["entity_id"], ",".join(ids)])


# ───────────────────────── BUILD ONE SPLIT (TRAIN or TEST) ─────────────────────────

def build_split(n_s1, target_s2, target_s3, country_weights, used_sig, s1_prefix, s2_prefix, s3_prefix):
    s1_pool = generate_s1_pool(n_s1, country_weights, used_sig)
    s2_records, s3_records, gt = generate_matches_and_decoys(s1_pool)
    s2_records = pad_distractors(s2_records, target_s2, country_weights)
    s3_records = pad_distractors(s3_records, target_s3, country_weights)
    assign_ids(s1_pool, s1_prefix)
    assign_ids(s2_records, s2_prefix)
    assign_ids(s3_records, s3_prefix)
    return s1_pool, s2_records, s3_records, gt


# ───────────────────────── VALIDATION ─────────────────────────

def validate(train_s1, train_s2, train_s3, train_gt_rows, test_s1, test_s2, test_s3):
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    for name, path in [("train_source1", f"{OUT_TRAIN}/train_source1.tsv"),
                        ("train_source2", f"{OUT_TRAIN}/train_source2.tsv"),
                        ("train_source3", f"{OUT_TRAIN}/train_source3.tsv"),
                        ("train_ground_truth", f"{OUT_TRAIN}/train_ground_truth.tsv"),
                        ("test_source1", f"{OUT_TEST}/test_source1.tsv"),
                        ("test_source2", f"{OUT_TEST}/test_source2.tsv"),
                        ("test_source3", f"{OUT_TEST}/test_source3.tsv")]:
        check(os.path.exists(path), f"{name} exists at {path}")

    s1_ids = {r["entity_id"] for r in train_s1}
    s2_ids = {r["entity_id"] for r in train_s2}
    s3_ids = {r["entity_id"] for r in train_s3}
    check(all(i.startswith("S1-") for i in s1_ids), "all train S1 ids prefixed S1-")
    check(all(i.startswith("S2-") for i in s2_ids), "all train S2 ids prefixed S2-")
    check(all(i.startswith("S3-") for i in s3_ids), "all train S3 ids prefixed S3-")
    check(len(s1_ids) == len(train_s1), "no duplicate S1 ids")
    check(len(s2_ids) == len(train_s2), "no duplicate S2 ids")
    check(len(s3_ids) == len(train_s3), "no duplicate S3 ids")

    gt_s1_ids = {row[0] for row in train_gt_rows}
    check(gt_s1_ids.issubset(s1_ids), "every GT source1_entity_id exists in train_source1")
    referenced = set()
    for _, matched in train_gt_rows:
        if matched:
            referenced.update(matched.split(","))
    check(referenced.issubset(s2_ids | s3_ids), "every GT-referenced S2/S3 id exists in its source file")

    train_sigs = {(r["business_name"], r["business_address"]) for r in train_s1}
    test_sigs = {(r["business_name"], r["business_address"]) for r in test_s1}
    check(train_sigs.isdisjoint(test_sigs), "train/test S1 businesses are disjoint")

    n_with_match = sum(1 for _, matched in train_gt_rows if matched)
    n_singleton = len(train_gt_rows) - n_with_match
    check(n_with_match > 0 and n_singleton > 0, "GT contains both matches and singletons")

    return ok


# ───────────────────────── MAIN ─────────────────────────

def main():
    global OUT_TRAIN, OUT_TEST, NOISE_PROFILE

    parser = argparse.ArgumentParser(description="Generate synthetic entity-resolution data.")
    parser.add_argument("--train-s1-count", type=int, default=TRAIN_S1_COUNT)
    parser.add_argument("--test-s1-count", type=int, default=None,
                        help="Defaults to 35%% of train S1 count.")
    parser.add_argument("--s2-train-count", type=int, default=None)
    parser.add_argument("--s3-train-count", type=int, default=None)
    parser.add_argument("--s2-test-count", type=int, default=None)
    parser.add_argument("--s3-test-count", type=int, default=None)
    parser.add_argument("--train-dir", default=OUT_TRAIN)
    parser.add_argument("--test-dir", default=OUT_TEST)
    parser.add_argument("--noise-profile", choices=["standard", "aggressive"], default=NOISE_PROFILE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_s1_count <= 0:
        parser.error("--train-s1-count must be positive")
    test_s1_count = args.test_s1_count or math.ceil(args.train_s1_count * 0.35)
    target_s2_train = args.s2_train_count or math.ceil(args.train_s1_count * 2.2)
    target_s3_train = args.s3_train_count or math.ceil(args.train_s1_count * 2.2)
    target_s2_test = args.s2_test_count or math.ceil(test_s1_count * (800 / 350))
    target_s3_test = args.s3_test_count or math.ceil(test_s1_count * (800 / 350))
    if min(test_s1_count, target_s2_train, target_s3_train,
           target_s2_test, target_s3_test) <= 0:
        parser.error("all dataset counts must be positive")

    random.seed(args.seed)
    OUT_TRAIN = args.train_dir
    OUT_TEST = args.test_dir
    NOISE_PROFILE = args.noise_profile
    used_sig = set()

    train_s1, train_s2, train_s3, train_gt = build_split(
        args.train_s1_count, target_s2_train, target_s3_train,
        {"India": 0.5, "US": 0.5}, used_sig, "S1", "S2", "S3")

    test_s1, test_s2, test_s3, test_gt = build_split(
        test_s1_count, target_s2_test, target_s3_test,
        {"India": 0.45, "US": 0.40, "France": FRANCE_FRACTION_TEST}, used_sig, "S1", "S2", "S3")

    cols = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(f"{OUT_TRAIN}/train_source1.tsv", train_s1, cols)
    write_tsv(f"{OUT_TRAIN}/train_source2.tsv", train_s2, cols)
    write_tsv(f"{OUT_TRAIN}/train_source3.tsv", train_s3, cols)
    write_ground_truth(f"{OUT_TRAIN}/train_ground_truth.tsv", train_s1, train_gt, train_s2, train_s3)

    write_tsv(f"{OUT_TEST}/test_source1.tsv", test_s1, cols)
    write_tsv(f"{OUT_TEST}/test_source2.tsv", test_s2, cols)
    write_tsv(f"{OUT_TEST}/test_source3.tsv", test_s3, cols)
    # NOTE: no test ground truth file is written — matches the real competition (test has no labels)

    train_gt_rows = []
    with open(f"{OUT_TRAIN}/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1_id, matched = line.rstrip("\n").split("\t") if "\t" in line else (line.strip(), "")
            train_gt_rows.append((s1_id, matched))

    print("\n=== VALIDATION ===")
    passed = validate(train_s1, train_s2, train_s3, train_gt_rows, test_s1, test_s2, test_s3)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")

    n_singleton = sum(1 for _, m in train_gt_rows if not m)
    n_matched = len(train_gt_rows) - n_singleton
    n_pos_pairs = sum(len(m.split(",")) for _, m in train_gt_rows if m)

    print("\n=== STATISTICS ===")
    print(f"Train S1: {len(train_s1)} | S2: {len(train_s2)} | S3: {len(train_s3)}")
    print(f"Test  S1: {len(test_s1)} | S2: {len(test_s2)} | S3: {len(test_s3)}")
    print(f"S1 with matches: {n_matched} | Singletons: {n_singleton}")
    print(f"Positive matching pairs: {n_pos_pairs}")
    print(f"Noise profile: {NOISE_PROFILE}")
    print(f"Forced hard-negative decoys: ~{int(args.train_s1_count*(DECOY_SAME_NAME_FRACTION+DECOY_SAME_ADDR_FRACTION))} in train")
    print(f"Generic-name collisions: ~{int(args.train_s1_count*GENERIC_NAME_FRACTION)} | Shared-address collisions: ~{int(args.train_s1_count*SHARED_ADDRESS_FRACTION)}")


if __name__ == "__main__":
    main()