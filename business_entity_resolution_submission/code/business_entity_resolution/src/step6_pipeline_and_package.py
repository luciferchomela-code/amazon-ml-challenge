import os, sys, zipfile, subprocess, argparse

DOC_TEMPLATE_PATH = "Documentation_template.md"


def generate_requirements():
    with open("requirements.txt", "w") as f:
        f.write("polars>=0.20.0\npyarrow>=14.0.0\nflashtext>=2.7\njellyfish>=1.0.3\n"
                "rapidfuzz>=3.0.0\nscikit-learn>=1.3.0\nscipy>=1.11.0\nlightgbm>=4.0.0\n"
                "pandas>=2.0.0\nnumpy>=1.24.0\n")


def generate_readme():
    with open("README.md", "w") as f:
        f.write("# Business Entity Resolution Pipeline\n\n"
                "```bash\npip install -r requirements.txt\n"
                "python src/step6_pipeline_and_package.py --run-all --train-dir dataset/train --test-dir dataset/test\n```\n")


def run_full_pipeline(train_dir, test_dir, force_reload=False,
                      cache_dir="output/cache", max_cands=35,
                      output_dir="output"):
    import step1_ingestion, step2_normalization, step3_blocking, step4_features, step5_model
    d = step1_ingestion.run_ingestion(
        train_dir, test_dir, cache_dir=cache_dir, force_reload=force_reload)
    d = step2_normalization.run_normalization(
        d, cache_dir=cache_dir, force_reload=force_reload)
    d = step3_blocking.run_blocking(d, cache_dir=cache_dir, max_cands=max_cands)
    d = step4_features.run_feature_engineering(
        d, cache_dir=cache_dir, force_reload=force_reload)
    os.makedirs(output_dir, exist_ok=True)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    if os.path.exists("output/candidate_pairs.tsv") and os.path.abspath("output/candidate_pairs.tsv") != os.path.abspath(candidate_path):
        import shutil
        shutil.copyfile("output/candidate_pairs.tsv", candidate_path)

    s1_train_cache = os.path.join(cache_dir, "train_source1.parquet")
    gt_cache = os.path.join(cache_dir, "train_ground_truth.parquet")
    model, best_thresh = step5_model.train_model(
        os.path.join(cache_dir, "features_train"),
        s1_train_path=s1_train_cache, gt_path=gt_cache)
    step5_model.predict_and_export(
        model, best_thresh, os.path.join(cache_dir, "features_test"),
        d["s1_test"], os.path.join(output_dir, "matching_results.tsv"))


def execute_competition_validator(output_dir="output", test_dir="dataset/test",
                                  validator_path="utils/validate_submission.py"):
    if os.path.exists(validator_path):
        cmd = [sys.executable, validator_path,
               "--matching", os.path.join(output_dir, "matching_results.tsv"),
               "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
               "--test-dir", test_dir]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit("[FATAL] Validation FAILED.")
        print("[SUCCESS] Validator Passed.")
    else:
        print("[WARN] utils/validate_submission.py not found.")


def create_submission_zip():
    zip_name = "business_entity_resolution_submission.zip"
    if not os.path.exists("output/matching_results.tsv") or not os.path.exists("output/candidate_pairs.tsv"):
        sys.exit("[FATAL] Missing TSV outputs. Run --run-all first.")

    generate_requirements()
    generate_readme()

    # FIX #6: methodology doc is mandatory at zip root
    if not os.path.exists(DOC_TEMPLATE_PATH):
        sys.exit(f"[FATAL] '{DOC_TEMPLATE_PATH}' not found at project root — required by the rules.")

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write("output/matching_results.tsv", "output/matching_results.tsv")
        zf.write("output/candidate_pairs.tsv", "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution"
        zf.write("requirements.txt", f"{base}/requirements.txt")
        zf.write("README.md", f"{base}/README.md")
        for root, _, files in os.walk("src"):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    zf.write(os.path.join(root, file), f"{base}/src/{file}")
        zf.write(DOC_TEMPLATE_PATH, "Documentation_template.md")

    print(f"[SUCCESS] Created {zip_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--max-cands", type=int, default=35)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if args.run_all:
        run_full_pipeline(
            args.train_dir, args.test_dir, args.force_reload,
            cache_dir=args.cache_dir, max_cands=args.max_cands,
            output_dir=args.output_dir)
    execute_competition_validator(args.output_dir, args.test_dir)
    if args.output_dir == "output":
        create_submission_zip()
