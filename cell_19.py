print("=" * 75)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print(f"Results stored in catalog matching notebook name: '{RESULTS_DIR}'")
print("=" * 75)

artifacts = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")))
print(f"Found {len(artifacts)} saved artifacts in '{RESULTS_DIR}':")
for art in artifacts:
    size_kb = os.path.getsize(art) / 1024.0
    print(f"  - {os.path.basename(art)} ({size_kb:.1f} KB)")
print("=" * 75)
