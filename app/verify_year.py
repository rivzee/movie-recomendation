import sys
import os
import pandas as pd

# Add app directory to path
sys.path.append(os.path.join(os.getcwd(), 'app'))

try:
    from recommender import recommend
    
    print("\n--- Testing Year Filter: 'Action' + Year='2023' ---")
    matched_label, results, chart_path = recommend("Action", year="2023", top_n=5)
    
    if not results:
        print("WARNING: No results found for 2023.")
    
    success = True
    for res in results:
        print(f"Title: {res['title']} | Year: {res['year']}")
        if res['year'] != 2023:
            print(f"    ^ FAILURE: Expected 2023, got {res['year']}")
            success = False
            
    if success and results:
        print("\nSUCCESS: Only 2023 anime returned.")

except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
