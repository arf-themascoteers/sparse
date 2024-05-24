import pandas as pd
import os

for f in os.listdir("../data"):
    d = os.path.join("../data", f)
    main_df = pd.read_csv(d)
    print(d, len(main_df), len(main_df.columns))