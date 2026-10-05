# User preprocessing integration

AquaGuard has no built-in/default preprocessing recipe. Install the user's exact, trusted Python implementation as `backend/models/user_preprocessing_pipeline.joblib`. The artifact must expose `preprocess_frame(frame: pandas.DataFrame) -> pandas.DataFrame`, preserve source row indexes/order for retained records, and return processed features plus target needed for supervised experiments.

The app executes this local artifact on a deep copy only. It does not invent or append missing-value deletion, imputation, duplicate removal, scaling, encoding, or other transforms. If the pipeline leaves incomplete selected inputs, AquaGuard blocks the experiment. Supervised train/test experiments require at least 12 complete processed rows.

The optional promotion interface additionally requires `transform_features(raw_frame) -> pandas.DataFrame` and exact ordered raw `input_features` metadata. Artifacts must be installed by a trusted operator; CSV uploads are never executed or deserialized.