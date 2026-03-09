MS-ASL100 Dataset Split

We divide the MS-ASL100 dataset into 4 parts.
Each member processes one part independently.

Each part contains about 4000 clips.

Current progress:
The full pipeline has been tested on part_0 and successfully runs end-to-end.

Pipeline includes:
1. Download videos
2. Cut sign clips
3. Extract MediaPipe hand landmarks (both hands)
4. Generate feature tensors
5. Train baseline model

Directory structure:

splits/
    part_*_urls.txt
    part_*_manifest.jsonl

Each member should run the following steps.

--------------------------------------------------

STEP 1: Download raw videos

Install yt-dlp.

Example:

yt-dlp -a ../splits/part_1_urls.txt -P videos_raw -o "%(id)s.%(ext)s"

--------------------------------------------------

STEP 2: Cut sign clips

Use:

python cut_clips.py

Input:
    videos_raw/
    part_*_manifest.jsonl

Output:
    clips/

--------------------------------------------------

STEP 3: Extract MediaPipe hand keypoints

Use:

python extract_all_clips_mp_tasks_dual.py

Output:
    features_mp/
    features_index.csv

Each clip becomes a tensor:

(64,126)

64 frames
126 features (both hands)

--------------------------------------------------

STEP 4: Train baseline model

Two baseline models are provided:

python train_gru_baseline.py
python train_gru_norm_vel.py

The improved model uses:

- landmark normalization
- velocity features

Current baseline result on part_0:

validation accuracy ≈ 0.16 when MIN_MAX_DETECT_RATE = 0.0 and 0.2
* Possibly because the data amount is too low for now.

--------------------------------------------------

Final Step:

After all parts are processed, we will merge:

features_index.csv
features_mp/

and train the final model for MS-ASL100.
The next step might be move on to larger data sets.
