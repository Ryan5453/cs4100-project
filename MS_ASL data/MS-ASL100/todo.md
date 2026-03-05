I upload the splited MS-ASL100 word set, and I took part0.
I might update details in each part after I finish them.

TO-DO:
1. Download videos from your part by url file
-I choose yt-dlp, by: 
yt-dlp -a ../splits/part_0_urls.txt -P videos_raw -o "%(id)s.%(ext)s"
2. Cut_clips use part_*_manifest.jsonl
3. Extract mediapipe keypoints, use both hands
4. Gather all parts of data and train first baseline
