from ultralytics import YOLO
import cv2
import numpy as np
import json
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, model, batch_size=20, max_frames=100, max_persons = 10, img_sz = 960, show_stream = True):
        self.model = model
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.output = None
        self.orig_shape = None
        self.keypoints = []
        self.keypoints_batch = []
        self.max_persons = max_persons
        self.max_frames = max_frames
        self.show_stream = show_stream


    def _convert_result_to_string(self, result):
        output = '['
        for i, r in enumerate(result):
            if i == self.max_frames:
                break
            output += r.tojson() + ','
        output = output[:-1] + ']'
        if self.show_stream:
            cv2.destroyAllWindows()
        return eval(output), r.orig_shape


    def _extract_keypoints(self, data):
        
        d = {}

        for i, frame in enumerate(data):
            track_id_keypoints = []

            for person_id in range(self.max_persons):
                if person_id >= len(frame):
                    track_id_keypoints = d.get(person_id, [])
                    track_id_keypoints.append(np.zeros(34))
                    d[person_id] = track_id_keypoints
                else:
                    keypoints = frame[person_id]["keypoints"]
                    x = np.array(keypoints["x"]) / self.orig_shape[1]
                    y = np.array(keypoints["y"]) / self.orig_shape[0]
                    XY = np.asarray([j for j in zip(x, y)]).reshape(34)

                    track_id_keypoints = d.get(person_id, [])
                    track_id_keypoints.append(XY)
                    d[person_id] = track_id_keypoints

        self.keypoints = np.asarray(list(d.values()))
        return self.keypoints


    def process_video(self, video_path):
        try:
            # print("Running inference:", end='\t')

            result = self.model.track(
                video_path, show=self.show_stream, save=False, stream=True,
                imgsz=self.img_sz, device='0', verbose=False,
                tracker="bytetrack.yaml", conf=0.5, iou=0.5
            )

            self.data, self.orig_shape = self._convert_result_to_string(result)
            
            
            self._extract_keypoints(self.data)
            
            return self.keypoints

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None
    

    def process_video_batch(self, video_paths):
        # Initialize tqdm with the total number of videos
        self.keypoints_batch = []
        skipped_id = []
        with tqdm(total=len(video_paths), desc='Processing Videos') as pbar:

            for i, video_path in enumerate(video_paths):
                keypoints = self.process_video(video_path)
                try:
                    if keypoints.shape[1] != self.max_frames:
                        skipped_id.append(i)
                        pbar.update(1)
                        pbar.set_postfix(skipped=f"Too short! (Skipped: {len(i)})")
                        continue
                except:
                    skipped_id.append(i)
                    pbar.update(1)
                    pbar.set_postfix(skipped=f"Too short! (Skipped: {len(skipped_id)})")
                    continue
                self.keypoints_batch.append(keypoints.tolist())

                # Update the progress bar
                pbar.update(1)

        return self.keypoints_batch, skipped_id
            
    def process_image(self, image):
        try:
            # print("Running inference:", end='\t')

            result = self.model.track(
                image, show=self.show_stream,
                imgsz=self.img_sz, device='0', verbose=False,
                tracker="bytetrack.yaml", conf=0.5, iou=0.5
            )

            self.data, self.orig_shape = self._convert_result_to_string(result)
            
            
            self._extract_keypoints(self.data)
            
            return self.keypoints

        except Exception as e:
            print(f"Error processing Image: {str(e)}")
            return None
        
# Example usage:
# video_path = 'examples/videos/nv143.MOV'

# model = YOLO('models/yolov8x-pose-p6.pt') 
# video_processor = VideoProcessor(model)
# # batch = video_processor.process_video(video_path)
# batch = video_processor.process_video_batch([video_path, video_path])

# with open("batch.json", 'w') as f:
#     json.dump(batch, f, indent=2)
