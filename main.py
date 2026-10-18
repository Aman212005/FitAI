from __future__ import annotations
import os, math
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "FormFit Workout Pose Corrector"
app = FastAPI(title=APP_NAME, version="1.0.0")
origins_env = os.getenv("FRONTEND_ORIGINS", "*")
origins = ["*"] if origins_env.strip() == "*" else [x.strip() for x in origins_env.split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def joint_angle(a, b, c):
    ba, bc = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom == 0: return 0.0
    cosine = (ba[0]*bc[0] + ba[1]*bc[1]) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

class WorkoutAnalyzer:
    def __init__(self, exercise="Squats"):
        self.exercise = exercise if exercise in {"Squats", "Push-ups", "Bicep curls"} else "Squats"
        self.reps, self.stage, self.last_angle = 0, None, 0.0
        self.feedback = "Stand back so your body is visible"
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, min_detection_confidence=.55, min_tracking_confidence=.55)

    def close(self): self.pose.close()

    def point(self, landmarks, name, w, h):
        p = landmarks[getattr(mp_pose.PoseLandmark, name).value]
        return (p.x*w, p.y*h, p.visibility)

    def analyze(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        angle_value, rep_completed = None, False
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(82,216,137), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245,250,247), thickness=2, circle_radius=2))
            if self.exercise == "Squats":
                lv = min(self.point(lm,n,w,h)[2] for n in ("LEFT_HIP","LEFT_KNEE","LEFT_ANKLE"))
                rv = min(self.point(lm,n,w,h)[2] for n in ("RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"))
                side = "LEFT" if lv >= rv else "RIGHT"
                hip, knee, ankle = (self.point(lm,f"{side}_{n}",w,h) for n in ("HIP","KNEE","ANKLE"))
                angle_value = joint_angle(hip[:2], knee[:2], ankle[:2])
                if angle_value > 160:
                    rep_completed = self.stage == "down"
                    self.stage = "up"
                    self.feedback = "Rep complete — keep your movement controlled" if rep_completed else "Stand tall — prepare for the next rep"
                elif angle_value < 105:
                    if self.stage == "up":
                        self.stage, self.feedback = "down", "Good depth — now stand up"
                    else: self.feedback = "Push hips back and keep your balance"
                else: self.feedback = "Move steadily through the full range"
            elif self.exercise == "Push-ups":
                shoulder, elbow, wrist = (self.point(lm,f"LEFT_{n}",w,h) for n in ("SHOULDER","ELBOW","WRIST"))
                angle_value = joint_angle(shoulder[:2], elbow[:2], wrist[:2])
                if angle_value > 155:
                    rep_completed = self.stage == "down"
                    self.stage, self.feedback = "up", "Arms extended — lower with control"
                elif angle_value < 95:
                    if self.stage == "up": self.stage, self.feedback = "down", "Good lower position — push back up"
                    else: self.feedback = "Lower your chest a little more"
                else: self.feedback = "Keep your body aligned and move steadily"
            else:
                shoulder, elbow, wrist = (self.point(lm,f"LEFT_{n}",w,h) for n in ("SHOULDER","ELBOW","WRIST"))
                angle_value = joint_angle(shoulder[:2], elbow[:2], wrist[:2])
                if angle_value < 55: self.stage, self.feedback = "curled", "Squeeze at the top, then lower slowly"
                elif angle_value > 150:
                    rep_completed = self.stage == "curled"
                    self.stage, self.feedback = "extended", "Lower under control without swinging"
                else: self.feedback = "Keep your upper arm steady"
            if angle_value is not None:
                self.last_angle = float(angle_value)
                cv2.putText(frame, f"Joint angle: {int(angle_value)} deg", (18,34), cv2.FONT_HERSHEY_SIMPLEX,.72,(82,216,137),2,cv2.LINE_AA)
            if rep_completed: self.reps += 1
        else: self.feedback = "No pose detected — move back and make sure joints are visible"
        cv2.rectangle(frame,(12,48),(220,102),(14,30,24),-1)
        cv2.putText(frame,self.exercise,(22,70),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,250,247),1,cv2.LINE_AA)
        cv2.putText(frame,f"REPS  {self.reps}",(22,94),cv2.FONT_HERSHEY_SIMPLEX,.72,(82,216,137),2,cv2.LINE_AA)
        return frame, {"type":"stats","reps":self.reps,"angle":round(self.last_angle),
            "feedback":self.feedback,"pose_detected":bool(result.pose_landmarks),
            "exercise":self.exercise,"stage":self.stage or "ready"}

@app.get("/")
def root(): return {"status":"ok","app":APP_NAME}
@app.get("/health")
def health(): return {"status":"healthy"}

@app.websocket("/ws/workout")
async def workout_socket(ws: WebSocket):
    await ws.accept()
    analyzer = None
    try:
        analyzer = WorkoutAnalyzer(ws.query_params.get("exercise","Squats"))
        await ws.send_json({"type":"status","message":"Connected"})
        while True:
            payload = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"type":"error","message":"Invalid image frame"})
                continue
            h,w = frame.shape[:2]
            if max(h,w)>960:
                scale=960/max(h,w); frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
            processed, stats = analyzer.analyze(frame)
            ok, jpg = cv2.imencode(".jpg",processed,[int(cv2.IMWRITE_JPEG_QUALITY),78])
            if ok:
                await ws.send_json(stats)
                await ws.send_bytes(jpg.tobytes())
    except WebSocketDisconnect: pass
    except Exception as exc:
        try: await ws.send_json({"type":"error","message":f"Processing error: {type(exc).__name__}"})
        except Exception: pass
    finally:
        if analyzer: analyzer.close()
