"use client";

import { useEffect, useRef, useState } from "react";
import Modal from "./Modal";

interface CameraCaptureModalProps {
  open: boolean;
  onClose: () => void;
  onCapture: (blob: Blob) => void;
}

export default function CameraCaptureModal({
  open,
  onClose,
  onCapture,
}: CameraCaptureModalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [status, setStatus] = useState<
    "starting" | "ready" | "recording" | "denied"
  >("starting");

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    setStatus("starting");

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStatus("ready");
      })
      .catch(() => setStatus("denied"));

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [open]);

  function startRecording() {
    if (!streamRef.current) return;
    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: "video/webm",
    });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      onCapture(blob);
      handleClose();
    };
    recorder.start();
    recorderRef.current = recorder;
    setStatus("recording");
  }

  function stopRecording() {
    recorderRef.current?.stop();
  }

  function handleClose() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    onClose();
  }

  return (
    <Modal open={open} onClose={handleClose} fullScreen>
      <button
        onClick={handleClose}
        className="absolute top-4 left-4 z-10 text-white text-xl"
        aria-label="Back"
      >
        &larr;
      </button>

      <div className="flex-1 min-h-0 overflow-hidden relative flex items-center justify-center bg-gray-900">
        {status === "denied" && (
          <p className="text-white text-sm px-6 text-center">
            Camera access was denied. Please allow camera access and try
            again.
          </p>
        )}
        {status === "starting" && (
          <p className="text-white text-sm">Turning on camera…</p>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-contain ${
            status === "ready" || status === "recording" ? "block" : "hidden"
          }`}
        />
        {status === "recording" && (
          <span className="absolute top-4 right-4 flex items-center gap-2 text-white text-xs bg-red-500/80 px-2 py-1 rounded-full">
            <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
            Recording
          </span>
        )}
      </div>

      <div className="py-4 flex items-center justify-center bg-black">
        <button
          onClick={status === "recording" ? stopRecording : startRecording}
          disabled={status !== "ready" && status !== "recording"}
          className={`w-14 h-14 rounded-full border-4 border-white flex items-center justify-center disabled:opacity-30`}
          aria-label={status === "recording" ? "Stop recording" : "Start recording"}
        >
          <span
            className={
              status === "recording"
                ? "w-5 h-5 bg-red-500 rounded-sm"
                : "w-10 h-10 bg-red-500 rounded-full"
            }
          />
        </button>
      </div>
    </Modal>
  );
}
