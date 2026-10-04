"use client";

import { useRef, useState } from "react";
import Modal from "./Modal";

interface AttachFileModalProps {
  open: boolean;
  onClose: () => void;
  /** e.g. "application/pdf" or "video/mp4,video/quicktime" */
  accept: string;
  title: string;
  note: string;
  /** If set, video files longer than this (in seconds) are rejected client-side. */
  maxDurationSeconds?: number;
  onFileSelected: (file: File) => void;
}

export default function AttachFileModal({
  open,
  onClose,
  accept,
  title,
  note,
  maxDurationSeconds,
  onFileSelected,
}: AttachFileModalProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  function checkDurationThenAccept(file: File) {
    if (!maxDurationSeconds) {
      onFileSelected(file);
      onClose();
      return;
    }

    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      if (video.duration > maxDurationSeconds) {
        setError(
          `That video is ${video.duration.toFixed(
            1
          )}s long — please attach one under ${maxDurationSeconds}s.`
        );
      } else {
        onFileSelected(file);
        onClose();
      }
    };
    video.src = URL.createObjectURL(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    const file = e.target.files?.[0];
    if (file) checkDurationThenAccept(file);
    e.target.value = "";
  }

  return (
    <Modal open={open} onClose={onClose}>
      <button
        onClick={onClose}
        className="absolute top-3 right-3 text-gray-400 hover:text-gray-600"
        aria-label="Close"
      >
        ✕
      </button>
      <h2 className="font-medium mb-1">{title}</h2>
      <p className="text-xs text-gray-500 mb-4">{note}</p>
      {error && <p className="text-xs text-red-500 mb-3">{error}</p>}
      <button
        onClick={() => inputRef.current?.click()}
        className="px-4 py-2 rounded-full border text-sm hover:bg-gray-50"
      >
        Upload
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={handleChange}
      />
    </Modal>
  );
}
