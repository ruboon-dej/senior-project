"use client";

import Link from "next/link";
import { useState } from "react";
import { translateThSLToThai, ApiError } from "@/lib/api";
import AttachFileModal from "@/components/AttachFileModal";
import CameraCaptureModal from "@/components/CameraCaptureModal";

export default function ThSLToThaiPage() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultText, setResultText] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAttachModal, setShowAttachModal] = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);

  async function handleVideo(file: File | Blob) {
    setPreviewUrl(URL.createObjectURL(file));
    setResultText(null);
    setIsLoading(true);
    setError(null);
    try {
      const result = await translateThSLToThai(file);
      setResultText(result.text);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not reach the translation backend."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      <Link href="/" className="text-sm text-gray-500 hover:underline">
        &larr; Back
      </Link>

      <h1 className="text-xl font-semibold mt-2 mb-6">ThSL → Thai</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Input panel: attach file or record */}
        <div className="rounded-2xl border bg-white p-4 flex flex-col min-h-[320px]">
          <div className="text-xs font-medium text-gray-400 mb-2">
            {previewUrl
              ? "ATTACHED VIDEO"
              : "PLEASE CAPTURE A VIDEO OR ATTACH A FILE"}
          </div>
          <div className="flex-1 rounded-xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center overflow-hidden">
            {previewUrl ? (
              <video
                src={previewUrl}
                controls
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center text-gray-400 px-6">
                <div className="text-3xl mb-2">📎</div>
                <p className="text-sm">
                  Attach a video or record one with your camera
                </p>
              </div>
            )}
          </div>
          <div className="flex items-center justify-between mt-2 pt-2 border-t">
            <span className="text-xs text-gray-400">MP4 / MOV, max 15s</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowAttachModal(true)}
                title="Attach a video file"
                className="w-9 h-9 rounded-full flex items-center justify-center border bg-gray-50 text-gray-500 hover:bg-gray-100"
              >
                📎
              </button>
              <button
                onClick={() => setShowCameraModal(true)}
                title="Record with camera"
                className="w-9 h-9 rounded-full flex items-center justify-center border bg-gray-50 text-gray-500 hover:bg-gray-100"
              >
                📷
              </button>
            </div>
          </div>
        </div>

        {/* Output panel */}
        <div className="rounded-2xl border bg-white p-4 flex flex-col min-h-[320px]">
          <div className="text-xs font-medium text-gray-400 mb-2">
            THAI OUTPUT
          </div>
          <div className="flex-1 rounded-xl bg-gray-50 p-4 flex items-center justify-center text-center">
            {isLoading && (
              <span className="text-gray-400 text-sm animate-pulse">
                Recognizing signs…
              </span>
            )}
            {!isLoading && !resultText && (
              <span className="text-gray-300 text-sm">
                Recognized Thai text will appear here
              </span>
            )}
            {!isLoading && resultText && (
              <p className="text-lg">{resultText}</p>
            )}
          </div>
        </div>
      </div>

      {error && (
        <p className="mt-4 text-sm text-red-500">
          {error} — showing UI only; connect the backend to enable real
          translation.
        </p>
      )}

      <AttachFileModal
        open={showAttachModal}
        onClose={() => setShowAttachModal(false)}
        accept="video/mp4,video/quicktime"
        title="Attach a file (MP4 and MOV only)"
        note="The video could only be at most 15 seconds long."
        maxDurationSeconds={15}
        onFileSelected={handleVideo}
      />

      <CameraCaptureModal
        open={showCameraModal}
        onClose={() => setShowCameraModal(false)}
        onCapture={handleVideo}
      />
    </div>
  );
}
