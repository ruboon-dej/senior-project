"use client";

import Link from "next/link";
import { Suspense, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { translateThaiToThSL, ApiError } from "@/lib/api";
import AttachFileModal from "@/components/AttachFileModal";

function ThaiToThSLPageInner() {
  const searchParams = useSearchParams();
  const characterId = searchParams.get("character");

  const [inputText, setInputText] = useState("");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [glosses, setGlosses] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [showAttachModal, setShowAttachModal] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function runTranslation(text: string) {
    if (!text.trim()) {
      setVideoUrl(null);
      setGlosses([]);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const result = await translateThaiToThSL(text);
      setVideoUrl(result.videoUrl);
      setGlosses(result.glosses);
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

  function handleChange(value: string) {
    setInputText(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runTranslation(value), 600);
  }

  function handleMicClick() {
    // Placeholder for Web Speech API integration.
    // Real implementation: use SpeechRecognition (browser) and feed the
    // transcript into handleChange().
    setIsRecording((prev) => !prev);
  }

  function handlePdfSelected(file: File) {
    // TODO: replace with a real call that extracts text from the PDF
    // (e.g. send it to the backend and OCR/parse the written section).
    const mockExtractedText = `(Extracted from "${file.name}" — mock text)`;
    handleChange(mockExtractedText);
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-2">
        <Link href="/" className="text-sm text-gray-500 hover:underline">
          &larr; Back
        </Link>
        {characterId && (
          <span className="text-xs text-gray-400">
            Signing as: Character {characterId}
          </span>
        )}
      </div>

      <h1 className="text-xl font-semibold mb-6">Thai → ThSL</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Input panel */}
        <div className="rounded-2xl border bg-white p-4 flex flex-col min-h-[320px]">
          <div className="text-xs font-medium text-gray-400 mb-2">INPUT</div>
          <textarea
            className="flex-1 resize-none outline-none text-lg placeholder:text-gray-300"
            placeholder="พิมพ์ข้อความภาษาไทยที่นี่..."
            value={inputText}
            onChange={(e) => handleChange(e.target.value)}
          />
          <div className="flex items-center justify-between mt-2 pt-2 border-t">
            <span className="text-xs text-gray-400">
              {inputText.length} characters
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowAttachModal(true)}
                title="Attach a PDF"
                className="w-9 h-9 rounded-full flex items-center justify-center border bg-gray-50 text-gray-500 hover:bg-gray-100"
              >
                📎
              </button>
              <button
                onClick={handleMicClick}
                title="Speak instead of typing"
                className={`w-9 h-9 rounded-full flex items-center justify-center border transition ${
                  isRecording
                    ? "bg-red-500 text-white border-red-500"
                    : "bg-gray-50 text-gray-500 hover:bg-gray-100"
                }`}
              >
                🎤
              </button>
            </div>
          </div>
        </div>

        {/* Output panel */}
        <div className="rounded-2xl border bg-white p-4 flex flex-col min-h-[320px]">
          <div className="text-xs font-medium text-gray-400 mb-2">
            EMBEDDED VIDEO PLAYER
          </div>
          <div className="flex-1 rounded-xl bg-gray-900 flex items-center justify-center overflow-hidden relative">
            {isLoading && (
              <span className="text-white text-sm animate-pulse">
                Generating sign video…
              </span>
            )}
            {!isLoading && !videoUrl && (
              <span className="text-gray-400 text-sm px-6 text-center">
                Your translated ThSL video will appear here
              </span>
            )}
            {!isLoading && videoUrl && (
              <video
                key={videoUrl}
                src={videoUrl}
                controls
                autoPlay
                className="w-full h-full object-contain"
              />
            )}
          </div>
          <div className="flex items-center justify-between mt-2 pt-2 border-t">
            <span className="text-xs text-gray-400">
              {glosses.length > 0
                ? `Glosses: ${glosses.join(" · ")}`
                : "Speaker plays generated audio (if available)"}
            </span>
            <button
              title="Play audio"
              className="w-9 h-9 rounded-full flex items-center justify-center border bg-gray-50 text-gray-500 hover:bg-gray-100"
            >
              🔊
            </button>
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
        accept="application/pdf"
        title="Attach a file (PDF only)"
        note="Only the written section will be scanned."
        onFileSelected={handlePdfSelected}
      />
    </div>
  );
}

export default function ThaiToThSLPage() {
  return (
    <Suspense fallback={null}>
      <ThaiToThSLPageInner />
    </Suspense>
  );
}
